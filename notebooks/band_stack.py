"""v16 band re-scorer: re-score every uncertain pair (final p in [0.01, 0.995), all pair kinds) with a LightGBM that adds
the fine-tuned bi-encoder's neighbour context (kaggle/biencoder: rank / cosine of the pair in the record -> S1 and
S1 -> record top-5 lists over ALL S1 of the country, best / 2nd / 5th cosines, gaps), the zero-shot e5 reverse
neighbours (kaggle/embed_rknn), name-key tie counts over all S1, and the four raw cross-encoder logits. Then the v13
decision layer (one owner per record, per-kind thresholds, empty-S1 rescue, empty-address rescue model) on top.

    python notebooks/band_stack.py                       # val: 5-fold OOF report
    python notebooks/band_stack.py --test --out submissions/v16

Val (fold 0 of split_v1): v15 0.99016 -> 0.99075 (5-fold OOF). Test: the 5 fold models are averaged.
"""
import os
import argparse
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
sys.path.insert(0, str(ROOT / "notebooks"))
ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true")
ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--lo", type=float, default=0.01)
ap.add_argument("--hi", type=float, default=0.995)
ap.add_argument("--no-fr", action="store_true", help="keep v15 probabilities for French pairs")
ap.add_argument("--bk", default="bknn", help="neighbour file: bknn (top-5 both directions) or bknn2 (reverse top-5 + forward top-15)")
ap.add_argument("--addr", action="store_true", help="+ address equality flags and the S1's confident-copy count")
ap.add_argument("--norm", type=float, default=0.0, help="record-level normalisation strength for the threshold decision (0 = off)")
ap.add_argument("--contest", action="store_true", help="+ the record's competing S1 probabilities (sum / max / count of other S1)")
ap.add_argument("--xcos", action="store_true", help="+ exact bi-encoder cosine of every band pair (kaggle/bcos_kernel -> artifacts/kaggle_bcos/)")
ap.add_argument("--seeds", type=int, default=1, help="average this many LightGBM seeds per fold")
ap.add_argument("--raw", action="store_true", help="+ raw pair similarities from the per-kind feature sets")
ap.add_argument("--ext5", action="store_true", help="add ext5 pairs (forward ranks 6-15, run_ext3.py --tag g) as kind 6")
a = ap.parse_args()
_argv, sys.argv = sys.argv, ["final_v13.py", "--blend", "ens1234", "--ext-tags", "e", "f", "--fr-ext2", "--fr-empty"]
import final_v13 as F  # noqa: E402  (v13/v15 frames, selection and reporting)
sys.argv = _argv
from src.evaluate import write_id_list_tsv  # noqa: E402
from src.train import to_pairs  # noqa: E402

A, R, K = F.A, F.R, F.A / "kaggle_ext2"
KEY = ["s1", "m", "src"]
num = F.num
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)  # noqa: E731
FEATS = (["pf", "p", "kind"]
         + ["bk_cos", "bk_rrank", "bk_frank", "bk_r1", "bk_r2", "bk_r5", "bk_f1", "bk_f2", "bk_f5", "bk_rgap", "bk_fgap", "bk_r12", "bk_f12", "bk_is_top"]
         + ["rk_rcos", "rk_rrank", "rk_r1", "rk_r2", "rk_r5", "rk_rgap", "rk_r12", "rk_is_top"]
         + ["k_n", "kl_n", "sk_n", "leq", "neq", "keq", "heq", "r_empty", "s_empty"]
         + ["ce1", "ce2", "ce3", "ce4"])
if a.addr:
    FEATS += ["aeq", "steq", "loeq", "s1_nhi", "sup_k", "sup_h", "sup_a"]
if a.contest:
    FEATS += ["c_sum_other", "c_max_other", "c_n_other", "c_is_max"]
if a.xcos:
    FEATS += ["xcos", "x_rgap", "x_fgap"]
RAW = ["n_jw", "n_tset", "a_tset", "a_street_tset", "a_loc_tset", "hv_eq", "hv_diff", "hv_lev", "ho_diff", "nm_miss1", "nm_extra2", "nm_subst", "n_acr"]
if a.raw:
    FEATS += RAW
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=5)


# ------------------------------------------------------------------ features
def neighbours(frame: pl.DataFrame, path: Path, pre: str, both: bool) -> pl.DataFrame:
    """Neighbour context from a top-5 list: dir 0 = record -> S1 (all S1 of the country), dir 1 = S1 -> record."""
    d = pl.read_parquet(path, columns=["s1", "m", "src", "cos", "rank"] + (["dir"] if both else [])).with_columns(pl.col("cos").cast(pl.Float32))
    if not both:
        d = d.with_columns(pl.lit(0, pl.UInt8).alias("dir"))
    rv = d.filter(pl.col("dir") == 0).join(frame.select("m", "src").unique(), on=["m", "src"], how="semi")
    fw = d.filter(pl.col("dir") == 1).join(frame.select("s1").unique(), on="s1", how="semi")
    del d
    rec = rv.group_by("m", "src").agg(pl.col("cos").filter(pl.col("rank") == 1).first().alias(f"{pre}_r1"),
                                      pl.col("cos").filter(pl.col("rank") == 2).first().alias(f"{pre}_r2"),
                                      pl.col("cos").filter(pl.col("rank") == 5).first().alias(f"{pre}_r5"),
                                      pl.col("s1").filter(pl.col("rank") == 1).first().alias("_top"))
    x = frame.join(rv.select(*KEY, pl.col("cos").alias(f"{pre}_rcos"), pl.col("rank").alias(f"{pre}_rrank")), on=KEY, how="left")
    x = x.join(rec, on=["m", "src"], how="left")
    cos = pl.col(f"{pre}_rcos")
    if both:
        s1s = fw.group_by("s1").agg(pl.col("cos").filter(pl.col("rank") == 1).first().alias(f"{pre}_f1"),
                                    pl.col("cos").filter(pl.col("rank") == 2).first().alias(f"{pre}_f2"),
                                    pl.col("cos").filter(pl.col("rank") == 5).first().alias(f"{pre}_f5"))
        x = x.join(fw.select(*KEY, pl.col("cos").alias(f"{pre}_fcos"), pl.col("rank").alias(f"{pre}_frank")), on=KEY, how="left").join(s1s, on="s1", how="left")
        x = x.with_columns(pl.coalesce(f"{pre}_rcos", f"{pre}_fcos").alias(f"{pre}_cos"))
        cos = pl.col(f"{pre}_cos")
        x = x.with_columns(pl.col(f"{pre}_frank").fill_null(20).cast(pl.Float32), (pl.col(f"{pre}_f1") - cos).alias(f"{pre}_fgap"),
                           (pl.col(f"{pre}_f1") - pl.col(f"{pre}_f2")).alias(f"{pre}_f12"))
    return x.with_columns(pl.col(f"{pre}_rrank").fill_null(20).cast(pl.Float32), (pl.col(f"{pre}_r1") - cos).alias(f"{pre}_rgap"),
                          (pl.col(f"{pre}_r1") - pl.col(f"{pre}_r2")).alias(f"{pre}_r12"),
                          (pl.col("_top") == pl.col("s1")).fill_null(False).cast(pl.Int8).alias(f"{pre}_is_top")).drop("_top")


def name_context(frame: pl.DataFrame, split: str) -> pl.DataFrame:
    """Tie counts over ALL S1 of the country (records' name key, key + legal form; the S1's own key) and equality flags."""
    cols = ["entity_id", "name_key", "name_legal", "name_norm", "addr_empty", "addr_house_no", "addr_key", "addr_street", "addr_localities"]
    s1n = pl.read_parquet(A / f"normalized/{split}_source1.parquet", columns=cols + ["country"]).with_columns(num("entity_id").alias("s1")).drop("entity_id")
    kc = s1n.group_by("country", "name_key").agg(pl.len().alias("k_n"))
    klc = s1n.group_by("country", "name_key", "name_legal").agg(pl.len().alias("kl_n"))
    need = frame.select("m", "src").unique()
    recs = pl.concat([pl.read_parquet(A / f"normalized/{split}_source{s}.parquet", columns=cols)
                        .with_columns(num("entity_id").alias("m"), pl.lit(s, pl.UInt8).alias("src")).join(need, on=["m", "src"], how="semi")
                        .select("m", "src", pl.col("name_key").alias("rk"), pl.col("name_legal").alias("rl"), pl.col("name_norm").alias("rn"),
                                pl.col("addr_empty").alias("r_empty"), pl.col("addr_house_no").alias("rh"), pl.col("addr_key").alias("ra"),
                                pl.col("addr_street").alias("rst"), pl.col("addr_localities").alias("rlo")) for s in (2, 3)])
    x = frame.join(s1n.select("s1", "country", pl.col("name_key").alias("sk"), pl.col("name_legal").alias("sl"), pl.col("name_norm").alias("sn"),
                              pl.col("addr_empty").alias("s_empty"), pl.col("addr_house_no").alias("sh"), pl.col("addr_key").alias("sa"),
                              pl.col("addr_street").alias("sst"), pl.col("addr_localities").alias("slo")), on="s1", how="left")
    x = x.join(recs, on=["m", "src"], how="left")
    x = (x.join(kc.rename({"name_key": "rk"}), on=["country", "rk"], how="left")
          .join(klc.rename({"name_key": "rk", "name_legal": "rl"}), on=["country", "rk", "rl"], how="left")
          .join(kc.rename({"name_key": "sk", "k_n": "sk_n"}), on=["country", "sk"], how="left"))
    x = x.with_columns(pl.col("k_n", "kl_n", "sk_n").fill_null(0),
                       (pl.col("sl") == pl.col("rl")).fill_null(False).cast(pl.Int8).alias("leq"),
                       (pl.col("sn") == pl.col("rn")).fill_null(False).cast(pl.Int8).alias("neq"),
                       (pl.col("sk") == pl.col("rk")).fill_null(False).cast(pl.Int8).alias("keq"),
                       (pl.col("sh") == pl.col("rh")).fill_null(False).cast(pl.Int8).alias("heq"),
                       pl.col("r_empty").cast(pl.Int8), pl.col("s_empty").cast(pl.Int8),
                       ((pl.col("sa") == pl.col("ra")) & (pl.col("ra").str.len_chars() > 0)).fill_null(False).cast(pl.Int8).alias("aeq"),
                       ((pl.col("sst") == pl.col("rst")) & (pl.col("rst").str.len_chars() > 0)).fill_null(False).cast(pl.Int8).alias("steq"),
                       ((pl.col("slo") == pl.col("rlo")) & (pl.col("rlo").str.len_chars() > 0)).fill_null(False).cast(pl.Int8).alias("loeq"))
    return x.drop("sk", "sl", "sn", "rk", "rl", "rn", "rh", "sh", "sa", "ra", "sst", "rst", "slo", "rlo")


def ce_logits(frame: pl.DataFrame, split: str) -> pl.DataFrame:
    """Raw logits of the 4 cross-encoders wherever they scored the pair (main/ext band, ext2, ext3, ext4); else null."""
    s = "val" if split == "train" else "test"
    main = None
    for name, d, pat in [("ce1", A / "kaggle_ce", "ce_{s}.parquet"), ("ce2", A / "kaggle_ce2", "ce2_{s}.parquet"),
                         ("ce3", A / "kaggle_ce2", "ce3_{s}.parquet"), ("ce4", A / "kaggle_ce2", "ce4_{s}.parquet")]:
        sc = (pl.read_parquet(d / f"{s}_keys.parquet").select("id", *KEY)
                .join(pl.read_parquet(d / pat.format(s=s)).rename({"logit": name}), on="id").drop("id"))
        main = sc if main is None else main.join(sc, on=KEY, how="full", coalesce=True)
    parts = [main]
    for T in ("", "e", "f"):
        keys = pl.read_parquet(K / f"keys_{T}{s}.parquet").filter(pl.col("split") == s).select("id", *KEY)
        for c in ("ce1", "ce2", "ce3", "ce4"):
            f = K / f"out_{T}{s}/{c}_ext2.parquet"
            keys = keys.join(pl.read_parquet(f).rename({"logit": c}), on="id", how="left") if f.exists() else keys.with_columns(pl.lit(None, pl.Float32).alias(c))
        parts.append(keys.drop("id"))
    allc = pl.concat([p.select(*KEY, *[pl.col(c).cast(pl.Float32) for c in ("ce1", "ce2", "ce3", "ce4")]) for p in parts], how="vertical_relaxed")
    return frame.join(allc.unique(KEY, keep="first"), on=KEY, how="left")


def s1_support(band: pl.DataFrame, frame: pl.DataFrame, split: str) -> pl.DataFrame:
    """Confident copies the S1 already has (pf >= 0.9, excluding this pair) and how many of them share the record's name
    key / house number / address key: stable between val and test (true copies per S1 are the same; only distractors
    double on test)."""
    conf = frame.filter(pl.col("pf") >= 0.9).select("s1", "m", "src")
    need = pl.concat([conf.select("m", "src"), band.select("m", "src")]).unique()
    attr = pl.concat([pl.read_parquet(A / f"normalized/{split}_source{s}.parquet", columns=["entity_id", "name_key", "addr_house_no", "addr_key"])
                        .with_columns(num("entity_id").alias("m"), pl.lit(s, pl.UInt8).alias("src")).join(need, on=["m", "src"], how="semi")
                        .select("m", "src", pl.col("name_key").alias("_k"), pl.col("addr_house_no").alias("_h"), pl.col("addr_key").alias("_a")) for s in (2, 3)])
    ca = conf.join(attr, on=["m", "src"], how="left")
    x = band.join(conf.group_by("s1").agg(pl.len().alias("s1_nhi")), on="s1", how="left").join(attr, on=["m", "src"], how="left")
    for c, name in (("_k", "sup_k"), ("_h", "sup_h"), ("_a", "sup_a")):
        g = ca.filter(pl.col(c).is_not_null() & (pl.col(c).str.len_chars() > 0)).group_by("s1", c).agg(pl.len().alias(name))
        x = x.join(g, on=["s1", c], how="left")
    self_hi = (pl.col("pf") >= 0.9).cast(pl.Int64)
    return x.with_columns((pl.col("s1_nhi").fill_null(0).cast(pl.Int64) - self_hi).alias("s1_nhi"),
                          *[(pl.col(n).fill_null(0).cast(pl.Int64) - self_hi).clip(0).alias(n) for n in ("sup_k", "sup_h", "sup_a")]).drop("_k", "_h", "_a")


def contest(band: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """The record's other S1 in the candidate frame (pre-re-scoring pf): sum, max, count >= 0.3, is this S1 the argmax."""
    r = frame.select("m", "src", "pf").group_by("m", "src").agg(pl.col("pf").sum().alias("_s"), pl.col("pf").max().alias("_mx"),
                                                               (pl.col("pf") >= 0.3).sum().alias("_n"), pl.col("pf").top_k(2).alias("_t2"))
    x = band.join(r, on=["m", "src"], how="left")
    second = pl.col("_t2").list.get(1, null_on_oob=True).fill_null(0.0)
    return x.with_columns((pl.col("_s") - pl.col("pf")).clip(0).alias("c_sum_other"),
                          pl.when(pl.col("pf") >= pl.col("_mx")).then(second).otherwise(pl.col("_mx")).alias("c_max_other"),
                          (pl.col("_n") - (pl.col("pf") >= 0.3).cast(pl.UInt32)).alias("c_n_other"),
                          (pl.col("pf") >= pl.col("_mx")).cast(pl.Int8).alias("c_is_max")).drop("_s", "_mx", "_n", "_t2")


def raw_pair(band: pl.DataFrame, split: str) -> pl.DataFrame:
    """Pair similarities from the feature set each kind was scored with (main, extension, ext2, ext3, ext4[, ext5])."""
    s = "val" if split == "train" else "test"
    sets = [R / f"{s}_set.parquet", R / f"ext_{s}_set.parquet"] + [R / f"ext2{t}_{s}_set.parquet" for t in ("", "e", "f") + (("g",) if a.ext5 else ())]
    parts = [pl.read_parquet(f, columns=KEY + RAW).join(band.select(KEY), on=KEY, how="semi") for f in sets]
    return band.join(pl.concat(parts, how="vertical_relaxed").unique(KEY, keep="first"), on=KEY, how="left")


def features(band: pl.DataFrame, split: str) -> pl.DataFrame:
    x = neighbours(band, A / f"kaggle_{a.bk}/{a.bk}_{split}.parquet", "bk", True)
    x = neighbours(x, A / f"kaggle_rknn/rknn_{split}.parquet", "rk", False)
    x = ce_logits(name_context(x, split), split)
    if a.xcos:
        xc = pl.read_parquet(A / f"kaggle_bcos/bcos_{'val' if split == 'train' else 'test'}.parquet").select(
            pl.col("s1").cast(pl.UInt32), pl.col("m").cast(pl.UInt32), pl.col("src").cast(pl.UInt8), pl.col("xcos").cast(pl.Float32))
        x = x.join(xc.unique(KEY), on=KEY, how="left").with_columns((pl.col("bk_r1") - pl.col("xcos")).alias("x_rgap"),
                                                                    (pl.col("bk_f1") - pl.col("xcos")).alias("x_fgap"))
    return raw_pair(x, split) if a.raw else x


def in_band(lo: float, hi: float) -> pl.Expr:
    """Uncertain pairs. Main/extension pairs with refiner p >= 0.999 were never cross-encoder-blended on val (the CE band
    stops there); on test only French above-band pairs (v10 blend) have them in [lo, hi): out of distribution, kept."""
    return (pl.col("pf") >= lo) & (pl.col("pf") < hi) & ~((pl.col("kind") <= 1) & (pl.col("p") >= 0.999))


def select(frame: pl.DataFrame) -> pl.DataFrame:
    """v13 decision layer (one owner per record by pf, per-kind thresholds, empty-S1 rescue at 0.5). With --norm the
    threshold / rescue decisions use p' = pf / max(1, pf + norm * (sum of the record's pf over its other S1)): a record
    has one owner, so a winner whose runner-up is also likely is a coin flip. Val: contested winners at p 0.75-0.9 are
    true only 45%; val sees only val S1 (20%) as competitors, test sees all (contests 3x US/IN, 18x France per S1)."""
    if not a.norm:
        return F.select(frame)
    p = frame.with_columns(pl.col("kind").replace_strict(F.THR, return_dtype=pl.Float64).alias("t"),
                           (pl.col("pf") / pl.max_horizontal(pl.lit(1.0), pl.col("pf") + a.norm * (pl.col("pf").sum().over("m", "src") - pl.col("pf")))).alias("pn"))
    own = p.filter(pl.col("pf") >= pl.col("pf").max().over("m", "src"))
    sel = own.filter(pl.col("pn") >= pl.col("t"))
    resc = own.join(sel.select("s1").unique(), on="s1", how="anti").sort("pn", descending=True).group_by("s1").head(1).filter(pl.col("pn") >= 0.5)
    return pl.concat([sel, resc]).select("s1", "m", "src", "kind")


def restack(frame: pl.DataFrame, band_scores: pl.DataFrame) -> pl.DataFrame:
    return (frame.join(band_scores, on=KEY, how="left").with_columns(pl.coalesce("ps", "pf").alias("pf")).drop("ps"))


if __name__ == "__main__":
    ids = pl.read_parquet(A / "splits/split_v1.parquet").filter(pl.col("role") == "val").select("s1_id", "country")
    truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
    tp = (truth.drop_nulls("match_id").select(num("s1_id").alias("s1"), num("match_id").alias("m"),
                                               pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src")).with_columns(pl.lit(1, pl.Int8).alias("y")))
    v = F.val_frame()
    v = F.add_ext2(v, R / "ext2b_val_oof.parquet", None)
    for i, tag in enumerate(["e", "f"]):
        v = F.add_ext2(v, R / f"ext2{tag}b_val_oof.parquet", None, kind=4 + i)
    if a.ext5:
        F.THR[6] = 0.75
        v = F.add_ext2(v, R / "ext2g_val_oof.parquet", None, kind=6)
    v = v.select("s1", "m", "src", "p", "pf", "kind")
    F.report("val v15 stack (no empty model)", F.select(v), truth, ids)
    b = features(v.filter(in_band(a.lo, a.hi)), "train")
    b = s1_support(b, v, "train")
    if a.contest:
        b = contest(b, v)
    b = b.join(tp, on=KEY, how="left").with_columns(pl.col("y").fill_null(0))
    X = b.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
    y = b["y"].to_numpy()
    log(f"val band {b.height:,} pairs, pos {int(y.sum()):,}, {len(FEATS)} features")
    fold = (b["s1"].hash(seed=31) % 5).to_numpy()
    inner = (b["s1"].hash(seed=37) % 10).to_numpy() == 0
    oof = np.zeros(len(y))
    models = []
    for k in range(5):
        fit, es = (fold != k) & ~inner, (fold != k) & inner
        for sd in range(a.seeds):
            prm = {**PARAMS, "seed": PARAMS["seed"] + 101 * sd}
            mdl = lgb.train(prm, lgb.Dataset(X[fit], y[fit], feature_name=FEATS), 3000, valid_sets=[lgb.Dataset(X[es], y[es])],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[fold == k] += mdl.predict(X[fold == k], num_iteration=mdl.best_iteration) / a.seeds
            mdl.save_model(str(A / f"models/lgb_bandstack{'_ext5' if a.ext5 else ''}_f{k}_s{sd}.txt"), num_iteration=mdl.best_iteration)
            models.append(mdl)
    from sklearn.metrics import roc_auc_score
    log(f"band AUC pf {roc_auc_score(y, b['pf'].to_numpy()):.5f} -> re-scored {roc_auc_score(y, oof):.5f}")
    vs = restack(v, b.select(KEY).with_columns(pl.Series("ps", oof)))
    del X, b
    sv = select(vs)
    F.report("val v16 (no empty model)", sv, truth, ids)
    tv = F.ER.top_candidates("train", vs.filter(pl.col("kind") < 2), sv, R / "val_set.parquet", R / "ext_val_set.parquet")
    tv = tv.join(tp, on=KEY, how="left").with_columns(pl.col("y").fill_null(0))
    if not a.test:
        q, _ = F.ER.fit_apply(tv, None)
        add = tv.filter(pl.Series(q >= 0.75)).select(KEY)
        F.report(f"val v16 final (+{add.height:,} empty rescue)", pl.concat([sv.select(KEY), add]), truth, ids)
        pl.concat([sv.select(*KEY, "kind"), add.with_columns(pl.lit(3, pl.UInt8).alias("kind"))]).write_parquet(R / "bandstack_val_sel.parquet")
        vs.select(*KEY, "p", "pf", "kind").write_parquet(R / "bandstack_val_frame.parquet")
        sys.exit(0)

    del v, vs
    s1_all = pl.read_parquet(A / "normalized/test_source1.parquet", columns=["entity_id", "country"])
    cty = s1_all.select(num("entity_id").alias("s1"), "country")
    t = F.test_frame()
    t = F.add_ext2(t, A / "test_scores_ext2b.parquet", cty)
    for i, tag in enumerate(["e", "f"]):
        t = F.add_ext2(t, A / f"test_scores_ext2{tag}b.parquet", cty, kind=4 + i)
    if a.ext5:
        t = F.add_ext2(t, A / "test_scores_ext2g.parquet", cty, kind=6)
    t = t.select("s1", "m", "src", "p", "pf", "kind")
    log(f"test frame {t.height:,} pairs")
    bt = t.filter(in_band(a.lo, a.hi))
    if a.no_fr:
        bt = bt.join(cty.filter(pl.col("country") == "France").select("s1"), on="s1", how="anti")
    bt = s1_support(features(bt, "test"), t, "test")
    if a.contest:
        bt = contest(bt, t)
    log(f"test band {bt.height:,} pairs; ce1 coverage {bt['ce1'].is_not_null().mean():.3f}, bk rev coverage {bt['bk_rcos'].is_not_null().mean():.3f}")
    Xt = bt.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
    pt = np.mean([m.predict(Xt) for m in models], axis=0)
    bt.with_columns(pl.Series("ps", pt)).join(cty, on="s1", how="left").write_parquet(R / f"bandstack{'_ext5' if a.ext5 else ''}_test.parquet")
    bt = bt.select(KEY).with_columns(pl.Series("ps", pt))
    del Xt
    t = restack(t, bt)
    st = select(t)
    tt = F.ER.top_candidates("test", t.filter(pl.col("kind") < 2), st, R / "test_set.parquet", R / "ext_test_set.parquet")
    q, qt = F.ER.fit_apply(tv, tt)
    add_v = tv.filter(pl.Series(q >= 0.75)).select(KEY)
    F.report(f"val v16 final (+{add_v.height:,} empty rescue)", pl.concat([sv.select(KEY), add_v]), truth, ids)
    add_t = tt.filter(pl.Series(qt >= 0.75)).select(KEY).with_columns(pl.lit(3, pl.UInt8).alias("kind"))
    st = pl.concat([st.select(*KEY, "kind"), add_t], how="vertical_relaxed").join(cty, on="s1", how="left")
    print(st.group_by("country", "kind").agg(pl.len()).sort("country", "kind"))
    per = st.group_by("country").agg(pl.len().alias("pairs"), pl.col("s1").n_unique().alias("nonempty")).join(
        cty.group_by("country").len(), on="country").with_columns((pl.col("pairs") / pl.col("len")).round(4).alias("per_s1"),
                                                                  (1 - pl.col("nonempty") / pl.col("len")).round(4).alias("empty"))
    print(per.sort("country"))
    ref = ROOT / "submissions/v15/selected.parquet"
    if ref.exists():
        old = pl.read_parquet(ref).select(KEY).join(cty, on="s1", how="left")
        new = st.select(*KEY, "country")
        diff = (new.join(old, on=KEY, how="anti").group_by("country").agg(pl.len().alias("added"))
                   .join(old.join(new, on=KEY, how="anti").group_by("country").agg(pl.len().alias("removed")), on="country", how="full", coalesce=True)
                   .join(cty.group_by("country").len(), on="country").with_columns((pl.col("added") / pl.col("len") * 1000).round(2).alias("add_per_1k"),
                                                                                  (pl.col("removed") / pl.col("len") * 1000).round(2).alias("rem_per_1k")))
        print("vs v15:", diff.sort("country"))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        write_id_list_tsv(to_pairs(st), s1_all["entity_id"], a.out / "matching_results.tsv", "matched_entity_ids")
        st.select(*KEY, "kind").write_parquet(a.out / "selected.parquet")
        log(f"{st.height:,} pairs -> {a.out / 'matching_results.tsv'}")
        r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str((a.out / "matching_results.tsv").resolve()),
                            "--test-dir", "dataset/test"], cwd=ROOT / "student_resource", capture_output=True, text=True)
        print(r.stdout[-600:], r.stderr[-600:])
