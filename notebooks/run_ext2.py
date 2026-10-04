"""ext2 = record-centric reverse-blocking pairs (src/revblock.py) scored by their own LightGBM, on top of v11.

    python notebooks/run_ext2.py                 # val: featurize, 5-fold OOF, combined report with v11 val
    python notebooks/run_ext2.py --test          # + test ext2 pairs -> artifacts/test_scores_ext2.parquet

Pairs already in the main union candidates or the v5 extension are removed, so ext2 only adds new pairs.
Features: string similarities (src.features), refiner features (src.refine), the reverse-blocking scores and the
record's best / second-best S1 scores over ALL S1 of the country (computed before any output restriction), and
context from the v11 decision (is the record already confidently owned? how many matches does the S1 have?).
"""
import os
import argparse
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.evaluate import evaluate  # noqa: E402
from src.features import REC_COLS, load_records, pair_features  # noqa: E402
from src.refine import PARAMS3, pair_new_features  # noqa: E402
from src.train import one_owner, to_pairs  # noqa: E402

A = ROOT / "artifacts"
R = A / "refine"
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)  # noqa: E731
ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--val-main", default=None, help="parquet (s1, m, src, p, kind) of the current final val scores")
ap.add_argument("--tag", default="", help="variant suffix: reads revblock/*_<tag>, writes ext2<tag>_* files")
ap.add_argument("--neg-frac", type=float, default=1.0, help="train on all positives + this fraction of negatives (weighted)")
ap.add_argument("--reuse-models", action="store_true", help="--test only: load artifacts/models/lgb_ext2_f*.txt instead of refitting")
a = ap.parse_args()
SUF = f"_{a.tag}" if a.tag else ""
num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731


def _key(df) -> np.ndarray:
    """(s1, m, src) -> one sortable u64 (s1 < 2^30, m < 2^32, src in {2, 3})."""
    return ((df["s1"].cast(pl.UInt64).to_numpy() << np.uint64(33)) | (df["src"].cast(pl.UInt64).to_numpy() - np.uint64(2)) << np.uint64(32)
            | df["m"].cast(pl.UInt64).to_numpy())


def new_pairs(rev_dir: Path, union_glob: str, ext_path: Path, out: Path) -> pl.DataFrame:
    """Reverse-blocking pairs minus the main union candidates and the v5 extension; file by file in bounded memory."""
    if out.exists():
        return pl.read_parquet(out)
    files = sorted(rev_dir.glob("*.parquet"))
    s1s = pl.concat([pl.read_parquet(f, columns=["s1"]).unique() for f in files]).unique()
    have = np.sort(np.concatenate([_key(pl.scan_parquet(union_glob).select("s1", "m", "src").join(s1s.lazy(), on="s1", how="semi").collect()),
                                   _key(pl.read_parquet(ext_path, columns=["s1", "m", "src"]))]))
    parts = []
    for f in files:
        d = pl.read_parquet(f).unique(["s1", "m", "src"])
        k = _key(d)
        i = np.searchsorted(have, k).clip(0, len(have) - 1)
        parts.append(d.filter(pl.Series(have[i] != k)))
    rv = pl.concat(parts)
    del parts, have
    rv = rv.with_columns((pl.col("r_score") / pl.col("r_best1")).alias("r_rel1"), (pl.col("r_best1") - pl.col("r_best2")).alias("r_gap12"),
                         (pl.col("r_name_score") / pl.col("r_name_best")).alias("r_name_rel"),
                         (pl.col("r_addr_score") / pl.col("r_addr_best")).fill_nan(None).alias("r_addr_rel"))
    rv.write_parquet(out)
    return rv


def featurize(split: str, e: pl.DataFrame, main: pl.DataFrame, out: Path) -> pl.DataFrame:
    """main: (s1, m, src, p) current final scores of the split (context only)."""
    if out.exists():
        return pl.read_parquet(out)
    s1t, pool = pl.read_parquet(R / f"{split}_s1.parquet"), pl.read_parquet(R / f"{split}_pool.parquet")
    r1 = load_records(A / "normalized", split, 1, e["s1"].unique()).rename({c: c + "_1" for c in REC_COLS})
    r2 = pl.concat([load_records(A / "normalized", split, s, e.filter(pl.col("src") == s)["m"].unique()) for s in (2, 3)])
    r2 = r2.rename({c: c + "_2" for c in REC_COLS})
    parts = []
    for i in range(0, e.height, 1_000_000):
        chunk = e.slice(i, 1_000_000)
        wide = chunk.join(r1, on="s1", how="left").join(r2, on=["m", "src"], how="left")
        wide = wide.with_columns([pl.col(c + s).fill_null("") for c in REC_COLS if c not in ("name_is_domain", "name_indic", "addr_empty")
                                  for s in ("_1", "_2")])
        f = pl.concat([chunk, pair_features(wide)], how="horizontal")
        parts.append(pair_new_features(f, s1t, pool))
        log(f"  featurized {min(i + 1_000_000, e.height):,}/{e.height:,}")
    d = pl.concat(parts)
    own = one_owner(main)
    rec = main.group_by("m", "src").agg(pl.col("p").max().alias("x_rec_pmax"), (pl.col("p") >= 0.5).sum().alias("x_rec_nconf"))
    s1c = own.group_by("s1").agg((pl.col("p") >= 0.75).sum().alias("x_s1_nsel"), pl.col("p").max().alias("x_s1_pmax"))
    d = (d.join(rec, on=["m", "src"], how="left").join(s1c, on="s1", how="left")
          .with_columns(pl.col("x_rec_pmax", "x_s1_pmax").fill_null(0.0), pl.col("x_rec_nconf", "x_s1_nsel").fill_null(0)))
    # how many ext2 records each S1 receives (a per-record count would only see val S1 on val, every S1 on test)
    d = d.with_columns(pl.len().over("s1").alias("x2_s1_n"))
    d = d.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    d.write_parquet(out)
    return d


def featurize_predict(split: str, e: pl.DataFrame, main: pl.DataFrame, models, feats, out_scores: Path, out_slim: Path,
                      slim_cols=("a_empty_2", "hv_eq", "nm_subst", "n_key_eq", "a_house_eq"), chunk: int = 1_000_000):
    """Test path in bounded memory: per chunk, features -> context -> model average; keeps only scores + a few flags."""
    s1t, pool = pl.read_parquet(R / f"{split}_s1.parquet"), pl.read_parquet(R / f"{split}_pool.parquet")
    own = one_owner(main)
    rec = main.group_by("m", "src").agg(pl.col("p").max().alias("x_rec_pmax"), (pl.col("p") >= 0.5).sum().alias("x_rec_nconf"))
    s1c = own.group_by("s1").agg((pl.col("p") >= 0.75).sum().alias("x_s1_nsel"), pl.col("p").max().alias("x_s1_pmax"))
    s1n = e.group_by("s1").agg(pl.len().alias("x2_s1_n"))
    del own
    scores, slims = [], []
    for i in range(0, e.height, chunk):
        c = e.slice(i, chunk)
        r1 = load_records(A / "normalized", split, 1, c["s1"].unique()).rename({x: x + "_1" for x in REC_COLS})
        r2 = pl.concat([load_records(A / "normalized", split, s, c.filter(pl.col("src") == s)["m"].unique()) for s in (2, 3)])
        r2 = r2.rename({x: x + "_2" for x in REC_COLS})
        wide = c.join(r1, on="s1", how="left").join(r2, on=["m", "src"], how="left")
        wide = wide.with_columns([pl.col(x + s).fill_null("") for x in REC_COLS if x not in ("name_is_domain", "name_indic", "addr_empty")
                                  for s in ("_1", "_2")])
        f = pair_new_features(pl.concat([c, pair_features(wide)], how="horizontal"), s1t, pool)
        del wide, r1, r2
        f = (f.join(rec, on=["m", "src"], how="left").join(s1c, on="s1", how="left").join(s1n, on="s1", how="left")
              .with_columns(pl.col("x_rec_pmax", "x_s1_pmax").fill_null(0.0), pl.col("x_rec_nconf", "x_s1_nsel").fill_null(0)))
        X = f.select([pl.col(x).cast(pl.Float32) for x in feats]).to_numpy()
        scores.append(f.select("s1", "m", "src").with_columns(pl.Series("p", np.mean([m.predict(X) for m in models], axis=0), dtype=pl.Float32)))
        slims.append(f.select("s1", "m", "src", *slim_cols))
        del f, X
        log(f"  scored {min(i + chunk, e.height):,}/{e.height:,}")
    pl.concat(scores).write_parquet(out_scores)
    pl.concat(slims).write_parquet(out_slim)


def final_select(allp: pl.DataFrame, thr: dict, empty_thr: float = 0.5) -> pl.DataFrame:
    """allp: s1, m, src, p, kind (0 main, 1 ext, 2 ext2). One owner per record, per-kind threshold, empty-S1 rescue."""
    p = allp.with_columns(pl.col("kind").replace_strict(thr, return_dtype=pl.Float64).alias("t"))
    own = one_owner(p.select("s1", "m", "src", "p", "t", "kind"))
    sel = own.filter(pl.col("p") >= pl.col("t"))
    if empty_thr:
        resc = own.join(sel.select("s1").unique(), on="s1", how="anti").sort("p", descending=True).group_by("s1").head(1).filter(pl.col("p") >= empty_thr)
        sel = pl.concat([sel, resc])
    return sel


def v11_val() -> pl.DataFrame:
    """v11 val scores per pair: blended probability where the CE band applies, else the refiner / extension p."""
    m = pl.read_parquet(R / "val_oof_v4f.parquet", columns=["s1", "m", "src", "p"]).with_columns(pl.lit(0, pl.UInt8).alias("kind"))
    e = (pl.read_parquet(R / "ext_val_oof_v5f.parquet", columns=["s1", "m", "src", "p"]).join(m, on=["s1", "m", "src"], how="anti")
           .with_columns(pl.lit(1, pl.UInt8).alias("kind")))
    allv = pl.concat([m, e], how="vertical_relaxed").join(pl.read_parquet(R / "ens123_val_oof.parquet"), on=["s1", "m", "src"], how="left")
    return allv.with_columns(pl.coalesce("pb", "p").alias("p")).drop("pb")


def fit_report(dv, feats, allv):
    """5-fold OOF on val ext2 pairs, saved models, combined val report with the current final val scores."""
    X = dv.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
    y = dv["y"].to_numpy()
    fold = (dv["s1"].hash(seed=11) % a.folds).to_numpy()
    inner = (dv["s1"].hash(seed=13) % 10).to_numpy() == 0
    keep = (y == 1) | (np.random.default_rng(5).random(len(y)) < a.neg_frac)
    w = np.where(y == 1, 1.0, 1.0 / a.neg_frac).astype(np.float32)
    oof = np.zeros(dv.height, dtype=np.float32)
    models = []
    for k in range(a.folds):
        fit, es = (fold != k) & ~inner & keep, (fold != k) & inner & keep
        b = lgb.train(PARAMS3, lgb.Dataset(X[fit], y[fit], weight=w[fit], feature_name=feats), 3000,
                      valid_sets=[lgb.Dataset(X[es], y[es], weight=w[es])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[fold == k] = b.predict(X[fold == k], num_iteration=b.best_iteration)
        b.save_model(str(A / f"models/lgb_ext2{a.tag}_f{k}.txt"), num_iteration=b.best_iteration)
        models.append(b)
        log(f"ext2 fold {k}: {b.best_iteration} rounds")
    imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
    log("ext2 top gain: " + ", ".join(f"{f} {g/sum(x[1] for x in imp):.3f}" for f, g in imp[:15]))
    from sklearn.metrics import roc_auc_score
    log(f"ext2 OOF AUC {roc_auc_score(y, oof):.5f}")
    ext2_v = dv.select("s1", "m", "src").with_columns(pl.Series("p", oof, dtype=pl.Float64), pl.lit(2, pl.UInt8).alias("kind"))
    ext2_v.write_parquet(R / f"ext2{a.tag}_val_oof.parquet")

    base_sel = final_select(allv, {0: 0.75, 1: 0.7})
    r = evaluate(to_pairs(base_sel), truth, ids)
    log(f"v11 val: F0.5 {r['f05']:.5f} US {r['by_country']['US']['f05']:.5f} IN {r['by_country']['India']['f05']:.5f}")
    comb = pl.concat([allv.select("s1", "m", "src", "p", "kind"), ext2_v], how="vertical_relaxed")
    for t2 in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        r = evaluate(to_pairs(final_select(comb, {0: 0.75, 1: 0.7, 2: t2})), truth, ids)
        log(f"  + ext2 @{t2}: F0.5 {r['f05']:.5f} P {r['macro_p']:.5f} R {r['macro_r']:.5f} "
            f"US {r['by_country']['US']['f05']:.5f} IN {r['by_country']['India']['f05']:.5f}")

    return models


if __name__ == "__main__":
    split = pl.read_parquet(A / "splits/split_v1.parquet")
    ids = split.filter(pl.col("role") == "val").select("s1_id", "country")
    truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
    tp = (truth.drop_nulls("match_id").select(num("s1_id").alias("s1"), num("match_id").alias("m"),
                                               pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src")).with_columns(pl.lit(1, pl.Int8).alias("y")))
    allv = pl.read_parquet(a.val_main) if a.val_main else v11_val()
    ev = new_pairs(A / f"revblock/train_val{SUF}", str(A / "blocking/union_train/*.parquet"), R / "ext_val_pairs.parquet", R / f"ext2{a.tag}_val_pairs.parquet")
    log(f"val ext2 pairs {ev.height:,}, true {ev.join(tp, on=['s1', 'm', 'src'], how='semi').height:,}")
    dv = featurize("train", ev, allv.select("s1", "m", "src", "p"), R / f"ext2{a.tag}_val_set.parquet")
    dv = dv.drop("y", strict=False).join(tp, on=["s1", "m", "src"], how="left").with_columns(pl.col("y").fill_null(0))
    feats = [c for c in dv.columns if c not in {"s1", "m", "src", "y", "label"} and dv[c].dtype != pl.Utf8]
    log(f"val ext2 set {dv.height:,} pairs, pos {dv['y'].sum():,}, {len(feats)} features")
    if a.test and a.reuse_models:
        models = [lgb.Booster(model_file=str(A / f"models/lgb_ext2{a.tag}_f{k}.txt")) for k in range(a.folds)]
        feats = models[0].feature_name()
        log(f"reusing {len(models)} saved ext2 models ({len(feats)} features)")
    else:
        models = fit_report(dv, feats, allv)

    if a.test:
        et = new_pairs(A / f"revblock/test{SUF}", str(A / "blocking/union_test/*.parquet"), R / "ext_test_pairs.parquet", R / f"ext2{a.tag}_test_pairs.parquet")
        log(f"test ext2 pairs {et.height:,}")
        main_t = pl.concat([pl.read_parquet(A / f"test_scores_v11_{k}.parquet", columns=["s1", "m", "src", "p"]) for k in ("main", "ext")])
        del dv, allv
        featurize_predict("test", et, main_t, models, feats, A / f"test_scores_ext2{a.tag}.parquet", R / f"ext2{a.tag}_test_set.parquet")
        log("test ext2 scored")
