"""v7 = joint group-consistency pass over main (refiner) + extension pairs.

The main refiner and the extension model score their pairs independently and only meet in one_owner. Here every
candidate of an S1 (from either model) is compared with that S1's best OTHER candidates (anchors): true copies agree
with each other, siblings / other businesses do not. S1-centric only (no across-S1 record competition: on val the
fit S1 are unscored, which would make that feature look different from test).

    python notebooks/run_v7.py --main v4d --ext refine/ext_val_oof_v5d.parquet [--test --test-ext refine/test_scores_ext_v5d.parquet]
"""
import os
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.evaluate import evaluate  # noqa: E402
from src.refine import PARAMS3  # noqa: E402
from src.train import one_owner, to_pairs  # noqa: E402

A = ROOT / "artifacts"
R = A / "refine"
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)  # noqa: E731
ap = argparse.ArgumentParser()
ap.add_argument("--main", default="v4d")
ap.add_argument("--ext", default="refine/ext_val_oof_v5d.parquet")
ap.add_argument("--test", action="store_true")
ap.add_argument("--test-ext", default="refine/test_scores_ext_v5d.parquet")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--tag", default="v7")
a = ap.parse_args()

PAIR = ["hv_eq", "hv_diff", "hv_lev", "hv_sub", "nm_miss1", "nm_extra2", "nm_len1", "nm_len2", "nm_extra2_df", "nm_miss1_df",
        "n_acr", "r_sup_kh", "r_sup_k", "r_n_s1_key", "s1_n_key", "n_tset", "a_tset", "a_street_tset", "a_empty_2", "n_key_eq", "a_state_eq"]


def assemble(split, main_scores, main_set, ext_scores, ext_set):
    m = pl.read_parquet(main_set, columns=["s1", "m", "src"] + PAIR + (["label"] if split == "train" else []))
    m = m.with_columns(pl.read_parquet(main_scores, columns=["p"])["p"], pl.lit(0, pl.Int8).alias("is_ext"))
    e = pl.read_parquet(ext_set, columns=["s1", "m", "src"] + PAIR).join(pl.read_parquet(ext_scores), on=["s1", "m", "src"])
    e = e.with_columns(pl.lit(1, pl.Int8).alias("is_ext")).filter(pl.col("p") >= 0.02)
    if split == "train":
        num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731
        tp = (pl.read_parquet(A / "processed/train_gt_pairs.parquet").drop_nulls("match_id")
                .select(num("s1_id").alias("s1"), num("match_id").alias("m"), pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src"))
                .with_columns(pl.lit(1, pl.Int8).alias("label")))
        e = e.join(tp, on=["s1", "m", "src"], how="left").with_columns(pl.col("label").fill_null(0))
    e = e.join(m.select("s1", "m", "src"), on=["s1", "m", "src"], how="anti")
    d = pl.concat([m, e.select(m.columns)], how="vertical_relaxed")
    return d.with_columns(pl.col(pl.Float64).cast(pl.Float32))


def records(split):
    pool = pl.read_parquet(R / f"{split}_pool.parquet", columns=["m", "src", "name_core", "name_key", "house_v2"])
    an = pl.concat([pl.read_parquet(A / f"normalized/{split}_source{i}.parquet", columns=["entity_id", "addr_norm"])
                      .select(pl.col("entity_id").str.slice(3).cast(pl.UInt32).alias("m"), pl.lit(i, pl.UInt8).alias("src"), "addr_norm")
                    for i in (2, 3)])
    return pool.join(an, on=["m", "src"], how="left")


def group(d, recs):
    d = d.with_row_index("_i")
    r = d.select("_i", "s1", "m", "src", "p").sort(["s1", "p"], descending=[False, True]).with_columns(
        pl.int_range(1, pl.len() + 1).over("s1").alias("rk"))
    top = r.filter(pl.col("rk") <= 3).select("s1", "rk", pl.col("m").alias("am"), pl.col("src").alias("asrc"), pl.col("p").alias("ap"))
    anc = (r.select("_i", "s1", "rk").join(top, on="s1", suffix="_a").filter(pl.col("rk") != pl.col("rk_a"))
             .sort(["_i", "rk_a"]).with_columns(pl.int_range(1, pl.len() + 1).over("_i").alias("k")).filter(pl.col("k") <= 2))
    agg = d.group_by("s1").agg(pl.col("p").sum().alias("_s"), (pl.col("p") >= 0.5).sum().alias("_n5"), (pl.col("p") >= 0.9).sum().alias("_n9"),
                               pl.len().alias("j_ncand"))
    # majority house among the S1's confident (p>=0.9) candidates
    hh = (d.filter(pl.col("p") >= 0.9).join(recs.select("m", "src", "house_v2"), on=["m", "src"]).filter(pl.col("house_v2") != "")
            .group_by("s1", "house_v2").len().sort("len", descending=True).group_by("s1").first()
            .select("s1", pl.col("house_v2").alias("maj_house"), pl.col("len").alias("maj_n")))
    base = (r.select("_i", "s1", "rk", "p").join(agg, on="s1")
              .join(anc.filter(pl.col("k") == 1).select("_i", pl.col("ap").alias("j_a1")), on="_i", how="left")
              .join(anc.filter(pl.col("k") == 2).select("_i", pl.col("ap").alias("j_a2")), on="_i", how="left")
              .with_columns(pl.col("rk").cast(pl.Int16).alias("j_rank"), (pl.col("_s") - pl.col("p")).alias("j_sum_other"),
                            (pl.col("_n5") - (pl.col("p") >= 0.5).cast(pl.UInt32)).alias("j_n5_other"),
                            (pl.col("_n9") - (pl.col("p") >= 0.9).cast(pl.UInt32)).alias("j_n9_other"),
                            (pl.col("j_a1") - pl.col("p")).alias("j_gap")))
    rr = recs.rename({c: c + "_r" for c in ["name_core", "name_key", "house_v2", "addr_norm"]})
    ra = recs.rename({c: c + "_a" for c in ["name_core", "name_key", "house_v2", "addr_norm"]} | {"m": "am", "src": "asrc"})
    pr = (anc.select("_i", "am", "asrc").join(d.select("_i", "m", "src"), on="_i")
             .join(rr, on=["m", "src"], how="left").join(ra, on=["am", "asrc"], how="left"))
    f = lambda c: pr[c].fill_null("").to_list()  # noqa: E731
    sims = pr.select("_i").with_columns(
        pl.Series("nt", cpdist(f("name_core_r"), f("name_core_a"), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
        pl.Series("at", cpdist(f("addr_norm_r"), f("addr_norm_a"), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
        (pr["name_key_r"] == pr["name_key_a"]).cast(pl.Int8).alias("ke"),
        pl.when((pr["house_v2_r"] != "") & (pr["house_v2_a"] != "")).then((pr["house_v2_r"] == pr["house_v2_a"]).cast(pl.Int8)).alias("he"),
    ).group_by("_i").agg(pl.col("nt").max().alias("j_nt"), pl.col("at").max().alias("j_at"), pl.col("ke").max().alias("j_ke"),
                         pl.col("he").max().alias("j_he"), pl.col("nt").min().alias("j_nt_min"))
    out = (base.join(sims, on="_i", how="left").join(d.select("_i", "m", "src"), on="_i")
               .join(recs.select("m", "src", "house_v2"), on=["m", "src"], how="left").join(hh, on="s1", how="left")
               .with_columns(pl.when((pl.col("house_v2") != "") & pl.col("maj_house").is_not_null())
                               .then((pl.col("house_v2") == pl.col("maj_house")).cast(pl.Int8)).alias("j_maj_house"),
                             pl.col("maj_n").fill_null(0).alias("j_maj_n"))
               .sort("_i"))
    return pl.concat([d.drop("_i"), out.select(JCOLS)], how="horizontal")


JCOLS = ["j_rank", "j_sum_other", "j_n5_other", "j_n9_other", "j_a1", "j_a2", "j_gap", "j_ncand", "j_nt", "j_at", "j_ke", "j_he",
         "j_nt_min", "j_maj_house", "j_maj_n"]
FEATS = ["p", "is_ext"] + PAIR + JCOLS

out_v = R / f"{a.tag}_val_set.parquet"
if not out_v.exists():
    d = assemble("train", R / f"val_oof_{a.main}.parquet", R / "val_set.parquet", A / a.ext, R / "ext_val_set.parquet")
    d = group(d, records("train"))
    d.write_parquet(out_v)
d = pl.read_parquet(out_v)
log(f"val joint set {d.height:,} pairs, pos {d['label'].sum():,}")
X = d.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
y = d["label"].to_numpy()
fold = (d["s1"].hash(seed=21) % a.folds).to_numpy()
inner = (d["s1"].hash(seed=9) % 10).to_numpy() == 0
oof = np.zeros(d.height, dtype=np.float32)
models = []
for k in range(a.folds):
    fit, es = (fold != k) & ~inner, (fold != k) & inner
    b = lgb.train(PARAMS3, lgb.Dataset(X[fit], y[fit], feature_name=FEATS), 4000,
                  valid_sets=[lgb.Dataset(X[es], y[es])], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[fold == k] = b.predict(X[fold == k], num_iteration=b.best_iteration)
    models.append(b)
    log(f"fold {k}: {b.best_iteration} rounds")
imp = sorted(zip(FEATS, models[0].feature_importance("gain")), key=lambda x: -x[1])
log("top gain: " + ", ".join(f"{f} {g/sum(x[1] for x in imp):.3f}" for f, g in imp[:15]))
split = pl.read_parquet(A / "splits/split_v1.parquet")
ids = split.filter(pl.col("role") == "val").select("s1_id", "country")
truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
base = d.select("s1", "m", "src", "p", "is_ext").with_columns(pl.when(pl.col("is_ext") == 1).then(0.7).otherwise(0.75).alias("t"))
r = evaluate(to_pairs(one_owner(base).filter(pl.col("p") >= pl.col("t"))), truth, ids)
log(f"baseline (main 0.75 / ext 0.7): F0.5 {r['f05']:.5f} US {r['by_country']['US']['f05']:.5f} IN {r['by_country']['India']['f05']:.5f}")
pv = d.select("s1", "m", "src").with_columns(pl.Series("p", oof))
pv.write_parquet(R / f"{a.tag}_val_oof.parquet")
res = {}
for t in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
    r = evaluate(to_pairs(one_owner(pv).filter(pl.col("p") >= t)), truth, ids)
    res[t] = r["f05"]
    log(f"{a.tag} thr {t}: F0.5 {r['f05']:.5f} P {r['macro_p']:.4f} R {r['macro_r']:.4f} US {r['by_country']['US']['f05']:.5f} IN {r['by_country']['India']['f05']:.5f}")
(A / f"models/lgb_{a.tag}.json").write_text(json.dumps({"features": FEATS, "thr": max(res, key=res.get), "val": max(res.values())}))

if a.test:
    dt = assemble("test", A / f"test_scores_{a.main}.parquet", R / "test_set.parquet", A / a.test_ext, R / "ext_test_set.parquet")
    dt = group(dt, records("test"))
    Xt = dt.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
    pt = np.mean([m.predict(Xt) for m in models], axis=0)
    dt.select("s1", "m", "src", "is_ext").with_columns(pl.Series("p", pt, dtype=pl.Float32)).write_parquet(A / f"test_scores_{a.tag}.parquet")
    log("test scored")
