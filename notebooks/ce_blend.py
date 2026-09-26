"""Blend the Kaggle cross-encoder score into the current system (v5f) for pairs in the uncertain band.

    python notebooks/ce_blend.py            # val: 5-fold blend OOF, F0.5 vs v5f
    python notebooks/ce_blend.py --test     # + test: artifacts/test_scores_ce_{main,ext}.parquet
Inputs: artifacts/kaggle_ce/{val,test}_keys.parquet, ce_{val,test}.parquet (Kaggle output),
        v5f probabilities (main v4f + extension v5f), refiner pair features for context.
Pairs outside the band keep their v5f probability.
"""
import os
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.evaluate import evaluate  # noqa: E402
from src.train import one_owner, to_pairs  # noqa: E402

A, R, K = ROOT / "artifacts", ROOT / "artifacts" / "refine", ROOT / "artifacts" / "kaggle_ce"
ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true")
ap.add_argument("--folds", type=int, default=5)
a, _ = ap.parse_known_args()
num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
FEATS = ["lp", "ce", "kind", "hv_eq", "a_empty_2", "nm_subst"]


def band(split, main_path, ext_path, main_set, ext_set):
    keys = pl.read_parquet(K / f"{split}_keys.parquet")
    ce = pl.read_parquet(K / f"ce_{split}.parquet").rename({"logit": "ce"})
    m = pl.read_parquet(main_path, columns=["s1", "m", "src", "p"])
    m = m.with_columns(pl.read_parquet(main_set, columns=["hv_eq", "a_empty_2", "nm_subst"]))
    e = pl.read_parquet(ext_path, columns=["s1", "m", "src", "p"]).join(
        pl.read_parquet(ext_set, columns=["s1", "m", "src", "hv_eq", "a_empty_2", "nm_subst"]), on=["s1", "m", "src"], how="left")
    allp = pl.concat([m.with_columns(pl.lit(0, pl.UInt8).alias("kind")),
                      e.join(m, on=["s1", "m", "src"], how="anti").with_columns(pl.lit(1, pl.UInt8).alias("kind"))], how="vertical_relaxed")
    b = keys.drop("kind").join(ce, on="id").join(allp, on=["s1", "m", "src"], how="inner")
    b = b.with_columns((pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log().alias("lp"))
    return allp, b


def decide(allp, blended, ids, truth=None, thr_main=0.75, thr_ext=0.7):
    p = allp.join(blended.select("s1", "m", "src", pl.col("pb")), on=["s1", "m", "src"], how="left")
    p = p.with_columns(pl.coalesce("pb", "p").alias("p")).with_columns(
        pl.when(pl.col("kind") == 1).then(thr_ext).otherwise(thr_main).alias("t"))
    return one_owner(p.select("s1", "m", "src", "p", "t")).filter(pl.col("p") >= pl.col("t"))


if __name__ == "__main__":
    ids = pl.read_parquet(A / "splits/split_v1.parquet").filter(pl.col("role") == "val").select("s1_id", "country")
    truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
    tp = (truth.drop_nulls("match_id").select(num("s1_id").alias("s1"), num("match_id").alias("m"),
                                               pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src")).with_columns(pl.lit(1, pl.Int8).alias("y")))
    allv, bv = band("val", R / "val_oof_v4f.parquet", R / "ext_val_oof_v5f.parquet", R / "val_set.parquet", R / "ext_val_set.parquet")
    bv = bv.join(tp, on=["s1", "m", "src"], how="left").with_columns(pl.col("y").fill_null(0))
    print(f"val band pairs {bv.height:,}, pos {bv['y'].mean():.3f}")
    X = bv.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
    y = bv["y"].to_numpy()
    fold = (bv["s1"].hash(seed=33) % a.folds).to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    models = []
    for k in range(a.folds):
        tr = fold != k
        b = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=FEATS), 600)
        oof[~tr] = b.predict(X[~tr])
        models.append(b)
    from sklearn.metrics import roc_auc_score
    print("band AUC: v5f p", round(roc_auc_score(y, bv["p"].to_numpy()), 5), "| ce", round(roc_auc_score(y, bv["ce"].to_numpy()), 5),
          "| blend", round(roc_auc_score(y, oof), 5))
    bvb = bv.with_columns(pl.Series("pb", oof))
    for tm, te in [(0.75, 0.7), (0.7, 0.7), (0.8, 0.75)]:
        base = evaluate(to_pairs(decide(allv, bvb.head(0), ids, thr_main=tm, thr_ext=te)), truth, ids)
        new = evaluate(to_pairs(decide(allv, bvb, ids, thr_main=tm, thr_ext=te)), truth, ids)
        print(f"thr {tm}/{te}: v5f F0.5 {base['f05']:.5f} -> blend {new['f05']:.5f} "
              f"(US {new['by_country']['US']['f05']:.5f} IN {new['by_country']['India']['f05']:.5f})")
    if a.test:
        allt, bt = band("test", A / "test_scores_v4f.parquet", R / "test_scores_ext_v5f.parquet", R / "test_set.parquet", R / "ext_test_set.parquet")
        Xt = bt.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
        bt = bt.with_columns(pl.Series("pb", np.mean([m.predict(Xt) for m in models], axis=0)))
        out = allt.join(bt.select("s1", "m", "src", "pb"), on=["s1", "m", "src"], how="left").with_columns(pl.coalesce("pb", "p").alias("p"))
        out.filter(pl.col("kind") == 0).select("s1", "m", "src", "p").write_parquet(A / "test_scores_ce_main.parquet")
        out.filter(pl.col("kind") == 1).select("s1", "m", "src", "p").write_parquet(A / "test_scores_ce_ext.parquet")
        print("test blended scores written")
