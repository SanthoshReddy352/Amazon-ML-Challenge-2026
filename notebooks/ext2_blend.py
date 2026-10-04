"""Blend cross-encoder logits into the ext2 LightGBM probability for ext2 pairs in the uncertain band.

    python notebooks/ext2_blend.py                  # val: 5-fold blend + combined report (v11 val + ext2)
    python notebooks/ext2_blend.py --test           # + test -> artifacts/test_scores_ext2b.parquet
CE logits: artifacts/kaggle_ext2/out_{val,test}/ce{1,2,3}_ext2.parquet keyed by artifacts/kaggle_ext2/keys_{val,test}.parquet.
Output val: artifacts/refine/ext2b_val_oof.parquet (s1, m, src, p) = blended p in the band, ext2 p elsewhere.
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
sys.path.insert(0, str(ROOT / "notebooks"))
from ce_blend import PARAMS  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.train import to_pairs  # noqa: E402

A, R, K = ROOT / "artifacts", ROOT / "artifacts" / "refine", ROOT / "artifacts" / "kaggle_ext2"
ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true")
ap.add_argument("--ce", nargs="+", default=["ce1", "ce2", "ce3"])
ap.add_argument("--tag", default="", help="ext2 variant suffix (e.g. w)")
a = ap.parse_args()
T = a.tag
num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731
FLAGS = ["a_empty_2", "hv_eq", "nm_subst", "n_key_eq", "a_house_eq"]
FEATS = ["lp"] + FLAGS + a.ce


def tie_guard(split: str, scored: pl.DataFrame) -> pl.DataFrame:
    """The cross-encoder sees one pair, not the other S1 that share the record's name. For empty-address records whose
    name key is shared by 2+ S1 of the country keep the (tie-aware) LightGBM p: min(p_blend, p_lgb). Such pairs are
    ~absent from the val band (the blend never learned them) but common in France (2.9k test pairs)."""
    s1 = pl.read_parquet(A / f"normalized/{split}_source1.parquet", columns=["country", "name_key"])
    kc = s1.group_by("country", "name_key").agg(pl.len().alias("k_n")).filter(pl.col("k_n") >= 2)
    recs = pl.concat([pl.read_parquet(A / f"normalized/{split}_source{s}.parquet", columns=["entity_id", "country", "name_key", "addr_empty"])
                        .filter(pl.col("addr_empty")).select(num("entity_id").alias("m"), pl.lit(s, pl.UInt8).alias("src"), "country", "name_key")
                      for s in (2, 3)])
    tied = recs.join(kc, on=["country", "name_key"], how="semi").select("m", "src").with_columns(pl.lit(True).alias("tied"))
    out = scored.join(tied, on=["m", "src"], how="left").with_columns(
        pl.when(pl.col("tied")).then(pl.min_horizontal("p", "p_lgb")).otherwise(pl.col("p")).alias("p"))
    print(f"  tie guard ({split}): {out.filter(pl.col('tied') & (pl.col('p') < pl.col('p_pre'))).height:,} pairs capped at the LightGBM p")
    return out.drop("tied")


def band(tag: str, scores: pl.DataFrame, set_path: Path) -> pl.DataFrame:
    keys = pl.read_parquet(K / f"keys_{T}{tag}.parquet").filter(pl.col("split") == tag).select("id", "s1", "m", "src")
    for c in a.ce:
        keys = keys.join(pl.read_parquet(K / f"out_{T}{tag}/{c}_ext2.parquet").rename({"logit": c}), on="id", how="left")
    b = keys.drop("id").join(scores, on=["s1", "m", "src"], how="inner").join(
        pl.read_parquet(set_path, columns=["s1", "m", "src"] + FLAGS), on=["s1", "m", "src"], how="left")
    return b.with_columns((pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log().alias("lp"))


if __name__ == "__main__":
    sys.argv = [sys.argv[0]]  # run_ext2's own argparse must not see our flags
    from run_ext2 import final_select, v11_val  # noqa: E402
    ids = pl.read_parquet(A / "splits/split_v1.parquet").filter(pl.col("role") == "val").select("s1_id", "country")
    truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
    tp = (truth.drop_nulls("match_id").select(num("s1_id").alias("s1"), num("match_id").alias("m"),
                                               pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src")).with_columns(pl.lit(1, pl.Int8).alias("y")))
    ev = pl.read_parquet(R / f"ext2{T}_val_oof.parquet").select("s1", "m", "src", "p")
    bv = band("val", ev, R / f"ext2{T}_val_set.parquet").join(tp, on=["s1", "m", "src"], how="left").with_columns(pl.col("y").fill_null(0))
    X = bv.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
    y = bv["y"].to_numpy()
    fold = (bv["s1"].hash(seed=33) % 5).to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    models = []
    for k in range(5):
        b = lgb.train(PARAMS, lgb.Dataset(X[fold != k], y[fold != k], feature_name=FEATS), 400)
        oof[fold == k] = b.predict(X[fold == k])
        models.append(b)
    from sklearn.metrics import roc_auc_score
    print(f"val band {bv.height:,} pairs, pos {int(y.sum()):,} | AUC ext2 p {roc_auc_score(y, bv['p'].to_numpy()):.5f} | " +
          " ".join(f"{c} {roc_auc_score(y, bv[c].fill_null(-20).to_numpy()):.5f}" for c in a.ce) + f" | blend {roc_auc_score(y, oof):.5f}")
    evb = ev.join(bv.select("s1", "m", "src").with_columns(pl.Series("pb", oof)), on=["s1", "m", "src"], how="left").with_columns(
        pl.col("p").alias("p_lgb"), pl.coalesce("pb", "p").alias("p")).drop("pb").with_columns(pl.col("p").alias("p_pre"))
    evb = tie_guard("train", evb).drop("p_lgb", "p_pre")
    evb.write_parquet(R / f"ext2{T}b_val_oof.parquet")
    allv = v11_val()
    base = evaluate(to_pairs(final_select(allv, {0: 0.75, 1: 0.7})), truth, ids)["f05"]
    for name, e in [("ext2 lgb", ev), ("ext2 blend", evb)]:
        comb = pl.concat([allv.select("s1", "m", "src", "p", "kind"), e.with_columns(pl.lit(2, pl.UInt8).alias("kind"))], how="vertical_relaxed")
        for t2 in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
            r = evaluate(to_pairs(final_select(comb, {0: 0.75, 1: 0.7, 2: t2})), truth, ids)
            print(f"{name} @{t2}: F0.5 {r['f05']:.5f} (+{r['f05'] - base:.5f}) US {r['by_country']['US']['f05']:.5f} IN {r['by_country']['India']['f05']:.5f}")
    if a.test:
        et = pl.read_parquet(A / f"test_scores_ext2{T}.parquet").select("s1", "m", "src", "p")
        bt = band("test", et, R / f"ext2{T}_test_set.parquet")
        Xt = bt.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
        bt = bt.select("s1", "m", "src").with_columns(pl.Series("pb", np.mean([m.predict(Xt) for m in models], axis=0)))
        et = et.join(bt, on=["s1", "m", "src"], how="left").with_columns(pl.col("p").alias("p_lgb"), pl.coalesce("pb", "p").alias("p")).drop("pb")
        et = tie_guard("test", et.with_columns(pl.col("p").alias("p_pre"))).drop("p_lgb", "p_pre")
        et.write_parquet(A / f"test_scores_ext2{T}b.parquet")
        print("test ext2 blended:", bt.height, "band pairs of", et.height)
