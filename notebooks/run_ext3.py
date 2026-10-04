"""ext3 = record-centric EMBEDDING neighbours (kaggle/embed_rknn: every S2/S3 record's top-5 S1 by multilingual-e5
cosine) that are not yet candidates (main union, v5 extension, ext2), scored by their own LightGBM.

    python notebooks/run_ext3.py                        # val: pairs, features, 5-fold OOF, report on the v13 stack
    python notebooks/run_ext3.py --test --reuse-models  # test -> artifacts/test_scores_ext2e.parquet
    python notebooks/run_ext3.py --source bknn --tag f  # ext4: fine-tuned bi-encoder neighbours (both directions)
    python notebooks/run_ext3.py --source bknn2 --max-rank 15 --tag g  # ext5: + forward top-15 (kaggle/bfwd_kernel)
Files follow the ext2 naming with tag "e" (ext2e_*) so ce_export_ext2.py / ext2_blend.py / final_v13.py reuse them.
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
sys.path.insert(0, str(ROOT / "notebooks"))
ap = argparse.ArgumentParser()
ap.add_argument("--test", action="store_true")
ap.add_argument("--reuse-models", action="store_true")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--max-rank", type=int, default=5)
ap.add_argument("--source", default="rknn", choices=["rknn", "bknn", "bknn2"], help="rknn = zero-shot reverse e5 (ext3); bknn = fine-tuned, both directions (ext4); bknn2 = bknn reverse top-5 + forward top-15 (ext5)")
ap.add_argument("--tag", default="e", help="file tag: ext2<tag>_* (e = ext3, f = ext4)")
a = ap.parse_args()
_argv, sys.argv = sys.argv, [sys.argv[0]]
import run_ext2 as X2  # noqa: E402  (its argparse sees no flags)
sys.argv = _argv
from src.evaluate import evaluate  # noqa: E402
from src.refine import PARAMS3  # noqa: E402
from src.train import to_pairs  # noqa: E402

A, R = ROOT / "artifacts", ROOT / "artifacts" / "refine"
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)  # noqa: E731
num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731


def rknn_pairs(split: str, s1_keep: pl.Series | None, existing: list, out: Path) -> pl.DataFrame:
    """New embedding-neighbour pairs (not in `existing`) with neighbour features; bounded memory (test: 67M rows)."""
    if out.exists():
        return pl.read_parquet(out)
    src_file = A / f"kaggle_{a.source}/{a.source}_{split}.parquet"
    d = pl.read_parquet(src_file).filter(pl.col("rank") <= a.max_rank)
    if "dir" not in d.columns:
        d = d.with_columns(pl.lit(0, pl.UInt8).alias("dir"))
    # record- and S1-level neighbour context over ALL S1 (rank-1 / rank-2 rows only)
    rec = (d.filter((pl.col("dir") == 0) & (pl.col("rank") == 1)).select("m", "src", pl.col("cos").cast(pl.Float32).alias("e_best1"))
            .join(d.filter((pl.col("dir") == 0) & (pl.col("rank") == 2)).select("m", "src", pl.col("cos").cast(pl.Float32).alias("e_best2")),
                  on=["m", "src"], how="left"))
    s1b = d.filter((pl.col("dir") == 1) & (pl.col("rank") == 1)).select("s1", pl.col("cos").cast(pl.Float32).alias("f_best1")).unique("s1")
    if s1_keep is not None:
        d = d.filter(pl.col("s1").is_in(s1_keep.implode()))
    have = []
    s1set = d.select("s1").unique()
    for glob_or_path in existing:
        if "*" in str(glob_or_path):
            have.append(X2._key(pl.scan_parquet(glob_or_path).select("s1", "m", "src").join(s1set.lazy(), on="s1", how="semi").collect()))
        elif Path(glob_or_path).exists():
            have.append(X2._key(pl.read_parquet(glob_or_path, columns=["s1", "m", "src"])))
    arr = np.concatenate(have)
    del have
    arr.sort()  # in place (np.unique's hash table does not fit 16 GB on test)
    have = arr
    parts = []
    for i in range(0, d.height, 10_000_000):
        c = d.slice(i, 10_000_000)
        k = X2._key(c)
        j = np.searchsorted(have, k).clip(0, len(have) - 1)
        parts.append(c.filter(pl.Series(have[j] != k)))
    del have, d
    d = pl.concat(parts)
    del parts
    d = d.group_by("s1", "m", "src").agg(pl.col("cos").max().cast(pl.Float32).alias("e_cos"),
                                         pl.col("rank").filter(pl.col("dir") == 0).min().alias("e_rank"),
                                         pl.col("rank").filter(pl.col("dir") == 1).min().alias("f_rank"))
    d = d.join(rec, on=["m", "src"], how="left").join(s1b, on="s1", how="left")
    d = d.with_columns(pl.col("e_best2").fill_null(0.0), (pl.col("e_cos") / pl.col("e_best1")).alias("e_rel1"),
                       (pl.col("e_best1") - pl.col("e_best2").fill_null(0.0)).alias("e_gap12"),
                       (pl.col("e_cos") / pl.col("f_best1")).alias("f_rel1"))
    if a.source == "rknn":
        d = d.drop("f_rank", "f_best1", "f_rel1")
    d.write_parquet(out)
    return d


def v13_val() -> pl.DataFrame:
    """v13 val scores (kind 0 main, 1 ext, 2 ext2) with the final probability in p."""
    m = pl.read_parquet(R / "val_oof_v4f.parquet", columns=["s1", "m", "src", "p"]).with_columns(pl.lit(0, pl.UInt8).alias("kind"))
    e = (pl.read_parquet(R / "ext_val_oof_v5f.parquet", columns=["s1", "m", "src", "p"]).join(m, on=["s1", "m", "src"], how="anti")
           .with_columns(pl.lit(1, pl.UInt8).alias("kind")))
    v = pl.concat([m, e], how="vertical_relaxed").join(pl.read_parquet(R / "ens1234_val_oof.parquet"), on=["s1", "m", "src"], how="left")
    v = v.with_columns(pl.coalesce("pb", "p").alias("p")).drop("pb")
    x2 = pl.read_parquet(R / "ext2b_val_oof.parquet").select("s1", "m", "src", pl.col("p").cast(pl.Float64)).join(v, on=["s1", "m", "src"], how="anti")
    return pl.concat([v, x2.with_columns(pl.lit(2, pl.UInt8).alias("kind"))], how="vertical_relaxed")


if __name__ == "__main__":
    split = pl.read_parquet(A / "splits/split_v1.parquet")
    ids = split.filter(pl.col("role") == "val").select("s1_id", "country")
    truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
    tp = (truth.drop_nulls("match_id").select(num("s1_id").alias("s1"), num("match_id").alias("m"),
                                               pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src")).with_columns(pl.lit(1, pl.Int8).alias("y")))
    allv = v13_val()
    val_ids = split.filter(pl.col("role") == "val")["s1_id"].str.slice(3).cast(pl.UInt32)
    ev = rknn_pairs("train", val_ids, [str(A / "blocking/union_train/*.parquet"), R / "ext_val_pairs.parquet", R / "ext2_val_pairs.parquet"]
                    + ([R / "ext2e_val_pairs.parquet"] if a.tag != "e" else []) + ([R / "ext2f_val_pairs.parquet"] if a.tag not in ("e", "f") else []),
                    R / f"ext2{a.tag}_val_pairs.parquet")
    log(f"val ext3 pairs {ev.height:,}, true {ev.join(tp, on=['s1', 'm', 'src'], how='semi').height:,}")
    dv = X2.featurize("train", ev, allv.select("s1", "m", "src", "p"), R / f"ext2{a.tag}_val_set.parquet")
    dv = dv.drop("y", strict=False).join(tp, on=["s1", "m", "src"], how="left").with_columns(pl.col("y").fill_null(0))
    feats = [c for c in dv.columns if c not in {"s1", "m", "src", "y", "label"} and dv[c].dtype != pl.Utf8]
    log(f"val ext3 set {dv.height:,} pairs, pos {dv['y'].sum():,}, {len(feats)} features")
    if a.test and a.reuse_models:
        models = [lgb.Booster(model_file=str(A / f"models/lgb_ext2{a.tag}_f{k}.txt")) for k in range(a.folds)]
        feats = models[0].feature_name()
    else:
        X = dv.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
        y = dv["y"].to_numpy()
        fold = (dv["s1"].hash(seed=11) % a.folds).to_numpy()
        inner = (dv["s1"].hash(seed=13) % 10).to_numpy() == 0
        oof = np.zeros(dv.height, dtype=np.float32)
        models = []
        for k in range(a.folds):
            fit, es = (fold != k) & ~inner, (fold != k) & inner
            b = lgb.train(PARAMS3, lgb.Dataset(X[fit], y[fit], feature_name=feats), 3000,
                          valid_sets=[lgb.Dataset(X[es], y[es])], callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[fold == k] = b.predict(X[fold == k], num_iteration=b.best_iteration)
            b.save_model(str(A / f"models/lgb_ext2{a.tag}_f{k}.txt"), num_iteration=b.best_iteration)
            models.append(b)
            log(f"ext3 fold {k}: {b.best_iteration} rounds")
        imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
        log("ext3 top gain: " + ", ".join(f"{f} {g/sum(x[1] for x in imp):.3f}" for f, g in imp[:12]))
        from sklearn.metrics import roc_auc_score
        log(f"ext3 OOF AUC {roc_auc_score(y, oof):.5f}")
        e3 = dv.select("s1", "m", "src").with_columns(pl.Series("p", oof, dtype=pl.Float64))
        e3.write_parquet(R / f"ext2{a.tag}_val_oof.parquet")
        r = evaluate(to_pairs(X2.final_select(allv, {0: 0.75, 1: 0.7, 2: 0.75})), truth, ids)
        log(f"v13 stack (no empty model): F0.5 {r['f05']:.5f} US {r['by_country']['US']['f05']:.5f} IN {r['by_country']['India']['f05']:.5f}")
        comb = pl.concat([allv, e3.with_columns(pl.lit(4, pl.UInt8).alias("kind"))], how="vertical_relaxed")
        for t3 in (0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
            r = evaluate(to_pairs(X2.final_select(comb, {0: 0.75, 1: 0.7, 2: 0.75, 4: t3})), truth, ids)
            log(f"  + ext3 @{t3}: F0.5 {r['f05']:.5f} US {r['by_country']['US']['f05']:.5f} IN {r['by_country']['India']['f05']:.5f}")
    if a.test:
        et = rknn_pairs("test", None, [str(A / "blocking/union_test/*.parquet"), R / "ext_test_pairs.parquet", R / "ext2_test_pairs.parquet"]
                        + ([R / "ext2e_test_pairs.parquet"] if a.tag != "e" else []) + ([R / "ext2f_test_pairs.parquet"] if a.tag not in ("e", "f") else []),
                        R / f"ext2{a.tag}_test_pairs.parquet")
        log(f"test ext3 pairs {et.height:,}")
        main_t = pl.concat([pl.read_parquet(A / f"test_scores_ens1234_{k}.parquet", columns=["s1", "m", "src", "p"]) for k in ("main", "ext")]
                           + [pl.read_parquet(A / "test_scores_ext2b.parquet").select("s1", "m", "src", pl.col("p").cast(pl.Float64))])
        del dv, allv
        X2.featurize_predict("test", et, main_t, models, feats, A / f"test_scores_ext2{a.tag}.parquet", R / f"ext2{a.tag}_test_set.parquet")
        log("test ext3 scored")
