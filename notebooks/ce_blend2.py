"""Blend several cross-encoder scores into v5f for pairs in the uncertain band (ensemble version of ce_blend.py).

    python notebooks/ce_blend2.py --ce ce1 ce2            # val report
    python notebooks/ce_blend2.py --ce ce1 ce2 ce3 --test # + test scores -> artifacts/test_scores_ens_{main,ext}.parquet
CE sources (keys + logits): ce1 = artifacts/kaggle_ce (band [0.02, 0.995)), ce2 / ce3 = artifacts/kaggle_ce2 (band
[0.01, 0.999)) with logits in ce2_{split}.parquet / ce3_{split}.parquet. The band is the union of the sources' bands;
a pair missing from a source gets a null score (LightGBM handles it).
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
from ce_blend import PARAMS, decide  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.train import one_owner, to_pairs  # noqa: E402

A, R = ROOT / "artifacts", ROOT / "artifacts" / "refine"
SRC = {"ce1": (A / "kaggle_ce", "ce_{s}.parquet"), "ce2": (A / "kaggle_ce2", "ce2_{s}.parquet"), "ce3": (A / "kaggle_ce2", "ce3_{s}.parquet")}
ap = argparse.ArgumentParser()
ap.add_argument("--ce", nargs="+", default=["ce1"])
ap.add_argument("--test", action="store_true")
ap.add_argument("--empty-thr", type=float, default=0.5)
ap.add_argument("--tag", default="ens")
ap.add_argument("--fr-extra", default=None, help="dir with fr_keys.parquet + ce1_fr/ce2_fr.parquet: extra (French) test pairs scored above the band")
a = ap.parse_args()
num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731


def allpairs(main_path, ext_path, main_set, ext_set):
    m = pl.read_parquet(main_path, columns=["s1", "m", "src", "p"]).with_columns(pl.read_parquet(main_set, columns=["hv_eq", "a_empty_2", "nm_subst"]))
    e = pl.read_parquet(ext_path, columns=["s1", "m", "src", "p"]).join(
        pl.read_parquet(ext_set, columns=["s1", "m", "src", "hv_eq", "a_empty_2", "nm_subst"]), on=["s1", "m", "src"], how="left")
    return pl.concat([m.with_columns(pl.lit(0, pl.UInt8).alias("kind")),
                      e.join(m, on=["s1", "m", "src"], how="anti").with_columns(pl.lit(1, pl.UInt8).alias("kind"))], how="vertical_relaxed")


def band(split, allp):
    b = None
    for name in a.ce:
        d, pat = SRC[name]
        sc = (pl.read_parquet(d / f"{split}_keys.parquet").select("id", "s1", "m", "src")
                .join(pl.read_parquet(d / pat.format(s=split)).rename({"logit": name}), on="id").drop("id"))
        b = sc if b is None else b.join(sc, on=["s1", "m", "src"], how="full", coalesce=True)
    if split == "test" and a.fr_extra:
        d = Path(a.fr_extra)
        ex = pl.read_parquet(d / "fr_keys.parquet")
        for name in a.ce:
            f = d / f"{name}_fr.parquet"
            ex = ex.join(pl.read_parquet(f).rename({"logit": name}), on="id", how="inner") if f.exists() else ex.with_columns(pl.lit(None, pl.Float32).alias(name))
        ex = ex.drop("id").join(b.select("s1", "m", "src"), on=["s1", "m", "src"], how="anti")
        print(f"  + {ex.height:,} extra French pairs scored above the band")
        b = pl.concat([b, ex.select(b.columns)], how="vertical_relaxed")
    b = b.join(allp, on=["s1", "m", "src"], how="inner")
    return b.with_columns((pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log().alias("lp"))


FEATS = ["lp", "kind", "hv_eq", "a_empty_2", "nm_subst"] + a.ce


def final(allp, bb, empty_thr):
    p = allp.join(bb.select("s1", "m", "src", "pb"), on=["s1", "m", "src"], how="left").with_columns(pl.coalesce("pb", "p").alias("p"))
    p = p.with_columns(pl.when(pl.col("kind") == 1).then(0.7).otherwise(0.75).alias("t"))
    own = one_owner(p.select("s1", "m", "src", "p", "t", "kind"))
    sel = own.filter(pl.col("p") >= pl.col("t"))
    if empty_thr:
        resc = own.join(sel.select("s1").unique(), on="s1", how="anti").sort("p", descending=True).group_by("s1").head(1).filter(pl.col("p") >= empty_thr)
        sel = pl.concat([sel, resc])
    return p, sel


if __name__ == "__main__":
    ids = pl.read_parquet(A / "splits/split_v1.parquet").filter(pl.col("role") == "val").select("s1_id", "country")
    truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
    tp = (truth.drop_nulls("match_id").select(num("s1_id").alias("s1"), num("match_id").alias("m"),
                                               pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src")).with_columns(pl.lit(1, pl.Int8).alias("y")))
    allv = allpairs(R / "val_oof_v4f.parquet", R / "ext_val_oof_v5f.parquet", R / "val_set.parquet", R / "ext_val_set.parquet")
    bv = band("val", allv).join(tp, on=["s1", "m", "src"], how="left").with_columns(pl.col("y").fill_null(0))
    X = bv.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
    y = bv["y"].to_numpy()
    fold = (bv["s1"].hash(seed=33) % 5).to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    models = []
    for k in range(5):
        b = lgb.train(PARAMS, lgb.Dataset(X[fold != k], y[fold != k], feature_name=FEATS), 600)
        oof[fold == k] = b.predict(X[fold == k])
        models.append(b)
    from sklearn.metrics import roc_auc_score
    print(f"val band {bv.height:,} pairs | AUC p {roc_auc_score(y, bv['p'].to_numpy()):.5f} | " +
          " ".join(f"{c} {roc_auc_score(y, bv[c].fill_null(-20).to_numpy()):.5f}" for c in a.ce) + f" | blend {roc_auc_score(y, oof):.5f}")
    bvb = bv.with_columns(pl.Series("pb", oof))
    for et in (None, 0.5):
        _, sel = final(allv, bvb, et)
        r = evaluate(to_pairs(sel), truth, ids)
        print(f"{'+'.join(a.ce)} empty_thr {et}: F0.5 {r['f05']:.5f} US {r['by_country']['US']['f05']:.5f} IN {r['by_country']['India']['f05']:.5f}")
    bvb.select("s1", "m", "src", "pb").write_parquet(R / f"{a.tag}_val_oof.parquet")
    if a.test:
        allt = allpairs(A / "test_scores_v4f.parquet", R / "test_scores_ext_v5f.parquet", R / "test_set.parquet", R / "ext_test_set.parquet")
        bt = band("test", allt)
        Xt = bt.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
        bt = bt.with_columns(pl.Series("pb", np.mean([m.predict(Xt) for m in models], axis=0)))
        p, _ = final(allt, bt, None)
        p.filter(pl.col("kind") == 0).select("s1", "m", "src", "p").write_parquet(A / f"test_scores_{a.tag}_main.parquet")
        p.filter(pl.col("kind") == 1).select("s1", "m", "src", "p").write_parquet(A / f"test_scores_{a.tag}_ext.parquet")
        print("test scores written")
