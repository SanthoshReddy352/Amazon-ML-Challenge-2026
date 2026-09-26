"""Export text pairs for the Kaggle cross-encoder (Step 10.2).

train : refiner rows of fit_sample3 + fit_sample4 (S1 disjoint from val): every uncertain pair (p3 < 0.995)
        + 250k confident pairs, with labels.
val   : val pairs in the uncertain band of the current system (main v4e p or extension p in [0.02, 0.995)).
test  : test pairs in the same band.
Text = "name , address" of each side (raw strings). Local key files map row ids back to (s1, m, src, kind).
Outputs artifacts/kaggle_ce/{train,val,test}.parquet (+ *_keys.parquet kept locally, not uploaded).
"""
from pathlib import Path

import os
import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
A, R = ROOT / "artifacts", ROOT / "artifacts" / "refine"
OUT = A / "kaggle_ce"
OUT.mkdir(exist_ok=True)
num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731
LO, HI = 0.02, 0.995


def texts(split):
    t = lambda d: pl.col("business_name").fill_null("") + " , " + pl.col("business_address").fill_null("")  # noqa: E731
    s1 = pl.read_parquet(A / f"processed/{split}_source1.parquet").select(num("entity_id").alias("s1"), t(None).alias("a"))
    pool = pl.concat([pl.read_parquet(A / f"processed/{split}_source{i}.parquet")
                        .select(num("entity_id").alias("m"), pl.lit(i, pl.UInt8).alias("src"), t(None).alias("b")) for i in (2, 3)])
    return s1, pool


def write(name, keys, split, label=None):
    s1, pool = texts(split)
    d = keys.with_row_index("id").join(s1, on="s1", how="left").join(pool, on=["m", "src"], how="left")
    cols = ["id", "a", "b"] + (["label"] if label else [])
    d.select(cols).write_parquet(OUT / f"{name}.parquet", compression="zstd")
    keys.with_row_index("id").write_parquet(OUT / f"{name}_keys.parquet")
    print(name, d.height, d.select(pl.col("a").is_null().sum(), pl.col("b").is_null().sum()).row(0))


# train: fit3 + fit4 refiner rows
tr = pl.concat([pl.read_parquet(R / f"fit{n}_set.parquet", columns=["s1", "m", "src", "label", "p3"]) for n in (3, 4)])
tr = pl.concat([tr.filter(pl.col("p3") < HI), tr.filter(pl.col("p3") >= HI).sample(250_000, seed=5)]).sample(fraction=1.0, shuffle=True, seed=6)
write("train", tr.select("s1", "m", "src", "label"), "train", label=True)

for name, split, main, ext in [("val", "train", R / "val_oof_v4e.parquet", R / "ext_val_oof_v5e.parquet"),
                               ("test", "test", A / "test_scores_v4e.parquet", R / "test_scores_ext_v5e.parquet")]:
    m = pl.read_parquet(main, columns=["s1", "m", "src", "p"]).filter((pl.col("p") >= LO) & (pl.col("p") < HI)).with_columns(pl.lit(0, pl.UInt8).alias("kind"))
    e = pl.read_parquet(ext, columns=["s1", "m", "src", "p"]).filter((pl.col("p") >= LO) & (pl.col("p") < HI)).with_columns(pl.lit(1, pl.UInt8).alias("kind"))
    k = pl.concat([m, e.join(m, on=["s1", "m", "src"], how="anti")]).select("s1", "m", "src", "kind")
    write(name, k, split)
