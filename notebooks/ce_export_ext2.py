"""Export ext2 pairs in the uncertain band (ext2 LightGBM p in [LO, HI)) for cross-encoder scoring on Kaggle.

    python notebooks/ce_export_ext2.py val   # -> kaggle/ce_ext2_upload/ext2.parquet [id, a, b] + artifacts/kaggle_ext2/keys_val.parquet
    python notebooks/ce_export_ext2.py test  # same for the test pairs (keys_test.parquet)
Val pairs use the 5-fold OOF p (artifacts/refine/ext2_val_oof.parquet), test pairs artifacts/test_scores_ext2.parquet.
Text = "name , address" of each side (raw strings), as in ce_export2.py.
"""
import os
import sys
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
A, R = ROOT / "artifacts", ROOT / "artifacts" / "refine"
OUT = A / "kaggle_ext2"
OUT.mkdir(exist_ok=True)
UP = ROOT / "kaggle" / "ce_ext2_upload"
num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731
LO, HI = 0.005, 0.999


def texts(split):
    t = pl.col("business_name").fill_null("") + " , " + pl.col("business_address").fill_null("")
    s1 = pl.read_parquet(A / f"processed/{split}_source1.parquet").select(num("entity_id").alias("s1"), t.alias("a"))
    pool = pl.concat([pl.read_parquet(A / f"processed/{split}_source{i}.parquet")
                        .select(num("entity_id").alias("m"), pl.lit(i, pl.UInt8).alias("src"), t.alias("b")) for i in (2, 3)])
    return s1, pool


parts, keys = [], []
TAG = os.environ.get("EXT2_TAG", "")  # variant suffix (e.g. w)
jobs = {"val": ("train", R / f"ext2{TAG}_val_oof.parquet"), "test": ("test", A / f"test_scores_ext2{TAG}.parquet")}
for tag in sys.argv[1:]:
    split, path = jobs[tag]
    k = pl.read_parquet(path, columns=["s1", "m", "src", "p"]).filter((pl.col("p") >= LO) & (pl.col("p") < HI)).select("s1", "m", "src")
    s1, pool = texts(split)
    d = k.join(s1, on="s1", how="left").join(pool, on=["m", "src"], how="left")
    keys.append(k.with_columns(pl.lit(tag).alias("split")))
    parts.append(d.select("a", "b"))
    print(tag, k.height, "pairs in band", d.select(pl.col("a").is_null().sum(), pl.col("b").is_null().sum()).row(0))
keys = pl.concat(keys).with_row_index("id")
for tag in sys.argv[1:]:  # one key file per split name (ext2_blend.py filters on the split column)
    keys.write_parquet(OUT / f"keys_{TAG}{tag}.parquet")
pl.concat(parts).with_row_index("id").select("id", "a", "b").write_parquet(UP / "ext2.parquet", compression="zstd")
print("wrote", keys.height, "pairs")
