"""ext5 input: merge the bi-encoder's record -> S1 top-5 (kaggle/biencoder, dir 0) with its deeper S1 -> record top-15
(kaggle/bfwd_kernel, dir 1) into artifacts/kaggle_bknn2/bknn2_{train,test}.parquet for run_ext3.py --source bknn2 and
band_stack.py --bk bknn2.

    python notebooks/bfwd_merge.py
"""
import os
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
A = ROOT / "artifacts"
(A / "kaggle_bknn2").mkdir(exist_ok=True)
COLS = ["s1", "m", "src", "cos", "rank", "dir"]
for split in ("train", "test"):
    rev = pl.read_parquet(A / f"kaggle_bknn/bknn_{split}.parquet", columns=COLS).filter(pl.col("dir") == 0)
    fwd = pl.read_parquet(A / f"kaggle_bfwd/bfwd_{split}.parquet", columns=COLS)
    out = pl.concat([rev, fwd.select(rev.columns).cast(rev.schema)])
    out.write_parquet(A / f"kaggle_bknn2/bknn2_{split}.parquet", compression="zstd")
    print(split, f"reverse {rev.height:,} + forward {fwd.height:,} (S1 {fwd['s1'].n_unique():,})")
