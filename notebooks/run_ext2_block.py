"""ext2 step 1: record-centric reverse blocking (src/revblock.py) for val S1 (train split) and for test.

    python notebooks/run_ext2_block.py train     # -> artifacts/revblock/train_val/  (pairs of val S1 only)
    python notebooks/run_ext2_block.py test      # -> artifacts/revblock/test/
    python notebooks/run_ext2_block.py train --tag w --k-total 10 --k-name 3 --k-addr 3   # wider variant -> train_val_w/
"""
import argparse
import os
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.revblock import generate  # noqa: E402

A = ROOT / "artifacts"
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)  # noqa: E731
ap = argparse.ArgumentParser()
ap.add_argument("split", choices=["train", "test"])
ap.add_argument("--tag", default="")
ap.add_argument("--k-total", type=int, default=3)
ap.add_argument("--k-name", type=int, default=1)
ap.add_argument("--k-addr", type=int, default=1)
ap.add_argument("--k-empty", type=int, default=10)
a = ap.parse_args()
kw = dict(k_total=a.k_total, k_name=a.k_name, k_addr=a.k_addr, k_empty=a.k_empty, log=log)
if a.split == "train":
    keep = pl.read_parquet(A / "splits/split_v1.parquet").filter(pl.col("role") == "val")["s1_id"].str.slice(3).cast(pl.UInt32)
    generate(A / "normalized", "train", A / f"revblock/train_val{('_' + a.tag) if a.tag else ''}", s1_keep=keep, **kw)
else:
    generate(A / "normalized", "test", A / f"revblock/test{('_' + a.tag) if a.tag else ''}", **kw)
log("done")
