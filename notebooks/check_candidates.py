"""Lean replacement for the validator's soft candidate check: every final (S1, record) pair must be one of the pairs the
models scored (the sources write_candidates.py writes to candidate_pairs.tsv), plus a row/header check of the TSV.

    python notebooks/check_candidates.py --selected submissions/v17/selected.parquet --candidates output/candidate_pairs.tsv
"""
import os
import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
A, R = ROOT / "artifacts", ROOT / "artifacts" / "refine"
ap = argparse.ArgumentParser()
ap.add_argument("--selected", type=Path, required=True)
ap.add_argument("--candidates", type=Path, required=True)
ap.add_argument("--ext2", nargs="*", default=["", "e", "f"])
a = ap.parse_args()
K = ["s1", "m", "src"]
sel = pl.read_parquet(a.selected, columns=K)
left = sel
for src in [str(A / "blocking/union_test/*.parquet"), R / "ext_test_pairs.parquet"] + [R / f"ext2{t}_test_pairs.parquet" for t in a.ext2]:
    left = left.join(pl.scan_parquet(src).select(K).join(left.lazy(), on=K, how="semi").collect(), on=K, how="anti")
    print(f"after {Path(str(src)).name}: {left.height:,} selected pairs not yet found", flush=True)
n_s1 = pl.scan_parquet(A / "normalized/test_source1.parquet").select(pl.len()).collect().item()
with open(a.candidates, encoding="utf-8") as f:
    header = f.readline().rstrip("\n")
    rows = sum(1 for _ in f)
ok = left.height == 0 and rows == n_s1 and header == "source1_entity_id\tcandidate_entity_ids"
print(f"selected {sel.height:,} pairs, outside candidates {left.height:,}; candidate TSV rows {rows:,} (test S1 {n_s1:,}), header ok {header.startswith('source1_entity_id')}")
print("CANDIDATES OK" if ok else "CANDIDATES FAIL")
sys.exit(0 if ok else 1)
