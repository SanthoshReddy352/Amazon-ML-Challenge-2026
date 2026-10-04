"""Write output/candidate_pairs.tsv = every pair the final models scored: union blocking candidates (blocking +
embedding neighbours) + blocking-extension pairs (+ ext2 record-centric reverse-blocking pairs with --ext2). One row per test S1 (empty list when none), streamed by country.

    python notebooks/write_candidates.py --out output/candidate_pairs.tsv
"""
import os
import argparse
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
A = ROOT / "artifacts"
ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=ROOT / "output/candidate_pairs.tsv")
ap.add_argument("--ext2", nargs="*", default=None, help="ext2 variant tags to include ('' = ext2, 'w' = ext2w): artifacts/refine/ext2<tag>_test_pairs.parquet")
a = ap.parse_args()

s1 = pl.read_parquet(A / "normalized/test_source1.parquet", columns=["entity_id", "country"]).with_columns(
    pl.col("entity_id").str.slice(3).cast(pl.UInt32).alias("s1"))
ext_files = [A / "refine/ext_test_pairs.parquet"] + [A / f"refine/ext2{tag}_test_pairs.parquet" for tag in (a.ext2 if a.ext2 is not None else [])]
a.out.parent.mkdir(parents=True, exist_ok=True)
seen, total = 0, 0
with open(a.out, "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for c in s1["country"].unique().sort().to_list():
        ids = s1.filter(pl.col("country") == c).select("s1", "entity_id")
        main = pl.scan_parquet(A / f"blocking/union_test/{c}_*.parquet").select("s1", "m", "src").collect()
        exts = [pl.scan_parquet(f).select("s1", "m", "src").join(ids.lazy().select("s1"), on="s1", how="semi").collect() for f in ext_files]
        pairs = pl.concat([main] + exts).unique()
        del exts
        total += pairs.height
        agg = (pairs.with_columns((pl.lit("S") + pl.col("src").cast(pl.Utf8) + "-" + pl.col("m").cast(pl.Utf8)).alias("mid"))
                    .group_by("s1").agg(pl.col("mid").sort().str.join(",")))
        rows = ids.join(agg, on="s1", how="left")
        for eid, mid in rows.select("entity_id", "mid").iter_rows():
            f.write(f"{eid}\t{mid or ''}\n")
        seen += rows.height
        print(f"{c}: {rows.height:,} S1, {pairs.height:,} pairs", flush=True)
        del main, pairs, agg, rows
print(f"wrote {seen:,} rows, {total:,} candidate pairs -> {a.out}")
