"""Write output/candidate_pairs.tsv = every pair the final models scored: union blocking candidates (blocking +
embedding neighbours) + blocking-extension pairs. One row per test S1 (empty list when none), streamed by country.

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
a = ap.parse_args()

s1 = pl.read_parquet(A / "normalized/test_source1.parquet", columns=["entity_id", "country"]).with_columns(
    pl.col("entity_id").str.slice(3).cast(pl.UInt32).alias("s1"))
ext = pl.read_parquet(A / "refine/ext_test_pairs.parquet", columns=["s1", "m", "src"])
a.out.parent.mkdir(parents=True, exist_ok=True)
seen, total = 0, 0
with open(a.out, "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for c in s1["country"].unique().sort().to_list():
        ids = s1.filter(pl.col("country") == c).select("s1", "entity_id")
        main = pl.scan_parquet(A / f"blocking/union_test/{c}_*.parquet").select("s1", "m", "src").collect()
        pairs = pl.concat([main, ext.join(ids.select("s1"), on="s1", how="semi")]).unique()
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
