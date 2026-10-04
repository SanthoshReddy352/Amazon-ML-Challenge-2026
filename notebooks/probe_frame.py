"""Save the v17 test decision frame once (re-scored pf, raw p, kind, country, the record's pf sum over ALL its S1) so
LB probes / variants of the decision layer can be generated in seconds by notebooks/probe_make.py.

    python notebooks/probe_frame.py      # -> artifacts/refine/v17_test_frame.parquet (pairs with pf >= 0.005)
"""
import os
import sys
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "notebooks"))
sys.argv = ["final_v13.py", "--blend", "ens1234", "--ext-tags", "e", "f", "--fr-ext2", "--fr-empty"]
import final_v13 as F  # noqa: E402

A, R = F.A, F.R
K = ["s1", "m", "src"]
cty = pl.read_parquet(A / "normalized/test_source1.parquet", columns=["entity_id", "country"]).select(F.num("entity_id").alias("s1"), "country")
t = F.test_frame()
t = F.add_ext2(t, A / "test_scores_ext2b.parquet", cty)
for i, tag in enumerate(["e", "f"]):
    t = F.add_ext2(t, A / f"test_scores_ext2{tag}b.parquet", cty, kind=4 + i)
t = t.select(*K, "p", "pf", "kind")
bt = pl.read_parquet(R / "bandstack_test.parquet", columns=[*K, "ps"])
t = t.join(bt, on=K, how="left").with_columns(pl.col("pf").alias("pf0"), pl.coalesce("ps", "pf").alias("pf")).drop("ps")
t = t.with_columns(pl.col("pf").sum().over("m", "src").alias("rsum"), pl.col("pf").max().over("m", "src").alias("rmax"))
t = t.filter(pl.col("pf") >= 0.005).join(cty, on="s1", how="left")
t.write_parquet(R / "v17_test_frame.parquet")
print(t.height, t.group_by("country").len().sort("country"))
