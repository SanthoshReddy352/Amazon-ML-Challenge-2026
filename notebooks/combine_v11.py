"""v11 test scores: 3-encoder blend (ens123) for the band pairs, v10's blended probability (ens12fr) for the French
pairs above the band (scored only by CE v2), so no pair reaches the blend with a feature pattern it never trained on.

    python notebooks/combine_v11.py            # -> artifacts/test_scores_v11_{main,ext}.parquet
"""
import os
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
A = ROOT / "artifacts"
extra = pl.read_parquet(A / "kaggle_cefr/fr_keys.parquet", columns=["s1", "m", "src"])
k2 = pl.read_parquet(A / "kaggle_ce2/test_keys.parquet", columns=["s1", "m", "src"])
extra = extra.join(k2, on=["s1", "m", "src"], how="anti")   # French pairs outside the CE band
for part in ("main", "ext"):
    base = pl.read_parquet(A / f"test_scores_ens123_{part}.parquet")
    fr = pl.read_parquet(A / f"test_scores_ens12fr_{part}.parquet").rename({"p": "p_fr"}).join(extra, on=["s1", "m", "src"], how="semi")
    out = base.join(fr, on=["s1", "m", "src"], how="left").with_columns(pl.coalesce("p_fr", "p").alias("p")).drop("p_fr")
    out.write_parquet(A / f"test_scores_v11_{part}.parquet")
    print(part, out.height, "French above-band pairs taken from v10:", fr.height)
