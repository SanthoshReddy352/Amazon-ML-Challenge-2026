"""Decision layer for v4+: one owner per record -> threshold -> matching_results.tsv (+ official validator).

    python notebooks/make_submission.py --scores artifacts/test_scores_v4.parquet --thr 0.75 --out submissions/v4
    python notebooks/make_submission.py --scores artifacts/test_scores_v4.parquet --ext artifacts/test_scores_ext.parquet ...

--ext: extra (s1, m, src, p) pairs from the blocking extension; they compete with the main pairs for ownership.
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.evaluate import write_id_list_tsv  # noqa: E402
from src.train import one_owner, to_pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scores", type=Path, required=True)
ap.add_argument("--ext", type=Path, default=None)
ap.add_argument("--thr", type=float, default=0.75)
ap.add_argument("--ext-thr", type=float, default=None)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--country-thr", nargs="*", default=[], help="per-country override, e.g. France:0.9 (main pairs)")
ap.add_argument("--country-ext-thr", nargs="*", default=[], help="per-country override for extension pairs")
ap.add_argument("--empty-thr", type=float, default=None, help="for S1 that would be empty, add their best owned candidate if p >= this")
ap.add_argument("--drop-contested", nargs="*", default=[], help="COUNTRY:P2 - drop a selection when the record's 2nd-best S1 has p >= P2")
a = ap.parse_args()

pred = pl.read_parquet(a.scores, columns=["s1", "m", "src", "p"]).with_columns(pl.lit(a.thr).alias("thr"), pl.lit(False).alias("is_ext"))
if a.ext:
    ext = pl.read_parquet(a.ext, columns=["s1", "m", "src", "p"]).with_columns(pl.lit(a.ext_thr or a.thr).alias("thr"), pl.lit(True).alias("is_ext"))
    pred = pl.concat([pred, ext.join(pred, on=["s1", "m", "src"], how="anti")])
s1_all = pl.read_parquet(ROOT / "artifacts/normalized/test_source1.parquet", columns=["entity_id", "country"])
cty = s1_all.select(pl.col("entity_id").str.slice(3).cast(pl.UInt32).alias("s1"), "country")
pred = pred.join(cty, on="s1", how="left")
for spec, is_ext in [(x, False) for x in a.country_thr] + [(x, True) for x in a.country_ext_thr]:
    c, t = spec.split(":")
    cond = (pl.col("country") == c) & (pl.col("is_ext") == is_ext)
    pred = pred.with_columns(pl.when(cond).then(float(t)).otherwise(pl.col("thr")).alias("thr"))
own = one_owner(pred)
sel = own.filter(pl.col("p") >= pl.col("thr"))
if a.empty_thr is not None:
    rescue = (own.join(sel.select("s1").unique(), on="s1", how="anti").sort("p", descending=True).group_by("s1").head(1)
                 .filter(pl.col("p") >= a.empty_thr))
    print(f"empty-S1 rescue (p>={a.empty_thr}): +{rescue.height:,} pairs")
    sel = pl.concat([sel, rescue.select(sel.columns)])
if a.drop_contested:
    p2 = (pred.sort("p", descending=True).with_columns(pl.int_range(pl.len()).over("m", "src").alias("_r"))
              .filter(pl.col("_r") == 1).select("m", "src", pl.col("p").alias("p2")))
    sel = sel.join(p2, on=["m", "src"], how="left").with_columns(pl.col("p2").fill_null(0.0))
    for spec in a.drop_contested:
        c, t = spec.split(":")
        bad = (pl.col("country") == c) & (pl.col("p2") >= float(t))
        print(f"drop-contested {c} p2>={t}: {sel.filter(bad).height:,} pairs")
        sel = sel.filter(~bad)
a.out.mkdir(parents=True, exist_ok=True)
write_id_list_tsv(to_pairs(sel), s1_all["entity_id"], a.out / "matching_results.tsv", "matched_entity_ids")
st = sel.group_by("country").agg(pl.len().alias("pairs"), pl.col("s1").n_unique().alias("nonempty"))
st = st.join(cty.group_by("country").len(), on="country").with_columns(
    (pl.col("pairs") / pl.col("len")).round(3).alias("per_s1"), (1 - pl.col("nonempty") / pl.col("len")).round(4).alias("empty"))
print(st.sort("country"))
print(f"{sel.height:,} pairs written to {a.out / 'matching_results.tsv'}")
r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str((a.out / "matching_results.tsv").resolve()),
                    "--test-dir", "dataset/test"], cwd=ROOT / "student_resource", capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-800:])
