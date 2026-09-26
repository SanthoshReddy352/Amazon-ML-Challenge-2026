"""France-only LB experiments on top of an LB-scored base whose French rows are v5c's.

US/India rows are copied from --base; French rows are re-selected from the v5c scores (v4c main + v5c extension)
with ONE rule, so LB(new) - LB(base) measures that rule on France alone.

    python notebooks/fr_exp.py --out submissions/frX_loose55 --fr-thr 0.55
    python notebooks/fr_exp.py --out submissions/frY_strict92 --fr-thr 0.92
    python notebooks/fr_exp.py --out submissions/frZ_single95 --single-max 0.95
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.train import one_owner  # noqa: E402

A = ROOT / "artifacts"
ap = argparse.ArgumentParser()
ap.add_argument("--base", type=Path, default=ROOT / "submissions/v5e2_usin")
ap.add_argument("--main", type=Path, default=A / "test_scores_v4c.parquet")
ap.add_argument("--ext", type=Path, default=A / "refine/ext_v3_top5/test_scores_ext.parquet")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--fr-thr", type=float, default=0.75, help="French main + extension threshold (v5c: 0.75 / 0.7)")
ap.add_argument("--single-max", type=float, default=None, help="French S1 with exactly one pick whose p < this -> empty")
ap.add_argument("--check", action="store_true", help="no rule: verify French rows reproduce the base")
a = ap.parse_args()

num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731
s1 = pl.read_parquet(A / "normalized/test_source1.parquet", columns=["entity_id", "country"])
fr = s1.filter(pl.col("country") == "France").select(num("entity_id").alias("s1"))
et = 0.7 if a.fr_thr == 0.75 else a.fr_thr
main = pl.read_parquet(a.main, columns=["s1", "m", "src", "p"]).with_columns(pl.lit(a.fr_thr).alias("t"))
ext = pl.read_parquet(a.ext, columns=["s1", "m", "src", "p"]).with_columns(pl.lit(et).alias("t"))
pred = pl.concat([main, ext.join(main, on=["s1", "m", "src"], how="anti")]).join(fr, on="s1")
sel = one_owner(pred).filter(pl.col("p") >= pl.col("t"))
n0 = sel.height
if a.single_max is not None:
    one = sel.group_by("s1").agg(pl.len().alias("k"), pl.col("p").max().alias("pm")).filter((pl.col("k") == 1) & (pl.col("pm") < a.single_max))
    sel = sel.join(one.select("s1"), on="s1", how="anti")
    print(f"single-pick rule: {one.height:,} French S1 set empty")
print(f"French picks {n0:,} -> {sel.height:,} ({sel.height / fr.height:.3f} per S1)")
rows = (sel.with_columns((pl.lit("S") + pl.col("src").cast(pl.Utf8) + "-" + pl.col("m").cast(pl.Utf8)).alias("mid"))
           .group_by("s1").agg(pl.col("mid").sort().str.join(","))
           .select((pl.lit("S1-") + pl.col("s1").cast(pl.Utf8)).alias("source1_entity_id"), pl.col("mid").alias("fr")))
base = pl.read_csv(a.base / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
fr_ids = s1.filter(pl.col("country") == "France")["entity_id"].implode()
isfr = pl.col("source1_entity_id").is_in(fr_ids)
out = base.join(rows, on="source1_entity_id", how="left").with_columns(
    pl.when(isfr).then(pl.col("fr")).otherwise(pl.col("matched_entity_ids")).alias("x"))
if a.check:
    norm = lambda c: pl.col(c).fill_null("").str.split(",").list.sort().list.join(",")  # noqa: E731
    same = out.filter(isfr).select((norm("x") == norm("matched_entity_ids")).alias("s"))["s"]
    print(f"CHECK: French rows identical to base: {same.mean():.6f}")
a.out.mkdir(parents=True, exist_ok=True)
with open(a.out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for x, y in out.select("source1_entity_id", "x").iter_rows():
        f.write(f"{x}\t{y or ''}\n")
r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str((a.out / "matching_results.tsv").resolve()),
                    "--test-dir", "dataset/test"], cwd=ROOT / "student_resource", capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1])
