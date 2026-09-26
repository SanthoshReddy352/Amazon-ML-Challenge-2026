"""France-only submission variants: US/India rows are copied from a base submission, French rows are re-selected
from the scores under one rule, so the LB difference to the base measures that rule on France alone.

    python notebooks/france_variant.py --base submissions/v5e --scores artifacts/test_scores_v4e.parquet \
        --ext artifacts/refine/test_scores_ext_v5e.parquet --out submissions/fr_A --drop-contested 0.5
Rules (combine freely):
    --fr-thr T / --fr-ext-thr T          French thresholds (default 0.75 / 0.7)
    --drop-contested P2                  drop a French pick if the record's 2nd-best S1 has p >= P2
    --drop-class CLASS:PMIN              drop French picks of CLASS with p < PMIN
    --add-class CLASS:PMIN               also accept owned French pairs of CLASS with p >= PMIN
Classes: empty_addr, sib_shift, house_conflict, acr_domain, all_new_name, word_swap, add_word, drop_word, same_words
"""
import argparse
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.train import one_owner  # noqa: E402

A, R = ROOT / "artifacts", ROOT / "artifacts" / "refine"
ap = argparse.ArgumentParser()
ap.add_argument("--base", type=Path, required=True)
ap.add_argument("--scores", type=Path, required=True)
ap.add_argument("--ext", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--fr-thr", type=float, default=0.75)
ap.add_argument("--fr-ext-thr", type=float, default=0.7)
ap.add_argument("--drop-contested", type=float, default=None)
ap.add_argument("--drop-class", nargs="*", default=[])
ap.add_argument("--add-class", nargs="*", default=[])
a = ap.parse_args()

num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731
s1 = pl.read_parquet(A / "normalized/test_source1.parquet", columns=["entity_id", "country"])
fr = s1.filter(pl.col("country") == "France").select(num("entity_id").alias("s1"))
FC = ["s1", "m", "src", "hv_eq", "hv_diff", "nm_subst", "nm_miss1", "nm_extra2", "nm_len1", "a_empty_2", "n_acr", "n_is_domain_2"]
main = pl.read_parquet(a.scores, columns=["s1", "m", "src", "p"]).with_columns(pl.lit(a.fr_thr).alias("thr"))
main = main.with_columns(pl.read_parquet(R / "test_set.parquet", columns=FC[3:]))  # same row order as test_set
ext = (pl.read_parquet(a.ext, columns=["s1", "m", "src", "p"]).with_columns(pl.lit(a.fr_ext_thr).alias("thr"))
         .join(pl.read_parquet(R / "ext_test_set.parquet", columns=FC), on=["s1", "m", "src"], how="left"))
pred = pl.concat([main, ext.join(main, on=["s1", "m", "src"], how="anti").select(main.columns)], how="vertical_relaxed")
cls = (pl.when(pl.col("a_empty_2") == 1).then(pl.lit("empty_addr"))
         .when((pl.col("hv_diff") >= 1) & (pl.col("hv_diff") <= 25)).then(pl.lit("sib_shift"))
         .when(pl.col("hv_eq") == 0).then(pl.lit("house_conflict"))
         .when((pl.col("n_acr") == 1) | (pl.col("n_is_domain_2") == 1)).then(pl.lit("acr_domain"))
         .when((pl.col("nm_miss1") == pl.col("nm_len1")) & (pl.col("nm_len1") > 0)).then(pl.lit("all_new_name"))
         .when(pl.col("nm_subst") == 1).then(pl.lit("word_swap"))
         .when(pl.col("nm_extra2") > 0).then(pl.lit("add_word"))
         .when(pl.col("nm_miss1") > 0).then(pl.lit("drop_word"))
         .otherwise(pl.lit("same_words")))
pred = pred.with_columns(cls.alias("cls"))
for spec in a.add_class:
    c, t = spec.split(":")
    pred = pred.with_columns(pl.when(pl.col("cls") == c).then(pl.min_horizontal(pl.col("thr"), pl.lit(float(t)))).otherwise(pl.col("thr")).alias("thr"))
own = one_owner(pred)
sel = own.filter(pl.col("p") >= pl.col("thr")).join(fr, on="s1")
base_fr = sel.height
if a.drop_contested is not None:
    p2 = (pred.sort("p", descending=True).with_columns(pl.int_range(pl.len()).over("m", "src").alias("_r"))
              .filter(pl.col("_r") == 1).select("m", "src", pl.col("p").alias("p2")))
    sel = sel.join(p2, on=["m", "src"], how="left").filter(pl.col("p2").fill_null(0) < a.drop_contested)
for spec in a.drop_class:
    c, t = spec.split(":")
    sel = sel.filter(~((pl.col("cls") == c) & (pl.col("p") < float(t))))
print(f"French picks: {base_fr:,} under thresholds -> {sel.height:,} after rules ({sel.height / fr.height:.3f} per S1)")
fr_rows = (sel.with_columns((pl.lit("S") + pl.col("src").cast(pl.Utf8) + "-" + pl.col("m").cast(pl.Utf8)).alias("mid"))
              .group_by("s1").agg(pl.col("mid").sort().str.join(",")))
base = pl.read_csv(a.base / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
fr_ids = s1.filter(pl.col("country") == "France")["entity_id"].implode()
fr_map = fr_rows.select((pl.lit("S1-") + pl.col("s1").cast(pl.Utf8)).alias("source1_entity_id"), pl.col("mid").alias("fr"))
out = (base.join(fr_map, on="source1_entity_id", how="left")
           .with_columns(pl.when(pl.col("source1_entity_id").is_in(fr_ids)).then(pl.col("fr")).otherwise(pl.col("matched_entity_ids")).alias("x"))
           .select("source1_entity_id", "x"))
a.out.mkdir(parents=True, exist_ok=True)
with open(a.out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for x, y in out.iter_rows():
        f.write(f"{x}\t{y or ''}\n")
r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str((a.out / "matching_results.tsv").resolve()),
                    "--test-dir", "dataset/test"], cwd=ROOT / "student_resource", capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1])
