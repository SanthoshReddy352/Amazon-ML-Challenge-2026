"""Decision-layer variants on the saved v17 test frame (notebooks/probe_frame.py), per country, for leaderboard probes.
Countries not mentioned keep v17's settings, so an LB difference isolates the changed country / class.

    python notebooks/probe_make.py --name v17_check                          # must reproduce v17 exactly
    python notebooks/probe_make.py --name p_fr_thr_up --fr-shift 0.10        # French thresholds +0.10
    python notebooks/probe_make.py --name p_fr_alpha2 --fr-alpha 2
Decision (as band_stack.py --norm): one owner per record by pf; keep if pf / max(1, pf + alpha * (record sum - pf)) >=
kind threshold (+ shift); empty-S1 rescue at empty_thr; v17's empty-address rescue picks (kind 3) kept when unclaimed.
"""
import os
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.evaluate import write_id_list_tsv  # noqa: E402
from src.train import to_pairs  # noqa: E402

A, R = ROOT / "artifacts", ROOT / "artifacts" / "refine"
K = ["s1", "m", "src"]
THR = {0: 0.75, 1: 0.7, 2: 0.75, 4: 0.75, 5: 0.75}
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
for c in ("fr", "us", "in"):
    ap.add_argument(f"--{c}-shift", type=float, default=0.0, help="threshold shift for all kinds")
    ap.add_argument(f"--{c}-alpha", type=float, default=1.0)
    ap.add_argument(f"--{c}-empty-thr", type=float, default=0.5)
    ap.add_argument(f"--{c}-no-resc3", action="store_true", help="drop v17's empty-address rescue picks")
    ap.add_argument(f"--{c}-drop-empty-ties", type=int, default=0, help="drop picks of empty-address records whose name key is shared by >= N S1")
    ap.add_argument(f"--{c}-raw", action="store_true", help="decide on the pre-re-scorer probability (v16c without the band re-scorer)")
    ap.add_argument(f"--{c}-drop-kinds", type=int, nargs="*", default=[], help="drop picks of these pair kinds (2 ext2, 3 empty rescue, 4 ext3, 5 ext4)")
a = ap.parse_args()
CODE = {"France": "fr", "US": "us", "India": "in"}


def decide(f: pl.DataFrame, shift: float, alpha: float, empty_thr: float) -> pl.DataFrame:
    f = f.with_columns((pl.col("kind").replace_strict(THR, return_dtype=pl.Float64) + shift).alias("t"),
                       (pl.col("pf") / pl.max_horizontal(pl.lit(1.0), pl.col("pf") + alpha * (pl.col("rsum") - pl.col("pf")))).alias("pn"))
    own = f.filter(pl.col("pf") >= pl.col("rmax"))
    sel = own.filter(pl.col("pn") >= pl.col("t"))
    resc = own.join(sel.select("s1").unique(), on="s1", how="anti").sort("pn", descending=True).group_by("s1").head(1).filter(pl.col("pn") >= empty_thr)
    return pl.concat([sel, resc]).select(*K, "kind")


def empty_ties(min_k: int) -> pl.DataFrame:
    num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731
    s1 = pl.read_parquet(A / "normalized/test_source1.parquet", columns=["country", "name_key"])
    kc = s1.group_by("country", "name_key").agg(pl.len().alias("k_n")).filter(pl.col("k_n") >= min_k)
    recs = pl.concat([pl.read_parquet(A / f"normalized/test_source{s}.parquet", columns=["entity_id", "country", "name_key", "addr_empty"])
                        .filter(pl.col("addr_empty")).select(num("entity_id").alias("m"), pl.lit(s, pl.UInt8).alias("src"), "country", "name_key") for s in (2, 3)])
    return recs.join(kc, on=["country", "name_key"], how="semi").select("m", "src")


if __name__ == "__main__":
    fr = pl.read_parquet(R / "v17_test_frame.parquet")
    v17 = pl.read_parquet(ROOT / "submissions/v17/selected.parquet")
    resc3 = v17.filter(pl.col("kind") == 3).select(*K)
    cty = pl.read_parquet(A / "normalized/test_source1.parquet", columns=["entity_id", "country"]).select(
        pl.col("entity_id").str.slice(3).cast(pl.UInt32).alias("s1"), "country")
    parts = []
    for country, c in CODE.items():
        g = vars(a)
        fc = fr.filter(pl.col("country") == country)
        if g[f"{c}_raw"]:   # pre-re-scorer probabilities; record max / sum recomputed over the saved frame (pf >= 0.005)
            fc = fc.with_columns(pl.col("pf0").alias("pf")).with_columns(pl.col("pf").sum().over("m", "src").alias("rsum"), pl.col("pf").max().over("m", "src").alias("rmax"))
        s = decide(fc, g[f"{c}_shift"], g[f"{c}_alpha"], g[f"{c}_empty_thr"])
        if not g[f"{c}_no_resc3"]:
            r3 = resc3.join(cty.filter(pl.col("country") == country), on="s1", how="semi").join(s.select("m", "src"), on=["m", "src"], how="anti")
            s = pl.concat([s, r3.with_columns(pl.lit(3, pl.UInt8).alias("kind"))])
        if g[f"{c}_drop_kinds"]:
            s = s.filter(~pl.col("kind").is_in(g[f"{c}_drop_kinds"]))
        if g[f"{c}_drop_empty_ties"]:
            s = s.join(empty_ties(g[f"{c}_drop_empty_ties"]), on=["m", "src"], how="anti")
        parts.append(s)
    st = pl.concat(parts).join(cty, on="s1", how="left")
    old = v17.select(*K).join(cty, on="s1", how="left")
    new = st.select(*K, "country")
    add = new.join(old, on=K, how="anti").group_by("country").len("added")
    rem = old.join(new, on=K, how="anti").group_by("country").len("removed")
    print(f"{a.name}: {st.height:,} pairs; vs v17:", cty.group_by("country").len().join(add, on="country", how="left").join(rem, on="country", how="left").fill_null(0).sort("country").rows())
    out = ROOT / "submissions" / a.name
    out.mkdir(parents=True, exist_ok=True)
    s1_all = pl.read_parquet(A / "normalized/test_source1.parquet", columns=["entity_id"])["entity_id"]
    write_id_list_tsv(to_pairs(st), s1_all, out / "matching_results.tsv", "matched_entity_ids")
    st.select(*K, "kind").write_parquet(out / "selected.parquet")
    shutil.copy(ROOT / "submissions/v17/v17_code.zip", out / f"{a.name}_code.zip")
    r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str((out / "matching_results.tsv").resolve()), "--test-dir", "dataset/test"],
                       cwd=ROOT / "student_resource", capture_output=True, text=True)
    print([ln for ln in r.stdout.splitlines() if "PASS" in ln or "FAIL" in ln or "ERROR" in ln])
