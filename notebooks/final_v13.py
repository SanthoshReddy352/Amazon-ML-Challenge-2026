"""Final assembly (v13): main + extension scores (cross-encoder blend), ext2 reverse-blocking pairs (blended),
one owner per record, per-kind thresholds, empty-S1 rescue, then the empty-address record rescue model.

    python notebooks/final_v13.py --blend ens123 --no-ext2 --no-empty --test --out submissions/_v11check   # must equal v11
    python notebooks/final_v13.py --blend ens1234 --test --out submissions/v13

Val (fold 0 of split_v1) is scored with every component; test uses the same thresholds. French pairs above the
original CE band keep v10's blended decisions (as in v11). --fr-ext2 / --fr-empty control whether ext2 and the
empty-address rescue also apply to France (a country without training labels).
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
sys.path.insert(0, str(ROOT / "notebooks"))
from src.evaluate import evaluate, write_id_list_tsv  # noqa: E402
from src.train import one_owner, to_pairs  # noqa: E402
import empty_rescue as ER  # noqa: E402

A, R = ROOT / "artifacts", ROOT / "artifacts" / "refine"
ap = argparse.ArgumentParser()
ap.add_argument("--blend", default="ens1234")
ap.add_argument("--no-ext2", action="store_true")
ap.add_argument("--no-empty", action="store_true")
ap.add_argument("--ext2-thr", type=float, default=0.75)
ap.add_argument("--ext2-tag", default="", help="ext2 variant suffix (e.g. w)")
ap.add_argument("--ext-tags", nargs="*", default=[], help="extra embedding-neighbour pair sets ext2<tag>b: e = ext3 (kind 4), f = ext4 (kind 5)")
ap.add_argument("--ext3-thr", type=float, default=0.75)
ap.add_argument("--empty-q", type=float, default=0.75)
ap.add_argument("--fr-ext2", action="store_true")
ap.add_argument("--fr-empty", action="store_true")
ap.add_argument("--test", action="store_true")
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()
num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731
THR = {0: 0.75, 1: 0.7, 2: a.ext2_thr, **{4 + i: a.ext3_thr for i in range(len(a.ext_tags))}}


def select(allp: pl.DataFrame, empty_thr: float = 0.5) -> pl.DataFrame:
    p = allp.with_columns(pl.col("kind").replace_strict(THR, return_dtype=pl.Float64).alias("t"))
    own = one_owner(p.select("s1", "m", "src", pl.col("pf").alias("p"), "t", "kind"))
    sel = own.filter(pl.col("p") >= pl.col("t"))
    resc = own.join(sel.select("s1").unique(), on="s1", how="anti").sort("p", descending=True).group_by("s1").head(1).filter(pl.col("p") >= empty_thr)
    return pl.concat([sel, resc]).select("s1", "m", "src", "kind")


def val_frame() -> pl.DataFrame:
    m = pl.read_parquet(R / "val_oof_v4f.parquet", columns=["s1", "m", "src", "p"]).with_columns(pl.lit(0, pl.UInt8).alias("kind"))
    e = (pl.read_parquet(R / "ext_val_oof_v5f.parquet", columns=["s1", "m", "src", "p"]).join(m, on=["s1", "m", "src"], how="anti")
           .with_columns(pl.lit(1, pl.UInt8).alias("kind")))
    v = pl.concat([m, e], how="vertical_relaxed").join(pl.read_parquet(R / f"{a.blend}_val_oof.parquet"), on=["s1", "m", "src"], how="left")
    return v.with_columns(pl.coalesce("pb", "p").alias("pf")).drop("pb")


def test_frame() -> pl.DataFrame:
    """Refiner / extension p plus the final blended p; French pairs above the CE band take v10's blended p (as v11)."""
    extra = pl.read_parquet(A / "kaggle_cefr/fr_keys.parquet", columns=["s1", "m", "src"])
    extra = extra.join(pl.read_parquet(A / "kaggle_ce2/test_keys.parquet", columns=["s1", "m", "src"]), on=["s1", "m", "src"], how="anti")
    parts = []
    for kind, part, raw in [(0, "main", A / "test_scores_v4f.parquet"), (1, "ext", R / "test_scores_ext_v5f.parquet")]:
        base = pl.read_parquet(A / f"test_scores_{a.blend}_{part}.parquet").rename({"p": "pf"})
        fr = pl.read_parquet(A / f"test_scores_ens12fr_{part}.parquet").rename({"p": "p_fr"}).join(extra, on=["s1", "m", "src"], how="semi")
        base = base.join(fr, on=["s1", "m", "src"], how="left").with_columns(pl.coalesce("p_fr", "pf").alias("pf")).drop("p_fr")
        base = base.join(pl.read_parquet(raw, columns=["s1", "m", "src", "p"]), on=["s1", "m", "src"], how="left")
        parts.append(base.with_columns(pl.lit(kind, pl.UInt8).alias("kind")))
    return pl.concat(parts, how="vertical_relaxed")


def add_ext2(frame: pl.DataFrame, path: Path, cty: pl.DataFrame | None, kind: int = 2) -> pl.DataFrame:
    e = pl.read_parquet(path).select("s1", "m", "src", pl.col("p").cast(pl.Float64).alias("pf"))
    if cty is not None and not a.fr_ext2:
        e = e.join(cty.filter(pl.col("country") == "France").select("s1"), on="s1", how="anti")
    e = e.join(frame.select("s1", "m", "src"), on=["s1", "m", "src"], how="anti")
    return pl.concat([frame, e.with_columns(pl.col("pf").alias("p"), pl.lit(kind, pl.UInt8).alias("kind"))], how="diagonal_relaxed")


def report(tag, sel, truth, ids):
    r = evaluate(to_pairs(sel), truth, ids)
    print(f"{tag}: F0.5 {r['f05']:.5f} P {r['macro_p']:.5f} R {r['macro_r']:.5f} US {r['by_country']['US']['f05']:.5f} "
          f"IN {r['by_country']['India']['f05']:.5f}", flush=True)
    return r["f05"]


if __name__ == "__main__":
    ids = pl.read_parquet(A / "splits/split_v1.parquet").filter(pl.col("role") == "val").select("s1_id", "country")
    truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
    tp = (truth.drop_nulls("match_id").select(num("s1_id").alias("s1"), num("match_id").alias("m"),
                                               pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src")).with_columns(pl.lit(1, pl.Int8).alias("y")))
    v = val_frame()
    report(f"val {a.blend} main+ext", select(v), truth, ids)
    if not a.no_ext2:
        v = add_ext2(v, R / f"ext2{a.ext2_tag}b_val_oof.parquet", None)
    for i, tag in enumerate(a.ext_tags):
        v = add_ext2(v, R / f"ext2{tag}b_val_oof.parquet", None, kind=4 + i)
    sv = select(v)
    report("val + ext2" + "".join(f" + ext2{t}" for t in a.ext_tags) if not a.no_ext2 else "val", sv, truth, ids)
    tv = None
    if not a.no_empty:
        tv = ER.top_candidates("train", v.filter(pl.col("kind") < 2), sv, R / "val_set.parquet", R / "ext_val_set.parquet")
        tv = tv.join(tp, on=["s1", "m", "src"], how="left").with_columns(pl.col("y").fill_null(0))

    if not a.test:
        if tv is not None:
            q, _ = ER.fit_apply(tv, None)
            add = tv.filter(pl.Series(q >= a.empty_q)).select("s1", "m", "src")
            report(f"val + empty rescue q>={a.empty_q} (+{add.height:,})", pl.concat([sv.select("s1", "m", "src"), add]), truth, ids)
        sys.exit(0)

    s1_all = pl.read_parquet(A / "normalized/test_source1.parquet", columns=["entity_id", "country"])
    cty = s1_all.select(num("entity_id").alias("s1"), "country")
    t = test_frame()
    if not a.no_ext2:
        t = add_ext2(t, A / f"test_scores_ext2{a.ext2_tag}b.parquet", cty)
    for i, tag in enumerate(a.ext_tags):
        t = add_ext2(t, A / f"test_scores_ext2{tag}b.parquet", cty, kind=4 + i)
    st = select(t)
    if tv is not None:
        tt = ER.top_candidates("test", t.filter(pl.col("kind") < 2), st, R / "test_set.parquet", R / "ext_test_set.parquet")
        if not a.fr_empty:
            tt = tt.filter(pl.col("country") != "France")
        q, qt = ER.fit_apply(tv, tt)
        add_v = tv.filter(pl.Series(q >= a.empty_q)).select("s1", "m", "src")
        report(f"val + empty rescue q>={a.empty_q} (+{add_v.height:,})", pl.concat([sv.select("s1", "m", "src"), add_v]), truth, ids)
        add_t = tt.filter(pl.Series(qt >= a.empty_q)).select("s1", "m", "src").with_columns(pl.lit(3, pl.UInt8).alias("kind"))
        st = pl.concat([st, add_t], how="vertical_relaxed")
    st = st.join(cty, on="s1", how="left")
    print(st.group_by("country", "kind").agg(pl.len()).sort("country", "kind"))
    per = st.group_by("country").agg(pl.len().alias("pairs"), pl.col("s1").n_unique().alias("nonempty")).join(
        cty.group_by("country").len(), on="country").with_columns((pl.col("pairs") / pl.col("len")).round(4).alias("per_s1"),
                                                                  (1 - pl.col("nonempty") / pl.col("len")).round(4).alias("empty"))
    print(per.sort("country"))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        write_id_list_tsv(to_pairs(st), s1_all["entity_id"], a.out / "matching_results.tsv", "matched_entity_ids")
        st.select("s1", "m", "src", "kind").write_parquet(a.out / "selected.parquet")
        print(f"{st.height:,} pairs -> {a.out / 'matching_results.tsv'}")
        r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str((a.out / "matching_results.tsv").resolve()),
                            "--test-dir", "dataset/test"], cwd=ROOT / "student_resource", capture_output=True, text=True)
        print(r.stdout[-600:], r.stderr[-600:])
