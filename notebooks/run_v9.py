"""v9 = v8 + a second blocking extension aimed at the pairs v8 never sees (isolated: writes only to artifacts/v9/).

Val error analysis (STATUS 13.4): the non-empty pairs v8 misses are mostly gibberish / transliterated rebrands, house-number
changes and missing house numbers, whose street is often typo'd ('249 Aqeduct Saint' vs '383 Aqueduct Street'). New keys:
    H  house_v2 | locality word          (street typo'd or dropped)
    K  street-word skeleton | locality   (vowel-robust street: 'aqueduct' / 'aqeduct' -> 'aqdct')
    L  house_v2 | street-word skeleton
plus the v5 key types with a looser rare-key cap. Pairs already in the union candidates or the v5 extension are excluded.

    python notebooks/run_v9.py gen          # val pairs + recall report on v8's misses
    python notebooks/run_v9.py fit          # features + 5-fold LightGBM + combined val report
    python notebooks/run_v9.py test         # test pairs + scores -> artifacts/v9/test_scores_x2.parquet
"""
import os
import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.blockext import _key_types, load  # noqa: E402

A, R, V = ROOT / "artifacts", ROOT / "artifacts" / "refine", ROOT / "artifacts" / "v9"
V.mkdir(exist_ok=True)
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)  # noqa: E731
num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731
TOP, MAX_DF = int(os.environ.get("V9_TOP", 5)), int(os.environ.get("V9_MAXDF", 100))


def skel(col: str) -> pl.Expr:
    """List of street words -> first letter + consonants, repeats collapsed ('aqueduct' -> 'aqdct')."""
    e = pl.element().str.slice(1).str.replace_all(r"[aeiouy]", "")
    for ch in "bcdfghjklmnpqrstvwxz":  # polars regex has no backreferences
        e = e.str.replace_all(ch + "+", ch)
    return pl.col(col).list.eval(pl.element().str.slice(0, 1) + e).list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).list.unique()


def key_types(df: pl.DataFrame, ids: list[str]) -> list[pl.LazyFrame]:
    h, lf = pl.col("house_v2"), df.lazy().with_columns(skel("sw").alias("sk"))
    hk = lambda *parts: pl.concat_str(parts).hash(seed=17).alias("k")  # noqa: E731
    return _key_types(df, ids) + [
        lf.filter(h != "").explode("lw").drop_nulls("lw").select(*ids, hk(pl.lit("H|"), h, pl.lit("|"), pl.col("lw"))),
        lf.explode("sk").drop_nulls("sk").explode("lw").drop_nulls("lw").select(*ids, hk(pl.lit("K|"), pl.col("sk"), pl.lit("|"), pl.col("lw"))),
        lf.filter(h != "").explode("sk").drop_nulls("sk").select(*ids, hk(pl.lit("L|"), h, pl.lit("|"), pl.col("sk"))),
    ]


def pairs(s1: pl.DataFrame, pool: pl.DataFrame, existing: pl.DataFrame, chunk: int = 100_000) -> pl.DataFrame:
    """Same contract as blockext.extension_pairs, with the extra key types and TOP / MAX_DF from the environment."""
    pks = []
    for sk, pk in zip(key_types(s1, ["s1"]), key_types(pool, ["m", "src"])):
        pk = pk.join(sk.select("k").unique(), on="k", how="semi")
        dfc = pk.group_by("k").len().filter(pl.col("len") <= MAX_DF).select("k")
        pks.append(pk.join(dfc, on="k", how="semi").unique().collect())
    pk = pl.concat(pks).unique()
    del pks
    out = []
    ids = s1["s1"]
    for i in range(0, s1.height, chunk):
        part = s1.filter(pl.col("s1").is_in(ids.slice(i, chunk).implode()))
        sk = pl.concat([f.collect() for f in key_types(part, ["s1"])]).unique()
        pr = (sk.join(pk, on="k").group_by("s1", "m", "src").agg(pl.len().alias("xk"))
                .join(existing, on=["s1", "m", "src"], how="anti"))
        if pr.height == 0:
            continue
        w = (pr.join(part.select("s1", pl.col("name_core").alias("n1"), pl.col("addr_norm").alias("a1")), on="s1")
               .join(pool.select("m", "src", pl.col("name_core").alias("n2"), pl.col("addr_norm").alias("a2")), on=["m", "src"]))
        ns = cpdist(w["n1"].to_list(), w["n2"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        ad = cpdist(w["a1"].fill_null("").to_list(), w["a2"].fill_null("").to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        w = w.select("s1", "m", "src", "xk").with_columns(pl.Series("xs_name", ns), pl.Series("xs_addr", ad))
        w = w.with_columns((pl.col("xs_name") + pl.col("xs_addr") + 20 * pl.col("xk")).alias("xs"),
                           (pl.col("xs_addr") + 20 * pl.col("xk")).alias("xs_a"))
        # the name+address rank favours distractors (S1 name + descriptor, same street), so also keep the best by address
        # alone: gibberish / transliterated rebrands share only the address
        out.append(pl.concat([w.sort("xs", descending=True).group_by("s1").head(TOP),
                              w.sort("xs_a", descending=True).group_by("s1").head(TOP)]).unique(["s1", "m", "src"]).drop("xs_a"))
    return pl.concat(out) if out else pl.DataFrame()


def gen(split, s1_ids, cand_glob, ext_path, out):
    if out.exists():
        return pl.read_parquet(out)
    countries = pl.read_parquet(A / "normalized" / f"{split}_source1.parquet", columns=["country"])["country"].unique().sort().to_list()
    parts = []
    for c in countries:
        part = out.with_name(f"{out.stem}_{c}.parquet")
        if not part.exists():
            s1c = load(A / "normalized", R, split, 1, c)
            if s1_ids is not None:
                s1c = s1c.filter(pl.col("s1").is_in(s1_ids.implode()))
            if s1c.height == 0:
                continue
            pool = pl.concat([load(A / "normalized", R, split, s, c) for s in (2, 3)])
            keep = s1c.select("s1")
            ex = pl.concat([pl.scan_parquet(cand_glob).select("s1", "m", "src").join(keep.lazy(), on="s1", how="semi").collect(),
                            pl.read_parquet(ext_path, columns=["s1", "m", "src"]).join(keep, on="s1", how="semi")],
                           how="vertical_relaxed").unique()
            pairs(s1c, pool, ex).write_parquet(part)
            del ex, pool, s1c
            gc.collect()
        parts.append(pl.read_parquet(part))
        log(f"  {split} {c}: {parts[-1].height:,} new pairs")
    e = pl.concat(parts)
    e.write_parquet(out)
    return e


if __name__ == "__main__":
    step = sys.argv[1]
    split = pl.read_parquet(A / "splits/split_v1.parquet")
    val_ids = split.filter(pl.col("role") == "val")["s1_id"].str.slice(3).cast(pl.UInt32)
    if step == "gen":
        tag = f"t{TOP}d{MAX_DF}"
        ev = gen("train", val_ids, str(A / "blocking/union_train/*.parquet"), R / "ext_val_pairs.parquet", V / f"x2_val_pairs_{tag}.parquet")
        log(f"val new pairs {ev.height:,} ({ev.height / len(val_ids):.2f} per S1)")
        truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
        tp = truth.drop_nulls("match_id").select(num("s1_id").alias("s1"), num("match_id").alias("m"),
                                                 pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src"))
        hit = ev.join(tp, on=["s1", "m", "src"], how="semi")
        log(f"true pairs among new: {hit.height:,} ({hit.height / max(ev.height, 1):.3f} of new pairs)")
        cats = os.environ.get("V9_CATS")
        if cats and Path(cats).exists():
            c = pl.read_parquet(cats).filter(pl.col("hit") == 0)
            c = c.join(hit.select("s1", "m", "src").with_columns(pl.lit(1).alias("rec")), on=["s1", "m", "src"], how="left").with_columns(pl.col("rec").fill_null(0))
            print(c.group_by("cat").agg(pl.len().alias("missed"), pl.col("rec").sum().alias("recovered")).sort("cat"))
