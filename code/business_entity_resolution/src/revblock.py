"""Record-centric (reverse) blocking: every S2/S3 record queries an index of the S1 records of its country.

The main blocker (src/blocking.py) keeps, for each S1, its top-k records. A copy whose name is typo'd or rebranded
is then crowded out of its S1's list by exact-name records from elsewhere, although, seen from the record, that S1
is still its best match. Val: 3.1% of true pairs never reach the matcher. With the record's own top-3 S1 (by total
token-overlap score) plus its best S1 by name and by address, ~22% of them come back
(35k missed -> 7.6k recovered at max_df 300).

Tokens and weights are those of the main blocker (same typed tokens, IDF over the S1 side), built symmetrically.
Output (per country): s1, m, src, r_score, r_name_score, r_addr_score, r_rank, r_rank_name, r_rank_addr.
"""
import time
from pathlib import Path

import polars as pl

from .blocking import NORM_COLS, TYPE_CODE, TYPE_WEIGHT, query, tokens_chunked

DEFAULTS = dict(max_df=300, k_total=3, k_name=1, k_addr=1, k_empty=10)


def _num(col: str) -> pl.Expr:
    return pl.col(col).str.slice(3).cast(pl.UInt32)


def build_s1_index(s1: pl.DataFrame, max_df: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    toks = tokens_chunked(s1, "query")  # symmetric: 5-grams for every long concatenated name
    n = s1.height
    df = toks.group_by("tok", "typ").agg(pl.len().alias("df"))
    tw = pl.DataFrame({"typ": [TYPE_CODE[t] for t in TYPE_WEIGHT], "tw": list(TYPE_WEIGHT.values())},
                      schema={"typ": pl.UInt8, "tw": pl.Float64})
    weights = (df.filter(pl.col("df") <= max_df).join(tw, on="typ")
                 .select("tok", "typ", (pl.col("tw") * (pl.lit(n + 1) / pl.col("df")).log()).cast(pl.Float32).alias("wt")))
    return toks.join(weights.select("tok"), on="tok").select("tok", "rid"), weights


def generate(norm_dir: Path, split: str, out_dir: Path, s1_keep: pl.Series | None = None, max_df: int = 300,
             k_total: int = 3, k_name: int = 1, k_addr: int = 1, k_empty: int = 10, q_chunk: int = 1_000_000, log=print):
    """Writes out_dir/<country>_<j>.parquet. s1_keep (u32 ids) restricts the OUTPUT to those S1 (ranks are still
    computed against every S1 of the country, as on test)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = NORM_COLS + ["addr_empty"]
    s1_all = pl.read_parquet(norm_dir / f"{split}_source1.parquet", columns=NORM_COLS)
    pool_lf = pl.scan_parquet([norm_dir / f"{split}_source{i}.parquet" for i in (2, 3)]).select(cols)
    for country in s1_all["country"].unique().sort().to_list():
        done = out_dir / f"{country}.done"
        if done.exists():
            continue
        t0 = time.time()
        s1 = s1_all.filter(pl.col("country") == country).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
        postings, weights = build_s1_index(s1, max_df)
        smap = s1.select("rid", _num("entity_id").alias("s1"))
        recs = pool_lf.filter(pl.col("country") == country).collect()
        n = 0
        for j in range(0, recs.height, q_chunk):
            q = recs.slice(j, q_chunk).with_row_index("rid").with_columns(pl.col("rid").cast(pl.UInt32))
            qmap = q.select(pl.col("rid").alias("qid"), _num("entity_id").alias("m"),
                            pl.col("entity_id").str.slice(1, 1).cast(pl.UInt8).alias("src"), "addr_empty")
            parts = []
            for res in query(q, postings, weights, pl.Series([1] * s1.height, dtype=pl.UInt8), max(k_total, k_empty), k_name, k_addr):
                # record-level context over ALL S1 of the country, before any output restriction
                best = res.group_by("qid").agg(
                    pl.col("score").filter(pl.col("rank") == 1).first().alias("r_best1"),
                    pl.col("score").filter(pl.col("rank") == 2).first().alias("r_best2"),
                    pl.col("name_score").max().alias("r_name_best"), pl.col("addr_score").max().alias("r_addr_best"))
                r = res.drop("src").join(qmap, on="qid").filter(
                    (pl.col("rank") <= k_total) | (pl.col("rank_name") <= k_name) | (pl.col("rank_addr") <= k_addr)
                    | (pl.col("addr_empty") & (pl.col("rank") <= k_empty)))
                r = r.join(smap, on="rid").join(best, on="qid")
                if s1_keep is not None:
                    r = r.filter(pl.col("s1").is_in(s1_keep.implode()))
                parts.append(r.select("s1", "m", "src", pl.col("score").alias("r_score"), pl.col("name_score").alias("r_name_score"),
                                      pl.col("addr_score").alias("r_addr_score"), pl.col("rank").alias("r_rank"),
                                      pl.col("rank_name").alias("r_rank_name"), pl.col("rank_addr").alias("r_rank_addr"),
                                      "r_best1", pl.col("r_best2").fill_null(0.0), "r_name_best", "r_addr_best"))
            part = pl.concat(parts)
            part.write_parquet(out_dir / f"{country}_{j // q_chunk:03d}.parquet")
            n += part.height
            log(f"  {split} {country}: records {min(j + q_chunk, recs.height):,}/{recs.height:,}, pairs {n:,}, {time.time() - t0:.0f}s")
        done.write_text(str(n))
        del postings, weights, recs
    return out_dir
