"""Export for the fine-tuned bi-encoder retriever (kaggle/biencoder): texts + training pairs.

    python notebooks/biencoder_export.py      # -> kaggle/biencoder_upload/{train_text2,test_text2,fit_pairs}.parquet

Text = "normalised name | normalised address" (address cut to 90 chars) for every S1/S2/S3 record of both splits.
Training pairs come ONLY from fit S1 (the validation S1 stay unseen): every fit true pair that the current
candidate set misses (the hard positives the retriever must learn) + a uniform sample of the others, 1.2M in total.
Each pair carries a hard negative S1: the other S1 whose blocking score for the record is highest.
"""
import os
from pathlib import Path

import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
A = ROOT / "artifacts"
UP = ROOT / "kaggle" / "biencoder_upload"
UP.mkdir(parents=True, exist_ok=True)
num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731
N_PAIRS = 1_200_000

for split in ("train", "test"):
    parts = []
    for s in (1, 2, 3):
        d = pl.read_parquet(A / f"normalized/{split}_source{s}.parquet", columns=["entity_id", "country", "name_norm", "addr_norm"])
        parts.append(d.select(num("entity_id").alias("id"), pl.lit(s, pl.UInt8).alias("src"), "country",
                              (pl.col("name_norm").fill_null("") + " | " + pl.col("addr_norm").fill_null("").str.slice(0, 90)).alias("text")))
    t = pl.concat(parts)
    t.write_parquet(UP / f"{split}_text2.parquet", compression="zstd")
    print(split, t.height)

fit = pl.read_parquet(A / "splits/split_v1.parquet").filter(pl.col("role") == "fit").select(num("s1_id").alias("a"))
gt = (pl.read_parquet(A / "processed/train_gt_pairs.parquet").drop_nulls("match_id")
        .select(num("s1_id").alias("a"), num("match_id").alias("m"), pl.col("match_id").str.slice(1, 1).cast(pl.UInt8).alias("src"))
        .join(fit, on="a", how="semi"))
inc = pl.scan_parquet(A / "blocking/union_train/*.parquet").select(pl.col("s1").alias("a"), "m", "src").join(gt.lazy(), on=["a", "m", "src"], how="semi").collect()
miss = gt.join(inc, on=["a", "m", "src"], how="anti")
rest = gt.join(inc, on=["a", "m", "src"], how="semi").sample(N_PAIRS - miss.height, seed=7)
pairs = pl.concat([miss.with_columns(pl.lit(1, pl.Int8).alias("hard")), rest.with_columns(pl.lit(0, pl.Int8).alias("hard"))])
print("fit true pairs", gt.height, "missed by candidates", miss.height, "sampled", pairs.height)
# hard negative: the competing S1 with the highest blocking score for the same record
comp = (pl.scan_parquet(A / "blocking/union_train/*.parquet").select("s1", "m", "src", "score")
          .join(pairs.lazy().select("m", "src", "a"), on=["m", "src"], how="inner").filter(pl.col("s1") != pl.col("a"))
          .sort("score", descending=True).group_by("m", "src").agg(pl.col("s1").first().alias("b")).collect())
pairs = pairs.join(comp, on=["m", "src"], how="left").sample(fraction=1.0, shuffle=True, seed=8)
print("with a hard negative:", pairs["b"].is_not_null().sum())
pairs.write_parquet(UP / "fit_pairs.parquet")
