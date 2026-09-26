"""Extra refiner training data: 440k fresh fit S1 (unseen by stage-1 and stage-2, so p_v3 is out-of-sample).

Run after `python -m src.features --split train --subset fit_sample3 ... --out artifacts/features/fit_sample3_u`:
    python notebooks/build_fit3.py
Outputs artifacts/features/fit_sample3_s2u (group features), artifacts/refine/fit3_set.parquet (refiner rows).
"""
import os
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.refine import pair_new_features  # noqa: E402
from src.stage2 import build  # noqa: E402

A = ROOT / "artifacts"
F, M, R = A / "features", A / "models", A / "refine"
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)  # noqa: E731
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="3")
N = ap.parse_args().name

s2 = F / f"fit_sample{N}_s2u"
if not any(s2.glob("*.parquet")):
    build(M / "lgb_s1_u.txt", F / f"fit_sample{N}_u", A / "normalized", "train", s2, log=lambda m: None)
log("group features ready")
meta = json.loads((M / "lgb_v3_u.json").read_text())
f2 = meta["features"]
b = lgb.Booster(model_file=str(M / "lgb_v3_u.txt"))
cols = ["s1", "m", "src"] + [c for c in f2 if c != "src"] + ["label"]
parts = []
for p in sorted(s2.glob("*.parquet")):
    d = pl.read_parquet(p)
    p3 = b.predict(d.select([pl.col(c).cast(pl.Float32) for c in f2]).to_numpy())
    d = d.select(cols).with_columns(pl.Series("p3", p3, dtype=pl.Float32)).filter(pl.col("p3") >= 0.02)
    parts.append(d.with_columns(pl.col(pl.Float64).cast(pl.Float32)))
d = pl.concat(parts)
log(f"fit{N} pairs with p3>=0.02: {d.height:,} (pos {d['label'].sum():,})")
s1, pool = pl.read_parquet(R / "train_s1.parquet"), pl.read_parquet(R / "train_pool.parquet")
d = pair_new_features(d, s1, pool)
d.write_parquet(R / f"fit{N}_set.parquet")
log(f"fit{N}_set written")
