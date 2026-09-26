"""Quick OOF comparison of refiner hyper-parameters on the cached val set (no test).
    python notebooks/tune_refiner.py '{"num_leaves":127,"learning_rate":0.03}'
"""
import json, sys, time
from pathlib import Path
import lightgbm as lgb, numpy as np, polars as pl
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))
from src.evaluate import evaluate
from src.refine import PARAMS3
from src.train import one_owner, to_pairs
A = ROOT / "artifacts"; R = A / "refine"
feats = json.loads((A / "models/lgb_v4c.json").read_text())["features"]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
params = PARAMS3 | json.loads(sys.argv[1] if len(sys.argv) > 1 else "{}")
val = pl.read_parquet(R / "val_set.parquet")
X = val.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy(); y = val["label"].to_numpy()
fold = (val["s1"].hash(seed=3) % K).to_numpy(); inner = (val["s1"].hash(seed=9) % 10).to_numpy() == 0
oof = np.zeros(val.height, dtype=np.float32); t0 = time.time()
for k in range(K):
    fit, es = (fold != k) & ~inner, (fold != k) & inner
    b = lgb.train(params, lgb.Dataset(X[fit], y[fit]), 6000, valid_sets=[lgb.Dataset(X[es], y[es])], callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[fold == k] = b.predict(X[fold == k], num_iteration=b.best_iteration); print(f"fold {k}: {b.best_iteration} rounds {time.time()-t0:.0f}s", flush=True)
split = pl.read_parquet(A / "splits/split_v1.parquet"); ids = split.filter(pl.col("role") == "val").select("s1_id", "country")
truth = pl.read_parquet(A / "processed/train_gt_pairs.parquet")
main = val.select("s1", "m", "src").with_columns(pl.Series("p", oof), pl.lit(0.75).alias("t"))
ext = pl.read_parquet(R / "ext_val_oof.parquet").with_columns(pl.lit(0.7).alias("t"))
for t in (0.7, 0.75, 0.8):
    r = evaluate(to_pairs(one_owner(pl.concat([main.with_columns(pl.lit(t).alias("t")), ext])).filter(pl.col("p") >= pl.col("t"))), truth, ids)
    print(f"{params} thr {t}: F0.5 {r['f05']:.5f} US {r['by_country']['US']['f05']:.5f} IN {r['by_country']['India']['f05']:.5f}", flush=True)
np.save(R / f"oof_tune_{abs(hash(json.dumps(params,sort_keys=True)))%10**6}.npy", oof)
