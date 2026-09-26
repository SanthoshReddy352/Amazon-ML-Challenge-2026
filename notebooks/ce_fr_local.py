"""CPU fallback for the French above-band scoring: score the high-risk French classes (word swaps, house conflicts,
new names, empty addresses, sibling shifts) with both trained cross-encoders locally.

    python notebooks/ce_fr_local.py
Input : artifacts/kaggle_cefr/fr_keys.parquet (+ kaggle/ce_fr_upload/fr.parquet texts), refine/fr_pair_classes.parquet
Output: artifacts/kaggle_cefr/ce{1,2}_fr.parquet [id, logit] for the scored subset.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
A = ROOT / "artifacts"
OUT = A / "kaggle_cefr"
CLASSES = ["swap_samehouse", "swap_other", "house_conflict", "allnew", "empty", "sib_shift"]
torch.set_num_threads(os.cpu_count())

keys = pl.read_parquet(OUT / "fr_keys.parquet")
cls = pl.read_parquet(A / "refine/fr_pair_classes.parquet")
sub = keys.join(cls, on=["s1", "m", "src"], how="left").filter(pl.col("cls").is_in(CLASSES))
txt = pl.read_parquet(ROOT / "kaggle/ce_fr_upload/fr.parquet").join(sub.select("id"), on="id")
print(f"scoring {txt.height:,} French pairs locally", flush=True)
a, b = txt["a"].to_list(), txt["b"].to_list()
order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])
for tag, d in (("ce2", A / "kaggle_ce2/out/ce_model"),):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d).eval()
    out = np.empty(len(a), dtype=np.float32)
    bs = 128
    with torch.inference_mode():
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([a[j] for j in idx], [b[j] for j in idx], truncation=True, max_length=128, padding=True, return_tensors="pt")
            out[idx] = model(**enc).logits.squeeze(-1).numpy()
            if (i // bs) % 50 == 0:
                print(f"  [{tag}] {i + len(idx):,}/{len(a):,} ({time.time()-t0:.0f}s)", flush=True)
    pl.DataFrame({"id": txt["id"], "logit": out}).write_parquet(OUT / f"{tag}_fr_local.parquet")
    print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)
