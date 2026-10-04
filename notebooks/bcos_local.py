"""CPU fallback of kaggle/bcos_kernel: exact cosine of the fine-tuned bi-encoder (artifacts/kaggle_bknn/bienc) for every
band pair of val / test -> artifacts/kaggle_bcos/bcos_{split}.parquet [s1, m, src, xcos].

    python notebooks/bcos_local.py val|test [--bench]
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
split = sys.argv[1]
MODEL = ROOT / "artifacts/kaggle_bknn/bienc"
torch.set_num_threads(os.cpu_count())
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).eval()


@torch.inference_mode()
def embed(texts, bs=512):
    out = []
    t0 = time.time()
    for i in range(0, len(texts), bs):
        b = tok(["query: " + t for t in texts[i:i + bs]], padding=True, truncation=True, max_length=48, return_tensors="pt")
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        out.append(F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=-1).numpy().astype(np.float32))
        if i // bs % 100 == 0:
            print(f"  {i + bs:,}/{len(texts):,} ({time.time() - t0:.0f}s)", flush=True)
    return np.concatenate(out)


if "--bench" in sys.argv:
    t0 = time.time()
    embed(["Rocky Capstone Inc | 112 Colvin Street, Rochester, NY"] * 2048)
    print(f"2048 texts in {time.time() - t0:.1f}s")
    sys.exit(0)
keys = pd.read_parquet(ROOT / f"kaggle/bcos_upload/band_{split}.parquet")
df = pd.read_parquet(ROOT / f"kaggle/biencoder_upload/{'train' if split == 'val' else 'test'}_text2.parquet", columns=["id", "src", "text"])
df["k"] = df["id"].astype(np.int64) * 4 + df["src"].astype(np.int64)
ka = keys["s1"].values.astype(np.int64) * 4 + 1
kb = keys["m"].values.astype(np.int64) * 4 + keys["src"].values.astype(np.int64)
sub = df[df["k"].isin(np.unique(np.concatenate([ka, kb])))].drop_duplicates("k")
del df
print(f"{split}: {len(keys):,} pairs, {len(sub):,} texts", flush=True)
emb = embed(sub["text"].tolist())
pos = pd.Series(np.arange(len(sub)), index=sub["k"].values)
ia, ib = pos.reindex(ka).values, pos.reindex(kb).values
ok = ~(np.isnan(ia) | np.isnan(ib))
x = np.full(len(keys), np.nan, dtype=np.float32)
x[ok] = (emb[ia[ok].astype(np.int64)] * emb[ib[ok].astype(np.int64)]).sum(1)
keys["xcos"] = x
(ROOT / "artifacts/kaggle_bcos").mkdir(exist_ok=True)
keys.to_parquet(ROOT / f"artifacts/kaggle_bcos/bcos_{split}.parquet")
print(f"done: missing {int((~ok).sum())}", flush=True)
