"""Kaggle GPU job (v16b): exact cosine of the fine-tuned bi-encoder (output of kernel amlc2026-biencoder, bienc/)
for every band pair (final p in [0.01, 0.995)) of val and test, so the band re-scorer sees the similarity even when the
pair is outside both top-5 neighbour lists.

Input: dataset amlc2026-text2 (train_text2 / test_text2 [id, src, country, text]), dataset amlc2026-bandkeys
(band_{val,test}.parquet [s1, m, src]), kernel output amlc2026-biencoder (bienc/).
Output: bcos_{val,test}.parquet [s1, m, src, xcos].
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

t0 = time.time()
TXT = os.path.dirname(glob.glob("/kaggle/input/**/train_text2.parquet", recursive=True)[0])
KEYS = os.path.dirname(glob.glob("/kaggle/input/**/band_val.parquet", recursive=True)[0])
MODEL = os.path.dirname(glob.glob("/kaggle/input/**/bienc/config.json", recursive=True)[0])
print("inputs", TXT, KEYS, MODEL, flush=True)
dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(dev).eval().half()


@torch.no_grad()
def embed(texts):
    out = []
    for i in range(0, len(texts), 2048):
        b = tok(["query: " + t for t in texts[i:i + 2048]], padding=True, truncation=True, max_length=48, return_tensors="pt").to(dev)
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        out.append(F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=-1).float().cpu().numpy())
    return np.concatenate(out)


for split, fn in (("val", "train_text2.parquet"), ("test", "test_text2.parquet")):
    keys = pd.read_parquet(f"{KEYS}/band_{split}.parquet")
    df = pd.read_parquet(f"{TXT}/{fn}", columns=["id", "src", "text"])
    df["k"] = df["id"].astype(np.int64) * 4 + df["src"].astype(np.int64)
    need = np.unique(np.concatenate([keys["s1"].values.astype(np.int64) * 4 + 1, keys["m"].values.astype(np.int64) * 4 + keys["src"].values.astype(np.int64)]))
    sub = df[df["k"].isin(need)].drop_duplicates("k")
    del df
    emb = embed(sub["text"].tolist())
    pos = pd.Series(np.arange(len(sub)), index=sub["k"].values)
    ia = pos.reindex(keys["s1"].values.astype(np.int64) * 4 + 1).values
    ib = pos.reindex(keys["m"].values.astype(np.int64) * 4 + keys["src"].values.astype(np.int64)).values
    ok = ~(np.isnan(ia) | np.isnan(ib))
    x = np.full(len(keys), np.nan, dtype=np.float32)
    x[ok] = (emb[ia[ok].astype(np.int64)] * emb[ib[ok].astype(np.int64)]).sum(1)
    keys["xcos"] = x
    keys.to_parquet(f"/kaggle/working/bcos_{split}.parquet")
    print(f"{split}: {len(keys):,} pairs, {len(sub):,} texts, missing {int((~ok).sum())} ({time.time()-t0:.0f}s)", flush=True)
print("DONE", flush=True)
