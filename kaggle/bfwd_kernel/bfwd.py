"""Kaggle GPU job (ext5): deeper S1 -> record retrieval with the fine-tuned bi-encoder (kernel amlc2026-biencoder,
bienc/): top-15 records per S1 (ext4 kept top-5). Train: only validation S1 are queried (all records embedded).

Input: dataset amlc2026-text2 (train_text2 / test_text2), dataset amlc2026-bandkeys (val_s1.parquet [s1]),
kernel output amlc2026-biencoder (bienc/). Output: bfwd_{test,train}.parquet [s1, m, src, cos, rank, dir=1].
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

K, MAXLEN = 15, 48
t0 = time.time()
TXT = os.path.dirname(glob.glob("/kaggle/input/**/train_text2.parquet", recursive=True)[0])
KEYS = os.path.dirname(glob.glob("/kaggle/input/**/val_s1.parquet", recursive=True)[0])
MODEL = os.path.dirname(glob.glob("/kaggle/input/**/bienc/config.json", recursive=True)[0])
print("inputs", TXT, KEYS, MODEL, "gpus", torch.cuda.device_count(), flush=True)


def mean_pool(model, tok, texts, dev):
    b = tok(["query: " + t for t in texts], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
    out = model(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).to(out.dtype)
    return F.normalize((out * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=-1)


def _embed_worker(gpu, texts, path):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(f"cuda:{gpu}").eval().half()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 2048):
            out.append(mean_pool(model, tok, texts[i:i + 2048], f"cuda:{gpu}").cpu().numpy().astype(np.float16))
    np.save(path, np.concatenate(out) if out else np.zeros((0, 384), np.float16))


def embed(texts):
    import torch.multiprocessing as mp
    n = max(torch.cuda.device_count(), 1)
    chunks = np.array_split(np.arange(len(texts)), n)
    ctx = mp.get_context("spawn")
    paths = [f"/tmp/emb_{g}.npy" for g in range(n)]
    ps = [ctx.Process(target=_embed_worker, args=(g, [texts[i] for i in c], paths[g])) for g, c in enumerate(chunks)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
        assert p.exitcode == 0, f"embed worker failed: {p.exitcode}"
    out = np.concatenate([np.load(pth) for pth in paths])
    for pth in paths:
        os.remove(pth)
    return out


@torch.no_grad()
def topk(q, p, k, qb=4096, pb=500_000):
    P = [torch.from_numpy(p[i:i + pb]).cuda() for i in range(0, len(p), pb)]
    out_s, out_i = [], []
    for i in range(0, len(q), qb):
        Q = torch.from_numpy(q[i:i + qb]).cuda()
        best_s = best_i = None
        off = 0
        for blk in P:
            v, ix = (Q @ blk.T).topk(min(k, blk.shape[0]), dim=1)
            ix = ix + off
            off += blk.shape[0]
            if best_s is None:
                best_s, best_i = v, ix
            else:
                cs, ci = torch.cat([best_s, v], 1), torch.cat([best_i, ix], 1)
                best_s, j = cs.topk(k, dim=1)
                best_i = ci.gather(1, j)
        out_s.append(best_s.float().cpu().numpy())
        out_i.append(best_i.cpu().numpy())
    del P
    torch.cuda.empty_cache()
    return np.concatenate(out_s), np.concatenate(out_i)


def main():
    val_s1 = set(pd.read_parquet(f"{KEYS}/val_s1.parquet")["s1"].astype(np.int64).tolist())
    for split in ("test", "train"):
        df = pd.read_parquet(f"{TXT}/{split}_text2.parquet")
        if split == "train":
            df = df[(df["src"] != 1) | df["id"].astype(np.int64).isin(val_s1)].reset_index(drop=True)
        emb = embed(df["text"].tolist())
        print(f"{split}: embedded {emb.shape} ({time.time()-t0:.0f}s)", flush=True)
        res = []
        for country in sorted(df["country"].unique()):
            s1 = np.where((df["country"] == country).values & (df["src"] == 1).values)[0]
            rec = np.where((df["country"] == country).values & (df["src"] != 1).values)[0]
            if len(s1) == 0 or len(rec) == 0:
                continue
            s, ix = topk(emb[s1], emb[rec], K)
            res.append(pd.DataFrame({"s1": np.repeat(df["id"].values[s1], K).astype(np.uint32),
                                     "m": df["id"].values[rec][ix.ravel()].astype(np.uint32),
                                     "src": df["src"].values[rec][ix.ravel()].astype(np.uint8),
                                     "cos": s.ravel().astype(np.float16),
                                     "rank": np.tile(np.arange(1, K + 1, dtype=np.uint8), len(s1)), "dir": np.uint8(1)}))
            print(f"  {country}: {len(rec)} records x {len(s1)} S1 ({time.time()-t0:.0f}s)", flush=True)
        pd.concat(res).to_parquet(f"/kaggle/working/bfwd_{split}.parquet", compression="zstd")
        print(f"  wrote bfwd_{split}.parquet ({time.time()-t0:.0f}s)", flush=True)
        del emb, df, res
    print("DONE", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
