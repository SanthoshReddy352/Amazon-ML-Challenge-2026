"""Kaggle GPU job (ext4): fine-tune multilingual-e5-small as a bi-encoder retriever on our own training pairs, then
exact top-K neighbours in BOTH directions (record -> S1 and S1 -> record), per country.

Input  (dataset amlc2026-text2): train_text2.parquet, test_text2.parquet [id, src, country, text = "name | address"],
                                  fit_pairs.parquet [a (S1 id), m, src, hard, b (hard-negative S1 id or null)]
Output (/kaggle/working): bienc/ (fine-tuned model), bknn_{split}.parquet [s1, m, src, cos, rank, dir (0 rev, 1 fwd)]
Training uses only fit S1 (validation S1 never seen). Loss: symmetric in-batch contrastive (InfoNCE, tau 0.05) with the
hard-negative S1 appended to the record -> S1 softmax. Model licence: MIT (118M parameters).
"""
import glob
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BASE, MAXLEN, BS, LR, TAU, K = "intfloat/multilingual-e5-small", 48, 384, 3e-5, 0.05, 5


def mean_pool(model, tok, texts, dev):
    b = tok(["query: " + t for t in texts], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(dev)
    out = model(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).to(out.dtype)
    return F.normalize((out * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=-1)


def train(in_dir, t0):
    tr = pd.read_parquet(f"{in_dir}/train_text2.parquet")
    s1_text = pd.Series(tr.loc[tr.src == 1, "text"].values, index=tr.loc[tr.src == 1, "id"].values)
    rec = tr[tr.src != 1]
    rec_text = pd.Series(rec["text"].values, index=rec["id"].values.astype(np.int64) * 4 + rec["src"].values)
    pairs = pd.read_parquet(f"{in_dir}/fit_pairs.parquet")
    a_txt = s1_text.reindex(pairs["a"].values).values
    p_txt = rec_text.reindex(pairs["m"].values.astype(np.int64) * 4 + pairs["src"].values).values
    rng = np.random.default_rng(3)
    b_ids = pairs["b"].values
    rand_ids = s1_text.index.values[rng.integers(0, len(s1_text), len(pairs))]
    b_ids = np.where(pd.isna(b_ids), rand_ids, b_ids).astype(np.int64)
    b_txt = s1_text.reindex(b_ids).values
    ok = pd.notna(a_txt) & pd.notna(p_txt) & pd.notna(b_txt)
    a_txt, p_txt, b_txt = a_txt[ok], p_txt[ok], b_txt[ok]
    print(f"train pairs {ok.sum():,} ({time.time()-t0:.0f}s)", flush=True)
    del tr, rec, s1_text, rec_text
    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = len(a_txt) // BS
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.02, (steps - s) / steps))
    scaler = torch.cuda.amp.GradScaler()
    order = rng.permutation(len(a_txt))
    labels = torch.arange(BS, device=dev)
    for s in range(steps):
        ix = order[s * BS:(s + 1) * BS]
        with torch.autocast("cuda", dtype=torch.float16):
            ea = mean_pool(model, tok, list(a_txt[ix]), dev)
            ep = mean_pool(model, tok, list(p_txt[ix]), dev)
            eb = mean_pool(model, tok, list(b_txt[ix]), dev)
            l_rs = (ep @ torch.cat([ea, eb]).T) / TAU        # record -> S1 (+ hard negatives)
            l_sr = (ea @ ep.T) / TAU                          # S1 -> record
            loss = (F.cross_entropy(l_rs.float(), labels) + F.cross_entropy(l_sr.float(), labels)) / 2
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if s % 200 == 0:
            print(f"  step {s}/{steps} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.save_pretrained("/kaggle/working/bienc")
    tok.save_pretrained("/kaggle/working/bienc")
    del model, opt
    torch.cuda.empty_cache()
    print(f"trained ({time.time()-t0:.0f}s)", flush=True)


def _embed_worker(gpu, texts, path):
    tok = AutoTokenizer.from_pretrained("/kaggle/working/bienc")
    model = AutoModel.from_pretrained("/kaggle/working/bienc").to(f"cuda:{gpu}").eval().half()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 2048):
            out.append(mean_pool(model, tok, texts[i:i + 2048], f"cuda:{gpu}").cpu().numpy().astype(np.float16))
    np.save(path, np.concatenate(out) if out else np.zeros((0, 384), np.float16))


def embed(texts):
    """Both GPUs in parallel processes; each writes its slice to /tmp (a Queue would pickle GBs)."""
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
def topk(q: np.ndarray, p: np.ndarray, k: int, qb: int = 4096, pb: int = 500_000):
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
    t0 = time.time()
    in_dir = os.path.dirname(glob.glob("/kaggle/input/**/train_text2.parquet", recursive=True)[0])
    print("input dir", in_dir, "gpus", torch.cuda.device_count(), flush=True)
    train(in_dir, t0)
    for split in ("test", "train"):
        df = pd.read_parquet(f"{in_dir}/{split}_text2.parquet")
        emb = embed(df["text"].tolist())
        print(f"{split}: embedded {emb.shape} ({time.time()-t0:.0f}s)", flush=True)
        res = []
        for country in sorted(df["country"].unique()):
            s1 = np.where((df["country"] == country).values & (df["src"] == 1).values)[0]
            rec = np.where((df["country"] == country).values & (df["src"] != 1).values)[0]
            if len(s1) == 0 or len(rec) == 0:
                continue
            s, ix = topk(emb[rec], emb[s1], K)                     # reverse: record -> S1
            res.append(pd.DataFrame({"s1": df["id"].values[s1][ix.ravel()].astype(np.uint32),
                                     "m": np.repeat(df["id"].values[rec], K).astype(np.uint32),
                                     "src": np.repeat(df["src"].values[rec], K).astype(np.uint8),
                                     "cos": s.ravel().astype(np.float16),
                                     "rank": np.tile(np.arange(1, K + 1, dtype=np.uint8), len(rec)), "dir": np.uint8(0)}))
            s, ix = topk(emb[s1], emb[rec], K)                     # forward: S1 -> record
            res.append(pd.DataFrame({"s1": np.repeat(df["id"].values[s1], K).astype(np.uint32),
                                     "m": df["id"].values[rec][ix.ravel()].astype(np.uint32),
                                     "src": df["src"].values[rec][ix.ravel()].astype(np.uint8),
                                     "cos": s.ravel().astype(np.float16),
                                     "rank": np.tile(np.arange(1, K + 1, dtype=np.uint8), len(s1)), "dir": np.uint8(1)}))
            print(f"  {country}: {len(rec)} records x {len(s1)} S1 ({time.time()-t0:.0f}s)", flush=True)
        pd.concat(res).to_parquet(f"/kaggle/working/bknn_{split}.parquet", compression="zstd")
        print(f"  wrote bknn_{split}.parquet ({time.time()-t0:.0f}s)", flush=True)
        del emb, df, res
    print("DONE", f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
