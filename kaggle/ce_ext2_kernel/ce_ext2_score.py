"""Kaggle GPU job: score the ext2 (reverse-blocking) pairs with every trained cross-encoder.
Input : dataset amlc2026-ce-ext2 (ext2.parquet [id, a, b]); kernel outputs amlc2026-cross-encoder{,2,3,4} (ce_model/).
Output: /kaggle/working/{tag}_ext2.parquet [id, logit]. GPU 0 and GPU 1 each work through their own model queue.
"""
import glob
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def run(jobs, gpu):
    df = pd.read_parquet(glob.glob("/kaggle/input/**/ext2.parquet", recursive=True)[0])
    order = np.argsort(df["a"].str.len().values + df["b"].str.len().values)
    rows = list(zip(df["a"].values[order], df["b"].values[order]))
    for tag, model_dir in jobs:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(f"cuda:{gpu}").eval().half()
        col = lambda b: tok([x[0] for x in b], [x[1] for x in b], truncation=True, max_length=128, padding=True, return_tensors="pt")  # noqa: E731
        out = []
        with torch.no_grad():
            for i, enc in enumerate(DataLoader(rows, batch_size=512, collate_fn=col, num_workers=0)):
                enc = {k: v.to(f"cuda:{gpu}") for k, v in enc.items()}
                out.append(model(**enc).logits.float().squeeze(-1).cpu().numpy())
                if i % 200 == 0:
                    print(f"[{tag}] {min((i + 1) * 512, len(rows)):,}/{len(rows):,} {time.time()-t0:.0f}s", flush=True)
        logit = np.empty(len(rows), dtype=np.float32)
        logit[order] = np.concatenate(out)
        pd.DataFrame({"id": df["id"].values, "logit": logit}).to_parquet(f"/kaggle/working/{tag}_ext2.parquet")
        print(f"[{tag}] done {time.time()-t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    dirs = {}
    for p in glob.glob("/kaggle/input/**/ce_model/config.json", recursive=True):
        d = p.rsplit("/", 1)[0]
        tag = "ce4" if "cross-encoder4" in d else "ce3" if "cross-encoder3" in d else "ce2" if "cross-encoder2" in d else "ce1"
        dirs[tag] = d
    print(dirs, flush=True)
    mp.set_start_method("spawn")
    # xlm-r models are ~1.7x faster than mdeberta: GPU0 = ce3 (+ce4), GPU1 = ce1 + ce2
    q0 = [(t, dirs[t]) for t in ("ce3", "ce4") if t in dirs]
    q1 = [(t, dirs[t]) for t in ("ce1", "ce2") if t in dirs]
    ngpu = max(torch.cuda.device_count(), 1)
    procs = [mp.Process(target=run, args=(q, i % ngpu)) for i, q in enumerate([q0, q1]) if q]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
