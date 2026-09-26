"""Kaggle GPU job: score French pairs above the cross-encoders' original band with the two trained cross-encoders.
Input : dataset amlc2026-ce-fr (fr.parquet [id, a, b]); kernel outputs amlc2026-cross-encoder{,2} (ce_model/).
Output: /kaggle/working/ce1_fr.parquet, ce2_fr.parquet [id, logit]. One model per GPU in parallel processes.
"""
import glob
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def run(tag, model_dir, gpu):
    t0 = time.time()
    df = pd.read_parquet(glob.glob("/kaggle/input/**/fr.parquet", recursive=True)[0])
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(f"cuda:{gpu}").eval().half()
    order = np.argsort(df["a"].str.len().values + df["b"].str.len().values)
    rows = list(zip(df["a"].values[order], df["b"].values[order]))
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
    pd.DataFrame({"id": df["id"].values, "logit": logit}).to_parquet(f"/kaggle/working/{tag}_fr.parquet")
    print(f"[{tag}] done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    dirs = {}
    for p in glob.glob("/kaggle/input/**/ce_model/config.json", recursive=True):
        d = p.rsplit("/", 1)[0]
        dirs["ce2" if "cross-encoder2" in d else "ce1"] = d
    print(dirs, flush=True)
    mp.set_start_method("spawn")
    dirs = {k: v for k, v in dirs.items() if k == "ce2"}  # the blend uses only CE v2 above its band
    procs = [mp.Process(target=run, args=(t, d, i % max(torch.cuda.device_count(), 1))) for i, (t, d) in enumerate(sorted(dirs.items()))]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
