"""Kaggle GPU job: fine-tune a multilingual cross-encoder on (S1 text, record text) pairs and score val/test pairs.

Input  (dataset amlc2026-ce-pairs): train.parquet [id, a, b, label], val.parquet / test.parquet [id, a, b]
Output (/kaggle/working): ce_val.parquet, ce_test.parquet [id, logit]; ce_log.txt
Model: xlm-roberta-base (MIT, 278M params) with a single-logit head, BCE loss, 1 epoch, AMP, both T4s (DataParallel).
Training pairs come from S1 disjoint from val (fit_sample3/4), so val scores are out-of-sample.
"""
import glob
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL, MAXLEN, BS, LR, EPOCHS = "xlm-roberta-base", 128, 128, 3e-5, 1
t0 = time.time()


def log(m):
    msg = f"[{time.time()-t0:7.0f}s] {m}"
    print(msg, flush=True)
    with open("/kaggle/working/ce_log.txt", "a") as f:
        f.write(msg + "\n")


def find(name):
    return glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0]


def collate(tok):
    def f(batch):
        a = [x[0] for x in batch]
        b = [x[1] for x in batch]
        enc = tok(a, b, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        if len(batch[0]) > 2:
            enc["labels"] = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        return enc
    return f


@torch.no_grad()
def predict(model, tok, df, bs=1024):
    model.eval()
    order = np.argsort(df["a"].str.len().values + df["b"].str.len().values)
    rows = list(zip(df["a"].values[order], df["b"].values[order]))
    dl = DataLoader(rows, batch_size=bs, shuffle=False, collate_fn=collate(tok), num_workers=2)
    out = []
    for i, enc in enumerate(dl):
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(**enc).logits.float().squeeze(-1).cpu().numpy())
        if i % 200 == 0:
            log(f"  predict {min((i + 1) * bs, len(rows)):,}/{len(rows):,}")
    logit = np.empty(len(rows), dtype=np.float32)
    logit[order] = np.concatenate(out)
    return logit


if __name__ == "__main__":
    log(f"GPUs: {torch.cuda.device_count()}")
    tr = pd.read_parquet(find("train.parquet"))
    rng = np.random.default_rng(0)
    hold = rng.random(len(tr)) < 0.01
    dev, tr = tr[hold], tr[~hold]
    log(f"train {len(tr):,} dev {len(dev):,} pos {tr['label'].mean():.3f}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).cuda()
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    rows = list(zip(tr["a"].values, tr["b"].values, tr["label"].values.astype(np.float32)))
    dl = DataLoader(rows, batch_size=BS, shuffle=True, collate_fn=collate(tok), num_workers=2, drop_last=True)
    steps = len(dl) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    step = 0
    for ep in range(EPOCHS):
        net.train()
        run = 0.0
        for enc in dl:
            y = enc.pop("labels").cuda(non_blocking=True)
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logit = net(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
            if step % 500 == 0:
                log(f"ep {ep} step {step}/{steps} loss {run:.4f}")
        d = predict(model, tok, dev)
        p = 1 / (1 + np.exp(-d))
        y = dev["label"].values
        ll = -np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1)))
        acc = np.mean((p >= 0.5) == y)
        log(f"epoch {ep} dev logloss {ll:.4f} acc {acc:.4f}")
    model.save_pretrained("/kaggle/working/ce_model")
    tok.save_pretrained("/kaggle/working/ce_model")
    for name in ("val", "test"):
        df = pd.read_parquet(find(f"{name}.parquet"))
        log(f"scoring {name}: {len(df):,}")
        pd.DataFrame({"id": df["id"].values, "logit": predict(model, tok, df)}).to_parquet(f"/kaggle/working/ce_{name}.parquet")
        log(f"{name} written")
    log("done")
