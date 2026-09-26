"""Class-aware France fit: per-class logit offsets for the French pair probabilities, fitted to the real LB.

    python notebooks/fr_fit2.py
Writes artifacts/refine/fr_fit2_grid.parquet (all combos, sorted by loss).
"""
import itertools
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from fr_fit import LB, fr, load, score  # noqa: E402
from lbsim import prepare, simulate  # noqa: E402

R = ROOT / "artifacts" / "refine"
pairs, sets = load()
cls = pl.read_parquet(R / "fr_pair_classes.parquet")
pairs = pairs.join(cls, on=["s1", "m", "src"], how="left").with_columns(pl.col("cls").fill_null("rest"))
P = prepare(pairs, fr, sets)
rows = []
for b, mu, bsw, bsib, ban in itertools.product([-1.0, -0.5, 0.0], [0.1, 0.25], [0.0, 1.0, 2.0, 3.0], [0.0, -1.0, -2.0, -3.0], [0.0, -1.5]):
    sim, sing = simulate(P, 1.0, b, mu, draws=2, seed=1, bcls={"swap_samehouse": bsw, "sib_shift": bsib, "allnew": ban})
    d, res = score(sim, sing)
    loss = sum((r / max(LB[k][1], 0.002)) ** 2 for k, r in res.items())
    rows.append(dict(b=b, mu=mu, b_swap=bsw, b_sib=bsib, b_allnew=ban, d=round(d, 5), sing=round(sing, 4), loss=round(loss, 3),
                     **{f"F_{k}": round(sim[k] - 0.0025, 4) for k in sim if k != "empty"}, **{f"r_{k}": round(r, 5) for k, r in res.items()}))
    print(rows[-1], flush=True)
out = pl.DataFrame(rows).sort("loss")
out.write_parquet(R / "fr_fit2_grid.parquet")
pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_cols(20)
print(out.head(15))
