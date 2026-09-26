"""Leaderboard simulator: expected macro F0.5 of prediction sets under a probabilistic truth.

Truth model: every record (m, src) belongs to at most one S1. Given candidate pairs with probabilities q (after an
optional calibration q' = sigmoid(a * logit(q) + b)), each record's owner is drawn from its candidates with
P(S1 i) = q'_i / max(1, sum q'), and "no owner" with the remaining mass. On top, every S1 gets Poisson(mu) extra true
matches that no prediction set contains (blocking misses). Each prediction set is scored with the exact macro F0.5
of the leaderboard (a singleton scores 1 only when predicted empty). Averaged over Monte-Carlo draws.
"""
import numpy as np
import polars as pl


def calibrate(q: np.ndarray, a: float = 1.0, b: float = 0.0) -> np.ndarray:
    q = np.clip(q, 1e-6, 1 - 1e-6)
    z = a * np.log(q / (1 - q)) + b
    return 1.0 / (1.0 + np.exp(-z))


def prepare(pairs: pl.DataFrame, s1_ids: pl.Series, sets: dict[str, pl.DataFrame]):
    """pairs: s1, m, src, q (all candidate pairs, one row per pair). sets: name -> (s1, m, src) predictions.
    Pairs predicted by a set but absent from `pairs` are added with q = 0.01."""
    extra = pl.concat([d.select("s1", "m", "src") for d in sets.values()]).unique().join(pairs.select("s1", "m", "src"), on=["s1", "m", "src"], how="anti")
    if "cls" not in pairs.columns:
        pairs = pairs.with_columns(pl.lit("rest").alias("cls"))
    pairs = pl.concat([pairs.select("s1", "m", "src", pl.col("q").cast(pl.Float64), "cls"),
                       extra.with_columns(pl.lit(0.01, pl.Float64).alias("q"), pl.lit("unscored").alias("cls"))])
    s1_index = {v: i for i, v in enumerate(s1_ids.to_list())}
    pairs = pairs.filter(pl.col("s1").is_in(s1_ids.implode()))
    pairs = pairs.with_columns((pl.col("m").cast(pl.Int64) * 4 + pl.col("src").cast(pl.Int64)).alias("rid")).sort("rid")
    rid = pairs["rid"].to_numpy()
    s1i = np.array([s1_index[x] for x in pairs["s1"].to_list()], dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(rid)]))
    key = pairs.select("s1", "m", "src").with_row_index("_row")
    masks = {}
    for name, d in sets.items():
        hit = key.join(d.select("s1", "m", "src").unique(), on=["s1", "m", "src"], how="semi")["_row"].to_numpy()
        mk = np.zeros(len(rid), dtype=bool)
        mk[hit] = True
        masks[name] = mk
    return dict(q=pairs["q"].to_numpy().astype(np.float64), s1i=s1i, grp=grp, starts=starts, n_s1=len(s1_index), masks=masks,
                cls=pairs["cls"].to_numpy())


def simulate(P: dict, a: float = 1.0, b: float = 0.0, mu: float = 0.0, draws: int = 8, seed: int = 0, beta: float = 0.5,
             bcls: dict | None = None, qover: dict | None = None):
    """bcls: extra logit offset per pair class (e.g. {"swap_samehouse": 1.0})."""
    rng = np.random.default_rng(seed)
    bb = np.full(len(P["q"]), b, dtype=np.float64)
    for c, off in (bcls or {}).items():
        bb[P["cls"] == c] += off
    q = calibrate(P["q"], a, bb)
    for c, val in (qover or {}).items():  # replace the probability of a whole class by a constant truth rate
        q = np.where(P["cls"] == c, val, q)
    grp, starts, s1i, n = P["grp"], P["starts"], P["s1i"], P["n_s1"]
    tot = np.add.reduceat(q, starts)
    scale = np.maximum(tot, 1.0)
    qn = q / scale[grp]
    cum = np.cumsum(qn)
    base = np.r_[0.0, cum[starts[1:] - 1]] if len(starts) > 1 else np.array([0.0])
    cum_in = cum - base[grp]          # cumulative prob within each record group
    prev = cum_in - qn
    b2 = beta * beta
    out = {k: [] for k in P["masks"]}
    sing = []
    for d in range(draws):
        u = rng.random(len(starts))[grp]
        true = (u >= prev) & (u < cum_in)          # at most one owner per record
        n_true = np.bincount(s1i[true], minlength=n)
        if mu > 0:  # blocking misses fall on S1 that have matches, proportionally to how many they have
            n_true = n_true + rng.poisson(mu * n_true / max(n_true.mean(), 1e-9))
        sing.append((n_true == 0).mean())
        for k, mk in P["masks"].items():
            n_pred = np.bincount(s1i[mk], minlength=n)
            tp = np.bincount(s1i[mk & true], minlength=n)
            with np.errstate(divide="ignore", invalid="ignore"):
                p_ = tp / n_pred
                r_ = tp / n_true
                f = np.where(tp == 0, 0.0, (1 + b2) * p_ * r_ / (b2 * p_ + r_))
            f = np.where((n_true == 0) & (n_pred == 0), 1.0, f)
            out[k].append(f.mean())
    return {k: float(np.mean(v)) for k, v in out.items()}, float(np.mean(sing))
