"""Empty-address rescue: for an empty-address record that no S1 claimed, accept its top candidate S1?

Val: 46.8% of empty-address true pairs are missed. Most misses are name ties (several S1 share the record's name
key), where the pair model splits its probability. A record-level LightGBM decides from the final pair probability
and tie-break evidence computed over ALL S1 of the country: how many S1 share the record's name key, the key plus
its legal form, or the exact normalised name; whether the top S1 shares the legal form / normalised name; plus
the pair features. Val-only competition features (2nd-best probability etc.) are left out: on val they only see
val S1, on test every S1. 5-fold OOF on val: +0.0003 F0.5 at q >= 0.75 (adds 3.0k pairs, 90% precise).

Used by notebooks/final_v13.py: fit_apply(val_frame, test_frame) -> (val_add, test_add).
"""
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
A, R = ROOT / "artifacts", ROOT / "artifacts" / "refine"
num = lambda c: pl.col(c).str.slice(3).cast(pl.UInt32)  # noqa: E731
FC = ["n_ratio", "n_tset", "n_jw", "n_key_eq", "n_core_eq", "n_legal_eq", "n_legal_jacc", "n_legal_missing_2", "n_is_domain_2",
      "n_indic_2", "n_skel_tset", "n_alt_tset", "n_tok_jacc", "n_ntok_diff", "nm_miss1", "nm_extra2", "n_acr", "r_n_s1_key",
      "s1_n_key", "r_sup_k"]
FM = ["rev_n_s1", "rev_margin", "rev_is_best", "rev_score_rel", "rev_name_rel", "knn_cos", "ctx_n_cands"]
FEATS = ["pf", "p", "kind", "leq", "neq", "keq", "k_n", "kl_n", "nn_n", "s1_nsel"] + FC + FM
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)


def top_candidates(split: str, allp: pl.DataFrame, sel: pl.DataFrame, set_main: Path, set_ext: Path,
                   set_extra: tuple = ()) -> pl.DataFrame:
    """allp: s1, m, src, p (model), pf (final), kind (0 main / 1 ext / 2 ext2 ...). sel: selected (s1, m, src).
    set_extra: feature sets of further pair kinds (e.g. ext2). Returns one row per unclaimed empty-address record:
    its top-pf candidate with FEATS."""
    fm = pl.read_parquet(set_main, columns=["s1", "m", "src", "a_empty_2"] + FC + FM)
    f = fm
    for extra in (set_ext, *set_extra):
        fe = pl.read_parquet(extra, columns=["s1", "m", "src", "a_empty_2"] + FC)
        f = pl.concat([f, fe.join(f.select("s1", "m", "src"), on=["s1", "m", "src"], how="anti")], how="diagonal_relaxed")
    x = allp.join(f, on=["s1", "m", "src"], how="inner").filter(pl.col("a_empty_2") == 1)
    x = x.join(sel.select("m", "src").unique(), on=["m", "src"], how="anti")
    x = x.sort("pf", descending=True).group_by("m", "src", maintain_order=True).head(1)
    s1n = pl.read_parquet(A / f"normalized/{split}_source1.parquet", columns=["entity_id", "country", "name_key", "name_legal", "name_norm"])
    s1n = s1n.with_columns(num("entity_id").alias("s1")).drop("entity_id")
    kc = s1n.group_by("country", "name_key").agg(pl.len().alias("k_n"))
    klc = s1n.group_by("country", "name_key", "name_legal").agg(pl.len().alias("kl_n"))
    nnc = s1n.group_by("country", "name_norm").agg(pl.len().alias("nn_n"))
    recs = pl.concat([pl.read_parquet(A / f"normalized/{split}_source{s}.parquet", columns=["entity_id", "name_key", "name_legal", "name_norm"])
                        .join(x.filter(pl.col("src") == s).select((pl.lit(f"S{s}-") + pl.col("m").cast(pl.Utf8)).alias("entity_id")), on="entity_id", how="semi")
                        .select(num("entity_id").alias("m"), pl.lit(s, pl.UInt8).alias("src"), pl.col("name_key").alias("rk"),
                                pl.col("name_legal").alias("rl"), pl.col("name_norm").alias("rn")) for s in (2, 3)])
    x = x.join(s1n.select("s1", "country", pl.col("name_key").alias("sk"), pl.col("name_legal").alias("sl"), pl.col("name_norm").alias("sn")), on="s1")
    x = x.join(recs, on=["m", "src"])
    x = (x.join(kc.rename({"name_key": "rk"}), on=["country", "rk"], how="left")
          .join(klc.rename({"name_key": "rk", "name_legal": "rl"}), on=["country", "rk", "rl"], how="left")
          .join(nnc.rename({"name_norm": "rn"}), on=["country", "rn"], how="left"))
    x = x.with_columns(pl.col("k_n", "kl_n", "nn_n").fill_null(0),
                       (pl.col("sl") == pl.col("rl")).fill_null(False).cast(pl.Int8).alias("leq"),
                       (pl.col("sn") == pl.col("rn")).fill_null(False).cast(pl.Int8).alias("neq"),
                       (pl.col("sk") == pl.col("rk")).fill_null(False).cast(pl.Int8).alias("keq"))
    nsel = sel.group_by("s1").agg(pl.len().alias("s1_nsel"))
    return x.join(nsel, on="s1", how="left").with_columns(pl.col("s1_nsel").fill_null(0))


def fit_apply(tv: pl.DataFrame, tt: pl.DataFrame | None, folds: int = 5):
    """tv: val top candidates with label y; tt: test top candidates. Returns (val OOF q, test q)."""
    X = tv.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
    y = tv["y"].to_numpy()
    fold = (tv["s1"].hash(seed=21) % folds).to_numpy()
    oof = np.zeros(len(y))
    models = []
    for k in range(folds):
        b = lgb.train(PARAMS, lgb.Dataset(X[fold != k], y[fold != k], feature_name=FEATS), 400)
        oof[fold == k] = b.predict(X[fold == k])
        models.append(b)
    qt = None
    if tt is not None and tt.height:
        Xt = tt.select([pl.col(c).cast(pl.Float32) for c in FEATS]).to_numpy()
        qt = np.mean([m.predict(Xt) for m in models], axis=0)
    return oof, qt
