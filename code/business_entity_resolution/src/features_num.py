"""Stage 3c: house-number relation features (v4, from the v2 error analysis).

The base features only say whether house numbers are *equal*. The data perturbs
numbers in two different ways:
  * true variants often truncate a number: "1629" -> "162", "1650" -> "650";
  * near-duplicate distractors shift it slightly: "35553" -> "35557", "84" -> "105".
On the out-of-fold v2 scores the model under-rates truncations (true-match rate 0.77-0.87
vs mean probability 0.63-0.66) and over-rates small shifts (0.31 vs 0.38).
These features describe how the numbers relate: prefix / suffix truncation and the
size of the numeric difference. They are pure digit arithmetic (country-agnostic).
"""
import os

import numpy as np
import polars as pl

K = 3  # house numbers compared per record (the first K of the address)


def number_table(norm):
    """Per record: its first K house numbers as int64 (-1 = none) and their digit counts."""
    toks = pl.col("addr_nums").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    d = norm.select("entity_id", toks.alias("_n"))
    cols = []
    for j in range(K):
        v = pl.col("_n").list.get(j, null_on_oob=True)
        cols += [v.cast(pl.Int64, strict=False).fill_null(-1).alias(f"n{j}"),
                 v.str.len_chars().fill_null(0).cast(pl.Int64).alias(f"l{j}")]
    return d.select("entity_id", *cols)


def _relations(a, la, b, lb):
    """Element-wise relation of two digit strings given as (value, length) arrays.
    Returns (both present, equal, b is a proper prefix of a or vice versa, proper suffix, |a - b|)."""
    ok = (a >= 0) & (b >= 0)
    eq = ok & (a == b)
    p10 = 10 ** np.arange(8, dtype=np.int64)
    da, db = np.clip(la - lb, 0, 7), np.clip(lb - la, 0, 7)
    prefix = ok & ~eq & (((la > lb) & (a // p10[da] == b)) | ((lb > la) & (b // p10[db] == a)))
    suffix = ok & ~eq & (((la > lb) & (a % p10[np.clip(lb, 0, 7)] == b)) | ((lb > la) & (b % p10[np.clip(la, 0, 7)] == a)))
    diff = np.where(ok, np.abs(a - b), -1)
    return ok, eq, prefix, suffix, diff


def num_features(pairs, tab):
    """House-number relation features for (s1_id, cand_id) pairs; row order is preserved."""
    A = pairs.select("s1_id").join(tab.rename({"entity_id": "s1_id"}), on="s1_id", how="left", maintain_order="left")
    B = pairs.select("cand_id").join(tab.rename({"entity_id": "cand_id"}), on="cand_id", how="left", maintain_order="left")
    a = [A[f"n{j}"].fill_null(-1).to_numpy() for j in range(K)]
    la = [A[f"l{j}"].fill_null(0).to_numpy() for j in range(K)]
    b = [B[f"n{j}"].fill_null(-1).to_numpy() for j in range(K)]
    lb = [B[f"l{j}"].fill_null(0).to_numpy() for j in range(K)]
    n = pairs.height
    any_eq = np.zeros(n, bool)
    any_prefix = np.zeros(n, bool)
    any_suffix = np.zeros(n, bool)
    trunc_len = np.zeros(n, np.int64)          # digits kept by the longest prefix/suffix truncation
    min_diff = np.full(n, np.iinfo(np.int64).max)
    min_diff_samelen = np.full(n, np.iinfo(np.int64).max)
    for i in range(K):
        for j in range(K):
            ok, eq, pre, suf, diff = _relations(a[i], la[i], b[j], lb[j])
            any_eq |= eq
            any_prefix |= pre
            any_suffix |= suf
            trunc_len = np.maximum(trunc_len, np.where(pre | suf, np.minimum(la[i], lb[j]), 0))
            ne = ok & ~eq
            min_diff = np.where(ne, np.minimum(min_diff, diff), min_diff)
            min_diff_samelen = np.where(ne & (la[i] == lb[j]), np.minimum(min_diff_samelen, diff), min_diff_samelen)
    _, eq0, pre0, suf0, diff0 = _relations(a[0], la[0], b[0], lb[0])
    big = np.iinfo(np.int64).max

    def logd(x):
        """log1p of a difference, -1 when there is none."""
        return np.where((x == big) | (x < 0), -1.0, np.log1p(np.where(x == big, 0, np.maximum(x, 0))))

    return pl.DataFrame({
        "f_nr_first_prefix": pre0.astype(np.float32),
        "f_nr_first_suffix": suf0.astype(np.float32),
        "f_nr_first_logdiff": np.where(eq0, 0.0, logd(diff0)).astype(np.float32),
        "f_nr_first_samelen": ((la[0] == lb[0]) & (la[0] > 0)).astype(np.float32),
        "f_nr_any_prefix": any_prefix.astype(np.float32),
        "f_nr_any_suffix": any_suffix.astype(np.float32),
        "f_nr_trunc_len": trunc_len.astype(np.float32),
        "f_nr_only_near": ((~any_eq) & (min_diff != big)).astype(np.float32),
        "f_nr_min_logdiff": logd(min_diff).astype(np.float32),
        "f_nr_min_logdiff_samelen": logd(min_diff_samelen).astype(np.float32),
    })


def build_numrel(work_dir, split, part_files, log=print):
    """Write <work>/<split>_numrel/part_XXX.parquet aligned row-by-row with the feature parts."""
    out_dir = os.path.join(work_dir, f"{split}_numrel")
    os.makedirs(out_dir, exist_ok=True)
    tab = number_table(pl.read_parquet(os.path.join(work_dir, f"{split}_norm.parquet"),
                                       columns=["entity_id", "addr_nums"]))
    for i, f in enumerate(part_files):
        p = pl.read_parquet(f, columns=["s1_id", "cand_id"])
        num_features(p, tab).write_parquet(os.path.join(out_dir, f"part_{i:03d}.parquet"))
        log(f"  {split} number-relation features part {i + 1}/{len(part_files)}")
    return out_dir
