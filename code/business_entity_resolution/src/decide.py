"""Stage 5: turn pair probabilities into final matches.

1. one-owner rule: every S2/S3 record belongs to at most one S1 entity (true for
   100% of the training labels), so a candidate is only kept for the S1 entity
   that gives it the highest probability;
2. keep the pair if its probability clears a threshold tuned for macro F0.5.
   v4 uses a rank rule: the most probable candidate of each S1 gets a lower
   threshold than its other candidates (an S1 with no match at all loses a whole
   point of macro F0.5, one extra candidate much less).
"""
import numpy as np
import polars as pl

from evaluate import macro_f05


def assign_owner(scored):
    """One-owner rule: keep every candidate only for the S1 entity that gives it the highest probability."""
    return scored.filter(pl.col("prob") == pl.col("prob").max().over("cand_id")).unique(["cand_id"], keep="first")


def decide(scored, thr=None):
    """thr: a number; or None to use per-pair threshold columns: `thr` (one threshold), or
    `thr_top` for the best remaining candidate of each S1 and `thr_rest` for its others."""
    owned = assign_owner(scored)
    if thr is not None:
        return owned.filter(pl.col("prob") >= thr)
    if "thr_top" in owned.columns:
        return keep_by_rank(owned, pl.col("thr_top"), pl.col("thr_rest")).drop("thr_top", "thr_rest")
    return owned.filter(pl.col("prob") >= pl.col("thr")).drop("thr")


def keep_by_rank(owned, thr_top, thr_rest):
    """Rank rule on owned pairs: the most probable candidate of each S1 needs prob >= thr_top,
    the others prob >= thr_rest (numbers or column expressions)."""
    r = pl.col("prob").rank("ordinal", descending=True).over("s1_id")
    return owned.filter(pl.col("prob") >= pl.when(r == 1).then(thr_top).otherwise(thr_rest))


def tune_rank_thresholds(scored, gt_pairs, s1_ids, top_grid=None, rest_grid=None, log=print):
    """Grid-search (thr_top, thr_rest) of the rank rule for macro F0.5 on out-of-fold scores.
    Returns (thr_top, thr_rest, best_score)."""
    top_grid = top_grid if top_grid is not None else np.round(np.arange(0.40, 0.751, 0.05), 2)
    rest_grid = rest_grid if rest_grid is not None else np.round(np.arange(0.60, 0.851, 0.05), 2)
    owned = assign_owner(scored).select("s1_id", "cand_id", "prob")
    best = (None, None, -1.0)
    for t_top in top_grid:
        for t_rest in rest_grid:
            if t_top > t_rest:
                continue
            m = macro_f05(keep_by_rank(owned, float(t_top), float(t_rest)), gt_pairs, s1_ids)
            log(f"  top={t_top:.2f} rest={t_rest:.2f}  macroF0.5={m['macro_f05']:.5f}  "
                f"P={m['micro_precision']:.4f}  R={m['micro_recall']:.4f}  singletons={m['singleton_acc']:.4f}")
            if m["macro_f05"] > best[2]:
                best = (float(t_top), float(t_rest), m["macro_f05"])
    return best


def decide_expected_f(scored, alpha=1.0, floor=0.05):
    """Per S1 entity choose the top-m candidates maximising the expected F0.5
        E[F](m) ~ 1.25 * sum_{j<=m} p_j / (0.25 * alpha * sum_j p_j + m)
    and predict nothing when P(no match) = prod_j (1 - p_j) is larger."""
    owned = assign_owner(scored).filter(pl.col("prob") >= floor)
    d = owned.sort(["s1_id", "prob"], descending=[False, True]).with_columns(
        pl.col("prob").cum_sum().over("s1_id").alias("_cs"),
        pl.int_range(1, pl.len() + 1).over("s1_id").alias("_m"),
        pl.col("prob").sum().over("s1_id").alias("_T"),
        (1 - pl.col("prob")).log().sum().over("s1_id").exp().alias("_p0"),
    ).with_columns(
        (1.25 * pl.col("_cs") / (0.25 * alpha * pl.col("_T") + pl.col("_m"))).alias("_ef")
    ).with_columns(
        pl.col("_ef").max().over("s1_id").alias("_best"),
        pl.col("_m").filter(pl.col("_ef") == pl.col("_ef").max()).first().over("s1_id").alias("_mbest"),
    )
    return d.filter((pl.col("_m") <= pl.col("_mbest")) & (pl.col("_best") > pl.col("_p0"))).select(scored.columns)


def tune_threshold(scored, gt_pairs, s1_ids, grid=None, log=print):
    """Grid-search the probability threshold that maximises macro F0.5 on out-of-fold scores.

    Returns ((best_threshold, best_score), [(threshold, metrics), ...])."""
    grid = grid if grid is not None else np.round(np.arange(0.20, 0.91, 0.05), 2)
    owned = assign_owner(scored)
    best = (None, -1.0)
    rows = []
    for thr in grid:
        m = macro_f05(owned.filter(pl.col("prob") >= thr), gt_pairs, s1_ids)
        rows.append((thr, m))
        log(f"  thr={thr:.2f}  macroF0.5={m['macro_f05']:.5f}  P={m['micro_precision']:.4f}  "
            f"R={m['micro_recall']:.4f}  singletons={m['singleton_acc']:.4f}")
        if m["macro_f05"] > best[1]:
            best = (float(thr), m["macro_f05"])
    return best, rows
