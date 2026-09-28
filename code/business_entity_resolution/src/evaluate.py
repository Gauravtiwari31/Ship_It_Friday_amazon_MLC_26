"""Macro F0.5 exactly as defined by the challenge.

Per S1 entity: F0.5 = 1.25*P*R / (0.25*P + R) = 1.25*tp / (0.25*n_true + n_pred);
an entity with no true matches scores 1.0 for an empty prediction, 0.0 otherwise.
"""
import polars as pl


def macro_f05(pred_pairs, gt_pairs, s1_ids):
    """pred_pairs / gt_pairs: frames with (s1_id, cand_id); s1_ids: all S1 ids evaluated."""
    base = pl.DataFrame({"s1_id": list(s1_ids)})
    gt = gt_pairs.filter(pl.col("s1_id").is_in(base["s1_id"].implode()))
    pr = pred_pairs.filter(pl.col("s1_id").is_in(base["s1_id"].implode())).select("s1_id", "cand_id").unique()
    tp = pr.join(gt, on=["s1_id", "cand_id"], how="semi").group_by("s1_id").len("tp")
    n_pred = pr.group_by("s1_id").len("n_pred")
    n_true = gt.group_by("s1_id").len("n_true")
    d = (base.join(tp, on="s1_id", how="left").join(n_pred, on="s1_id", how="left")
         .join(n_true, on="s1_id", how="left").fill_null(0))
    f = pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0).otherwise(
        1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")).clip(1e-9))
    d = d.with_columns(f.alias("f05"))
    tot_tp, tot_pred, tot_true = d["tp"].sum(), d["n_pred"].sum(), d["n_true"].sum()
    return {
        "macro_f05": d["f05"].mean(),
        "micro_precision": tot_tp / max(1, tot_pred),
        "micro_recall": tot_tp / max(1, tot_true),
        "singleton_acc": d.filter(pl.col("n_true") == 0)["f05"].mean(),
        "n": d.height,
    }
