"""Stage 4b: probability-context features for a second-stage model.

The stage-1 probability of a pair is put in the context of its competitors:
the best competing S1 for the same candidate, the rank of the pair inside the
S1 entity and inside the candidate, and how many confident matches the S1 entity
has. On train the stage-1 probabilities are out-of-fold, on test they are the
average of the two fold models.
"""
import os

import numpy as np
import polars as pl


def prob_context(scored):
    """scored: (s1_id, cand_id, prob) in part order -> p1_* feature columns in the same order."""
    d = scored.select("s1_id", "cand_id", pl.col("prob").alias("p1_prob")).with_row_index("_ord")
    d = d.with_columns(
        pl.col("p1_prob").rank("ordinal", descending=True).over("cand_id").alias("p1_rank_cand"),
        pl.col("p1_prob").rank("ordinal", descending=True).over("s1_id").alias("p1_rank_s1"),
        pl.col("p1_prob").max().over("s1_id").alias("p1_max_s1"),
        pl.col("p1_prob").sum().over("s1_id").alias("p1_sum_s1"),
        (pl.col("p1_prob") > 0.5).sum().over("s1_id").alias("p1_n_high_s1"),
        pl.col("p1_prob").sum().over("cand_id").alias("p1_sum_cand"),
    ).with_columns(
        pl.col("p1_prob").filter(pl.col("p1_rank_cand") != 1).max().over("cand_id").fill_null(0.0).alias("_second"),
        pl.col("p1_prob").max().over("cand_id").alias("_best"),
    ).with_columns(
        pl.when(pl.col("p1_rank_cand") == 1).then(pl.col("p1_prob") - pl.col("_second"))
        .otherwise(pl.col("p1_prob") - pl.col("_best")).alias("p1_margin_cand"),
        (pl.col("p1_sum_s1") - pl.col("p1_prob")).alias("p1_sum_s1_others"),
    ).sort("_ord")
    cols = [c for c in d.columns if c.startswith("p1_")]
    return d.select([pl.col(c).cast(pl.Float32) for c in cols])


def source_context(scored, cand_src):
    """Source-aware stage-1 context (v5): almost every S1 entity with matches has at least one
    record in *each* of S2 and S3 (only 7.9% / 6.9% of them have none), so an ambiguous
    candidate more likely belongs to the competing S1 that has no confident match yet from
    the candidate's source.

    scored: (s1_id, cand_id, prob) in part order; cand_src: (entity_id, src) of the candidates.
    Returns p1_src_* / p1_osrc_* columns in the same order (confident = probability > 0.5)."""
    d = (scored.select("s1_id", "cand_id", "prob").with_row_index("_ord")
         .join(cand_src.select(pl.col("entity_id").alias("cand_id"), pl.col("src").alias("_src")),
               on="cand_id", how="left", maintain_order="left"))
    hi = (pl.col("prob") > 0.5).cast(pl.Int32)
    grp = ["s1_id", "_src"]
    d = d.with_columns(
        (hi.sum().over(grp) - hi).alias("p1_src_n_high_s1"),
        (hi.sum().over("s1_id") - hi.sum().over(grp)).alias("p1_osrc_n_high_s1"),
        pl.col("prob").rank("ordinal", descending=True).over(grp).alias("_rk_src"),
        pl.col("prob").rank("ordinal", descending=True).over("cand_id").alias("_rk_cand"),
    ).with_columns(
        pl.when(pl.col("_rk_src") == 1)
        .then(pl.col("prob").filter(pl.col("_rk_src") != 1).max().over(grp))
        .otherwise(pl.col("prob").max().over(grp)).fill_null(0.0).alias("p1_src_max_s1"),
    )
    comp = []
    for f in ("p1_src_n_high_s1", "p1_src_max_s1", "p1_osrc_n_high_s1"):
        first = pl.col(f).filter(pl.col("_rk_cand") == 1).first().over("cand_id")
        second = pl.col(f).filter(pl.col("_rk_cand") == 2).first().over("cand_id")
        comp.append(pl.when(pl.col("_rk_cand") == 1).then(second).otherwise(first)
                    .fill_null(-1).alias(f.replace("_s1", "_comp")))
    d = d.with_columns(comp).sort("_ord")
    cols = ["p1_src_n_high_s1", "p1_src_max_s1", "p1_osrc_n_high_s1",
            "p1_src_n_high_comp", "p1_src_max_comp", "p1_osrc_n_high_comp"]
    return d.select([pl.col(c).cast(pl.Float32) for c in cols])


SIB_THRESHOLD = 0.9  # a "confident sibling" is another candidate of the same S1 with stage-1 probability >= this


def sibling_context(scored, cand_attr):
    """Sibling context (v9): the S2/S3 records of one entity often share an address variant that
    differs from the S1 address. On v7 out-of-fold scores, a pair whose candidate has exactly the
    address of a confident sibling (another candidate of the same S1) is a true match far more
    often than its probability says (0.45-0.7 band: 77% true vs 58% predicted), while the other
    pairs are slightly over-predicted.

    scored: (s1_id, cand_id, prob) in part order; cand_attr: (entity_id, addr_tokens, addr_nums,
    name_core) of the candidates. Returns p1_sib_* columns in the same order: the number of other
    confident candidates of the S1 (and of the best competing S1 of the candidate) with the same
    address tokens / house numbers / core name (0 when the value is empty)."""
    # memory-lean: ids and attribute values become 64-bit hashes (empty value -> null), so the
    # window functions over 46M training pairs run on numbers instead of strings
    def h(col):
        """64-bit hash of a string column; null for an empty value."""
        return pl.when(pl.col(col).fill_null("") != "").then(pl.col(col).hash(seed=3)).otherwise(None)

    attr = cand_attr.select(pl.col("entity_id").hash(seed=3).alias("_c"), h("addr_tokens").alias("_a"),
                            h("addr_nums").alias("_n"), h("name_core").alias("_m"))
    d = (scored.select(pl.col("s1_id").hash(seed=3).alias("_s"), pl.col("cand_id").hash(seed=3).alias("_c"),
                       pl.col("prob").cast(pl.Float32)).with_row_index("_ord")
         .join(attr, on="_c", how="left", maintain_order="left"))
    del attr
    confident = pl.col("prob") >= SIB_THRESHOLD
    for key, name in (("_a", "addr"), ("_n", "nums"), ("_m", "name")):
        # confident candidates per (S1, value): a small table (confident pairs only) joined back
        cnt = d.filter(confident & pl.col(key).is_not_null()).group_by(["_s", key]).len("_cnt")
        self_conf = (confident & pl.col(key).is_not_null()).cast(pl.Int32)
        d = (d.join(cnt, on=["_s", key], how="left", maintain_order="left")
             .with_columns((pl.col("_cnt").fill_null(0).cast(pl.Int32) - self_conf).alias(f"p1_sib_{name}_s1"))
             .drop("_cnt", key))
        del cnt
    d = d.with_columns(pl.col("prob").rank("ordinal", descending=True).over("_c").alias("_rk_cand"))
    comp = []
    for name in ("addr", "nums", "name"):
        f = f"p1_sib_{name}_s1"
        first = pl.col(f).filter(pl.col("_rk_cand") == 1).first().over("_c")
        second = pl.col(f).filter(pl.col("_rk_cand") == 2).first().over("_c")
        comp.append(pl.when(pl.col("_rk_cand") == 1).then(second).otherwise(first)
                    .fill_null(-1).alias(f"p1_sib_{name}_comp"))
    d = d.with_columns(comp).sort("_ord")
    cols = [f"p1_sib_{n}_{w}" for w in ("s1", "comp") for n in ("addr", "nums", "name")]
    return d.select([pl.col(c).cast(pl.Float32) for c in cols])


def ce_context(ce):
    """Cross-encoder context (v12) over the uncertain pairs of a split, same row order as `ce`
    (s1_id, cand_id, ce_logit): the logit's rank among the S1's uncertain candidates, its gap to the
    best other candidate of the S1, the S1's number of positive logits, its gap to the best other S1
    claiming the same record, and how many S1 entities claim the record."""
    c = pl.col("ce_logit")
    d = ce.select("s1_id", "cand_id", "ce_logit").with_columns(
        c.rank("ordinal", descending=True).over("s1_id").cast(pl.Float32).alias("p2_ce_rank_s1"),
        (c > 0).sum().over("s1_id").cast(pl.Float32).alias("p2_ce_npos_s1"),
        pl.len().over("cand_id").cast(pl.Float32).alias("p2_ce_ncl_cand"),
        c.max().over("s1_id").alias("_m1"), c.max().over("cand_id").alias("_m2"))
    second = lambda key, name: d.group_by(key).agg(c.top_k(2).alias("_t")) \
        .select(key, pl.col("_t").list.sort(descending=True).list.get(1, null_on_oob=True).alias(name))
    d = d.join(second("s1_id", "_s1b"), on="s1_id", how="left", maintain_order="left") \
        .join(second("cand_id", "_s2b"), on="cand_id", how="left", maintain_order="left")
    other1 = pl.when(c == pl.col("_m1")).then(pl.col("_s1b")).otherwise(pl.col("_m1"))
    other2 = pl.when(c == pl.col("_m2")).then(pl.col("_s2b")).otherwise(pl.col("_m2"))
    return d.select((c - other1).cast(pl.Float32).alias("p2_ce_gap_s1"), "p2_ce_rank_s1", "p2_ce_npos_s1",
                    (c - other2).cast(pl.Float32).alias("p2_ce_gap_cand"), "p2_ce_ncl_cand")


def write_sibling_context_parts(scored, cand_attr, part_sizes, out_dir):
    """Split the sibling context table back into row-aligned parts."""
    os.makedirs(out_dir, exist_ok=True)
    ctx = sibling_context(scored, cand_attr)
    offs = np.concatenate([[0], np.cumsum(part_sizes)])
    for i in range(len(part_sizes)):
        ctx.slice(int(offs[i]), int(part_sizes[i])).write_parquet(os.path.join(out_dir, f"part_{i:03d}.parquet"))
    return out_dir


def write_source_context_parts(scored, cand_src, part_sizes, out_dir):
    """Split the source-aware context table back into row-aligned parts."""
    os.makedirs(out_dir, exist_ok=True)
    ctx = source_context(scored, cand_src)
    offs = np.concatenate([[0], np.cumsum(part_sizes)])
    for i in range(len(part_sizes)):
        ctx.slice(int(offs[i]), int(part_sizes[i])).write_parquet(os.path.join(out_dir, f"part_{i:03d}.parquet"))
    return out_dir


def write_context_parts(scored, part_sizes, out_dir):
    """Split the context table back into row-aligned parts."""
    os.makedirs(out_dir, exist_ok=True)
    ctx = prob_context(scored)
    offs = np.concatenate([[0], np.cumsum(part_sizes)])
    for i in range(len(part_sizes)):
        ctx.slice(int(offs[i]), int(part_sizes[i])).write_parquet(os.path.join(out_dir, f"part_{i:03d}.parquet"))
    return out_dir
