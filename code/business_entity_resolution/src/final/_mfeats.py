# shared invariant group features for the France decision layer (see loo_meta.py)
def mfeats(f, META):
    """Invariant group features over final probabilities (edges with prob >= 0.01)."""
    d = f.filter(pl.col("prob") >= 0.01).with_columns(ORIG.alias("_o"))
    pc = pl.col("prob").clip(1e-6, 1 - 1e-6)
    d = d.with_columns((pc / (1 - pc)).log().alias("lp"), pl.len().over("cand_id").alias("r_n"),
                       pl.col("prob").rank("ordinal", descending=True).over("cand_id").alias("r_rank"),
                       pl.col("prob").max().over("cand_id").alias("r_best"), pl.col("prob").sum().over("cand_id").alias("r_sum"),
                       pl.len().over("s1_id").alias("s_n"), pl.col("prob").rank("ordinal", descending=True).over("s1_id").alias("s_rank"),
                       pl.col("prob").sum().over("s1_id").alias("s_sum"), (pl.col("prob") >= 0.5).sum().over("s1_id").alias("s_n05"),
                       pl.col("f_bi_top1").sum().over("s1_id").alias("s_bi_n"))
    d = d.with_columns(pl.col("prob").top_k(2).get(1, null_on_oob=True).over("cand_id").fill_null(0).alias("r_second"))
    d = d.with_columns(pl.when(pl.col("r_rank") == 1).then(pl.col("prob") - pl.col("r_second"))
                       .otherwise(pl.col("prob") - pl.col("r_best")).alias("r_margin"))
    d = d.join(META, on="_o", how="left")
    conf = d.filter((pl.col("prob") >= 0.9) & (pl.col("r_rank") == 1))
    cs = conf.group_by("s1_id").agg(pl.len().alias("sib_n"), (pl.col("csrc") == 2).sum().alias("sib_n2"), (pl.col("csrc") == 3).sum().alias("sib_n3"))
    d = d.join(cs, on="s1_id", how="left").with_columns(pl.col(["sib_n", "sib_n2", "sib_n3"]).fill_null(0))
    sc = ((pl.col("prob") >= 0.9) & (pl.col("r_rank") == 1)).cast(pl.Int32)
    d = d.with_columns((pl.col("sib_n") - sc).alias("sib_n"),
                       pl.when(pl.col("csrc") == 2).then(pl.col("sib_n2") - sc).otherwise(pl.col("sib_n3") - sc).alias("sib_same"),
                       pl.when(pl.col("csrc") == 2).then(pl.col("sib_n3")).otherwise(pl.col("sib_n2")).alias("sib_other")).drop("sib_n2", "sib_n3")
    cc = ["prob", "sib_n", "lp", "f_bi_cos1", "s_sum"]
    t2 = d.filter(pl.col("r_rank") <= 2).select("cand_id", "r_rank", *cc)
    c1 = t2.filter(pl.col("r_rank") == 1).drop("r_rank"); c2 = t2.filter(pl.col("r_rank") == 2).drop("r_rank")
    d = d.join(c1.rename({c: "c1_" + c for c in cc}), on="cand_id", how="left").join(c2.rename({c: "c2_" + c for c in cc}), on="cand_id", how="left")
    for c in cc:
        d = d.with_columns(pl.when(pl.col("r_rank") == 1).then(pl.col("c2_" + c)).otherwise(pl.col("c1_" + c)).alias("comp_" + c))
    d = d.drop([x for x in d.columns if x.startswith("c1_") or x.startswith("c2_")])
    d = d.with_columns((pl.col("sib_n") - pl.col("comp_sib_n")).alias("d_sib_n"), (pl.col("lp") - pl.col("comp_lp")).alias("d_lp"),
                       (pl.col("f_bi_cos1") - pl.col("comp_f_bi_cos1")).alias("d_bi"), pl.col("p3").is_not_null().cast(pl.Int8).alias("in_zone"))
    return d


MF = ["prob", "lp", "chain", "p1", "p3", "p2_ce_logit", "f_bi_top1", "f_bi_cos1", "f_bi_gap", "r_n", "r_rank", "r_best", "r_second", "r_margin",
      "r_sum", "s_n", "s_rank", "s_sum", "s_n05", "s_bi_n", "addr_empty", "csrc", "name_owners", "sib_n", "sib_same", "sib_other", "comp_prob",
      "comp_sib_n", "comp_lp", "comp_f_bi_cos1", "comp_s_sum", "d_sib_n", "d_lp", "d_bi", "in_zone"]


