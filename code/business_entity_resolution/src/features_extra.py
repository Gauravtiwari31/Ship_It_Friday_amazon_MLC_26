"""Stage 3b: extra pair features added in v2 (from the error analysis).

* rarity   : how many S1 / S2-S3 records share the exact (order-free) core name,
             the address token set, or the name + first house number. A unique
             name with an empty address is almost surely a match; a common one is not.
* word-risk: target encoding of the words present in only one of the two names
             ("extra" words of the candidate, "missing" words of S1). The data
             contains near-duplicate distractors that add words such as
             "Greater"/"Southside", while true variants add "Services"/"Center".
             Encoded out-of-fold on train (fold k uses statistics of fold 1-k);
             test uses the statistics of all training pairs. Unseen words fall
             back to the prior, so an unseen country (France) is handled neutrally.
"""
import os

import polars as pl

PRIOR_WEIGHT = 20.0


def _name_key():
    """Order-free name key: the sorted set of core-name words."""
    return (pl.col("name_core").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
            .list.unique().list.sort().list.join(" "))


def _addr_key():
    """Order-free address key: the sorted set of address tokens."""
    return (pl.col("addr_tokens").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
            .list.unique().list.sort().list.join(" "))


def rarity_tables(norm):
    """Per-record rarity counts (within the record's country)."""
    d = norm.select(
        "entity_id", "country", (pl.col("src") == 1).alias("is_s1"),
        _name_key().alias("nk"), _addr_key().alias("ak"),
        pl.col("addr_nums").str.split(" ").list.first().fill_null("").alias("n1"),
    ).with_columns((pl.col("nk") + "#" + pl.col("n1")).alias("nnk"))
    out = d.select("entity_id", "country", "nk", "ak", "nnk")
    for key, name in (("nk", "name"), ("ak", "addr"), ("nnk", "namenum")):
        cnt = d.group_by("country", key).agg(
            pl.col("is_s1").sum().alias(f"{name}_s1"), (~pl.col("is_s1")).sum().alias(f"{name}_oth"))
        out = out.join(cnt, on=["country", key], how="left")
    return out


def token_stats(pairs_with_tokens):
    """pairs: (label, extra: list[str], missing: list[str]) -> per-token (n, pos) for each role."""
    res = {}
    for role in ("extra", "missing"):
        res[role] = (pairs_with_tokens.select("label", role).explode(role).drop_nulls(role)
                     .group_by(role).agg(pl.len().alias("n"), pl.col("label").sum().alias("pos"))
                     .rename({role: "tok"}))
    return res


def diff_tokens(df):
    """Add 'extra' (candidate-only core words) and 'missing' (S1-only core words) lists."""
    ta = pl.col("name_core_a").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    tb = pl.col("name_core_b").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    return df.with_columns(tb.list.set_difference(ta).alias("extra"), ta.list.set_difference(tb).alias("missing"))


def encode(df, stats, prior):
    """Aggregate smoothed match-rates of the extra / missing words of every pair."""
    df = df.with_row_index("_rid")
    for role in ("extra", "missing"):
        st = stats[role].with_columns(
            ((pl.col("pos") + PRIOR_WEIGHT * prior) / (pl.col("n") + PRIOR_WEIGHT)).alias("rate"))
        e = (df.select("_rid", role).explode(role).drop_nulls(role).rename({role: "tok"})
             .join(st.select("tok", "rate"), on="tok", how="left").with_columns(pl.col("rate").fill_null(prior))
             .group_by("_rid").agg(pl.col("rate").min().alias(f"f_{role}_rate_min"),
                                   pl.col("rate").mean().alias(f"f_{role}_rate_mean"),
                                   pl.len().alias(f"f_{role}_n")))
        df = df.join(e, on="_rid", how="left").with_columns(
            pl.col(f"f_{role}_rate_min").fill_null(-1.0), pl.col(f"f_{role}_rate_mean").fill_null(-1.0),
            pl.col(f"f_{role}_n").fill_null(0))
    return df.drop("_rid")


def add_rarity(df, rar):
    """Attach the rarity counts of both records of each pair, plus exact name-key / address-key equality."""
    a = rar.select(pl.col("entity_id").alias("s1_id"),
                   pl.col("name_s1").alias("f_rar_name_s1_a"), pl.col("name_oth").alias("f_rar_name_oth_a"),
                   pl.col("addr_s1").alias("f_rar_addr_s1_a"), pl.col("namenum_s1").alias("f_rar_namenum_s1_a"),
                   pl.col("nk").alias("_nk_a"), pl.col("ak").alias("_ak_a"))
    b = rar.select(pl.col("entity_id").alias("cand_id"),
                   pl.col("name_s1").alias("f_rar_name_s1_b"), pl.col("name_oth").alias("f_rar_name_oth_b"),
                   pl.col("addr_s1").alias("f_rar_addr_s1_b"), pl.col("namenum_s1").alias("f_rar_namenum_s1_b"),
                   pl.col("nk").alias("_nk_b"), pl.col("ak").alias("_ak_b"))
    return (df.join(a, on="s1_id", how="left").join(b, on="cand_id", how="left")
            .with_columns((pl.col("_nk_a") == pl.col("_nk_b")).alias("f_name_key_eq"),
                          ((pl.col("_ak_a") == pl.col("_ak_b")) & (pl.col("_ak_a") != "")).alias("f_addr_key_eq"))
            .drop("_nk_a", "_nk_b", "_ak_a", "_ak_b"))


def build_extra(work_dir, split, part_files, log=print):
    """Write <work>/<split>_extra/part_XXX.parquet aligned row-by-row with the feature parts."""
    out_dir = os.path.join(work_dir, f"{split}_extra")
    os.makedirs(out_dir, exist_ok=True)
    norm = pl.read_parquet(os.path.join(work_dir, f"{split}_norm.parquet"),
                           columns=["entity_id", "src", "country", "name_core", "addr_tokens", "addr_nums"])
    rar = rarity_tables(norm)
    names = norm.select("entity_id", "name_core")
    del norm

    def with_names(p):
        """Attach the core names of both records of each pair."""
        return (p.join(names.rename({"entity_id": "s1_id", "name_core": "name_core_a"}), on="s1_id", how="left")
                .join(names.rename({"entity_id": "cand_id", "name_core": "name_core_b"}), on="cand_id", how="left"))

    stats_path = os.path.join(work_dir, "model", "word_risk_stats.parquet")
    if split == "train":
        # per-fold statistics for out-of-fold encoding, and all-train statistics for test
        tok = []
        for f in part_files:
            p = pl.read_parquet(f, columns=["s1_id", "cand_id", "label", "fold"])
            tok.append(diff_tokens(with_names(p)).select("label", "fold", "extra", "missing"))
        tok = pl.concat(tok)
        fold_stats = {k: token_stats(tok.filter(pl.col("fold") == k)) for k in (0, 1)}
        fold_prior = {k: tok.filter(pl.col("fold") == k)["label"].mean() for k in (0, 1)}
        all_stats = token_stats(tok)
        prior_all = tok["label"].mean()
        pl.concat([all_stats[r].with_columns(pl.lit(r).alias("role")) for r in all_stats]) \
            .with_columns(pl.lit(prior_all).alias("prior")).write_parquet(stats_path)
        del tok
    else:
        s = pl.read_parquet(stats_path)
        all_stats = {r: s.filter(pl.col("role") == r).select("tok", "n", "pos") for r in ("extra", "missing")}
        prior_all = s["prior"][0]

    for i, f in enumerate(part_files):
        p = pl.read_parquet(f, columns=["s1_id", "cand_id"] + (["fold"] if split == "train" else []))
        d = diff_tokens(with_names(p))
        if split == "train":
            enc = []
            for k in (0, 1):
                sub = d.with_row_index("_ord").filter(pl.col("fold") == k)
                enc.append(encode(sub, fold_stats[1 - k], fold_prior[1 - k]))
            d = pl.concat(enc).sort("_ord").drop("_ord")
        else:
            d = encode(d, all_stats, prior_all)
        d = add_rarity(d, rar)
        cols = [c for c in d.columns if c.startswith("f_")]
        d.select([pl.col(c).cast(pl.Float32) for c in cols]).write_parquet(os.path.join(out_dir, f"part_{i:03d}.parquet"))
        log(f"  {split} extra features part {i + 1}/{len(part_files)}")
    return out_dir
