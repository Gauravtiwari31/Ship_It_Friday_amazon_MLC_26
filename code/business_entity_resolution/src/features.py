"""Stage 3: pair features for (S1 record, candidate record).

All features are country-agnostic similarity measures, so they transfer to
countries not seen in training (e.g. France in the test set).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

NORM_COLS = ["name_core", "name_main", "name_alt", "name_legal", "name_concat", "name_native",
             "addr_tokens", "addr_words", "addr_nums", "addr_state", "addr_empty"]


def _cp(a, b, scorer, **kw):
    """Row-wise RapidFuzz score of two equally long string lists (parallel, float32)."""
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def context_features(pairs):
    """Features that describe the competition around a pair (computed on the full candidate set)."""
    return pairs.with_columns(
        pl.len().over("s1_id").alias("ctx_n_cand"),
        pl.len().over("cand_id").alias("ctx_n_s1"),
        pl.col("blk_comb").rank("ordinal", descending=True).over("s1_id").alias("ctx_rank_in_s1"),
        pl.col("blk_comb").rank("ordinal", descending=True).over("cand_id").alias("ctx_rank_in_cand"),
        (pl.col("blk_comb") - pl.col("blk_comb").max().over("s1_id")).alias("ctx_gap_s1_best"),
        (pl.col("blk_comb") - pl.col("blk_comb").max().over("cand_id")).alias("ctx_gap_cand_best"),
    ).with_columns(
        # margin over the best *other* S1 competing for this candidate
        (pl.col("blk_comb") - pl.col("blk_comb").filter(pl.col("ctx_rank_in_cand") != 1).max().over("cand_id")
         .fill_null(0.0)).alias("ctx_margin_cand"),
    ).with_columns(
        pl.when(pl.col("ctx_rank_in_cand") == 1).then(pl.col("ctx_margin_cand")).otherwise(pl.col("ctx_gap_cand_best"))
        .alias("ctx_margin_cand")
    )


def pair_features(pairs, norm):
    """pairs: frame with s1_id, cand_id (+ blk_*/ctx_* columns). norm: normalised records.
    Returns pairs with added f_* feature columns (float32)."""
    left = norm.select(pl.col("entity_id").alias("s1_id"), *[pl.col(c).alias(c + "_a") for c in NORM_COLS])
    right = norm.select(pl.col("entity_id").alias("cand_id"), pl.col("src").alias("src_b"),
                        *[pl.col(c).alias(c + "_b") for c in NORM_COLS])
    df = pairs.join(left, on="s1_id", how="left").join(right, on="cand_id", how="left")

    na, nb = df["name_core_a"].to_list(), df["name_core_b"].to_list()
    ma, mb, alt_b = df["name_main_a"].to_list(), df["name_main_b"].to_list(), df["name_alt_b"].to_list()
    aa, ab = df["addr_tokens_a"].to_list(), df["addr_tokens_b"].to_list()
    wa, wb = df["addr_words_a"].to_list(), df["addr_words_b"].to_list()
    nsa = [s.replace(" ", "") for s in na]
    nsb = [s.replace(" ", "") for s in nb]

    f = {}
    f["f_name_ratio"] = _cp(na, nb, fuzz.ratio)
    f["f_name_tsort"] = _cp(na, nb, fuzz.token_sort_ratio)
    f["f_name_tset"] = _cp(na, nb, fuzz.token_set_ratio)
    f["f_name_partial"] = _cp(na, nb, fuzz.partial_ratio)
    f["f_name_jw"] = _cp(na, nb, JaroWinkler.normalized_similarity)
    f["f_name_nospace_ratio"] = _cp(nsa, nsb, fuzz.ratio)
    f["f_name_nospace_partial"] = _cp(nsa, nsb, fuzz.partial_ratio)
    f["f_name_main_tset"] = _cp(ma, mb, fuzz.token_set_ratio)
    alt_score = _cp(ma, alt_b, fuzz.token_set_ratio)
    has_alt = np.array([bool(x) for x in alt_b])
    f["f_name_alt_tset"] = np.where(has_alt, alt_score, -1).astype(np.float32)
    f["f_addr_ratio"] = _cp(aa, ab, fuzz.ratio)
    f["f_addr_tset"] = _cp(aa, ab, fuzz.token_set_ratio)
    f["f_addr_tsort"] = _cp(aa, ab, fuzz.token_sort_ratio)
    f["f_addr_partial"] = _cp(aa, ab, fuzz.partial_ratio)
    f["f_addr_words_tset"] = _cp(wa, wb, fuzz.token_set_ratio)
    del na, nb, ma, mb, alt_b, aa, ab, wa, wb, nsa, nsb

    def toks(c):
        """Split a space-separated column into a list of non-empty tokens."""
        return pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))

    df = df.with_columns(pl.DataFrame(f)).with_columns(
        toks("name_core_a").alias("_ta"), toks("name_core_b").alias("_tb"),
        toks("addr_nums_a").alias("_na"), toks("addr_nums_b").alias("_nb"),
        toks("name_legal_a").alias("_la"), toks("name_legal_b").alias("_lb"),
        toks("addr_state_a").alias("_sa"), toks("addr_state_b").alias("_sb"),
    ).with_columns(
        pl.col("_ta").list.len().alias("f_ntok_a"),
        pl.col("_tb").list.len().alias("f_ntok_b"),
        pl.col("_ta").list.set_intersection("_tb").list.len().alias("f_ntok_common"),
        pl.col("_ta").list.set_union("_tb").list.len().alias("_tu"),
        (pl.col("_ta").list.first() == pl.col("_tb").list.first()).fill_null(False).alias("f_first_tok_eq"),
        pl.col("_na").list.len().alias("f_nnum_a"),
        pl.col("_nb").list.len().alias("f_nnum_b"),
        pl.col("_na").list.set_intersection("_nb").alias("_ni"),
        pl.col("_na").list.set_union("_nb").list.len().alias("_nu"),
        (pl.col("_na").list.first() == pl.col("_nb").list.first()).fill_null(False).alias("f_first_num_eq"),
        pl.col("_nb").list.first().is_in(pl.col("_na")).fill_null(False).alias("f_first_num_b_in_a"),
        pl.col("_la").list.len().alias("f_nlegal_a"),
        pl.col("_lb").list.len().alias("f_nlegal_b"),
        pl.col("_la").list.set_intersection("_lb").list.len().alias("f_legal_common"),
        pl.col("_sa").list.len().alias("f_has_state_a"),
        pl.col("_sb").list.len().alias("f_has_state_b"),
        pl.col("_sa").list.set_intersection("_sb").list.len().alias("f_state_common"),
    ).with_columns(
        (pl.col("f_ntok_common") / pl.col("_tu").clip(1)).alias("f_name_jacc"),
        pl.col("_ni").list.len().alias("f_num_common"),
        (pl.col("_ni").list.len() / pl.col("_nu").clip(1)).alias("f_num_jacc"),
        pl.col("_ni").list.eval(pl.element().str.len_chars()).list.max().fill_null(0).alias("f_num_longest_common"),
        ((pl.col("f_nlegal_a") > 0) & (pl.col("f_nlegal_b") > 0) & (pl.col("f_legal_common") == 0)).alias("f_legal_conflict"),
        ((pl.col("f_has_state_a") > 0) & (pl.col("f_has_state_b") > 0) & (pl.col("f_state_common") == 0)).alias("f_state_conflict"),
        pl.col("name_native_a").alias("f_native_a"),
        pl.col("name_native_b").alias("f_native_b"),
        (pl.col("name_concat_b") != "").alias("f_concat_b"),
        (pl.col("name_alt_b") != "").alias("f_alt_b"),
        pl.col("addr_empty_a").alias("f_addr_empty_a"),
        pl.col("addr_empty_b").alias("f_addr_empty_b"),
        pl.col("addr_tokens_a").str.count_matches(" ").alias("f_addr_len_a"),
        pl.col("addr_tokens_b").str.count_matches(" ").alias("f_addr_len_b"),
        pl.col("src_b").alias("f_src_b"),
    )
    keep = [c for c in pairs.columns] + sorted(c for c in df.columns if c.startswith("f_"))
    return df.select(keep).with_columns(
        [pl.col(c).cast(pl.Float32) for c in keep if c.startswith(("f_", "blk_", "ctx_"))]
    )


# State presence is only detectable for countries covered by the hand-written state
# tables (US, India). For any other country (e.g. France in the test set) these
# would always be 0 on both sides, a pattern never seen in training, so they are
# not used; f_state_conflict stays (it is 0 = "no evidence" when states are unknown).
DROP_FEATURES = {"f_has_state_a", "f_has_state_b", "f_state_common"}


# Word-risk encodings are learned from the vocabulary of the training countries. In a
# country without training labels almost every word is unseen (prior value), and the
# leave-one-country-out experiment shows these features then cause over-matching, so
# the model used for unseen countries is trained without them.
WORD_RISK_FEATURES = {"f_extra_rate_min", "f_extra_rate_mean", "f_extra_n",
                      "f_missing_rate_min", "f_missing_rate_mean", "f_missing_n"}


def feature_columns(df, exclude=()):
    """Names of the model input columns (by prefix), minus dropped and explicitly excluded features."""
    return [c for c in df.columns if c.startswith(("f_", "blk_", "ctx_", "p1_", "p2_"))
            and c not in DROP_FEATURES and c not in exclude]


def compute_in_chunks(pairs, norm, chunk=3_000_000, log=print):
    """pair_features() over a large pair table in chunks, to bound memory."""
    parts = []
    for s in range(0, pairs.height, chunk):
        parts.append(pair_features(pairs.slice(s, chunk), norm))
        log(f"  features {min(s + chunk, pairs.height):,}/{pairs.height:,}")
    return pl.concat(parts)
