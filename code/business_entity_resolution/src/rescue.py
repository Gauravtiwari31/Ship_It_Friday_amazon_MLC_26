"""Targeted rescue candidates, added to the normal blocking candidates inside blocking.generate_candidates (the
matcher decides). A rescue adds at most `k` candidates per affected record and never widens the normal top-k. Each
rescue is switched on separately in config.RESCUE and evaluated as its own experiment.

* translit (EXP-003): records whose name is written in a non-Latin script. Transliteration mostly works, but it
  yields spelling variants of the English S1 name (lakshmi / laxmi, jai / jay, shree / sree), and these names are
  shared by dozens of S1 businesses, so name keys cannot rank the right one. The record is paired with every S1 of
  the country that has the same *spelling-variant name key*; those S1 are ranked by the blocking address cosine and
  the best k with some address evidence (cosine >= min_addr_cos) become candidates. Script-based, not country-based.
"""
import re

import numpy as np
import polars as pl

from blocking import rowwise_dot

_DIGRAPHS = [("ksh", "x"), ("ks", "x"), ("sh", "s"), ("ph", "f"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"),
             ("gh", "g"), ("ch", "c"), ("jh", "j"), ("w", "v"), ("z", "j"), ("q", "k")]
_VOWELS = re.compile(r"(?<=.)[aeiouy]+")
_REPEAT = re.compile(r"(.)\1+")
_EMPTY = np.empty(0, np.int64)


def variant_key(token):
    """Spelling-variant key of a romanised word: common digraphs merged, vowels after the first letter dropped,
    repeated letters collapsed ("lakshmi" / "laxmi" -> "lxm", "shree" / "sree" -> "sr", "jai" / "jay" -> "j")."""
    for a, b in _DIGRAPHS:
        token = token.replace(a, b)
    return _REPEAT.sub(r"\1", _VOWELS.sub("", token))


def name_variant_key(core):
    """Order-free key of a core name: the sorted variant keys of its words ("" when empty)."""
    return " ".join(sorted({variant_key(t) for t in core.split() if t}))


def translit_keys(d1, d2, V, n2, k=1, min_addr_cos=0.05):
    """Pair keys (s1_row * n2 + record_row) of the translit rescue for one country (see the module docstring).
    d1 / d2: S1 / S2-S3 records of the country (name_core, name_native); V: the blocking views."""
    ci = np.flatnonzero(d2["name_native"].to_numpy())
    if len(ci) == 0:
        return _EMPTY
    rec = pl.DataFrame({"c": ci, "vk": [name_variant_key(s) for s in d2["name_core"].gather(ci).to_list()]})
    s1 = pl.DataFrame({"r": np.arange(d1.height), "vk": [name_variant_key(s) for s in d1["name_core"].to_list()]})
    pairs = rec.filter(pl.col("vk") != "").join(s1, on="vk").select("r", "c")
    if pairs.height == 0:
        return _EMPTY
    r, c = pairs["r"].to_numpy(), pairs["c"].to_numpy()
    cos = rowwise_dot(V["addr"][0], V["addr"][1], r, c)
    best = pl.DataFrame({"r": r, "c": c, "cos": cos}).filter(pl.col("cos") >= min_addr_cos) \
        .sort(["c", "cos"], descending=[False, True]).group_by("c", maintain_order=True).head(k)
    return best["r"].to_numpy().astype(np.int64) * n2 + best["c"].to_numpy().astype(np.int64)


def _name_group_size(names):
    """For every name: how many names of the list share its order-free core-name key."""
    key = pl.Series(names).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort().list.join(" ")
    return pl.DataFrame({"k": key}).with_columns(pl.len().over("k").alias("n"))["n"].to_numpy()


def _order_free(names):
    """Order-free core-name keys (sorted unique words)."""
    return pl.Series(names).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort().list.join(" ")


def typo_keys(d1, d2, V, n2, k=1, min_word_sim=75, min_addr_cos=0.05, min_len=4):
    """Typo rescue (EXP-004): a record word of >= min_len letters that occurs in no S1 name of the country is
    treated as a possible spelling corruption and replaced by its closest S1-vocabulary word (character-trigram
    retrieval, accepted when the edit similarity is >= min_word_sim). The corrected name is paired with every S1
    of the same order-free name key; a record with an address keeps the best k by blocking address cosine (>=
    min_addr_cos), a record without an address only a unique S1 name."""
    from rapidfuzz import fuzz
    from sklearn.feature_extraction.text import HashingVectorizer
    from sparse_dot_topn import sp_matmul_topn

    from blocking import N_FEATURES, _apply_tfidf, _idf
    names1 = d1["name_core"].to_list()
    vocab = sorted({t for s in names1 for t in s.split()})
    vset = set(vocab)
    names2 = d2["name_core"].to_list()
    ci = [i for i, s in enumerate(names2) if any(len(t) >= min_len and t not in vset for t in s.split())]
    if not ci:
        return _EMPTY
    oov = sorted({t for i in ci for t in names2[i].split() if len(t) >= min_len and t not in vset})
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=N_FEATURES, alternate_sign=False, norm=None, dtype=np.float32)
    cv, co = hv.transform(vocab), hv.transform(oov)
    idf = _idf((cv, co), 10 ** 9)
    C = sp_matmul_topn(_apply_tfidf(co, idf), _apply_tfidf(cv, idf).T.tocsr(), top_n=10, threshold=0.1, sort=True).tocsr()
    fix = {}
    for q, word in enumerate(oov):  # nearest vocabulary word by edit similarity among the 10 trigram neighbours
        s, e = C.indptr[q], C.indptr[q + 1]
        if s < e:
            cand = [vocab[j] for j in C.indices[s:e]]
            sims = [fuzz.ratio(word, w) for w in cand]
            b = int(np.argmax(sims))
            if sims[b] >= min_word_sim:
                fix[word] = cand[b]
    fixed = [(i, " ".join(fix.get(t, t) for t in names2[i].split())) for i in ci]
    fixed = [(i, s) for i, s in fixed if s != names2[i]]
    if not fixed:
        return _EMPTY
    rec = pl.DataFrame({"c": [i for i, _ in fixed], "key": _order_free([s for _, s in fixed])})
    s1 = pl.DataFrame({"r": np.arange(d1.height), "key": _order_free(names1)}).with_columns(pl.len().over("key").alias("grp"))
    pairs = rec.join(s1, on="key").select("r", "c", "grp")
    if pairs.height == 0:
        return _EMPTY
    r, c = pairs["r"].to_numpy().astype(np.int64), pairs["c"].to_numpy().astype(np.int64)
    empty = d2["addr_empty"].to_numpy()[c]
    cos = rowwise_dot(V["addr"][0], V["addr"][1], r, c)
    best = pl.DataFrame({"r": r, "c": c, "cos": cos, "empty": empty, "grp": pairs["grp"].to_numpy()}) \
        .filter(pl.when(pl.col("empty")).then(pl.col("grp") == 1).otherwise(pl.col("cos") >= min_addr_cos)) \
        .sort(["c", "cos"], descending=[False, True]).group_by("c", maintain_order=True).head(k)
    return best["r"].to_numpy().astype(np.int64) * n2 + best["c"].to_numpy().astype(np.int64)


def _num_word_keys(s):
    """Address cross keys "house number|address word" (first 3 numbers x first 10 words): an exact house number
    together with a street / locality word. Input: "numbers<TAB>address tokens"."""
    nums, _, toks = s.partition("\t")
    words = [t for t in toks.split() if not t.isdigit()][:10]
    return [n + "|" + w for n in nums.split()[:3] for w in dict.fromkeys(words)]


def dba_keys(d1, d2, V, n2, k=1, min_cos=0.3, min_margin=0.0):
    """Trade-name rescue (EXP-005): a record whose name contains no word of any S1 name of the country (an invented
    brand or trade name) cannot be found by name, so it is retrieved by its address alone, on "house number|word"
    cross keys (TF-IDF): only an exact house number together with the same street / locality words scores, shifted
    numbers never do. The best S1 becomes a candidate if its cosine is >= min_cos and beats the 2nd by min_margin."""
    from sparse_dot_topn import sp_matmul_topn

    from blocking import _apply_tfidf, _idf, hashed_counts
    vocab = {t for s in d1["name_core"].to_list() for t in s.split()}
    names2, nums2 = d2["name_core"].to_list(), d2["addr_nums"].to_list()
    ci = np.array([i for i, s in enumerate(names2) if s and nums2[i] and not any(t in vocab for t in s.split())], np.int64)
    if len(ci) == 0:
        return _EMPTY
    t1 = [a + "\t" + b for a, b in zip(d1["addr_nums"].to_list(), d1["addr_tokens"].to_list())]
    at2 = d2["addr_tokens"].to_list()
    t2 = [nums2[i] + "\t" + at2[i] for i in ci]
    c1, c2 = hashed_counts(t1, _num_word_keys), hashed_counts(t2, _num_word_keys)
    idf = _idf((c1, c2), 10 ** 9)
    M1, Q = _apply_tfidf(c1, idf), _apply_tfidf(c2, idf)
    C = sp_matmul_topn(Q, M1.T.tocsr(), top_n=2, threshold=0.01, sort=True).tocsr()
    best = []
    for q in range(C.shape[0]):
        s, e = C.indptr[q], C.indptr[q + 1]
        if s < e and C.data[s] >= min_cos and C.data[s] - (C.data[s + 1] if e - s > 1 else 0.0) >= min_margin:
            best.append((C.indices[s], ci[q]))
    if not best:
        return _EMPTY
    r, c = np.array(best, np.int64).T
    return r * n2 + c


def common_keys(d1, d2, V, n2, k=1, top_n=20, min_cos=0.3, min_margin=0.1, min_group=2):
    """Common-name + partial-address rescue (EXP-006): a record whose name is ambiguous (even its rarest name word
    occurs in >= min_group S1 names of the country) and whose address is partial (house number + a few locality
    words, which the pruned blocking address view scores ~0) is matched through exact evidence only: the S1 sharing
    a name word AND a house number (the blocking "name word x house number" view, top_n) are re-ranked by
    "house number|address word" cross keys (TF-IDF, frequent locality words kept). The best becomes a candidate if
    its cosine is >= min_cos and beats the 2nd of the retrieved set by min_margin. House numbers must be equal."""
    from sparse_dot_topn import sp_matmul_topn

    from blocking import _apply_tfidf, _idf, hashed_counts
    df = {}
    for s in d1["name_core"].to_list():
        for t in set(s.split()):
            df[t] = df.get(t, 0) + 1
    names2, nums2, empty2 = d2["name_core"].to_list(), d2["addr_nums"].to_list(), d2["addr_empty"].to_numpy()

    def ambiguous(s):
        f = [df[t] for t in s.split() if t in df]
        return bool(f) and min(f) >= min_group
    ci = np.array([i for i, s in enumerate(names2) if nums2[i] and not empty2[i] and ambiguous(s)], np.int64)
    if len(ci) == 0:
        return _EMPTY
    C = sp_matmul_topn(V["cross"][1][ci], V["cross"][0].T.tocsr(), top_n=top_n, threshold=0.01, sort=True).tocsr()
    q, r = np.repeat(np.arange(len(ci)), np.diff(C.indptr)), C.indices.astype(np.int64)
    if len(r) == 0:
        return _EMPTY
    at2 = d2["addr_tokens"].to_list()
    c1 = hashed_counts([a + "\t" + b for a, b in zip(d1["addr_nums"].to_list(), d1["addr_tokens"].to_list())], _num_word_keys)
    c2 = hashed_counts([nums2[i] + "\t" + at2[i] for i in ci], _num_word_keys)
    idf = _idf((c1, c2), 10 ** 9)
    cos = rowwise_dot(_apply_tfidf(c2, idf), _apply_tfidf(c1, idf), q, r)
    t = pl.DataFrame({"q": q, "r": r, "cos": cos}).sort(["q", "cos"], descending=[False, True]) \
        .with_columns(pl.col("cos").shift(-1).over("q").fill_null(0.0).alias("next"), pl.int_range(pl.len()).over("q").alias("rank"))
    best = t.filter((pl.col("rank") < k) & (pl.col("cos") >= min_cos) & (pl.col("cos") - pl.col("next") >= min_margin))
    return best["r"].to_numpy().astype(np.int64) * n2 + ci[best["q"].to_numpy()]


def bienc_keys(d1, d2, V, n2, path, k=1, min_cos=0.0):
    """Learned retrieval (leader_gap R1): every S2/S3 record's top-1 S1 of its country under a supervised multilingual
    bi-encoder (e5-small fine-tuned on training pairs with hard negatives, cross-fitted by S1 fold; see retrieval.py),
    precomputed in `path` (s1_id, cand_id, bi_cos). Record-centric: at most one S1 per record, so the candidate set
    grows by < 0.1 pairs per record while recovering most of the records the lexical views miss."""
    p = pl.read_parquet(path, columns=["s1_id", "cand_id", "bi_cos"]).filter(pl.col("bi_cos") >= min_cos)
    r1 = pl.DataFrame({"s1_id": d1["entity_id"], "_r": np.arange(d1.height, dtype=np.int64)})
    r2 = pl.DataFrame({"cand_id": d2["entity_id"], "_c": np.arange(d2.height, dtype=np.int64)})
    p = p.join(r1, on="s1_id").join(r2, on="cand_id")
    if p.height == 0:
        return _EMPTY
    return p["_r"].to_numpy() * n2 + p["_c"].to_numpy()


RESCUES = {"translit": translit_keys, "typo": typo_keys, "dba": dba_keys, "common": common_keys, "bienc": bienc_keys}
