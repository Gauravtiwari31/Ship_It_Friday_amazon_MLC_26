"""Stage 2: candidate generation (blocking).

For every country label present in the data (open set) we build TF-IDF vectors
over hashed "rare keys" (keys more frequent than max_df are dropped):
  * name  : core-name words, sorted word pairs, 4-char prefixes/suffixes of long
            words (typo tolerant) and the order-free concatenation of the words
            (matches website/hashtag names that glue the words together)
  * addr  : address word and number tokens, and adjacent token pairs
  * cross : name word x house number ("ram|18"), rare even for generic names
  * comb  : [name | addr | cross] concatenated
and retrieve, with a multi-threaded sparse top-k product (sparse_dot_topn):
  * forward : for each S1 record the top-K candidates by name, addr and comb
  * reverse : for each S2/S3 record the top-k S1 records by comb
The union of these pairs is the candidate set.
"""
import os

import numpy as np
import polars as pl
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

N_FEATURES = 2 ** 22
TAB = chr(9)


def name_keys(s):
    """Blocking keys of a core name: words, sorted word pairs, 4-char prefixes and
    suffixes of longer words (typo tolerant) and the order-free concatenation."""
    toks = s.split()
    if not toks:
        return []
    out = ["w" + t for t in toks]
    for t in toks:
        if len(t) >= 6:
            out.append("p" + t[:4])
            out.append("s" + t[-4:])
    u = sorted(set(toks))
    if 1 < len(u) <= 8:
        out += ["q" + a + "|" + b for i, a in enumerate(u) for b in u[i + 1:]]
    out.append("c" + "".join(u))
    return out


def cross_keys(s):
    """Name-word x house-number keys ("ram|18"): rare even when the name is generic
    and the address is short. Input: "core name<TAB>numbers"."""
    name, _, nums = s.partition(TAB)
    toks = list(dict.fromkeys(name.split()))[:6]
    nums = nums.split()[:4]
    if not toks or not nums:
        return []
    out = ["x" + t + "|" + n for t in toks for n in nums]
    cat = "".join(sorted(toks))
    out += ["y" + cat + "|" + n for n in nums]
    return out


def addr_keys(s):
    """Blocking keys of an address: tokens (words and numbers) and adjacent pairs."""
    toks = s.split()
    out = ["a" + t for t in toks]
    out += ["b" + toks[i] + "|" + toks[i + 1] for i in range(len(toks) - 1)]
    return out


def _hash_chunk(texts, analyzer):
    """Key counts of one chunk of texts as a sparse matrix (hashing trick, no vocabulary to store)."""
    hv = HashingVectorizer(analyzer=analyzer, n_features=N_FEATURES, alternate_sign=False,
                           norm=None, dtype=np.float32)
    return hv.transform(texts)


def hashed_counts(texts, analyzer, n_jobs=-1, chunk=200_000):
    """Sparse key-count matrix of all texts, computed in parallel chunks."""
    chunks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    mats = Parallel(n_jobs=n_jobs, max_nbytes=None)(delayed(_hash_chunk)(c, analyzer) for c in chunks)
    return sp.vstack(mats).tocsr() if mats else sp.csr_matrix((0, N_FEATURES), dtype=np.float32)


def _idf(counts_pair, max_df):
    """IDF from the key counts of both sides; keys more frequent than max_df get
    weight 0 (uninformative, and they dominate the cost of the top-k product)."""
    df = np.zeros(N_FEATURES, dtype=np.int64)
    n = 0
    for m in counts_pair:
        df += np.bincount(m.indices, minlength=N_FEATURES)
        n += m.shape[0]
    idf = (np.log((n + 1) / (df + 1)) + 1).astype(np.float32)
    idf[df > max_df] = 0
    return idf


def _apply_tfidf(m, idf):
    """In place: sublinear tf * idf, drop zeroed keys, L2-normalise rows."""
    m.data = (1 + np.log(m.data)) * idf[m.indices]
    m.eliminate_zeros()
    return normalize(m, norm="l2", copy=False)


def _texts(df):
    """The three texts the blocking keys are built from: name (core name, else full name),
    address tokens, and name + house numbers (for the name x number cross keys)."""
    name_txt = df.select(
        pl.when(pl.col("name_core") != "").then(pl.col("name_core")).otherwise(pl.col("name_full"))
    ).to_series().to_list()
    addr_txt = df["addr_tokens"].to_list()
    cross_txt = [a + TAB + b for a, b in zip(name_txt, df["addr_nums"].to_list())]
    return {"name": name_txt, "addr": addr_txt, "cross": cross_txt}


ANALYZERS = {"name": name_keys, "addr": addr_keys, "cross": cross_keys}


def build_views(d1, d2, max_df, log=print):
    """TF-IDF matrices for S1 records (d1) and S2/S3 records (d2), one view at a time
    to keep memory low. Returns {view: (M1, M2)} for name, addr, cross, comb."""
    t1, t2 = _texts(d1), _texts(d2)
    V = {}
    for view, fn in ANALYZERS.items():
        c1 = hashed_counts(t1[view], fn)
        c2 = hashed_counts(t2[view], fn)
        idf = _idf((c1, c2), max_df)
        V[view] = (_apply_tfidf(c1, idf), _apply_tfidf(c2, idf))
        log(f"    view {view}: nnz {V[view][0].nnz + V[view][1].nnz:,}")
    del t1, t2
    V["comb"] = tuple(
        normalize(sp.hstack([V["name"][i], V["addr"][i], V["cross"][i]], format="csr"), norm="l2", copy=False)
        for i in (0, 1))
    return V


def topk(A, B, k, thr=0.0, n_threads=None):
    """For each row of A, the k rows of B with largest cosine. Returns (rows, cols, vals)."""
    if A.shape[0] == 0 or B.shape[0] == 0:
        return (np.empty(0, np.int64),) * 2 + (np.empty(0, np.float32),)
    n_threads = n_threads or os.cpu_count()
    C = sp_matmul_topn(A, B.T.tocsr(), top_n=k, threshold=thr, sort=False, n_threads=n_threads)
    C = C.tocoo()
    return C.row.astype(np.int64), C.col.astype(np.int64), C.data.astype(np.float32)


def rowwise_dot(A, B, ia, ib, chunk=1_000_000):
    """cosine between row ia[t] of A and row ib[t] of B for all t (both L2-normalised)."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        a = A[ia[s:s + chunk]]
        b = B[ib[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def generate_candidates(norm_df, params, log=print, rescue=None):
    """Return polars frame (s1_id, cand_id, blk_name, blk_addr, blk_cross, blk_comb).
    Countries are processed independently (open set of labels). rescue: {method: params} of targeted rescue
    candidates (rescue.py, config.RESCUE); rescue pairs get the same blk_* cosines as any other candidate."""
    from rescue import RESCUES  # rescue.py imports this module
    results = []
    for country in sorted(norm_df["country"].unique().to_list()):
        cdf = norm_df.filter(pl.col("country") == country)
        d1, d2 = cdf.filter(pl.col("src") == 1), cdf.filter(pl.col("src") != 1)
        del cdf
        if d1.height == 0 or d2.height == 0:
            continue
        log(f"  {country}: {d1.height:,} S1 vs {d2.height:,} S2/S3 records")
        id1, id2 = d1["entity_id"].to_numpy(), d2["entity_id"].to_numpy()
        V = build_views(d1, d2, params["max_df"], log=log)
        if rescue:
            d1, d2 = d1.select("entity_id", "name_core", "addr_nums", "addr_tokens"), \
                d2.select("entity_id", "name_core", "name_native", "addr_nums", "addr_empty", "addr_tokens")
        else:
            del d1, d2
        keys = []
        n2 = np.int64(len(id2))
        for view, k in (("name", params["k_name"]), ("addr", params["k_addr"]), ("comb", params["k_comb"])):
            if k > 0:
                r, c, _ = topk(V[view][0], V[view][1], k, params["thr"])
                keys.append(r * n2 + c)
                log(f"    forward top-{k} by {view}: {len(r):,} pairs")
        if params["k_rev"] > 0:
            c, r, _ = topk(V["comb"][1], V["comb"][0], params["k_rev"], params["thr"])
            keys.append(r * n2 + c)
            log(f"    reverse top-{params['k_rev']} by comb: {len(r):,} pairs")
        key = np.unique(np.concatenate(keys))
        del keys
        for name, rp in (rescue or {}).items():  # targeted rescue candidates (rescue.py), one switch per method
            rk = RESCUES[name](d1, d2, V, n2, **rp)
            new = np.setdiff1d(rk, key)
            key = np.union1d(key, new)
            log(f"    rescue {name}: {len(rk):,} top pairs, {len(new):,} new candidates")
        if rescue:
            del d1, d2
        r, c = key // n2, key % n2
        feats = {"blk_" + v: rowwise_dot(V[v][0], V[v][1], r, c) for v in ("name", "addr", "cross", "comb")}
        del V
        results.append(pl.DataFrame({"s1_id": id1[r], "cand_id": id2[c], **feats}))
        log(f"    {country}: {len(r):,} unique candidate pairs ({len(r) / len(id1):.1f} per S1)")
    return pl.concat(results) if results else pl.DataFrame()
