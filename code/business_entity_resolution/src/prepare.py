"""Stage 1: load raw TSVs, learn transliteration map (train only), normalise all records."""
import json
import os
from multiprocessing import Pool

import polars as pl

from io_utils import load_ground_truth, load_split
from normalize import Segmenter, addr_features, expand_concat, learn_translit, name_features

_TMAP = None
_SEG = None

NAME_KEYS = ["name_full", "name_core", "name_main", "name_alt", "name_legal", "name_concat", "name_native"]
ADDR_KEYS = ["addr_tokens", "addr_words", "addr_nums", "addr_state", "addr_empty"]


def _init(tmap):
    """Worker initialiser: keep the transliteration map in a module global (sent once per process)."""
    global _TMAP
    _TMAP = tmap


def _norm_chunk(args):
    """Normalise one slice of the raw parquet cache in a worker; returns one column per name/address key.

    The worker reads its own rows (path, offset, length), so the parent never holds the raw
    strings as Python lists (that copy made the 12.5M-row train split run out of memory)."""
    path, offset, length = args
    raw = (pl.scan_parquet(path).slice(offset, length)
           .select("business_name", "business_address", "country").collect())
    cols = {k: [] for k in NAME_KEYS + ADDR_KEYS}
    for n, a, c in raw.iter_rows():
        nf = name_features(n, _TMAP, c)
        af = addr_features(a, _TMAP)
        for k in NAME_KEYS:
            cols[k].append(nf[k])
        for k in ADDR_KEYS:
            cols[k].append(af[k])
    return pl.DataFrame(cols)


def _init_seg(seg):
    """Worker initialiser: keep the word segmenter in a module global (sent once per process)."""
    global _SEG
    _SEG = seg


def _seg_chunk(rows):
    """Split the concatenated website/hashtag token of each name row into words (worker function)."""
    out = {k: [] for k in ("name_full", "name_core", "name_main")}
    for r in rows:
        nf = expand_concat(r, _SEG)
        for k in out:
            out[k].append(nf[k])
    return pl.DataFrame(out)


def build_translit(data_dir, work_dir, model_dir, sample=1_500_000):
    """Learn the native-script -> English word map from (a sample of) the training matches.

    Cached in <model_dir>/translit.json; uses only the training labels."""
    path = os.path.join(model_dir, "translit.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    tr = load_split(data_dir, "train", work_dir)
    gt = load_ground_truth(data_dir, work_dir)
    s1 = tr.filter(pl.col("src") == 1).select(
        pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("n1"),
        pl.col("business_address").alias("a1"))
    oth = tr.filter(pl.col("src") != 1).select(
        pl.col("entity_id").alias("cand_id"), pl.col("business_name").alias("n2"),
        pl.col("business_address").alias("a2"))
    p = gt.sample(min(sample, gt.height), seed=0).join(s1, on="s1_id").join(oth, on="cand_id")
    tmap = learn_translit(p.select("n1", "n2", "a1", "a2").iter_rows())
    os.makedirs(model_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tmap, f, ensure_ascii=False)
    return tmap


def normalize_split(data_dir, split, work_dir, tmap, n_jobs=None, chunk=100_000):
    """Normalise every record of a split in parallel and cache it as <work_dir>/<split>_norm.parquet.

    After the per-record normalisation, website/hashtag names are split into words with a
    unigram model of the ordinary name words of the same split (no labels involved).
    At most 8 workers: each one holds the transliteration map and its slice in memory."""
    out_path = os.path.join(work_dir, f"{split}_norm.parquet")
    if os.path.exists(out_path):
        return pl.read_parquet(out_path)
    raw_path = os.path.join(work_dir, f"{split}_raw.parquet")
    if not os.path.exists(raw_path):
        del_raw = load_split(data_dir, split, work_dir)   # builds the raw parquet cache
        del del_raw
    ids = pl.read_parquet(raw_path, columns=["entity_id", "src", "country"])
    jobs = [(raw_path, i, min(chunk, ids.height - i)) for i in range(0, ids.height, chunk)]
    n_jobs = n_jobs or max(1, min(8, (os.cpu_count() or 2) - 1))
    with Pool(n_jobs, initializer=_init, initargs=(tmap,)) as pool:
        parts = list(pool.imap(_norm_chunk, jobs, chunksize=1))   # imap keeps the row order
    df = pl.concat([ids, pl.concat(parts)], how="horizontal")
    del parts

    # unigram model of name words (from ordinary, non-concatenated names) for splitting domains
    vocab = (df.filter((pl.col("name_concat") == "") & (~pl.col("name_native")))
             .select(pl.col("name_full").str.split(" ").explode().alias("w"))
             .filter(pl.col("w") != "").group_by("w").len().filter(pl.col("len") >= 3))
    seg = Segmenter(dict(vocab.iter_rows()))
    del vocab
    mask = df["name_concat"] != ""
    rows = df.filter(mask).select(NAME_KEYS).to_dicts()
    jobs = [rows[i:i + 20_000] for i in range(0, len(rows), 20_000)]
    with Pool(n_jobs, initializer=_init_seg, initargs=(seg,)) as pool:
        seg_parts = pool.map(_seg_chunk, jobs, chunksize=1)
    del rows, jobs
    if seg_parts:
        seg_df = pl.concat(seg_parts)
        idx = mask.arg_true()
        cols = []
        for k in ("name_full", "name_core", "name_main"):
            s = df[k].clone()
            s.scatter(idx, seg_df[k])
            cols.append(s)
        df = df.with_columns(cols)
    df.write_parquet(out_path)
    return df
