"""Reading / writing helpers. All raw files are tab-separated with no quoting."""
import os
import polars as pl

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path):
    """Read a raw TSV: tab separator, no quote handling (quotes are data), every column as text."""
    return pl.read_csv(
        path, separator="\t", quote_char=None, infer_schema=False,
        missing_utf8_is_empty_string=True, truncate_ragged_lines=True,
    )


def load_split(data_dir, split, cache_dir):
    """Return one polars frame with all three sources of a split (train/test).

    Adds `src` (1/2/3). Cached as parquet for fast re-loading.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{split}_raw.parquet")
    if os.path.exists(cache):
        return pl.read_parquet(cache)
    frames = []
    for s in (1, 2, 3):
        df = read_tsv(os.path.join(data_dir, split, f"{split}_source{s}.tsv"))
        df = df.select(SRC_COLS).with_columns(
            pl.lit(s, dtype=pl.Int8).alias("src"),
            *[pl.col(c).fill_null("") for c in SRC_COLS],
        )
        frames.append(df)
    df = pl.concat(frames)
    df.write_parquet(cache)
    return df


def load_ground_truth(data_dir, cache_dir):
    """Return (s1_id, match_id) pairs frame for the training split."""
    cache = os.path.join(cache_dir, "train_gt_pairs.parquet")
    if os.path.exists(cache):
        return pl.read_parquet(cache)
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    pairs = (
        gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "cand_id"})
    )
    pairs.write_parquet(cache)
    return pairs


def write_id_lists(path, s1_ids, mapping, col_name):
    """Write a result TSV: one row per S1 id, comma-joined id list (may be empty)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col_name}\n")
        for s1 in s1_ids:
            ids = mapping.get(s1, ())
            seen, out = set(), []
            for x in ids:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
            f.write(f"{s1}\t{','.join(out)}\n")
