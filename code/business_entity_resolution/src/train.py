"""Stage 4: train the pair classifier with 2-fold cross-fitting.

* model_k is trained on (a sample of) the S1 entities of fold k;
* every training pair is scored by the model of the *other* fold (out-of-fold),
  so the matching decisions and the threshold are tuned on honest scores;
* on the test set the two models are averaged.

Features live in several row-aligned part directories (base features, extra
features, stage-2 probability context); `dirs` lists the ones to combine.

Two interchangeable backends (config.MODEL_BACKEND):
* "lightgbm": LightGBM on the CPU (v1-v4);
* "xgb_cuda": XGBoost on the NVIDIA GPU (depth-10 trees). On a 6M-pair benchmark it trains
  3x faster than LightGBM on 16 CPU threads with the same validation log-loss.
"""
import glob
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb

from config import LGB_MAX_ROUNDS, MODEL_BACKEND, SEED, TRAIN_SAMPLE_S1
from features import feature_columns

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.08,
    "num_leaves": 255,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "num_threads": os.cpu_count(),
    "verbose": -1,
    "seed": SEED,
}


XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "logloss"],  # the last metric drives early stopping
    "tree_method": "hist",
    "device": "cuda",
    "eta": 0.08,
    "max_depth": 10,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "max_bin": 255,
    "seed": SEED,
}


class _Chunks(xgb.DataIter):
    """Feeds a list of float32 matrices to XGBoost without concatenating them in RAM."""

    def __init__(self, Xs, ys, feats):
        """Xs, ys: lists of feature matrices and label vectors; feats: feature names."""
        self._Xs, self._ys, self._feats, self._i = Xs, ys, feats, 0
        super().__init__()

    def next(self, input_data):
        """Pass the next chunk to XGBoost; return False when all chunks are consumed."""
        if self._i == len(self._Xs):
            return False
        input_data(data=self._Xs[self._i], label=self._ys[self._i], feature_names=self._feats)
        self._i += 1
        return True

    def reset(self):
        """Restart from the first chunk."""
        self._i = 0


class _XgbLog(xgb.callback.TrainingCallback):
    """XGBoost callback that logs the validation metrics every `period` rounds."""

    def __init__(self, period, log):
        """period: rounds between log lines; log: logging function."""
        self.period, self.log = period, log
        super().__init__()

    def after_iteration(self, model, epoch, evals_log):
        """Called by XGBoost after every boosting round; returning False continues training."""
        if epoch % self.period == 0:
            res = "  ".join(f"{d}-{m}={v[-1]:.5f}" for d, ms in evals_log.items() for m, v in ms.items())
            self.log(f"    round {epoch:4d}  {res}")
        return False


class XGBModel:
    """An XGBoost booster behind the small part of the LightGBM Booster API the pipeline uses."""

    def __init__(self, booster):
        """booster: a trained xgboost.Booster with feature names."""
        self.booster = booster
        self.booster.set_param({"device": "cuda"})

    def feature_name(self):
        """Feature names in model input order."""
        return list(self.booster.feature_names)

    def predict(self, X, num_threads=None):
        """Match probabilities for a float matrix (computed on the GPU)."""
        return self.booster.inplace_predict(np.ascontiguousarray(X, dtype=np.float32))

    def feature_importance(self, kind="gain"):
        """Total gain per feature, in feature_name() order."""
        d = self.booster.get_score(importance_type="total_gain")
        return [d.get(f, 0.0) for f in self.feature_name()]

    def save_model(self, path):
        """Save the booster (UBJSON)."""
        self.booster.save_model(path)


def _train_xgb(Xtr, ytr, Xes, yes, feats, rounds, params, log):
    """Train XGBoost on the GPU with early stopping; returns (model cut at its best iteration,
    best iteration, early-stop log-loss)."""
    dtr = xgb.QuantileDMatrix(_Chunks(Xtr, ytr, feats), max_bin=XGB_PARAMS["max_bin"])
    des = xgb.QuantileDMatrix(_Chunks(Xes, yes, feats), ref=dtr, max_bin=XGB_PARAMS["max_bin"])
    Xtr.clear()
    Xes.clear()
    bst = xgb.train({**XGB_PARAMS, **(params or {})}, dtr, num_boost_round=rounds, evals=[(des, "early_stop")],
                    early_stopping_rounds=50, verbose_eval=False, callbacks=[_XgbLog(25, log)])
    best = bst[: bst.best_iteration + 1]
    best.feature_names = feats
    return XGBModel(best), bst.best_iteration, bst.best_score


def model_path(model_dir, prefix, fold):
    """File of a fold model; the extension depends on the backend."""
    ext = "ubj" if MODEL_BACKEND == "xgb_cuda" else "txt"
    return os.path.join(model_dir, f"{prefix}_fold{fold}.{ext}")


def load_model(path):
    """Load a fold model saved by train_all (LightGBM .txt or XGBoost .ubj)."""
    if path.endswith(".ubj"):
        b = xgb.Booster()
        b.load_model(path)
        return XGBModel(b)
    return lgb.Booster(model_file=path)


def part_files(d):
    """Sorted part files of a feature directory."""
    return sorted(glob.glob(os.path.join(d, "part_*.parquet")))


def load_part(i, dirs):
    """Horizontally combine part i of every directory (they are row-aligned)."""
    frames = [pl.read_parquet(part_files(dirs[0])[i])]
    for d in dirs[1:]:
        extra = pl.read_parquet(part_files(d)[i])
        frames.append(extra.select([c for c in extra.columns if c not in frames[0].columns]))
    return pl.concat(frames, how="horizontal")


def n_parts(dirs):
    """Number of parts (the same in every row-aligned directory)."""
    return len(part_files(dirs[0]))


def _log_every(period, log):
    """LightGBM callback that logs the validation metrics every `period` rounds."""
    def _cb(env):
        """Called by LightGBM after every boosting round."""
        if env.iteration % period == 0 or env.iteration == env.end_iteration - 1:
            res = "  ".join(f"{d}-{m}={v:.5f}" for d, m, v, _ in env.evaluation_result_list)
            log(f"    round {env.iteration:4d}  {res}")
    return _cb


def _fold_s1(dirs, fold):
    """Sorted unique S1 ids of a fold (reads only the id and fold columns)."""
    ids = pl.concat([pl.read_parquet(f, columns=["s1_id", "fold"]).filter(pl.col("fold") == fold)
                     for f in part_files(dirs[0])])
    return ids["s1_id"].unique().sort().to_numpy()


def train_fold(dirs, fold, n_s1, rounds=600, log=print, exclude=(), params=None):
    """Train one model (LightGBM or XGBoost, see MODEL_BACKEND) on a sample of n_s1 S1 entities
    of `fold` (all their candidate pairs); 5% of the sampled entities are held out for early stopping.

    Memory-lean: every part is filtered to the sampled entities as it is read and kept
    as a float32 matrix; both libraries bin the list of matrices without concatenating them."""
    rng = np.random.default_rng(SEED + fold)
    s1 = _fold_s1(dirs, fold)
    s1 = s1[rng.choice(len(s1), min(n_s1, len(s1)), replace=False)]
    n_es = max(1, len(s1) // 20)  # 5% of the sampled entities for early stopping
    es_ids, tr_ids = pl.Series(s1[:n_es]).implode(), pl.Series(s1[n_es:]).implode()
    feats = None
    Xtr, ytr, Xes, yes = [], [], [], []
    for i in range(n_parts(dirs)):
        df = load_part(i, dirs).filter(pl.col("fold") == fold)
        feats = feats or feature_columns(df, exclude)
        for ids, X, y in ((tr_ids, Xtr, ytr), (es_ids, Xes, yes)):
            sub = df.filter(pl.col("s1_id").is_in(ids))
            if sub.height:
                X.append(sub.select(pl.col(feats).cast(pl.Float32)).to_numpy())
                y.append(sub["label"].to_numpy())
        del df
    n_tr, n_pos, n_es_rows = sum(len(y) for y in ytr), int(sum(y.sum() for y in ytr)), sum(len(y) for y in yes)
    log(f"fold {fold}: train pairs {n_tr:,} (pos {n_pos:,}), early-stop pairs {n_es_rows:,}, "
        f"{len(feats)} features, backend {MODEL_BACKEND}")
    t = time.time()
    if MODEL_BACKEND == "xgb_cuda":
        booster, best_it, best_ll = _train_xgb(Xtr, ytr, Xes, yes, feats, rounds, params, log)
    else:
        dtr = lgb.Dataset(Xtr, np.concatenate(ytr), feature_name=feats, free_raw_data=True)
        des = lgb.Dataset(Xes, np.concatenate(yes), reference=dtr)
        del Xtr, Xes
        booster = lgb.train(
            {**LGB_PARAMS, **(params or {}), "metric": ["binary_logloss", "auc"]}, dtr, num_boost_round=rounds,
            valid_sets=[des], valid_names=["early_stop"],
            callbacks=[lgb.early_stopping(50, verbose=False), _log_every(25, log)],
        )
        best_it, best_ll = booster.best_iteration, booster.best_score["early_stop"]["binary_logloss"]
    log(f"fold {fold}: best iteration {best_it}, early-stop logloss {best_ll:.5f}, {time.time() - t:.0f}s")
    return booster, feats


def train_all(dirs, model_dir, prefix="lgb", log=print, exclude=(), n_s1=None, rounds=None, params=None):
    """Train (or load, if saved) the two cross-fitted models <prefix>_fold{0,1}.txt / .ubj.
    n_s1: S1 entities sampled per fold (default TRAIN_SAMPLE_S1 // 2; all when TRAIN_SAMPLE_S1 is None)."""
    os.makedirs(model_dir, exist_ok=True)
    n_s1 = n_s1 or (TRAIN_SAMPLE_S1 // 2 if TRAIN_SAMPLE_S1 else 10 ** 9)
    for fold in (0, 1):
        path = model_path(model_dir, prefix, fold)
        if os.path.exists(path):
            continue
        booster, feats = train_fold(dirs, fold, n_s1, rounds=rounds or LGB_MAX_ROUNDS, log=log, exclude=exclude,
                                    params=params)
        if isinstance(booster, XGBModel):
            booster.save_model(path)
        else:
            booster.save_model(path, num_iteration=booster.best_iteration)
        imp = sorted(zip(feats, booster.feature_importance("gain")), key=lambda x: -x[1])
        log("top features (gain): " + ", ".join(f"{n}={v / 1e3:.0f}k" for n, v in imp[:15]))
    return [load_model(model_path(model_dir, prefix, f)) for f in (0, 1)]


def score_parts(dirs, models, oof, log=print, s1_ids=None):
    """Return (s1_id, cand_id, prob[, label]) for all pairs of a split, in part order.
    oof=True: pairs of fold k are scored by the model of fold 1-k; else the average of both.
    s1_ids: optionally restrict to the pairs of these S1 entities."""
    if isinstance(dirs, str):
        dirs = [dirs]
    out = []
    feats = models[0].feature_name()
    keep_ids = pl.Series(list(s1_ids)) if s1_ids is not None else None
    for i in range(n_parts(dirs)):
        df = load_part(i, dirs)
        if keep_ids is not None:
            df = df.filter(pl.col("s1_id").is_in(keep_ids.implode()))
            if df.height == 0:
                continue
        X = df.select(pl.col(feats).cast(pl.Float32)).to_numpy()
        p0 = models[0].predict(X, num_threads=os.cpu_count())
        p1 = models[1].predict(X, num_threads=os.cpu_count())
        if oof:
            prob = np.where(df["fold"].to_numpy() == 0, p1, p0)
        else:
            prob = (p0 + p1) / 2
        keep = ["s1_id", "cand_id"] + (["label"] if "label" in df.columns else [])
        out.append(df.select(keep).with_columns(pl.Series("prob", prob.astype(np.float32))))
        log(f"  scored part {i + 1}/{n_parts(dirs)}")
    return pl.concat(out)
