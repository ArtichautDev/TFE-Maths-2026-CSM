#!/usr/bin/env python3

import argparse
import gc
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import torch
import yaml
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import MAE


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "paths": {
        "train_parquet": "/workspace/data/train.parquet",
        "val_parquet": "/workspace/data/val.parquet",
        "output_dir": "/workspace/nixtla/runs/tft_h128_2x3090",
    },
    "data": {
        "freq": 1,
        "max_train_rows": 2000000,
        "max_val_rows": 300000,
        "read_tail": True,
        "max_unique_ids": 5000,
    },
    "features": {
        "id_col": "group_id",
        "time_col": "time_idx",
        "target_col": "target",
        "drop_columns": ["datetime", "DELAY_ARR"],
    },
    "model": {
        "horizon": 2,
        "input_size": 14,
        "hidden_size": 128,
        "n_head": 4,
        "dropout": 0.1,
        "n_rnn_layers": 1,
        "rnn_type": "lstm",
        "scaler_type": "robust",
        "use_exog": True,
        "use_futr_exog": False,
    },
    "train": {
        "learning_rate": 0.001,
        "max_steps": 1200,
        "epochs": None,
        "resume_ckpt_path": "",
        "val_size": 0,
        "val_check_steps": 200,
        "early_stop_patience_steps": -1,
        "num_lr_decays": -1,
        "batch_size": 256,
        "valid_batch_size": 256,
        "windows_batch_size": 256,
        "inference_windows_batch_size": 256,
        "step_size": 2,
        "accelerator": "gpu",
        "devices": 2,
        "strategy": "ddp_find_unused_parameters_true",
        "precision": "16-mixed",
        "gradient_clip_val": 0.0,
        "log_every_n_steps": 20,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "enable_progress_bar": True,
        "enable_checkpointing": False,
        "num_sanity_val_steps": 0,
        "limit_val_batches": 0.0,
        "benchmark": True,
        "strict_devices": True,
    },
    "eval": {
        "compute_holdout_mae": True,
    },
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        deep_update(cfg, user_cfg)
    return cfg



def get_exog_splits(
    cfg: Dict[str, Any],
    exog_cols_all: List[str],
    cat_cols_all: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    feats = cfg.get("features", {})
    stat_cfg = feats.get("stat_exog_list") or []
    hist_cfg = feats.get("hist_exog_list") or []
    futr_cfg = feats.get("futr_exog_list") or []

    drop_high_card_idx = bool(feats.get("drop_high_cardinality_idx", True))
    allow_idx_features = set(feats.get("allow_idx_features", []) or [])
    cat_set = set(cat_cols_all)

    if drop_high_card_idx:
        eligible = [c for c in exog_cols_all if c not in cat_set or c in allow_idx_features]
    else:
        eligible = list(exog_cols_all)

    # If user provides explicit lists, respect them (with filtering to existing eligible cols)
    if stat_cfg or hist_cfg or futr_cfg:
        stat = [c for c in stat_cfg if c in eligible]
        futr = [c for c in futr_cfg if c in eligible]
        taken = set(stat) | set(futr)
        hist = [c for c in hist_cfg if c in eligible and c not in taken]
        return stat, hist, futr

    # Automatic split fallback (audit-aligned) when config lists are empty.
    stat_prefixes = (
        "nb_arrets_total",
        "position_dans_trajet",
        "est_gare_terminus",
        "temps_theorique_",
        "traverse_",
        "est_hub_majeur",
        "nb_connexions_gare",
    )
    hist_prefixes = (
        "retard_j_",
        "retard_moyen_",
        "volatilite_",
        "taux_ponctualite_",
        "lag_missing_",
        "meteo_observee_",
        "fiabilite_",
        "risque_retard_",
        "stress_reseau_",
        "indice_complexite",
    )
    futr_prefixes = (
        "heure_prevue_",
        "jour_semaine_",
        "mois_",
        "fest_",
        "is_jour_ferie",
        "is_vacances_scolaires",
        "is_heure_pointe",
        "nb_trains_prevus_",
        "densite_prevue_",
        "congestion_prevue_",
        "travaux_",
        "meteo_",
    )

    stat: List[str] = []
    hist: List[str] = []
    futr: List[str] = []
    for col in eligible:
        if col.startswith(stat_prefixes):
            stat.append(col)
        elif col.startswith(hist_prefixes):
            hist.append(col)
        elif col.startswith(futr_prefixes):
            futr.append(col)
        else:
            hist.append(col)

    taken = set(stat) | set(futr)
    hist = [c for c in hist if c not in taken]
    return stat, hist, futr


def infer_feature_lists(
    train_parquet: Path,
    id_col: str,
    time_col: str,
    target_col: str,
    drop_columns: List[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    schema_cols = pq.read_schema(train_parquet).names
    drop_set = set(drop_columns)
    protected = {id_col, time_col, target_col}
    exog_cols = [c for c in schema_cols if c not in protected and c not in drop_set]
    cat_cols = [c for c in exog_cols if c.endswith("_idx")]
    real_cols = [c for c in exog_cols if not c.endswith("_idx")]
    read_cols = [id_col, time_col, target_col] + exog_cols
    return read_cols, exog_cols, cat_cols, real_cols


def read_parquet_frame(
    parquet_path: Path,
    columns: List[str],
    max_rows: int | None,
    read_tail: bool,
) -> pd.DataFrame:
    if max_rows is None or int(max_rows) <= 0:
        table = pq.read_table(parquet_path, columns=columns)
        for i, field in enumerate(table.schema):
            if field.type == pa.float64():
                table = table.set_column(i, field.name, table.column(i).cast(pa.float32()))
        return table.to_pandas()

    max_rows = int(max_rows)
    parquet_file = pq.ParquetFile(parquet_path)
    selected_row_groups: List[int] = []
    collected = 0

    row_groups = range(parquet_file.num_row_groups - 1, -1, -1) if read_tail else range(parquet_file.num_row_groups)
    for rg in row_groups:
        rg_rows = parquet_file.metadata.row_group(rg).num_rows
        selected_row_groups.append(rg)
        collected += rg_rows
        if collected >= max_rows:
            break

    selected_row_groups = sorted(selected_row_groups)
    table = parquet_file.read_row_groups(selected_row_groups, columns=columns)
    for i, field in enumerate(table.schema):
        if field.type == pa.float64():
            table = table.set_column(i, field.name, table.column(i).cast(pa.float32()))
    df = table.to_pandas()

    if len(df) > max_rows:
        df = df.tail(max_rows).copy() if read_tail else df.head(max_rows).copy()
    return df


def cast_and_clean(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
    exog_cols: List[str],
    cat_cols: List[str],
) -> pd.DataFrame:
    work = df
    work[id_col] = work[id_col].astype("category")
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce").astype(np.int32)
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce").astype(np.float32)

    for col in exog_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").astype(np.float32)

    work = work.dropna(subset=[id_col, time_col, target_col])

    for col in exog_cols:
        if col in cat_cols:
            work[col] = work[col].fillna(-1).clip(lower=0).astype(np.float32)
        else:
            work[col] = work[col].fillna(0.0).astype(np.float32)

    work[target_col] = work[target_col].astype(np.float32)
    work[time_col] = work[time_col].astype(np.int32)

    work = work.drop_duplicates(subset=[id_col, time_col], keep="last")

    keep_cols = [id_col, time_col, target_col] + exog_cols
    drop_cols = [c for c in work.columns if c not in keep_cols]
    if drop_cols:
        work = work.drop(columns=drop_cols)

    return work


def to_nf_format_train(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
) -> Tuple[pd.DataFrame, Dict[Any, int]]:
    out = df.rename(columns={id_col: "unique_id", time_col: "ds", target_col: "y"})
    codes, uniques = pd.factorize(out["unique_id"], sort=True)
    id_map = {uid: int(i) for i, uid in enumerate(uniques.tolist())}

    out["unique_id"] = codes.astype(np.int32)
    out["ds"] = out["ds"].astype(np.int32)
    out["y"] = out["y"].astype(np.float32)
    out = out.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    return out, id_map


def to_nf_format_with_id_map(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
    id_map: Dict[Any, int],
) -> pd.DataFrame:
    out = df.rename(columns={id_col: "unique_id", time_col: "ds", target_col: "y"})
    out["unique_id"] = out["unique_id"].map(id_map)
    out = out.dropna(subset=["unique_id"])

    out["unique_id"] = out["unique_id"].astype(np.int32)
    out["ds"] = out["ds"].astype(np.int32)
    out["y"] = out["y"].astype(np.float32)
    out = out.sort_values(["unique_id", "ds"], kind="mergesort").reset_index(drop=True)
    return out


def estimate_total_windows(
    train_nf: pd.DataFrame,
    input_size: int,
    horizon: int,
    step_size: int,
) -> int:
    seq_len = int(input_size) + int(horizon)
    stride = max(1, int(step_size))
    lengths = train_nf.groupby("unique_id", sort=False).size().to_numpy(dtype=np.int64)

    total = 0
    for series_len in lengths:
        usable = int(series_len) - seq_len + 1
        if usable <= 0:
            continue
        total += 1 + (usable - 1) // stride
    return int(total)


def resolve_devices(train_cfg: Dict[str, Any]) -> int:
    accelerator = str(train_cfg.get("accelerator", "gpu"))
    requested_devices = int(train_cfg.get("devices", 1))
    strict_devices = bool(train_cfg.get("strict_devices", True))

    if accelerator != "gpu":
        return 1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible alors que accelerator=gpu.")

    available_devices = torch.cuda.device_count()
    if available_devices < requested_devices:
        if strict_devices:
            raise RuntimeError(
                f"GPU demandés={requested_devices} mais visibles={available_devices}. "
                "Vérifie CUDA_VISIBLE_DEVICES (attendu: 0,1)."
            )
        requested_devices = available_devices

    return max(1, requested_devices)


def build_model(
    cfg: Dict[str, Any],
    stat_exog_list: List[str],
    hist_exog_list: List[str],
    futr_exog_list: List[str],
    devices: int,
    default_root_dir: str | None = None,
) -> TFT:
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    num_workers = int(train_cfg.get("num_workers", 0))
    dataloader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))

    strategy = "auto"
    if str(train_cfg.get("accelerator", "gpu")) == "gpu" and devices > 1:
        strategy = str(train_cfg.get("strategy", "ddp_find_unused_parameters_true"))

    windows_batch_size = train_cfg.get("windows_batch_size", 1024)
    if windows_batch_size is not None:
        windows_batch_size = int(windows_batch_size)

    inference_windows_batch_size = train_cfg.get("inference_windows_batch_size", 1024)
    if inference_windows_batch_size is not None:
        inference_windows_batch_size = int(inference_windows_batch_size)

    trainer_kwargs: Dict[str, Any] = {}
    if default_root_dir:
        trainer_kwargs["default_root_dir"] = default_root_dir

    return TFT(
        loss=MAE(),
        h=int(model_cfg["horizon"]),
        input_size=int(model_cfg["input_size"]),
        hidden_size=int(model_cfg["hidden_size"]),
        n_head=int(model_cfg.get("n_head", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        attn_dropout=float(model_cfg.get("attn_dropout", 0.1)),
        n_rnn_layers=int(model_cfg.get("n_rnn_layers", 1)),
        rnn_type=str(model_cfg.get("rnn_type", "lstm")),
        stat_exog_list=stat_exog_list or None,
        hist_exog_list=hist_exog_list or None,
        futr_exog_list=futr_exog_list or None,
        
        scaler_type=str(model_cfg.get("scaler_type", "robust")),
        learning_rate=float(train_cfg["learning_rate"]),
        max_steps=int(train_cfg["max_steps"]),
        val_check_steps=int(train_cfg.get("val_check_steps", 100)),
        early_stop_patience_steps=int(train_cfg.get("early_stop_patience_steps", -1)),
        num_lr_decays=int(train_cfg.get("num_lr_decays", -1)),
        batch_size=int(train_cfg["batch_size"]),
        valid_batch_size=int(train_cfg.get("valid_batch_size", train_cfg["batch_size"])),
        windows_batch_size=windows_batch_size,
        inference_windows_batch_size=inference_windows_batch_size,
        step_size=int(train_cfg.get("step_size", 1)),
        random_seed=int(cfg.get("seed", 42)),
        dataloader_kwargs=dataloader_kwargs,
        accelerator=str(train_cfg.get("accelerator", "gpu")),
        devices=devices,
        strategy=strategy,
        precision=str(train_cfg.get("precision", "16-mixed")),
        gradient_clip_val=float(train_cfg.get("gradient_clip_val", 0.0)),
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 20)),
        enable_progress_bar=bool(train_cfg.get("enable_progress_bar", True)),
        enable_checkpointing=bool(train_cfg.get("enable_checkpointing", False)),
        num_sanity_val_steps=int(train_cfg.get("num_sanity_val_steps", 0)),
        limit_val_batches=float(train_cfg.get("limit_val_batches", 0.0)),
        benchmark=bool(train_cfg.get("benchmark", True)),
        enable_model_summary=False,
        logger=False,
        **trainer_kwargs,
    )


def holdout_eval_mae(
    nf: NeuralForecast,
    val_nf: pd.DataFrame,
    horizon: int,
    futr_exog_list: List[str],
    static_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    val_head = (
        val_nf[["unique_id", "ds", "y"]]
        .sort_values(["unique_id", "ds"], kind="mergesort")
        .groupby("unique_id", as_index=False)
        .head(horizon)
    )

    available_futr = [c for c in futr_exog_list if c in val_nf.columns]
    try:
        if available_futr:
            expected_futr = nf.make_future_dataframe(df=val_nf)
            global_val_futr = val_nf[["unique_id", "ds"] + available_futr].drop_duplicates(
                subset=["unique_id", "ds"]
            )
            futr_df = expected_futr[["unique_id", "ds"]].merge(
                global_val_futr,
                on=["unique_id", "ds"],
                how="left",
            )
            for col in available_futr:
                # First fill within each series, then fallback by timestamp and finally neutral zero.
                futr_df[col] = futr_df.groupby("unique_id", sort=False)[col].ffill().bfill()
                if futr_df[col].isna().any():
                    by_ds = global_val_futr.groupby("ds", sort=False)[col].mean()
                    futr_df[col] = futr_df[col].fillna(futr_df["ds"].map(by_ds))
                futr_df[col] = futr_df[col].fillna(0.0).astype(np.float32)

            missing_future = nf.get_missing_future(futr_df=futr_df, df=val_nf)
            if len(missing_future) > 0:
                for col in available_futr:
                    missing_future[col] = 0.0
                futr_df = pd.concat([futr_df, missing_future], ignore_index=True)
                futr_df = futr_df.drop_duplicates(subset=["unique_id", "ds"], keep="last")

            preds = nf.predict(df=val_nf, futr_df=futr_df, static_df=static_df)
        else:
            preds = nf.predict(df=val_nf, static_df=static_df)
    except Exception as exc:
        return {
            "matched_rows": 0,
            "finite_rows": 0,
            "nan_pred_rows": 0,
            "mae": None,
            "model_col": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    pred_cols = [c for c in preds.columns if c not in {"unique_id", "ds"}]
    if not pred_cols:
        return {"matched_rows": 0, "mae": None}

    model_col = pred_cols[0]
    merged = val_head.merge(preds[["unique_id", "ds", model_col]], on=["unique_id", "ds"], how="inner")

    if merged.empty:
        return {"matched_rows": 0, "mae": None, "model_col": model_col}

    y_true = merged["y"].to_numpy(dtype=np.float64)
    y_pred = merged[model_col].to_numpy(dtype=np.float64)
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    finite_rows = int(finite_mask.sum())
    nan_pred_rows = int((~np.isfinite(y_pred)).sum())

    if finite_rows == 0:
        mae = None
    else:
        mae = float(np.mean(np.abs(y_pred[finite_mask] - y_true[finite_mask])))

    return {
        "matched_rows": int(len(merged)),
        "finite_rows": finite_rows,
        "nan_pred_rows": nan_pred_rows,
        "mae": mae,
        "model_col": model_col,
    }


def run_training(config_path: Path) -> None:
    cfg = load_config(config_path)
    global_rank = int(os.environ.get("RANK", "0"))
    is_global_zero = global_rank == 0

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    train_parquet = Path(cfg["paths"]["train_parquet"])
    val_parquet = Path(cfg["paths"]["val_parquet"])
    output_root = Path(cfg["paths"]["output_dir"])
    run_name = str(cfg.get("paths", {}).get("run_name", "")).strip() or time.strftime("run_%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    id_col = str(cfg["features"]["id_col"])
    time_col = str(cfg["features"]["time_col"])
    target_col = str(cfg["features"]["target_col"])
    drop_columns = list(cfg["features"].get("drop_columns", []))

    read_cols_all, exog_cols_all, cat_cols_all, real_cols_all = infer_feature_lists(
        train_parquet=train_parquet,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        drop_columns=drop_columns,
    )

    use_exog = bool(cfg["model"].get("use_exog", True))
    if use_exog:
        stat_exog_list, hist_exog_list, futr_exog_list = get_exog_splits(cfg, exog_cols_all, cat_cols_all)
        use_futr_exog = bool(cfg["model"].get("use_futr_exog", False))
        if not use_futr_exog:
            futr_exog_list = []

        exog_cols = sorted(set(stat_exog_list + hist_exog_list + futr_exog_list))
        cat_cols = [c for c in exog_cols if c.endswith("_idx")]
        real_cols = [c for c in exog_cols if c not in cat_cols]
        read_cols = [id_col, time_col, target_col] + exog_cols
    else:
        stat_exog_list = []
        hist_exog_list = []
        futr_exog_list = []
        read_cols = [id_col, time_col, target_col]
        exog_cols = []
        cat_cols = []
        real_cols = []

    print("==== nixtla_tft setup ====")
    print(f"train_parquet={train_parquet}")
    print(f"val_parquet={val_parquet}")
    print(f"output_dir={output_dir}")
    print(f"CUDA_VISIBLE_DEVICES={torch.cuda.device_count()} visible GPUs in torch")
    print(
        f"exog_available={len(exog_cols_all)} exog_used={len(exog_cols)} "
        f"real_used={len(real_cols)} cat_idx_used={len(cat_cols)}"
    )
    print(
        f"stat_exog={len(stat_exog_list)} hist_exog={len(hist_exog_list)} "
        f"futr_exog={len(futr_exog_list)}"
    )

    data_cfg = cfg.get("data", {})
    max_train_rows = data_cfg.get("max_train_rows")
    max_val_rows = data_cfg.get("max_val_rows")
    read_tail = bool(data_cfg.get("read_tail", True))
    max_unique_ids = data_cfg.get("max_unique_ids")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_read_stagger_seconds = int(data_cfg.get("ddp_read_stagger_seconds", 0))

    print(f"max_train_rows={max_train_rows} max_val_rows={max_val_rows} read_tail={read_tail}")
    print(f"max_unique_ids={max_unique_ids}")
    print(f"global_rank={global_rank} world_size={world_size}")
    if world_size > 1 and ddp_read_stagger_seconds > 0 and global_rank > 0:
        sleep_for = int(ddp_read_stagger_seconds) * int(global_rank)
        print(f"ddp_rank_stagger_sleep_seconds={sleep_for}")
        time.sleep(sleep_for)

    devices = resolve_devices(cfg["train"])

    train_raw = read_parquet_frame(
        parquet_path=train_parquet,
        columns=read_cols,
        max_rows=max_train_rows,
        read_tail=read_tail,
    )
    train_df = cast_and_clean(
        train_raw,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        exog_cols=exog_cols,
        cat_cols=cat_cols,
    )

    if max_unique_ids is not None and int(max_unique_ids) > 0:
        top_ids = train_df[id_col].value_counts().head(int(max_unique_ids)).index
        train_df = train_df[train_df[id_col].isin(top_ids)].copy()

    static_df = None
    if stat_exog_list:
        static_df = (
            train_df[[id_col] + stat_exog_list]
            .groupby(id_col, as_index=False, sort=False, observed=True)
            .last()
            .copy()
        )

    train_nf, id_map = to_nf_format_train(train_df, id_col=id_col, time_col=time_col, target_col=target_col)

    if static_df is not None:
        static_df = static_df.rename(columns={id_col: "unique_id"})
        static_df["unique_id"] = static_df["unique_id"].map(id_map)
        static_df = static_df.dropna(subset=["unique_id"]).reset_index(drop=True)
        static_df["unique_id"] = static_df["unique_id"].astype(np.int32)

    del train_raw
    del train_df
    gc.collect()

    val_size_for_fit = int(cfg["train"].get("val_size", 0))
    dropped_short_series = 0
    min_len = int(cfg["model"]["input_size"]) + int(cfg["model"]["horizon"]) + val_size_for_fit
    if min_len > 0:
        sizes = train_nf.groupby("unique_id", sort=False).size()
        keep_ids = sizes[sizes >= min_len].index
        dropped_short_series = int(len(sizes) - len(keep_ids))
        if dropped_short_series > 0:
            train_nf = train_nf[train_nf["unique_id"].isin(keep_ids)].reset_index(drop=True)
            gc.collect()
            print(
                f"dropped_short_series={dropped_short_series} "
                f"(len < {min_len})"
            )

    train_series = int(train_nf["unique_id"].nunique())
    train_rows_loaded = int(len(train_nf))
    batch_size = int(cfg["train"]["batch_size"])
    step_size = int(cfg["train"].get("step_size", 1))
    windows_batch_size = cfg["train"].get("windows_batch_size", 1024)
    total_windows = estimate_total_windows(
        train_nf=train_nf,
        input_size=int(cfg["model"]["input_size"]),
        horizon=int(cfg["model"]["horizon"]),
        step_size=step_size,
    )

    steps_per_epoch_by_series = int(math.ceil(train_series / max(1, batch_size * max(1, devices))))
    if windows_batch_size is None:
        approx_steps_from_windows = int(math.ceil(total_windows / max(1, batch_size * max(1, devices))))
        window_mode = "all_windows"
    else:
        approx_steps_from_windows = int(math.ceil(total_windows / max(1, int(windows_batch_size) * max(1, devices))))
        window_mode = f"sampled_windows({int(windows_batch_size)})"

    print("==== dataset stats ====")
    print(f"train_rows_loaded={train_rows_loaded}")
    print(f"train_series={train_series}")
    print(f"estimated_total_windows={total_windows}")
    print(f"window_mode={window_mode}")
    print(f"approx_steps_per_epoch_by_series={steps_per_epoch_by_series}")
    print(f"approx_steps_per_epoch_by_windows={approx_steps_from_windows}")
    print("==== training config ====")
    print(f"requested_epochs={cfg['train'].get('epochs')}")
    print(f"configured_max_steps={cfg['train'].get('max_steps')}")
    print(f"val_size={cfg['train'].get('val_size', 0)}")
    print(f"val_check_steps={cfg['train'].get('val_check_steps', 100)}")
    print(f"early_stop_patience_steps={cfg['train'].get('early_stop_patience_steps', -1)}")
    print(f"limit_val_batches={cfg['train'].get('limit_val_batches', 1.0)}")

    requested_epochs = cfg["train"].get("epochs")
    step_budget_mode = str(cfg["train"].get("step_budget_mode", "series")).strip().lower()
    if requested_epochs is not None and int(requested_epochs) > 0:
        if step_budget_mode == "windows":
            step_budget_per_epoch = max(1, int(approx_steps_from_windows))
        else:
            step_budget_per_epoch = max(1, int(steps_per_epoch_by_series))
        effective_max_steps = int(step_budget_per_epoch * int(requested_epochs))
        cfg["train"]["max_steps"] = effective_max_steps
    else:
        effective_max_steps = int(cfg["train"].get("max_steps", 1))

    epochs_equivalent_series = (
        float(effective_max_steps) / float(steps_per_epoch_by_series)
        if steps_per_epoch_by_series > 0
        else 0.0
    )
    epochs_equivalent_windows = (
        float(effective_max_steps) / float(approx_steps_from_windows)
        if approx_steps_from_windows > 0
        else 0.0
    )
    epochs_equivalent = epochs_equivalent_series
    print(f"effective_max_steps={effective_max_steps}")
    print(f"step_budget_mode={step_budget_mode}")
    print(f"series_epochs_equivalent={epochs_equivalent_series:.3f}")
    print(f"windows_epochs_equivalent={epochs_equivalent_windows:.3f}")

    model = build_model(
        cfg,
        stat_exog_list,
        hist_exog_list,
        futr_exog_list,
        devices,
        default_root_dir=str(output_dir),
    )
    resume_ckpt_path = str(cfg["train"].get("resume_ckpt_path", "")).strip()
    resume_info: Dict[str, Any] = {}
    skip_fit = False
    if resume_ckpt_path:
        ckpt_path = Path(resume_ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint introuvable: {ckpt_path}")

        print("==== resume checkpoint ====")
        print(f"resume_ckpt_path={ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise RuntimeError("checkpoint invalide: state_dict absent")

        load_out = model.load_state_dict(state_dict, strict=False)
        missing_n = len(getattr(load_out, "missing_keys", []))
        unexpected_n = len(getattr(load_out, "unexpected_keys", []))

        resume_epoch = int(ckpt.get("epoch", -1))
        resume_global_step = int(ckpt.get("global_step", 0) or 0)
        remaining_steps = int(effective_max_steps) - resume_global_step
        if remaining_steps <= 0:
            skip_fit = True
            remaining_steps = 0
        else:
            model.max_steps = int(remaining_steps)
            model.trainer_kwargs["max_steps"] = int(remaining_steps)

        print(f"resume_epoch={resume_epoch}")
        print(f"resume_global_step={resume_global_step}")
        print(f"remaining_steps={remaining_steps}")
        print(f"state_dict_missing_keys={missing_n}")
        print(f"state_dict_unexpected_keys={unexpected_n}")

        resume_info = {
            "resume_ckpt_path": str(ckpt_path),
            "resume_epoch": resume_epoch,
            "resume_global_step": resume_global_step,
            "remaining_steps": int(remaining_steps),
            "state_dict_missing_keys": int(missing_n),
            "state_dict_unexpected_keys": int(unexpected_n),
            "skip_fit": bool(skip_fit),
        }

        del ckpt
        del state_dict
        gc.collect()

    freq = cfg["data"].get("freq", 1)
    nf = NeuralForecast(models=[model], freq=freq)

    t0 = time.time()
    if skip_fit:
        print("resume_global_step >= effective_max_steps, fit skipped")
    else:
        nf.fit(
            df=train_nf,
            static_df=static_df,
            val_size=int(cfg["train"].get("val_size", 0)),
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            verbose=True,
        )
    elapsed = time.time() - t0

    summary: Dict[str, Any] = {
        "train_rows": train_rows_loaded,
        "train_series": train_series,
        "dropped_short_series": int(dropped_short_series),
        "train_ds_min": int(train_nf["ds"].min()),
        "train_ds_max": int(train_nf["ds"].max()),
        "exog_total": int(len(exog_cols)),
        "stat_exog": int(len(stat_exog_list)),
        "hist_exog": int(len(hist_exog_list)),
        "futr_exog": int(len(futr_exog_list)),
        "estimated_total_windows": total_windows,
        "window_mode": window_mode,
        "approx_steps_per_epoch_by_series": steps_per_epoch_by_series,
        "approx_steps_per_epoch_by_windows": approx_steps_from_windows,
        "effective_max_steps": int(effective_max_steps),
        "step_budget_mode": step_budget_mode,
        "series_epochs_equivalent": float(epochs_equivalent),
        "windows_epochs_equivalent": float(epochs_equivalent_windows),
        "devices_used": int(devices),
        "hidden_size": int(cfg["model"]["hidden_size"]),
        "horizon": int(cfg["model"]["horizon"]),
        "input_size": int(cfg["model"]["input_size"]),
        "elapsed_seconds": float(elapsed),
    }
    if resume_info:
        summary["resume"] = resume_info

    if is_global_zero:
        model_dir = output_dir / "model_artifacts"
        model_dir.mkdir(parents=True, exist_ok=True)
        nf.save(path=str(model_dir), save_dataset=False, overwrite=True)

        if bool(cfg.get("eval", {}).get("compute_holdout_mae", True)) and val_parquet.exists():
            val_raw = read_parquet_frame(
                parquet_path=val_parquet,
                columns=read_cols,
                max_rows=max_val_rows,
                read_tail=read_tail,
            )
            val_df = cast_and_clean(
                val_raw,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                exog_cols=exog_cols,
                cat_cols=cat_cols,
            )
            del val_raw
            gc.collect()

            val_nf = to_nf_format_with_id_map(
                val_df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                id_map=id_map,
            )
            eval_info = holdout_eval_mae(
                nf=nf,
                val_nf=val_nf,
                horizon=int(cfg["model"]["horizon"]),
                futr_exog_list=futr_exog_list,
                static_df=static_df,
            )
            summary.update({"holdout_eval": eval_info})

        summary_path = output_dir / "run_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("==== training done ====")
        print(f"elapsed_seconds={elapsed:.2f}")
        print(f"saved_model_dir={model_dir}")
        print(f"summary_json={summary_path}")
    else:
        print(f"rank={global_rank} done (save/eval réservé au rank 0)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Nixtla TFT on parquet train/val data")
    parser.add_argument(
        "--config",
        type=str,
        default="/workspace/nixtla/config_tft_2x3090.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(Path(args.config))
