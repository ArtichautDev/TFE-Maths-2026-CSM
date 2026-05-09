#!/usr/bin/env python3

import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast

import train_nixtla_tft as t


def load_clean(path: Path, read_cols, id_col, time_col, target_col, exog_cols, cat_cols, max_rows, read_tail):
    raw = t.read_parquet_frame(path, columns=read_cols, max_rows=max_rows, read_tail=read_tail)
    cleaned = t.cast_and_clean(
        raw,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        exog_cols=exog_cols,
        cat_cols=cat_cols,
    )
    del raw
    gc.collect()
    return cleaned


def split_gap_stats(train_df, val_df, test_df, id_col, time_col):
    train_last = train_df.groupby(id_col)[time_col].max()
    val_first = val_df.groupby(id_col)[time_col].min()
    val_last = val_df.groupby(id_col)[time_col].max()
    test_first = test_df.groupby(id_col)[time_col].min()

    common_tv = train_last.index.intersection(val_first.index)
    common_vt = val_last.index.intersection(test_first.index)
    common_tt = train_last.index.intersection(test_first.index)

    def safe_mean(x: pd.Series):
        return float(x.mean()) if len(x) else None

    gap_tv = (val_first.loc[common_tv] - train_last.loc[common_tv]) if len(common_tv) else pd.Series(dtype=np.float64)
    gap_vt = (test_first.loc[common_vt] - val_last.loc[common_vt]) if len(common_vt) else pd.Series(dtype=np.float64)
    gap_tt = (test_first.loc[common_tt] - train_last.loc[common_tt]) if len(common_tt) else pd.Series(dtype=np.float64)

    return {
        "common_train_val_series": int(len(common_tv)),
        "val_is_next_step_count": int((gap_tv == 1).sum()) if len(gap_tv) else 0,
        "val_gap_mean": safe_mean(gap_tv),
        "common_val_test_series": int(len(common_vt)),
        "test_after_val_next_step_count": int((gap_vt == 1).sum()) if len(gap_vt) else 0,
        "test_after_val_gap_mean": safe_mean(gap_vt),
        "common_train_test_series": int(len(common_tt)),
        "test_after_train_next_step_count": int((gap_tt == 1).sum()) if len(gap_tt) else 0,
        "test_after_train_gap_mean": safe_mean(gap_tt),
    }


def evaluate_split(target_nf, preds, context_nf, model_col, h):
    target_head = (
        target_nf[["unique_id", "ds", "y"]]
        .sort_values(["unique_id", "ds"], kind="mergesort")
        .groupby("unique_id", as_index=False)
        .head(h)
    )
    merged = target_head.merge(preds[["unique_id", "ds", model_col]], on=["unique_id", "ds"], how="inner")

    if merged.empty:
        return {
            "matched_rows": 0,
            "finite_rows": 0,
            "nan_pred_rows": 0,
            "mae": None,
            "rmse": None,
            "naive_mae": None,
            "improvement_vs_naive_pct": None,
        }

    baseline = (
        context_nf[["unique_id", "ds", "y"]]
        .sort_values(["unique_id", "ds"], kind="mergesort")
        .groupby("unique_id", as_index=False)
        .tail(1)[["unique_id", "y"]]
        .rename(columns={"y": "naive_last"})
    )
    merged = merged.merge(baseline, on="unique_id", how="left")

    y_true = merged["y"].to_numpy(np.float64)
    y_pred = merged[model_col].to_numpy(np.float64)
    y_naive = merged["naive_last"].to_numpy(np.float64)

    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    finite_rows = int(finite_mask.sum())
    nan_pred_rows = int((~np.isfinite(y_pred)).sum())

    mae = None
    rmse = None
    if finite_rows:
        err = y_pred[finite_mask] - y_true[finite_mask]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))

    naive_mask = np.isfinite(y_true) & np.isfinite(y_naive)
    naive_mae = float(np.mean(np.abs(y_naive[naive_mask] - y_true[naive_mask]))) if naive_mask.any() else None

    improvement = None
    if mae is not None and naive_mae is not None and naive_mae != 0:
        improvement = float(100.0 * (naive_mae - mae) / naive_mae)

    return {
        "matched_rows": int(len(merged)),
        "finite_rows": finite_rows,
        "nan_pred_rows": nan_pred_rows,
        "mae": mae,
        "rmse": rmse,
        "naive_mae": naive_mae,
        "improvement_vs_naive_pct": improvement,
    }


def parse_efficiency(log_text: str, elapsed_seconds):
    text = log_text.replace("\r", "\n")
    progress_pattern = re.compile(r"Epoch\s+(\d+):\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[", re.S)
    progress = list(progress_pattern.finditer(text))

    epoch_last = None
    pct_last = None
    num_last = None
    den_last = None
    est_epochs_done = None
    est_steps_done = None
    est_batches_per_sec = None

    if progress:
        last = progress[-1]
        epoch_last = int(last.group(1))
        pct_last = int(last.group(2))
        num_last = int(last.group(3))
        den_last = int(last.group(4))
        est_epochs_done = epoch_last + (num_last / den_last if den_last else 0.0)
        est_steps_done = epoch_last * den_last + num_last
        if elapsed_seconds and elapsed_seconds > 0:
            est_batches_per_sec = est_steps_done / elapsed_seconds
    else:
        epochs = [int(x) for x in re.findall(r"Epoch\s+(\d+):", text)]
        if epochs:
            epoch_last = max(epochs)
            est_epochs_done = float(epoch_last + 1)

    early_stop = bool(re.search(r"early stopping|did not improve|signaling.*stop", text, flags=re.I))

    return {
        "epoch_last_seen": epoch_last,
        "epoch_percent_last_seen": pct_last,
        "batch_progress_num_last_seen": num_last,
        "batch_progress_den_last_seen": den_last,
        "estimated_epochs_done_from_log": est_epochs_done,
        "estimated_steps_done_from_log": est_steps_done,
        "estimated_batches_per_sec": est_batches_per_sec,
        "early_stopping_message_found": early_stop,
        "training_done_marker_found": "==== training done ====" in text,
    }


def main():
    cfg_path = Path("/workspace/nixtla/config_tft_1x4080.yaml")
    run_dir = Path("/workspace/nixtla/runs/tft_v4_3day_36m/run_20260313_122118")
    model_dir = run_dir / "model_artifacts"
    summary_path = run_dir / "run_summary.json"
    log_path = Path("/workspace/nixtla/train_4080_first.log")
    out_path = run_dir / "quality_report.json"

    cfg = t.load_config(cfg_path)

    id_col = str(cfg["features"]["id_col"])
    time_col = str(cfg["features"]["time_col"])
    target_col = str(cfg["features"]["target_col"])
    # Do not drop categorical variables that the model actually expects!
    drop_columns = ["datetime", "DELAY_ARR"]

    train_parquet = Path(cfg["paths"]["train_parquet"])
    val_parquet = Path(cfg["paths"]["val_parquet"])
    test_parquet = Path("/workspace/data/test.parquet")

    read_cols_all, exog_cols_all, cat_cols_all, _ = t.infer_feature_lists(
        train_parquet=train_parquet,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        drop_columns=drop_columns,
    )

    use_exog = bool(cfg["model"].get("use_exog", True))
    if use_exog:
        read_cols = read_cols_all
        exog_cols = exog_cols_all
        cat_cols = cat_cols_all
    else:
        read_cols = [id_col, time_col, target_col]
        exog_cols = []
        cat_cols = []

    read_tail = bool(cfg.get("data", {}).get("read_tail", True))
    max_train_rows = cfg.get("data", {}).get("max_train_rows")
    max_val_rows = cfg.get("data", {}).get("max_val_rows")

    train_df = load_clean(train_parquet, read_cols, id_col, time_col, target_col, exog_cols, cat_cols, max_train_rows, read_tail)
    val_df = load_clean(val_parquet, read_cols, id_col, time_col, target_col, exog_cols, cat_cols, max_val_rows, read_tail)
    test_df = load_clean(test_parquet, read_cols, id_col, time_col, target_col, exog_cols, cat_cols, max_val_rows, read_tail)

    max_unique_ids = cfg.get("data", {}).get("max_unique_ids")
    if max_unique_ids is not None and int(max_unique_ids) > 0:
        top_ids = train_df[id_col].value_counts().head(int(max_unique_ids)).index
        train_df = train_df[train_df[id_col].isin(top_ids)].copy()

    split_stats = split_gap_stats(train_df, val_df, test_df, id_col, time_col)

    stat_exog_ls, hist_exog_ls, futr_exog_ls = t.get_exog_splits(cfg, exog_cols, cat_cols)
    
    static_df = None
    if stat_exog_ls:
        static_df = (
            train_df[[id_col] + stat_exog_ls]
            .groupby(id_col, as_index=False, sort=False, observed=True)
            .last()
            .copy()
        )

    train_nf, id_map = t.to_nf_format_train(train_df, id_col=id_col, time_col=time_col, target_col=target_col)
    val_nf = t.to_nf_format_with_id_map(val_df, id_col=id_col, time_col=time_col, target_col=target_col, id_map=id_map)
    test_nf = t.to_nf_format_with_id_map(test_df, id_col=id_col, time_col=time_col, target_col=target_col, id_map=id_map)

    if static_df is not None:
        static_df = static_df.rename(columns={id_col: "unique_id"})
        static_df["unique_id"] = static_df["unique_id"].map(id_map)
        static_df = static_df.dropna(subset=["unique_id"]).reset_index(drop=True)
        static_df["unique_id"] = static_df["unique_id"].astype(np.int32)

    val_size_for_fit = int(cfg["train"].get("val_size", 0))
    if val_size_for_fit > 0:
        sizes = train_nf.groupby("unique_id", sort=False).size()
        keep_ids = set(sizes[sizes > val_size_for_fit].index.astype(np.int32).tolist())
    else:
        keep_ids = set(train_nf["unique_id"].unique().tolist())

    train_nf = train_nf[train_nf["unique_id"].isin(keep_ids)].reset_index(drop=True)
    val_nf = val_nf[val_nf["unique_id"].isin(keep_ids)].reset_index(drop=True)
    test_nf = test_nf[test_nf["unique_id"].isin(keep_ids)].reset_index(drop=True)

    nf = NeuralForecast.load(path=str(model_dir), verbose=False)
    for m in nf.models:
        m.devices = 1
        m.strategy = "auto"
        if hasattr(m, "trainer_kwargs"):
            m.trainer_kwargs["devices"] = 1
            if "strategy" in m.trainer_kwargs:
                m.trainer_kwargs["strategy"] = "auto"

    available_futr = [c for c in futr_exog_ls if c in train_nf.columns]
    
    # NeuralForecast predict() expects futr_df to have rows exactly matching the combinations 
    # of unique_id and (last_ds_in_train + 1 ... last_ds_in_train + horizon).
    if available_futr:
        # Create a reliable futr_df from the concatenated test/val splits to guarantee we have the horizons
        global_futr = pd.concat([val_nf, test_nf], ignore_index=True)[['unique_id', 'ds'] + available_futr]
        global_futr = global_futr.drop_duplicates(subset=['unique_id', 'ds'])
        
        # Generates exactly what NeuralForecast expects for missing periods
        expected_futr = nf.make_future_dataframe(df=train_nf)
        
        # Merge our true exogenous values onto the expected timeline
        futr_df_train_context = expected_futr[['unique_id', 'ds']].merge(
            global_futr, on=['unique_id', 'ds'], how='left'
        )
        
        # Any still-missing values can be forward-filled or zeroed, we backfill and fillna 0
        for col in available_futr:
            futr_df_train_context[col] = futr_df_train_context.groupby('unique_id')[col].bfill().fillna(0.0)
    else:
        futr_df_train_context = None

    if futr_df_train_context is not None:
        pred_train_context = nf.predict(df=train_nf, static_df=static_df, futr_df=futr_df_train_context, verbose=False)
    else:
        pred_train_context = nf.predict(df=train_nf, static_df=static_df, verbose=False)

    pred_cols = [c for c in pred_train_context.columns if c not in {"unique_id", "ds"}]
    if not pred_cols:
        raise RuntimeError("Aucune colonne de prédiction trouvée")
    model_col = pred_cols[0]

    h = int(cfg["model"]["horizon"])
    val_eval = evaluate_split(val_nf, pred_train_context, train_nf, model_col, h)
    test_eval_after_train = evaluate_split(test_nf, pred_train_context, train_nf, model_col, h)

    train_val_context = (
        pd.concat([train_nf, val_nf], ignore_index=True)
        .sort_values(["unique_id", "ds"], kind="mergesort")
        .drop_duplicates(subset=["unique_id", "ds"], keep="last")
        .reset_index(drop=True)
    )
    
    if available_futr:
        expected_futr_val = nf.make_future_dataframe(df=train_val_context)
        futr_df_val_context = expected_futr_val[['unique_id', 'ds']].merge(
            global_futr, on=['unique_id', 'ds'], how='left'
        )
        for col in available_futr:
            futr_df_val_context[col] = futr_df_val_context.groupby('unique_id')[col].bfill().fillna(0.0)
    else:
        futr_df_val_context = None

    if futr_df_val_context is not None:
        pred_train_val_context = nf.predict(df=train_val_context, static_df=static_df, futr_df=futr_df_val_context, verbose=False)
    else:
        pred_train_val_context = nf.predict(df=train_val_context, static_df=static_df, verbose=False)
        
    test_eval_after_val = evaluate_split(test_nf, pred_train_val_context, train_val_context, model_col, h)

    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    elapsed = summary.get("elapsed_seconds")
    log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    eff = parse_efficiency(log_text, elapsed)

    bad_quality = False
    reasons = []

    if val_eval["mae"] is None or val_eval["finite_rows"] == 0:
        bad_quality = True
        reasons.append("val_predictions_non_finite_or_missing")
    elif val_eval["naive_mae"] is not None and val_eval["mae"] >= 0.99 * val_eval["naive_mae"]:
        bad_quality = True
        reasons.append("val_not_better_than_naive")

    if (
        test_eval_after_val["matched_rows"] > 0
        and test_eval_after_val["mae"] is not None
        and test_eval_after_val["naive_mae"] is not None
        and test_eval_after_val["mae"] >= 0.99 * test_eval_after_val["naive_mae"]
    ):
        bad_quality = True
        reasons.append("test_after_val_not_better_than_naive")

    report = {
        "model_dir": str(model_dir),
        "model_col": model_col,
        "horizon": h,
        "split_alignment": split_stats,
        "rows": {
            "train_nf_rows": int(len(train_nf)),
            "val_nf_rows": int(len(val_nf)),
            "test_nf_rows": int(len(test_nf)),
            "train_nf_series": int(train_nf["unique_id"].nunique()),
            "val_nf_series": int(val_nf["unique_id"].nunique()),
            "test_nf_series": int(test_nf["unique_id"].nunique()),
        },
        "metrics": {
            "val_from_train_context": val_eval,
            "test_from_train_context": test_eval_after_train,
            "test_from_train_plus_val_context": test_eval_after_val,
        },
        "efficiency": {
            "elapsed_seconds": elapsed,
            "effective_max_steps_config": summary.get("effective_max_steps"),
            "series_epochs_equivalent_config": summary.get("series_epochs_equivalent"),
            **eff,
        },
        "decision": {
            "bad_quality": bad_quality,
            "reasons": reasons,
            "relaunch_recommended": bad_quality,
        },
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
