#!/usr/bin/env python3
"""
LightGBM baseline — comparaison avec Ridge et TFT.

Mêmes 75 features que le TFT. LightGBM est invariant à l'échelle,
pas besoin de normaliser les features.

Chargement : 10M dernières lignes de train.parquet (tail-read, comme le TFT).
Val       : 200K lignes de val.parquet (early stopping).
Test      : tout test.parquet (chunk par chunk pour la prédiction).
Loss      : MAE (regression_l1) — identique à la loss du TFT.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent.parent
# Ces chemins sont des defaults locaux ; utilisez --data-dir pour les overrider.
_DATA_DIR     = ROOT / "data"
TRAIN_PARQUET = _DATA_DIR / "train.parquet"
VAL_PARQUET   = _DATA_DIR / "val.parquet"
TEST_PARQUET  = _DATA_DIR / "test.parquet"
SCALER_PATH   = _DATA_DIR / "scaler_mappings.json"
TFT_RUN_DIR   = ROOT / "nixtla" / "runs" / "tft_v4_3day_36m"
RIDGE_REPORT  = ROOT / "nixtla" / "runs" / "linear_baseline_report.json"
OUT_DIR       = ROOT / "nixtla" / "runs"

# ---------------------------------------------------------------------------
STAT_EXOG = [
    "nb_arrets_total", "position_dans_trajet", "est_gare_terminus",
    "temps_theorique_depuis_depart", "temps_theorique_jusque_arrivee",
    "temps_theorique_trajet", "traverse_jonction_nord_midi",
    "traverse_hub_schaerbeek", "traverse_hub_bruxelles_midi",
    "traverse_hub_anvers", "traverse_hub_gent", "traverse_hub_liege",
    "traverse_hub_charleroi", "est_hub_majeur", "nb_connexions_gare",
]
HIST_EXOG = [
    "retard_j_1", "retard_j_2", "retard_j_3", "retard_j_4",
    "retard_j_5", "retard_j_6", "retard_j_7", "retard_j_14", "retard_j_21",
    "retard_moyen_meme_jour_semaine",
    "lag_missing_j1", "lag_missing_j2", "lag_missing_j3", "lag_missing_j4",
    "lag_missing_j5", "lag_missing_j6", "lag_missing_j7",
    "lag_missing_j14", "lag_missing_j21", "lag_missing_semaine",
    "retard_moyen_ligne_7j", "volatilite_retard_ligne_7j",
    "taux_ponctualite_ligne_7j", "retard_moyen_ligne_30j",
    "retard_moyen_gare_7j", "retard_moyen_slot_horaire_gare_7j",
    "meteo_observee_j_1", "fiabilite_train_historique",
    "risque_retard_score", "stress_reseau_score", "indice_complexite",
]
FUTR_EXOG = [
    "dt_hour", "heure_prevue_sin", "heure_prevue_cos",
    "jour_semaine_sin", "jour_semaine_cos", "mois_sin", "mois_cos",
    "fest_weekend", "is_jour_ferie", "is_vacances_scolaires", "is_heure_pointe",
    "nb_trains_prevus_ligne_heure",
    "densite_prevue_jonction_nord_midi", "densite_prevue_hub_schaerbeek",
    "densite_prevue_hub_bruxelles_midi", "densite_prevue_hub_anvers",
    "densite_prevue_hub_gent", "densite_prevue_hub_liege",
    "densite_prevue_hub_charleroi", "congestion_prevue_score",
    "travaux_nb_actifs_ligne", "travaux_niveau_impact",
    "travaux_distance_km", "travaux_actif_ligne",
    "meteo_temperature", "meteo_precipitations", "meteo_vent_vitesse",
    "meteo_nebulosite", "meteo_conditions_extremes",
]

TARGET_COL   = "target"
NAIVE_COL    = "retard_j_1"
ALL_FEATURES = STAT_EXOG + HIST_EXOG + FUTR_EXOG


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def available_cols(parquet_path: Path, wanted):
    schema  = pq.read_schema(parquet_path)
    present = set(schema.names)
    missing = [c for c in wanted if c not in present]
    if missing:
        print(f"  [WARN] absentes dans {parquet_path.name} : {missing}")
    return [c for c in wanted if c in present]


def load_tail(parquet_path: Path, cols, max_rows: int) -> pd.DataFrame:
    """Charge les max_rows dernières lignes via les derniers row groups."""
    pf   = pq.ParquetFile(parquet_path)
    n_rg = pf.num_row_groups

    selected = []
    collected = 0
    for rg in range(n_rg - 1, -1, -1):
        rg_rows = pf.metadata.row_group(rg).num_rows
        selected.append(rg)
        collected += rg_rows
        if collected >= max_rows:
            break
    selected = sorted(selected)

    print(f"  {parquet_path.name} : {len(selected)}/{n_rg} row groups "
          f"(tail ≤ {max_rows:,})")

    table = pf.read_row_groups(selected, columns=cols)
    for i, field in enumerate(table.schema):
        if field.type == pa.float64():
            table = table.set_column(i, field.name,
                                     table.column(i).cast(pa.float32()))
    df = table.to_pandas()
    del table
    gc.collect()

    df = df.dropna(subset=[TARGET_COL]).fillna(0.0)
    print(f"  → {len(df):,} lignes")
    return df


def iter_test_chunks(parquet_path: Path, cols, chunk_size=500_000):
    """Itère le test.parquet en chunks (pour prédiction sans tout charger)."""
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=chunk_size, columns=cols):
        table = pa.Table.from_batches([batch])
        for i, field in enumerate(table.schema):
            if field.type == pa.float64():
                table = table.set_column(i, field.name,
                                         table.column(i).cast(pa.float32()))
        chunk = table.to_pandas()
        chunk = chunk.dropna(subset=[TARGET_COL]).fillna(0.0)
        if len(chunk):
            yield chunk


def load_naive_scaler(target_mean: float = None):
    """Retourne (mean, std, scale_factor) pour dénormaliser retard_j_1.

    scale_factor = 60 si target est en secondes et retard_j_1 en minutes
    (détection automatique : target_mean >> naive_mean * 10).
    """
    if not SCALER_PATH.exists():
        return None, None, 1.0
    scalers = json.loads(SCALER_PATH.read_text())
    entry   = scalers.get(NAIVE_COL, {})
    sm, ss  = entry.get("mean"), entry.get("std")
    # Détection unité : si target_mean ≈ retard_j_1_mean * 60 → target en secondes
    scale = 1.0
    if sm is not None and target_mean is not None:
        naive_mean_raw = sm  # en minutes (d'après scaler)
        if target_mean > naive_mean_raw * 20:  # ratio > 20× → probablement secondes
            scale = 60.0
            print(f"  [auto-détect] target en SECONDES (mean={target_mean:.1f}), "
                  f"retard_j_1 en MINUTES (mean={naive_mean_raw:.3f}) "
                  f"→ naïf * 60")
        else:
            print(f"  [auto-détect] target et retard_j_1 dans la même unité "
                  f"(target mean={target_mean:.3f}, naive mean={naive_mean_raw:.3f})")
    return sm, ss, scale


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_naive):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {"mae": None, "rmse": None, "naive_mae": None,
                "improvement_vs_naive_pct": None, "n_rows": 0}
    err  = y_pred[mask] - y_true[mask]
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    nm   = np.isfinite(y_true) & np.isfinite(y_naive)
    naiv = float(np.mean(np.abs(y_naive[nm] - y_true[nm]))) if nm.any() else None
    impr = float(100.0 * (naiv - mae) / naiv) if (naiv and naiv != 0) else None
    return {"mae": mae, "rmse": rmse, "naive_mae": naiv,
            "improvement_vs_naive_pct": impr, "n_rows": int(mask.sum())}


# ---------------------------------------------------------------------------
# Chargement résultats existants
# ---------------------------------------------------------------------------

def eval_on_tft_subset(model, best_iter, feat_cols, h: int = 2):
    """Évalue LightGBM sur les mêmes lignes que le TFT :
      - premiers h timesteps (time_idx le plus bas) par group_id dans test.parquet
      - naïf = dernière valeur target par série dans train.parquet (même que TFT)
    """
    print(f"\n=== Éval TFT-subset (premiers h={h} timesteps/série) ===")

    # 1. Charger test complet avec group_id + time_idx pour sélectionner les bonnes lignes
    id_col   = "group_id"
    time_col = "time_idx"
    test_cols = [id_col, time_col, TARGET_COL] + feat_cols
    test_cols = [c for c in test_cols
                 if c in pq.read_schema(TEST_PARQUET).names]
    if id_col not in test_cols or time_col not in test_cols:
        print("  [SKIP] group_id ou time_idx absent de test.parquet")
        return None

    print(f"  Lecture test.parquet …")
    chunks = []
    for batch in pq.ParquetFile(TEST_PARQUET).iter_batches(
            batch_size=500_000, columns=test_cols):
        chunk = batch.to_pandas().dropna(subset=[TARGET_COL]).fillna(0.0)
        if len(chunk):
            chunks.append(chunk[[id_col, time_col, TARGET_COL] + feat_cols])
    test_df = pd.concat(chunks, ignore_index=True)
    del chunks; gc.collect()
    print(f"  {len(test_df):,} lignes test chargées")

    # 2. Premiers h timesteps par groupe (même logique que TFT : head(h) après sort)
    test_df = test_df.sort_values([id_col, time_col])
    subset  = test_df.groupby(id_col, sort=False).head(h).reset_index(drop=True)
    del test_df; gc.collect()
    print(f"  Subset : {len(subset):,} lignes ({subset[id_col].nunique():,} séries × h={h})")

    # 3. Naïf TFT = dernière target de la série dans train.parquet
    print(f"  Calcul naïf (dernière target par série depuis train.parquet) …")
    train_cols = [id_col, time_col, TARGET_COL]
    train_cols = [c for c in train_cols
                  if c in pq.read_schema(TRAIN_PARQUET).names]
    last_train = {}
    for batch in pq.ParquetFile(TRAIN_PARQUET).iter_batches(
            batch_size=500_000, columns=train_cols):
        chunk = batch.to_pandas().dropna(subset=[TARGET_COL])
        for gid, grp in chunk.groupby(id_col, sort=False):
            best_idx = grp[time_col].idxmax()
            last_val = grp.loc[best_idx, TARGET_COL]
            if gid not in last_train or grp.loc[best_idx, time_col] > last_train[gid][0]:
                last_train[gid] = (grp.loc[best_idx, time_col], float(last_val))
    naive_map = {gid: v for gid, (_, v) in last_train.items()}
    del last_train; gc.collect()
    print(f"  Naïf disponible pour {len(naive_map):,} séries")

    # 4. Prédiction LightGBM sur le subset
    X_sub  = subset[feat_cols].to_numpy(np.float32)
    y_sub  = subset[TARGET_COL].to_numpy(np.float64)
    yp_sub = model.predict(X_sub, num_iteration=best_iter)

    # 5. Naïf par ligne
    y_naive_sub = np.array([naive_map.get(gid, np.nan)
                             for gid in subset[id_col]])

    return compute_metrics(y_sub, yp_sub, y_naive_sub)


def load_ridge_results():
    if not RIDGE_REPORT.exists():
        return None
    data = json.loads(RIDGE_REPORT.read_text())
    r = data.get("ridge", {})
    n = data.get("naive", {})
    return {
        "mae": r.get("mae"), "rmse": r.get("rmse"),
        "naive_mae": r.get("naive_mae"),
        "improvement_vs_naive_pct": r.get("improvement_vs_naive_pct"),
        "n_rows": r.get("n_rows"),
        "alpha": data.get("config", {}).get("best_alpha"),
        "n_train": data.get("config", {}).get("max_train_rows"),
        "naive_mae_standalone": n.get("mae"),
    }


def load_tft_results():
    if not TFT_RUN_DIR.exists():
        return None
    reports = sorted(TFT_RUN_DIR.glob("run_*/quality_report.json"))
    if not reports:
        return None
    latest = reports[-1]
    data    = json.loads(latest.read_text())
    metrics = data.get("metrics", {})
    for key in ("test_from_train_plus_val_context",
                "test_from_train_context", "val_from_train_context"):
        m = metrics.get(key)
        if m and m.get("mae") is not None:
            return {"source": key, "run": str(latest.parent.name), **m}
    return None


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def print_table(rows):
    hdr = (f"{'Modèle':<44} {'MAE':>8} {'RMSE':>8} "
           f"{'MAE Naïf':>9} {'Δ vs Naïf':>11} {'N lignes':>11}")
    sep = "─" * len(hdr)
    print(); print(sep); print(hdr); print(sep)
    for label, mae, rmse, nm, imp, n in rows:
        ms  = f"{mae:.4f}"  if mae  is not None else "N/A"
        rs  = f"{rmse:.4f}" if rmse is not None else "N/A"
        ns  = f"{nm:.4f}"   if nm   is not None else "N/A"
        is_ = f"{imp:+.2f}%" if imp is not None else "N/A"
        nn  = f"{n:,}"       if n   is not None else "N/A"
        print(f"{label:<44} {ms:>8} {rs:>8} {ns:>9} {is_:>11} {nn:>11}")
    print(sep); print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-rows", type=int, default=10_000_000,
                        help="Lignes train (défaut 10M, tail-read)")
    parser.add_argument("--max-val-rows",   type=int, default=200_000,
                        help="Lignes val pour early stopping (défaut 200K)")
    parser.add_argument("--n-estimators",   type=int, default=1000)
    parser.add_argument("--num-leaves",     type=int, default=255)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--data-dir",       type=str, default=None,
                        help="Répertoire contenant train/val/test.parquet et scaler_mappings.json")
    parser.add_argument("--out-dir",        type=str, default=None,
                        help="Répertoire de sortie pour le rapport JSON")
    args = parser.parse_args()

    # Override des chemins si --data-dir fourni
    global TRAIN_PARQUET, VAL_PARQUET, TEST_PARQUET, SCALER_PATH, OUT_DIR
    if args.data_dir:
        d = Path(args.data_dir)
        TRAIN_PARQUET = d / "train.parquet"
        VAL_PARQUET   = d / "val.parquet"
        TEST_PARQUET  = d / "test.parquet"
        SCALER_PATH   = d / "scaler_mappings.json"
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    t0 = time.time()

    feat_cols = available_cols(TRAIN_PARQUET, ALL_FEATURES)
    print(f"\nFeatures : {len(feat_cols)}/{len(ALL_FEATURES)}")

    # Calcul target_mean pour auto-détection d'unité (petit échantillon)
    _pf = pq.ParquetFile(TRAIN_PARQUET)
    _b  = next(_pf.iter_batches(batch_size=200_000, columns=[TARGET_COL]))
    _target_mean = float(np.nanmean(_b.column(TARGET_COL).to_pylist()))
    del _pf, _b

    naive_sm, naive_ss, naive_scale = load_naive_scaler(target_mean=_target_mean)
    if naive_sm:
        print(f"  retard_j_1 scaler : mean={naive_sm:.4f} min, std={naive_ss:.4f} min, "
              f"scale×{naive_scale:.0f}")

    read_cols = feat_cols + [TARGET_COL]

    # -------------------------------------------------------------------
    # Chargement train
    # -------------------------------------------------------------------
    print(f"\n=== Chargement train ({args.max_train_rows//1_000_000}M lignes, tail) ===")
    train_df = load_tail(TRAIN_PARQUET, read_cols, args.max_train_rows)
    X_train  = train_df[feat_cols].to_numpy(np.float32)
    y_train  = train_df[TARGET_COL].to_numpy(np.float32)
    del train_df; gc.collect()

    # -------------------------------------------------------------------
    # Chargement val (early stopping)
    # -------------------------------------------------------------------
    print(f"\n=== Chargement val ({args.max_val_rows//1_000}K lignes, tail) ===")
    val_df  = load_tail(VAL_PARQUET, read_cols, args.max_val_rows)
    X_val   = val_df[feat_cols].to_numpy(np.float32)
    y_val   = val_df[TARGET_COL].to_numpy(np.float32)
    del val_df; gc.collect()

    # -------------------------------------------------------------------
    # Dataset LightGBM
    # -------------------------------------------------------------------
    print(f"\n=== Construction datasets LightGBM ===")
    dtrain = lgb.Dataset(X_train, label=y_train,
                         feature_name=feat_cols, free_raw_data=True)
    dval   = lgb.Dataset(X_val, label=y_val,
                         reference=dtrain, free_raw_data=True)
    del X_train, y_train, X_val, y_val; gc.collect()

    # -------------------------------------------------------------------
    # Entraînement
    # -------------------------------------------------------------------
    params = {
        "objective":        "regression_l1",   # MAE = même loss que TFT
        "num_leaves":       args.num_leaves,
        "learning_rate":    0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples": 50,
        "n_jobs":           -1,
        "seed":             args.seed,
        "verbose":          -1,
    }
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    print(f"\n=== Entraînement LightGBM (≤{args.n_estimators} rounds) ===")
    t_train = time.time()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    best_iter = model.best_iteration
    print(f"\n  Meilleure itération : {best_iter}  ({time.time()-t_train:.1f}s)")

    # -------------------------------------------------------------------
    # Sauvegarde du modèle
    # -------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "lgbm_model.txt"
    model.save_model(str(model_path))
    print(f"\n  Modèle sauvegardé : {model_path}")

    # -------------------------------------------------------------------
    # Importance des features (top 15)
    # -------------------------------------------------------------------
    imp = sorted(zip(feat_cols, model.feature_importance("gain")),
                 key=lambda x: -x[1])
    print("\n  Top 15 features (gain) :")
    for name, score in imp[:15]:
        bar = "█" * int(score / imp[0][1] * 30)
        print(f"    {name:<40s} {bar}")

    # -------------------------------------------------------------------
    # Évaluation sur test.parquet (chunk par chunk)
    # -------------------------------------------------------------------
    print(f"\n=== Évaluation sur test.parquet ===")
    t_eval = time.time()

    naive_idx = feat_cols.index(NAIVE_COL) if NAIVE_COL in feat_cols else None

    y_true_l, y_pred_l, y_naive_l = [], [], []
    for chunk in iter_test_chunks(TEST_PARQUET, read_cols):
        X  = chunk[feat_cols].to_numpy(np.float32)
        y  = chunk[TARGET_COL].to_numpy(np.float64)
        yp = model.predict(X, num_iteration=best_iter)

        y_true_l.append(y)
        y_pred_l.append(yp)

        if naive_idx is not None and naive_sm is not None:  # noqa (naive_scale hoisted)
            naive_raw = (chunk[NAIVE_COL].to_numpy(np.float64)
                         * naive_ss + naive_sm) * naive_scale
        elif naive_idx is not None:
            naive_raw = chunk[NAIVE_COL].to_numpy(np.float64)
        else:
            naive_raw = np.zeros(len(y))
        y_naive_l.append(naive_raw)

    y_true  = np.concatenate(y_true_l)
    y_pred  = np.concatenate(y_pred_l)
    y_naive = np.concatenate(y_naive_l)
    print(f"  Prédictions en {time.time()-t_eval:.1f}s")

    metrics_lgbm = compute_metrics(y_true, y_pred, y_naive)
    print(f"  LightGBM → MAE={metrics_lgbm['mae']:.4f}  "
          f"RMSE={metrics_lgbm['rmse']:.4f}  "
          f"Δnaïf={metrics_lgbm['improvement_vs_naive_pct']:+.2f}%  "
          f"({metrics_lgbm['n_rows']:,} lignes)")

    # Baseline naïve
    nm        = np.isfinite(y_true) & np.isfinite(y_naive)
    naive_mae = float(np.mean(np.abs(y_naive[nm] - y_true[nm]))) if nm.any() else None
    naive_rms = float(np.sqrt(np.mean((y_naive[nm]-y_true[nm])**2))) if nm.any() else None
    n_test    = int(nm.sum())

    # -------------------------------------------------------------------
    # Résultats Ridge et TFT
    # -------------------------------------------------------------------
    print("\n=== Chargement résultats comparaison ===")
    ridge = load_ridge_results()
    tft   = load_tft_results()
    if ridge: print(f"  Ridge chargé  (α={ridge.get('alpha')}, {(ridge.get('n_train') or 0)//1_000_000}M lignes)")
    if tft:   print(f"  TFT chargé    ({tft['run']})")

    # -------------------------------------------------------------------
    # Table finale
    # -------------------------------------------------------------------
    rows = [
        ("Naïf (retard_j_1 dénormalisé)",
         naive_mae, naive_rms, naive_mae, 0.0, n_test),
    ]
    if ridge:
        rows.append((
            f"Ridge (α={ridge.get('alpha'):.4g}, {(ridge.get('n_train') or 0)//1_000_000}M lignes)",
            ridge["mae"], ridge["rmse"], ridge["naive_mae"],
            ridge["improvement_vs_naive_pct"], ridge["n_rows"],
        ))
    rows.append((
        f"LightGBM (iter={best_iter}, {args.max_train_rows//1_000_000}M lignes)",
        metrics_lgbm["mae"], metrics_lgbm["rmse"],
        metrics_lgbm["naive_mae"], metrics_lgbm["improvement_vs_naive_pct"],
        metrics_lgbm["n_rows"],
    ))
    if tft:
        rows.append((
            f"TFT NeuralForecast ({tft['run']})",
            tft.get("mae"), tft.get("rmse"),
            tft.get("naive_mae"), tft.get("improvement_vs_naive_pct"),
            tft.get("finite_rows"),
        ))

    print("\n--- Éval sur tout le test ---")
    print_table(rows)

    # -------------------------------------------------------------------
    # Éval TFT-subset : premiers h=2 timesteps/série (même pop que TFT)
    # -------------------------------------------------------------------
    h = tft.get("horizon", 2) if tft else 2
    metrics_subset = eval_on_tft_subset(model, best_iter, feat_cols, h=h)

    if metrics_subset:
        print(f"\n--- Éval TFT-subset (h={h} premiers timesteps/série) ---")
        tft_naive = tft.get("naive_mae") if tft else None
        rows_subset = []
        if tft_naive:
            rows_subset.append(("Naïf TFT (last target/série)", tft_naive, None,
                                 tft_naive, 0.0, tft.get("finite_rows")))
        rows_subset.append((
            f"LightGBM subset (iter={best_iter})",
            metrics_subset["mae"], metrics_subset["rmse"],
            metrics_subset["naive_mae"], metrics_subset["improvement_vs_naive_pct"],
            metrics_subset["n_rows"],
        ))
        if tft:
            rows_subset.append((
                f"TFT NeuralForecast ({tft['run']})",
                tft.get("mae"), tft.get("rmse"),
                tft.get("naive_mae"), tft.get("improvement_vs_naive_pct"),
                tft.get("finite_rows"),
            ))
        print_table(rows_subset)
        print("  ✓ Même population de test — comparaison directe valide")
        print()

    # -------------------------------------------------------------------
    # Sauvegarde
    # -------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "max_train_rows": args.max_train_rows,
            "n_features": len(feat_cols),
            "feature_cols": feat_cols,
            "best_iteration": best_iter,
            "params": params,
        },
        "lgbm":         {"best_iteration": best_iter, **metrics_lgbm},
        "lgbm_subset":  metrics_subset,
        "ridge":        ridge,
        "naive":        {"mae": naive_mae, "rmse": naive_rms, "n_rows": n_test},
        "tft":          tft,
        "feature_importance_gain": {k: int(v) for k, v in imp},
        "elapsed_seconds": float(time.time() - t0),
    }
    out = OUT_DIR / "lgbm_baseline_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Rapport : {out}")
    print(f"Temps total : {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
