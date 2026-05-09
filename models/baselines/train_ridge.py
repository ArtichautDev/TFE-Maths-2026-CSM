#!/usr/bin/env python3
"""
Régression Ridge "best possible" — comparaison avec TFT et LightGBM.

v3 — améliorations :
  1. Features polynomiales deg-2 sur les 15 features les plus importantes
     (selon LightGBM gain) : 75 orig + 120 interactions = 195 features.
  2. Baseline naïve corrigée : auto-détecte si target en secondes
     (target_mean > retard_j_1_raw_mean × 20) → × 60.
  3. eval_on_tft_subset() : éval sur premiers h=2 timesteps/série
     (même population que TFT) pour comparaison directe.

Algorithme : équations normales incrémentales (chunks de 500K lignes)
  → Ridge exact sur toutes les lignes, mémoire O(p²) ≈ quelques MB.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parent.parent
TRAIN_PARQUET = ROOT / "data" / "train.parquet"
VAL_PARQUET   = ROOT / "data" / "val.parquet"
TEST_PARQUET  = ROOT / "data" / "test.parquet"
SCALER_PATH   = ROOT / "data" / "scaler_mappings.json"
TFT_RUN_DIR   = ROOT / "nixtla" / "runs" / "tft_v4_3day_36m"
OUT_DIR       = ROOT / "nixtla" / "runs"

# ---------------------------------------------------------------------------
# Features identiques au TFT (config_tft_1x4080.yaml)
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

# Top-15 features par gain LightGBM → expansion polynomiale deg-2
POLY_BASE = [
    "fiabilite_train_historique", "position_dans_trajet",
    "retard_moyen_meme_jour_semaine", "temps_theorique_depuis_depart",
    "temps_theorique_jusque_arrivee", "retard_j_1", "retard_j_7",
    "temps_theorique_trajet", "retard_j_14", "retard_j_6",
    "retard_j_2", "retard_j_21", "retard_j_4", "retard_j_3",
    "risque_retard_score",
]

TARGET_COL   = "target"
NAIVE_COL    = "retard_j_1"
ALL_FEATURES = STAT_EXOG + HIST_EXOG + FUTR_EXOG
CHUNK_SIZE   = 500_000
ALPHAS       = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# ---------------------------------------------------------------------------
# Helpers polynomiaux
# ---------------------------------------------------------------------------

def get_poly_idx(feat_cols):
    """Indices des POLY_BASE disponibles dans feat_cols."""
    return [feat_cols.index(f) for f in POLY_BASE if f in feat_cols]


def expand_poly(X_std, poly_idx):
    """
    X_std : (n, p) features standardisées
    Retourne (n, k*(k+1)/2) avec k = len(poly_idx) — produits croisés + carrés.
    """
    base = X_std[:, poly_idx]
    k    = base.shape[1]
    terms = []
    for i in range(k):
        for j in range(i, k):
            terms.append(base[:, i] * base[:, j])
    return np.column_stack(terms) if terms else np.empty((X_std.shape[0], 0))


# ---------------------------------------------------------------------------
# Dénormaliseur baseline naïve
# ---------------------------------------------------------------------------

def load_naive_scaler():
    if not SCALER_PATH.exists():
        return None, None
    scalers = json.loads(SCALER_PATH.read_text())
    entry = scalers.get(NAIVE_COL, {})
    return entry.get("mean"), entry.get("std")


# ---------------------------------------------------------------------------
# Utilitaires parquet
# ---------------------------------------------------------------------------

def available_cols(parquet_path, wanted):
    schema  = pq.read_schema(parquet_path)
    present = set(schema.names)
    missing = [c for c in wanted if c not in present]
    if missing:
        print(f"  [WARN] colonnes absentes : {missing}")
    return [c for c in wanted if c in present]


def iter_parquet_chunks(parquet_path, cols, max_rows=None, read_tail=True):
    pf   = pq.ParquetFile(parquet_path)
    n_rg = pf.num_row_groups

    if max_rows is not None and read_tail:
        selected, collected = [], 0
        for rg in range(n_rg - 1, -1, -1):
            selected.append(rg)
            collected += pf.metadata.row_group(rg).num_rows
            if collected >= max_rows:
                break
        rg_list = sorted(selected)
        print(f"  {parquet_path.name} : {len(rg_list)}/{n_rg} row groups (tail ≤ {max_rows:,})")
    else:
        rg_list = list(range(n_rg))

    for rg in rg_list:
        table = pf.read_row_group(rg, columns=cols)
        for i, field in enumerate(table.schema):
            if field.type == pa.float64():
                table = table.set_column(i, field.name,
                                         table.column(i).cast(pa.float32()))
        chunk = table.to_pandas()
        del table
        chunk = chunk.dropna(subset=[TARGET_COL]).fillna(0.0)
        if len(chunk):
            yield chunk
        gc.collect()


# ---------------------------------------------------------------------------
# Passe 1 : mean/std des features + mean de y
# ---------------------------------------------------------------------------

def compute_stats(parquet_path, feat_cols, max_rows=None, read_tail=True):
    cols   = feat_cols + [TARGET_COL]
    n      = 0
    f_mean = np.zeros(len(feat_cols), dtype=np.float64)
    f_M2   = np.zeros(len(feat_cols), dtype=np.float64)
    y_sum  = 0.0
    naive_raw_sum = 0.0
    naive_n       = 0

    naive_idx = feat_cols.index(NAIVE_COL) if NAIVE_COL in feat_cols else None

    n_chunks = 0
    for chunk in iter_parquet_chunks(parquet_path, cols, max_rows, read_tail):
        X  = chunk[feat_cols].to_numpy(np.float64)
        y  = chunk[TARGET_COL].to_numpy(np.float64)
        bn = X.shape[0]

        bm    = X.mean(axis=0)
        bv    = X.var(axis=0)
        delta = bm - f_mean
        total = n + bn
        f_mean += delta * (bn / total)
        f_M2   += bv * bn + delta**2 * (n * bn / total)
        y_sum  += y.sum()
        n      = total
        n_chunks += 1
        if n_chunks % 20 == 0:
            print(f"    stats pass: {n:,} lignes…", end="\r", flush=True)

        if naive_idx is not None:
            naive_raw_sum += X[:, naive_idx].sum()
            naive_n       += bn

    f_std = np.sqrt(f_M2 / max(n, 1))
    f_std[f_std == 0] = 1.0
    y_mean     = y_sum / n
    naive_mean = naive_raw_sum / naive_n if naive_n else 0.0
    print(f"    stats pass: {n:,} lignes — terminé.    ")
    return f_mean.astype(np.float32), f_std.astype(np.float32), float(y_mean), n, float(naive_mean)


# ---------------------------------------------------------------------------
# Passe 2 : mean/std des features polynomiales (sur X déjà standardisé)
# ---------------------------------------------------------------------------

def compute_poly_stats(parquet_path, feat_cols, f_mean, f_std, poly_idx,
                       max_rows=None, read_tail=True):
    cols = feat_cols + [TARGET_COL]
    k    = len(poly_idx)
    np_  = k * (k + 1) // 2

    n      = 0
    p_mean = np.zeros(np_, dtype=np.float64)
    p_M2   = np.zeros(np_, dtype=np.float64)

    n_chunks = 0
    for chunk in iter_parquet_chunks(parquet_path, cols, max_rows, read_tail):
        X    = (chunk[feat_cols].to_numpy(np.float64) - f_mean) / f_std
        P    = expand_poly(X, poly_idx)
        bn   = P.shape[0]
        bm   = P.mean(axis=0)
        bv   = P.var(axis=0)
        delta = bm - p_mean
        total = n + bn
        p_mean += delta * (bn / total)
        p_M2   += bv * bn + delta**2 * (n * bn / total)
        n      = total
        n_chunks += 1
        if n_chunks % 20 == 0:
            print(f"    poly stats: {n:,} lignes…", end="\r", flush=True)

    p_std = np.sqrt(p_M2 / max(n, 1))
    p_std[p_std == 0] = 1.0
    print(f"    poly stats: {n:,} lignes — terminé.    ")
    return p_mean.astype(np.float32), p_std.astype(np.float32)


# ---------------------------------------------------------------------------
# Passe 3 : équations normales (features orig + poly)
# ---------------------------------------------------------------------------

def compute_normal_equations(parquet_path, feat_cols, f_mean, f_std,
                              y_mean, poly_idx, p_mean, p_std,
                              max_rows=None, read_tail=True):
    p_orig = len(feat_cols)
    k      = len(poly_idx)
    np_    = k * (k + 1) // 2
    p_tot  = p_orig + np_

    XtX = np.zeros((p_tot, p_tot), dtype=np.float64)
    Xty = np.zeros(p_tot,          dtype=np.float64)
    n   = 0

    cols     = feat_cols + [TARGET_COL]
    n_chunks = 0
    for chunk in iter_parquet_chunks(parquet_path, cols, max_rows, read_tail):
        Xo = (chunk[feat_cols].to_numpy(np.float64) - f_mean) / f_std
        Xp = (expand_poly(Xo, poly_idx) - p_mean) / p_std
        X  = np.concatenate([Xo, Xp], axis=1)
        y  = chunk[TARGET_COL].to_numpy(np.float64) - y_mean

        XtX += X.T @ X
        Xty += X.T @ y
        n   += len(y)
        n_chunks += 1
        if n_chunks % 20 == 0:
            print(f"    normal eq: {n:,} lignes…", end="\r", flush=True)

    print(f"    normal eq: {n:,} lignes — terminé.   ")
    return XtX, Xty, n


# ---------------------------------------------------------------------------
# Ridge
# ---------------------------------------------------------------------------

def ridge_solve(XtX, Xty, alpha):
    p = XtX.shape[0]
    return np.linalg.solve(XtX + alpha * np.eye(p), Xty)


# ---------------------------------------------------------------------------
# Prédiction chunk par chunk
# ---------------------------------------------------------------------------

def predict_all(parquet_path, feat_cols, f_mean, f_std, weights, y_mean_train,
                poly_idx, p_mean, p_std,
                naive_scaler_mean, naive_scaler_std, naive_scale):
    cols      = feat_cols + [TARGET_COL]
    naive_idx = feat_cols.index(NAIVE_COL) if NAIVE_COL in feat_cols else None

    y_true_l, y_pred_l, y_naive_l = [], [], []

    for chunk in iter_parquet_chunks(parquet_path, cols,
                                     max_rows=None, read_tail=False):
        Xo = (chunk[feat_cols].to_numpy(np.float64) - f_mean) / f_std
        Xp = (expand_poly(Xo, poly_idx) - p_mean) / p_std
        X  = np.concatenate([Xo, Xp], axis=1)
        y  = chunk[TARGET_COL].to_numpy(np.float64)

        y_true_l.append(y)
        y_pred_l.append(X @ weights + y_mean_train)

        if naive_idx is not None and naive_scaler_mean is not None:
            naive_raw = (chunk[NAIVE_COL].to_numpy(np.float64)
                         * naive_scaler_std + naive_scaler_mean)
        elif naive_idx is not None:
            naive_raw = chunk[NAIVE_COL].to_numpy(np.float64)
        else:
            naive_raw = np.full(len(y), y_mean_train)
        y_naive_l.append(naive_raw * naive_scale)

    return (np.concatenate(y_true_l),
            np.concatenate(y_pred_l),
            np.concatenate(y_naive_l))


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
# Sélection alpha sur val
# ---------------------------------------------------------------------------

def select_alpha(XtX_tr, Xty_tr, feat_cols, f_mean, f_std, y_mean_train,
                 poly_idx, p_mean, p_std,
                 naive_sm, naive_ss, naive_scale, alphas):
    print(f"\n  Sélection alpha (validation) …")
    best_alpha, best_mae = None, float("inf")

    for alpha in alphas:
        w = ridge_solve(XtX_tr, Xty_tr, alpha)
        y_t, y_p, y_n = predict_all(VAL_PARQUET, feat_cols, f_mean, f_std, w,
                                     y_mean_train, poly_idx, p_mean, p_std,
                                     naive_sm, naive_ss, naive_scale)
        m    = compute_metrics(y_t, y_p, y_n)
        mark = " ←" if (m["mae"] is not None and m["mae"] < best_mae) else ""
        print(f"    alpha={alpha:>8.4g}  MAE={m['mae']:.4f}  "
              f"Δnaïf={m['improvement_vs_naive_pct']:+.2f}%{mark}")
        if m["mae"] is not None and m["mae"] < best_mae:
            best_mae   = m["mae"]
            best_alpha = alpha

    print(f"  → alpha optimal : {best_alpha}  (MAE val = {best_mae:.4f})")
    return best_alpha


# ---------------------------------------------------------------------------
# Évaluation sur le sous-ensemble TFT (h=2 premiers timesteps/série)
# ---------------------------------------------------------------------------

def eval_on_tft_subset(weights, feat_cols, f_mean, f_std, y_mean_train,
                       poly_idx, p_mean, p_std,
                       naive_sm, naive_ss, naive_scale, h=2):
    print(f"\n=== Éval TFT-subset (premiers h={h} timesteps/série) ===")
    schema = pq.read_schema(TEST_PARQUET)
    cols_needed = feat_cols + [TARGET_COL]

    group_col = "group_id" if "group_id" in schema.names else None
    time_col  = "time_idx"  if "time_idx"  in schema.names else None

    if group_col is None or time_col is None:
        print("  [SKIP] group_id ou time_idx absent du test.parquet")
        return None

    cols_load = list(set(cols_needed + [group_col, time_col]))
    naive_idx = feat_cols.index(NAIVE_COL) if NAIVE_COL in feat_cols else None

    print("  Lecture test.parquet …")
    pf  = pq.ParquetFile(TEST_PARQUET)
    dfs = []
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=cols_load)
        dfs.append(tbl.to_pandas())
    df_test = pd.concat(dfs, ignore_index=True)
    df_test = df_test.dropna(subset=[TARGET_COL]).fillna(0.0)
    print(f"  {len(df_test):,} lignes test chargées")

    df_test = df_test.sort_values([group_col, time_col])
    df_sub  = df_test.groupby(group_col, sort=False).head(h)
    print(f"  Subset : {len(df_sub):,} lignes ({df_sub[group_col].nunique():,} séries × h={h})")

    # Prédictions Ridge
    Xo  = (df_sub[feat_cols].to_numpy(np.float64) - f_mean) / f_std
    Xp  = (expand_poly(Xo, poly_idx) - p_mean) / p_std
    X   = np.concatenate([Xo, Xp], axis=1)
    y_sub  = df_sub[TARGET_COL].to_numpy(np.float64)
    yp_sub = X @ weights + y_mean_train

    # Baseline naïve : dernière target connue par série depuis train
    print("  Calcul naïf (dernière target par série depuis train.parquet) …")
    pf_tr = pq.ParquetFile(TRAIN_PARQUET)
    tr_cols = [group_col, TARGET_COL]
    last_target = {}
    for rg in range(pf_tr.num_row_groups):
        chunk = pf_tr.read_row_group(rg, columns=tr_cols).to_pandas()
        chunk = chunk.dropna(subset=[TARGET_COL])
        for gid, val in zip(chunk[group_col], chunk[TARGET_COL]):
            last_target[gid] = float(val)
    print(f"  Naïf disponible pour {len(last_target):,} séries")

    y_naive_sub = np.array([
        last_target.get(g, float("nan")) for g in df_sub[group_col]
    ])

    m_ridge = compute_metrics(y_sub, yp_sub, y_naive_sub)
    print(f"  Ridge subset → MAE={m_ridge['mae']:.4f}  RMSE={m_ridge['rmse']:.4f}  "
          f"Δnaïf={m_ridge['improvement_vs_naive_pct']:+.2f}%  ({m_ridge['n_rows']:,} lignes)")
    return m_ridge


# ---------------------------------------------------------------------------
# Résultats TFT existants
# ---------------------------------------------------------------------------

def load_tft_results():
    if not TFT_RUN_DIR.exists():
        return None
    reports = sorted(TFT_RUN_DIR.glob("run_*/quality_report.json"))
    if not reports:
        return None
    latest  = reports[-1]
    print(f"  TFT : {latest.parent.name}/quality_report.json")
    data    = json.loads(latest.read_text())
    metrics = data.get("metrics", {})
    for key in ("test_from_train_plus_val_context",
                "test_from_train_context", "val_from_train_context"):
        m = metrics.get(key)
        if m and m.get("mae") is not None:
            return {"source": key, "run": str(latest.parent.name), **m}
    return None


def load_lgbm_results():
    p = OUT_DIR / "lgbm_baseline_report.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def print_table(rows):
    hdr = (f"{'Modèle':<50} {'MAE':>8} {'RMSE':>8} "
           f"{'MAE Naïf':>9} {'Δ vs Naïf':>11} {'N lignes':>11}")
    sep = "─" * len(hdr)
    print(); print(sep); print(hdr); print(sep)
    for label, mae, rmse, nm, imp, n in rows:
        ms  = f"{mae:.4f}"   if mae  is not None else "N/A"
        rs  = f"{rmse:.4f}"  if rmse is not None else "N/A"
        ns  = f"{nm:.4f}"    if nm   is not None else "N/A"
        is_ = f"{imp:+.2f}%" if imp  is not None else "N/A"
        nn  = f"{n:,}"       if n    is not None else "N/A"
        print(f"{label:<50} {ms:>8} {rs:>8} {ns:>9} {is_:>11} {nn:>11}")
    print(sep); print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-rows", type=int, default=45_000_000)
    parser.add_argument("--max-val-rows",   type=int, default=1_500_000)
    parser.add_argument("--no-tail",        action="store_true")
    parser.add_argument("--data-dir",       type=str, default=None)
    parser.add_argument("--out-dir",        type=str, default=None)
    args = parser.parse_args()

    global TRAIN_PARQUET, VAL_PARQUET, TEST_PARQUET, SCALER_PATH, OUT_DIR
    if args.data_dir:
        d = Path(args.data_dir)
        TRAIN_PARQUET = d / "train.parquet"
        VAL_PARQUET   = d / "val.parquet"
        TEST_PARQUET  = d / "test.parquet"
        SCALER_PATH   = d / "scaler_mappings.json"
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    read_tail = not args.no_tail
    t0        = time.time()

    feat_cols = available_cols(TRAIN_PARQUET, ALL_FEATURES)
    poly_idx  = get_poly_idx(feat_cols)
    k         = len(poly_idx)
    np_terms  = k * (k + 1) // 2
    print(f"\nFeatures : {len(feat_cols)} orig + {np_terms} poly = {len(feat_cols)+np_terms} total")

    naive_sm, naive_ss = load_naive_scaler()
    if naive_sm is not None:
        print(f"  retard_j_1 scaler : mean={naive_sm:.4f} min, std={naive_ss:.4f} min")

    # -------------------------------------------------------------------
    # Passe 1 — statistiques features + y_mean
    # -------------------------------------------------------------------
    print(f"\n=== Passe 1/3 : stats features ({args.max_train_rows//1_000_000}M lignes) ===")
    f_mean, f_std, y_mean_train, n_train, naive_mean_raw = compute_stats(
        TRAIN_PARQUET, feat_cols, args.max_train_rows, read_tail)

    # Auto-détection unité : target en secondes si >> retard_j_1 en minutes
    naive_scale = 1.0
    if naive_mean_raw > 0 and y_mean_train > naive_mean_raw * 20:
        naive_scale = 60.0
        print(f"  [auto-détect] target en SECONDES (mean={y_mean_train:.1f}), "
              f"retard_j_1 en MINUTES (mean={naive_mean_raw:.3f}) → naïf × 60")
    print(f"  y_mean_train = {y_mean_train:.4f}  (intercept Ridge)")

    # -------------------------------------------------------------------
    # Passe 2 — statistiques features polynomiales
    # -------------------------------------------------------------------
    print(f"\n=== Passe 2/3 : stats poly ({np_terms} termes deg-2) ===")
    p_mean, p_std = compute_poly_stats(
        TRAIN_PARQUET, feat_cols, f_mean, f_std, poly_idx,
        args.max_train_rows, read_tail)

    # -------------------------------------------------------------------
    # Passe 3 — équations normales train
    # -------------------------------------------------------------------
    print(f"\n=== Passe 3/3 : X^TX, X^Ty (dim={len(feat_cols)+np_terms}) ===")
    XtX, Xty, _ = compute_normal_equations(
        TRAIN_PARQUET, feat_cols, f_mean, f_std, y_mean_train,
        poly_idx, p_mean, p_std,
        args.max_train_rows, read_tail)

    # -------------------------------------------------------------------
    # Sélection alpha sur val
    # -------------------------------------------------------------------
    print(f"\n=== Équations normales val ===")
    XtX_v, Xty_v, n_val = compute_normal_equations(
        VAL_PARQUET, feat_cols, f_mean, f_std, y_mean_train,
        poly_idx, p_mean, p_std,
        args.max_val_rows, read_tail=True)
    print(f"  {n_val:,} lignes val")

    best_alpha = select_alpha(
        XtX, Xty, feat_cols, f_mean, f_std, y_mean_train,
        poly_idx, p_mean, p_std,
        naive_sm, naive_ss, naive_scale, ALPHAS)

    weights = ridge_solve(XtX, Xty, best_alpha)
    p_tot   = len(feat_cols) + np_terms
    print(f"  ||w||_2 = {np.linalg.norm(weights):.4f}  intercept = {y_mean_train:.4f}  (dim={p_tot})")

    # -------------------------------------------------------------------
    # Évaluation test complet
    # -------------------------------------------------------------------
    print(f"\n=== Évaluation sur test.parquet ===")
    t_e = time.time()
    y_true, y_pred, y_naive = predict_all(
        TEST_PARQUET, feat_cols, f_mean, f_std, weights,
        y_mean_train, poly_idx, p_mean, p_std,
        naive_sm, naive_ss, naive_scale)
    print(f"  Prédictions en {time.time()-t_e:.1f}s")

    metrics = compute_metrics(y_true, y_pred, y_naive)
    print(f"  Ridge → MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
          f"Δnaïf={metrics['improvement_vs_naive_pct']:+.2f}%  "
          f"({metrics['n_rows']:,} lignes)")

    nm        = np.isfinite(y_true) & np.isfinite(y_naive)
    naive_mae  = float(np.mean(np.abs(y_naive[nm] - y_true[nm]))) if nm.any() else None
    naive_rmse = float(np.sqrt(np.mean((y_naive[nm]-y_true[nm])**2))) if nm.any() else None
    n_test     = int(nm.sum())

    # -------------------------------------------------------------------
    # Évaluation TFT-subset
    # -------------------------------------------------------------------
    m_sub = eval_on_tft_subset(
        weights, feat_cols, f_mean, f_std, y_mean_train,
        poly_idx, p_mean, p_std,
        naive_sm, naive_ss, naive_scale)

    # -------------------------------------------------------------------
    # Chargement comparaisons
    # -------------------------------------------------------------------
    print("\n=== Chargement résultats comparaison ===")
    tft  = load_tft_results()
    lgbm = load_lgbm_results()
    if lgbm:
        lgbm_m = lgbm.get("lgbm", {})
        lgbm_label = (f"LightGBM (iter={lgbm.get('config',{}).get('best_iteration','?')}, "
                      f"{lgbm.get('config',{}).get('n_train_rows',0)//1_000_000}M lignes)")
        print(f"  LightGBM chargé  ({lgbm_label})")
    if tft:
        print(f"  TFT chargé  (run_{tft['run']})")

    # -------------------------------------------------------------------
    # Table — tout le test
    # -------------------------------------------------------------------
    print("\n--- Éval sur tout le test ---")
    rows = [
        ("Naïf (retard_j_1 dénormalisé)",
         naive_mae, naive_rmse, naive_mae, 0.0, n_test),
        (f"Ridge poly (α={best_alpha:.4g}, {n_train//1_000_000}M lignes)",
         metrics["mae"], metrics["rmse"],
         metrics["naive_mae"], metrics["improvement_vs_naive_pct"],
         metrics["n_rows"]),
    ]
    if lgbm:
        rows.append((lgbm_label,
                     lgbm_m.get("mae"), lgbm_m.get("rmse"),
                     lgbm_m.get("naive_mae"), lgbm_m.get("improvement_vs_naive_pct"),
                     lgbm_m.get("n_rows")))
    if tft:
        rows.append((f"TFT NeuralForecast ({tft['run']})",
                     tft.get("mae"), tft.get("rmse"),
                     tft.get("naive_mae"), tft.get("improvement_vs_naive_pct"),
                     tft.get("finite_rows")))
    print_table(rows)

    # -------------------------------------------------------------------
    # Table — sous-ensemble TFT
    # -------------------------------------------------------------------
    if m_sub:
        lgbm_sub = lgbm.get("lgbm_subset") if lgbm else None
        tft_naive = tft.get("naive_mae") if tft else None
        print("--- Éval TFT-subset (h=2 premiers timesteps/série) ---")
        rows_sub = [
            ("Ridge poly subset",
             m_sub["mae"], m_sub["rmse"],
             m_sub["naive_mae"], m_sub["improvement_vs_naive_pct"],
             m_sub["n_rows"]),
        ]
        if lgbm_sub:
            rows_sub.append(("LightGBM subset",
                              lgbm_sub.get("mae"), lgbm_sub.get("rmse"),
                              lgbm_sub.get("naive_mae"), lgbm_sub.get("improvement_vs_naive_pct"),
                              lgbm_sub.get("n_rows")))
        if tft:
            rows_sub.append((f"TFT NeuralForecast ({tft['run']})",
                              tft.get("mae"), tft.get("rmse"),
                              tft.get("naive_mae"), tft.get("improvement_vs_naive_pct"),
                              tft.get("finite_rows")))
        print_table(rows_sub)

    # -------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "max_train_rows": args.max_train_rows,
            "n_features_orig": len(feat_cols),
            "n_features_poly": np_terms,
            "n_features_total": p_tot,
            "feature_cols": feat_cols,
            "poly_base": [feat_cols[i] for i in poly_idx],
            "best_alpha": float(best_alpha),
            "y_mean_train": float(y_mean_train),
            "naive_scale": naive_scale,
            "naive_scaler": {"mean": naive_sm, "std": naive_ss},
        },
        "ridge": {"alpha": float(best_alpha), **metrics},
        "ridge_subset": m_sub,
        "naive": {"mae": naive_mae, "rmse": naive_rmse, "n_rows": n_test},
        "tft": tft,
        "elapsed_seconds": float(time.time() - t0),
    }
    out = OUT_DIR / "linear_baseline_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Rapport : {out}")
    print(f"Temps total : {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
