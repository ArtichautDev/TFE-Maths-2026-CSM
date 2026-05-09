import polars as pl
import os
import sys

INPUT_PATH = 'code/output/datasets/dataset_moduleH_verified.parquet'
OUTPUT_PATH = 'code/output/datasets/dataset_final.parquet'

# Verification limits
SEUIL_FIABILITE_MINUTES = 10.0 # 600s
MAX_DENSITY_PER_HOUR = 50.0

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        sys.exit(1)
        
    print(f"Loading {INPUT_PATH}...")
    lf = pl.scan_parquet(INPUT_PATH)
    
    # Verify column existence for heure_pointe
    schema = lf.collect_schema().names()
    heure_pointe_col = 'is_heure_pointe' if 'is_heure_pointe' in schema else 'est_heure_pointe'
    print(f"Using heure_pointe column: {heure_pointe_col}")
    
    # -------------------------------------------------------------------------
    # 1. Fiabilite Train Historique (Feature 70)
    # -------------------------------------------------------------------------
    print("Calculating Fiabilite Train Historique...")
    
    lag_cols = [f'retard_j_{i}' for i in range(1, 8)]
    missing_cols = [f'lag_missing_j{i}' for i in range(1, 8)]
    
    # Logic:
    # - If all history is missing (lag_missing_jX is 1 everywhere), fiabilite = 0.5
    # - Else, calculate separate stats on valid lags.
    
    # Replace imputed 0s with null where missing flag is set, to compute correct stats
    lag_exprs = [
        pl.when(pl.col(m) == 1).then(None).otherwise(pl.col(l)) 
        for l, m in zip(lag_cols, missing_cols)
    ]
    
    # Mean of Positive Delays (Method: max(0, delay))
    # We ignore negative delays (early arrivals) for reliability calculation as per JS logic
    pos_lag_exprs = [pl.max_horizontal([l, pl.lit(0)]) for l in lag_exprs]
    
    # Count valid (non-null) lags
    valid_count_expr = pl.sum_horizontal([l.is_not_null().cast(pl.Int8) for l in lag_exprs])
    
    # Sum of positive delays
    sum_pos_delay_expr = pl.sum_horizontal([l.fill_null(0) for l in pos_lag_exprs])
    
    # Mean Delay (Minutes)
    mean_delay_expr = pl.when(valid_count_expr > 0).then(sum_pos_delay_expr / valid_count_expr).otherwise(0)
    
    # Variance Calculation: E[(x - mean)^2]
    # We need strictly the valid lags.
    sq_diff_exprs = [
        ((pl.max_horizontal([l, pl.lit(0)]) - mean_delay_expr).pow(2)).fill_null(0) * l.is_not_null().cast(pl.Int8)
        for l in lag_exprs
    ]
    variance_expr = pl.sum_horizontal(sq_diff_exprs) / valid_count_expr
    std_dev_expr = variance_expr.sqrt()
    
    # Penalty
    variance_penalty = pl.min_horizontal([pl.lit(0.2), std_dev_expr / SEUIL_FIABILITE_MINUTES])
    
    # Base Reliability: 1 - (mean / SEUIL)
    fiabilite_base = pl.max_horizontal([pl.lit(0), pl.lit(1) - (mean_delay_expr / SEUIL_FIABILITE_MINUTES)])
    
    # Final Formula
    # If count >= 3, subtract penalty.
    fiabilite_calc = pl.max_horizontal([
        pl.lit(0),
        fiabilite_base - pl.when(valid_count_expr >= 3).then(variance_penalty).otherwise(0)
    ])
    
    # Handle fully missing history case
    all_missing = pl.all_horizontal([pl.col(m) == 1 for m in missing_cols])
    
    fiabilite_final = pl.when(all_missing).then(0.5).otherwise(fiabilite_calc) # .clip(0, 1) implied by logic
    
    # -------------------------------------------------------------------------
    # 2. Score Risque Retard (Feature 71)
    # -------------------------------------------------------------------------
    print("Calculating Score Risque Retard...")
    # Weights: Hist=0.35, Congestion=0.25, Travaux=0.20, Meteo=0.10, Peak=0.10
    
    risk_weights = {
        'hist': 0.35,
        'cong': 0.25,
        'trav': 0.20,
        'meteo': 0.10,
        'peak': 0.10
    }
    
    histo_score = pl.min_horizontal([pl.lit(1), mean_delay_expr / SEUIL_FIABILITE_MINUTES])
    
    travaux_score = pl.when(pl.col('travaux_actif_ligne') == 1).then((pl.col('travaux_niveau_impact') + 1) / 4).otherwise(0)
    
    risk_score = (
        (histo_score * risk_weights['hist']) +
        (pl.col('congestion_prevue_score').fill_null(0) * risk_weights['cong']) +
        (travaux_score * risk_weights['trav']) +
        (pl.col('meteo_conditions_extremes').fill_null(0) * risk_weights['meteo']) +
        (pl.col(heure_pointe_col).fill_null(0) * 0.5 * risk_weights['peak']) # JS: 0.5 factor
    )
    
    
    # -------------------------------------------------------------------------
    # 3. Score Stress Reseau (Feature 72)
    # -------------------------------------------------------------------------
    print("Calculating Score Stress Reseau...")
    # Weights: Densite=0.4, Travaux=0.3, Meteo=0.2, Peak=0.1
    stress_weights = {'dens': 0.4, 'trav': 0.3, 'meteo': 0.2, 'peak': 0.1}
    
    # Density Score
    densities = [
        'densite_prevue_jonction_nord_midi',
        'densite_prevue_hub_bruxelles_midi',
        'densite_prevue_hub_schaerbeek',
        'nb_trains_prevus_ligne_heure'
    ]
    # Check if cols exist
    available_densities = [d for d in densities if d in schema]
    
    if available_densities:
        mean_density = pl.sum_horizontal([pl.col(d).fill_null(0) for d in available_densities]) / len(available_densities)
        density_score = pl.min_horizontal([pl.lit(1), mean_density / MAX_DENSITY_PER_HOUR])
    else:
        density_score = pl.lit(0)
        
    # Travaux
    travaux_stress_score = pl.min_horizontal([pl.lit(1), pl.col('travaux_nb_actifs_ligne').fill_null(0) / 3])
    
    # Meteo
    meteo_stress_score = (
        pl.when(pl.col('meteo_conditions_extremes') == 1).then(1)
        .when(pl.col('meteo_precipitations') > 5).then(0.5)
        .when(pl.col('meteo_vent_vitesse') > 50).then(0.5)
        .otherwise(0)
    )
    
    stress_score = (
        (density_score * stress_weights['dens']) +
        (travaux_stress_score * stress_weights['trav']) +
        (meteo_stress_score * stress_weights['meteo']) +
        (pl.col(heure_pointe_col).fill_null(0) * stress_weights['peak'])
    )
    
    # -------------------------------------------------------------------------
    # 4. Indice Complexite (Feature 73)
    # -------------------------------------------------------------------------
    print("Calculating Indice Complexite...")
    
    score_arret = pl.min_horizontal([pl.lit(1), pl.col('nb_arrets_total').fill_null(0) / 25])
    score_dist = pl.min_horizontal([pl.lit(1), pl.col('longueur_trajet_km').fill_null(0) / 100])
    
    hubs = [
        'traverse_hub_bruxelles_midi', 'traverse_hub_anvers', 'traverse_hub_gent',
        'traverse_hub_liege', 'traverse_hub_charleroi', 'traverse_hub_schaerbeek',
        'traverse_jonction_nord_midi'
    ]
    hubs_present = [h for h in hubs if h in schema]
    hubs_count = pl.sum_horizontal([pl.col(h).fill_null(0) for h in hubs_present])
    score_hubs = pl.min_horizontal([pl.lit(1), hubs_count / 3])
    
    score_conn = pl.min_horizontal([pl.lit(1), pl.col('nb_connexions_gare').fill_null(0) / 10])
    
    # Position: 1 - abs(0.5 - pos) * 2 -> * 0.5
    pos_score = (1 - (pl.lit(0.5) - pl.col('position_dans_trajet').fill_null(0)).abs() * 2) * 0.5
    
    complexite_final = (score_arret + score_dist + score_hubs + score_conn + pos_score) / 4.5
    
    # -------------------------------------------------------------------------
    # Apply and Save
    # -------------------------------------------------------------------------
    # CLIP TARGET to remove outliers [-20, +240] min
    if 'target' in schema:
        print("Clipping 'target' to range [-20, 240] min...")
        target_expr = pl.col('target').clip(-20.0, 240.0)
    else:
        print("Warning: 'target' column not found, skipping clip.")
        target_expr = pl.lit(None)

    lf = lf.with_columns([
        fiabilite_final.cast(pl.Float32).alias('fiabilite_train_historique'),
        risk_score.cast(pl.Float32).alias('risque_retard_score'),
        stress_score.cast(pl.Float32).alias('stress_reseau_score'),
        complexite_final.cast(pl.Float32).alias('indice_complexite')
    ])

    if 'target' in schema:
        lf = lf.with_columns(target_expr.cast(pl.Float32).alias('target'))
    
    # Keep all columns, including new ones
    print(f"Writing to {OUTPUT_PATH}...")
    # Use collect().write_parquet() to avoid schema mismatch issues in sink_parquet
    # caused by type inference differences during streaming
    lf.collect().write_parquet(OUTPUT_PATH)
    
    # Verification
    print("Verification - Sample of derived features:")
    df = pl.read_parquet(OUTPUT_PATH, n_rows=5)
    print(df.select([
        'numero_train', 
        'fiabilite_train_historique', 
        'risque_retard_score', 
        'stress_reseau_score', 
        'indice_complexite'
    ]))

if __name__ == '__main__':
    main()
