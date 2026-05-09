import os
import polars as pl
import numpy as np

INPUT_PATH = 'code/output/datasets/dataset_moduleB_C_verified.parquet'
OUTPUT_PATH = 'code/output/datasets/dataset_moduleD_verified.parquet'

# HUBS defined as (hub_key, traverse_col_name, density_col_name)
HUBS = [
    ('jonction', 'traverse_jonction_nord_midi', 'densite_prevue_jonction_nord_midi'),
    ('schaerbeek', 'traverse_hub_schaerbeek', 'densite_prevue_hub_schaerbeek'),
    ('midi', 'traverse_hub_bruxelles_midi', 'densite_prevue_hub_bruxelles_midi'),
    ('anvers', 'traverse_hub_anvers', 'densite_prevue_hub_anvers'),
    ('gent', 'traverse_hub_gent', 'densite_prevue_hub_gent'),
    ('liege', 'traverse_hub_liege', 'densite_prevue_hub_liege'),
    ('charleroi', 'traverse_hub_charleroi', 'densite_prevue_hub_charleroi')
]

def run():
    print('Loading Module B & C Verified Dataset...')
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")
        
    lf = pl.scan_parquet(INPUT_PATH)
    
    # ----------------------------------------------------
    # 1. Line Density: Trains scheduled on the same line in same hour
    # ----------------------------------------------------
    print("Computing Line Density (trains/hour/line)...")
    
    # Group by [date, hour, ligne_principale] -> count unique trains
    line_density = (
        lf.group_by(['date', 'dt_hour', 'ligne_principale'])
        .agg(pl.n_unique('numero_train').alias('nb_trains_prevus_ligne_heure'))
    )
    
    lf = lf.join(
        line_density,
        on=['date', 'dt_hour', 'ligne_principale'],
        how='left'
    )
    
    lf = lf.with_columns(pl.col('nb_trains_prevus_ligne_heure').fill_null(0).cast(pl.UInt16))
    
    # ----------------------------------------------------
    # 2. Hub Density: Trains traversing specific hubs in same hour
    # ----------------------------------------------------
    print("Computing Hub Densities...")
    
    # Global density lookup per hour
    # We create a dataframe of [date, dt_hour] -> [densite_jonction, densite_midi, ...]
    
    # Base for aggregation is all (date, hour)
    # We need to iterate hubs and merge their counts
    
    # Actually, iterate hubs, compute [date, hour, count], join to main LF
    
    for hub_key, traverse_col, density_col in HUBS:
        print(f"  - Analyzing {hub_key} ({traverse_col})...")
        
        # Filter rows where a train traverses this hub
        # Count unique trains traversing it per [date, hour]
        
        hub_activity = (
            lf.filter(pl.col(traverse_col) == 1)
            .group_by(['date', 'dt_hour'])
            .agg(pl.n_unique('numero_train').alias(density_col))
        )
        
        # Join back to main table (on date/hour - network wide feature)
        lf = lf.join(
            hub_activity,
            on=['date', 'dt_hour'],
            how='left'
        )
        
        # Fill nulls with 0
        lf = lf.with_columns(pl.col(density_col).fill_null(0).cast(pl.UInt16))

    # ----------------------------------------------------
    # 3. Congestion Score
    # ----------------------------------------------------
    print("Computing Congestion Score...")
    
    # Heuristic capacities (approx max trains/hour per hub)
    caps = {
        'densite_prevue_jonction_nord_midi': 90.0,
        'densite_prevue_hub_bruxelles_midi': 80.0,
        'densite_prevue_hub_schaerbeek': 60.0,
        'densite_prevue_hub_anvers': 50.0,
        'densite_prevue_hub_gent': 40.0,
        'densite_prevue_hub_liege': 30.0,
        'densite_prevue_hub_charleroi': 20.0
    }
    
    # Weights for global congestion index
    weights = {
        'densite_prevue_jonction_nord_midi': 0.3,
        'densite_prevue_hub_bruxelles_midi': 0.2, # Midi + Jonction = 50%
        'densite_prevue_hub_schaerbeek': 0.1,
        'densite_prevue_hub_anvers': 0.1,
        'densite_prevue_hub_gent': 0.1,
        'densite_prevue_hub_liege': 0.1,
        'densite_prevue_hub_charleroi': 0.1
    }
    
    # Construct score expression
    # Use fill_null(0) to be safe, though we did it above
    score_expr = pl.lit(0.0)
    for col_name, weight in weights.items():
        if col_name in lf.columns: # Should be there
            cap = caps.get(col_name, 50.0)
            score_expr = score_expr + (pl.col(col_name).cast(pl.Float32) / cap) * weight
            
    lf = lf.with_columns(score_expr.alias('congestion_prevue_score'))
    
    # ----------------------------------------------------
    # 4. Final Selection & Output
    # ----------------------------------------------------
    
    # Keep everything
    print(f'Writing Module D output to {OUTPUT_PATH}...')
    lf.sink_parquet(OUTPUT_PATH)

    # Verification sample
    print("Verifying Module D Output...")
    df_verif = pl.scan_parquet(OUTPUT_PATH).filter(pl.col('nb_trains_prevus_ligne_heure') > 0).head(1).collect()
    if df_verif.height > 0:
        r = df_verif.row(0, named=True)
        print('Sample density feature:')
        print(f"Train {r['numero_train']} @ {r['nom_gare']} hour {r['dt_hour']}")
        print(f"  - Line Density: {r['nb_trains_prevus_ligne_heure']}")
        print(f"  - Jonction Density: {r['densite_prevue_jonction_nord_midi']}")
        print(f"  - Midi Density: {r['densite_prevue_hub_bruxelles_midi']}")
        print(f"  - Congestion Score: {r['congestion_prevue_score']}")
    else:
        print('No density features found verified.')

if __name__ == '__main__':
    run()
