import os
import unicodedata
import re
import polars as pl
import datetime

INPUT_PATH = 'code/output/datasets/dataset_moduleG_verified.parquet'
OUTPUT_PATH = 'code/output/datasets/dataset_moduleH_verified.parquet'
RAW_PUNCTUALITY_PATH = 'code/output/datasets/dataset_optimized.parquet'

def normalize_gare(name: str) -> str:
    if not name: return ''
    normalized = str(name).upper().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return ''.join(c for c in unicodedata.normalize('NFD', normalized) if unicodedata.category(c) != 'Mn')

def compute_rolling_stats(hist_df):
    """
    Compute rolling statistics for Lines, Stations, and Station+Hour slots.
    Returns 3 DataFrames joinable by [key, date].
    """
    print("  -> Pre-computing daily aggregates for rolling stats...")
    
    # Ensure date_dt is valid
    hist = hist_df.with_columns([
        pl.col('retard_min').fill_null(0),
        (pl.col('retard_min') < 6).cast(pl.Float64).alias('is_on_time') # Ponctualite (<6 min)
    ])

    # -------------------------------------------------------------------------
    # A. LINE STATS (7j, 30j)
    # -------------------------------------------------------------------------
    print("  -> Computing Line Stats (7j, 30j)...")
    # 1. Daily Mean per Line
    daily_line = (
        hist.group_by(['ligne_dep_raw', 'date_dt'])
        .agg([
            pl.mean('retard_min').alias('daily_mean_delay'),
            pl.std('retard_min').alias('daily_std_delay'),
            pl.mean('is_on_time').alias('daily_punctuality')
        ])
        .sort(['ligne_dep_raw', 'date_dt'])
    )

    # 2. Rolling features
    # usage: rolling window over 'date_dt'
    # strict window: '7d' means previous 7 days
    # We use 'closed="left"' to exclude current day if we were running on same day
    
    line_rolling = daily_line.rolling(
        index_column='date_dt', 
        period='7d', 
        by='ligne_dep_raw', 
        closed='left' 
    ).agg([
        pl.mean('daily_mean_delay').alias('retard_moyen_ligne_7j'),
        pl.mean('daily_std_delay').alias('volatilite_retard_ligne_7j'),
        pl.mean('daily_punctuality').alias('taux_ponctualite_ligne_7j')
    ])
    
    line_rolling_30 = daily_line.rolling(
        index_column='date_dt', 
        period='30d', 
        by='ligne_dep_raw',
        closed='left'
    ).agg([
        pl.mean('daily_mean_delay').alias('retard_moyen_ligne_30j')
    ])
    
    # Join 7d and 30d
    line_stats = line_rolling.join(line_rolling_30, on=['ligne_dep_raw', 'date_dt'], how='left')

    # -------------------------------------------------------------------------
    # B. STATION STATS (7j)
    # -------------------------------------------------------------------------
    print("  -> Computing Station Stats (7j)...")
    daily_station = (
        hist.group_by(['gare_norm', 'date_dt'])
        .agg([
            pl.mean('retard_min').alias('daily_mean_delay')
        ])
        .sort(['gare_norm', 'date_dt'])
    )
    
    station_stats = daily_station.rolling(
        index_column='date_dt', 
        period='7d', 
        by='gare_norm',
        closed='left'
    ).agg([
        pl.mean('daily_mean_delay').alias('retard_moyen_gare_7j')
    ])

    # -------------------------------------------------------------------------
    # C. STATION + SLOT STATS (7j) - "Same hour at this station"
    # -------------------------------------------------------------------------
    print("  -> Computing Station+Slot Stats (7j)...")
    # Slot = Hour
    daily_station_slot = (
        hist.group_by(['gare_norm', 'dt_hour', 'date_dt'])
        .agg([
            pl.mean('retard_min').alias('daily_mean_delay')
        ])
        .sort(['gare_norm', 'dt_hour', 'date_dt'])
    )
    
    station_slot_stats = daily_station_slot.rolling(
        index_column='date_dt',
        period='7d',
        by=['gare_norm', 'dt_hour'], # Rolling per station+hour bucket !
        closed='left'
    ).agg([
        pl.mean('daily_mean_delay').alias('retard_moyen_slot_horaire_gare_7j')
    ])

    return line_stats, station_stats, station_slot_stats

def main():
    # Use streaming when possible but here we use collect() for stability with large RAM
    pl.Config.set_streaming_chunk_size(1000000)
    
    print(f'Loading punctuality base: {RAW_PUNCTUALITY_PATH}')
    
    # Build historical stats (J-7 etc)
    hist = pl.scan_parquet(RAW_PUNCTUALITY_PATH).select([
        pl.col('TRAIN_NO').alias('numero_train'),
        pl.col('PTCAR_LG_NM_NL').alias('nom_gare_raw'),
        pl.col('LINE_NO_DEP').alias('ligne_dep_raw'),
        pl.col('datetime'),
        pl.col('date_parsed'),
        pl.col('DELAY_ARR').alias('delay_arr_sec')
    ])
    
    hist = hist.with_columns([
        pl.coalesce([pl.col('date_parsed'), pl.col('datetime').dt.date()]).alias('date_dt'),
        pl.col('datetime').dt.hour().alias('dt_hour')
    ])
    
    # Normalize gare names
    hist = hist.with_columns(
        pl.col('nom_gare_raw').map_elements(normalize_gare, return_dtype=pl.String).alias('gare_norm')
    )
    
    hist = hist.with_columns(
        (pl.col('delay_arr_sec') / 60.0).alias('retard_min')
    )
    # Clip outliers for history calculation only [IMPORTANT FIX]
    hist = hist.with_columns(
        pl.col('retard_min').clip(-60, 240)
    )

    print("Collecting History for Stats Calculation (RAM usage expected)...")
    # We collet to perform complex rolling operations reliably
    hist_df = hist.collect()
    
    # Compute Advanced Stats
    print("Computing Advanced Rolling Stats...")
    line_stats, station_stats, station_slot_stats = compute_rolling_stats(hist_df)
    
    # Prepare Daily Stats for Lags (Simple Join)
    # Group ID for trains
    hist_df = hist_df.with_columns(
        pl.concat_str([pl.col('numero_train').cast(pl.String), pl.col('gare_norm')], separator='_').alias('hist_group_id')
    )
    daily_train_stats = (
        hist_df.group_by(['hist_group_id', 'date_dt'])
        .agg(pl.mean('retard_min').alias('retard_min'))
    )

    # Free memory of raw hist_df if possible, but we need daily_train_stats
    del hist_df
    import gc
    gc.collect()

    # -------------------------------------------------------------------------
    # LOAD TRAIN/TEST DATASET
    # -------------------------------------------------------------------------
    print(f'Loading input dataset: {INPUT_PATH}')
    lf = pl.scan_parquet(INPUT_PATH)
    
    # Ensure join keys exist
    lf = lf.with_columns([
        pl.concat_str([pl.col('numero_train').cast(pl.String), pl.col('nom_gare')], separator='_').alias('hist_group_id'),
        pl.col('datetime').dt.date().alias('date_dt'),
        pl.col('datetime').dt.hour().alias('dt_hour')
    ])

    # 1. Join Lag features (J-X)
    print('Joining Lag features...')
    # Re-enable full lag set
    lag_days = [1, 2, 3, 4, 5, 6, 7, 14, 21]
    
    # Reuse `daily_train_stats` (DataFrame)
    # We convert lf to LazyFrame (it is already), but join with DataFrame is supported.
    # Actually, joining Lazy with DataFrame puts DataFrame in query plan.
    # Ideally convert `daily_train_stats` to Lazy for the plan.
    daily_train_stats_lazy = daily_train_stats.lazy()

    for n in lag_days:
        lag_col = f'retard_j_{n}'
        lf = lf.with_columns((pl.col('date_dt') - pl.duration(days=n)).alias(f'date_minus_{n}'))
        
        lf = lf.join(
            daily_train_stats_lazy,
            left_on=['hist_group_id', f'date_minus_{n}'],
            right_on=['hist_group_id', 'date_dt'],
            how='left'
        ).rename({'retard_min': lag_col}).drop(f'date_minus_{n}')
        
    # Aggr features
    lf = lf.with_columns([
        pl.mean_horizontal(['retard_j_7', 'retard_j_14', 'retard_j_21']).alias('retard_moyen_meme_jour_semaine')
    ])
    
    # Missing Flags
    lag_cols_list = lag_days
    lf = lf.with_columns([
        pl.col(f'retard_j_{n}').is_null().cast(pl.Int32).alias(f'lag_missing_j{n}') 
        for n in lag_cols_list
    ])
    lf = lf.with_columns([
        pl.all_horizontal([pl.col(f'lag_missing_j{n}').cast(pl.Boolean) for n in [7, 14, 21]])
        .cast(pl.Int32)
        .alias('lag_missing_semaine')
    ])
    # Fill Nans for lags
    lag_col_names = [f'retard_j_{n}' for n in lag_cols_list]
    lf = lf.with_columns([pl.col(c).fill_null(0.0) for c in lag_col_names])

    # 2. Join Advanced Rolling Stats
    print("Joining Advanced Rolling Stats...")
    
    # Line Stats
    # Note: `line_rolling` has `date_dt` which corresponds to the DAY OF THE STATISTIC.
    # Since we used closed='left', the stat for `date_dt` includes previous days EXCLUDING `date_dt`.
    # Wait... closed='left' means [start, end).
    
    # Actually, let's verify polars meaning of 'closed'.
    # If I say `rolling(period='7d', closed='left')` on 2024-01-08:
    # It looks at window [2024-01-01, 2024-01-08).
    # Does it include 2024-01-08? No, it's open on right.
    # So the value computed for row '2024-01-08' is based in 01..07.
    # This is EXACTLY what we want for leakage prevention.
    # So we join on `date_dt` == `date_dt`.
    
    lf = lf.join(
        line_stats.lazy(),  # line_stats has ['ligne_dep_raw', 'date_dt', 'retard_moyen_ligne_7j', ...]
        left_on=['ligne_principale', 'date_dt'],
        right_on=['ligne_dep_raw', 'date_dt'],
        how='left'
    )

    # Station Stats
    # Normalize `nom_gare` in LF just to be sure
    lf = lf.with_columns(
         pl.col('nom_gare').map_elements(normalize_gare, return_dtype=pl.String).alias('gare_norm_join')
    )
    
    lf = lf.join(
        station_stats.lazy(), # station_stats has ['gare_norm', 'date_dt', 'retard_moyen_gare_7j']
        left_on=['gare_norm_join', 'date_dt'],
        right_on=['gare_norm', 'date_dt'],
        how='left'
    )
    
    lf = lf.join(
        station_slot_stats.lazy(), # station_slot_stats has ['gare_norm', 'dt_hour', 'date_dt', ...]
        left_on=['gare_norm_join', 'dt_hour', 'date_dt'],
        right_on=['gare_norm', 'dt_hour', 'date_dt'],
        how='left'
    )
    
    lf = lf.drop('gare_norm_join')

    # Fill Nulls in Stats (if no history, 0 is acceptable)
    stat_cols = [
        'retard_moyen_ligne_7j', 'retard_moyen_ligne_30j', 'volatilite_retard_ligne_7j', 
        'taux_ponctualite_ligne_7j', 'retard_moyen_gare_7j', 'retard_moyen_slot_horaire_gare_7j'
    ]
    lf = lf.with_columns([pl.col(c).fill_null(0.0) for c in stat_cols])
    
    # Target Propagation
    lf = lf.with_columns([
        pl.col('DELAY_ARR').fill_null(0).alias('target')
    ])
    
    # 3. Write
    print(f'Writing to {OUTPUT_PATH} with ZSTD compression...')
    # Use collect().write_parquet() for stability with complex joins + large RAM
    lf.collect().write_parquet(OUTPUT_PATH, compression='zstd')
    
    print('Done.')

if __name__ == '__main__':
    main()

