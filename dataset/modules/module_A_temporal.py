import polars as pl
import numpy as np
import math
import re
import sys
import datetime
import unicodedata
import os

def run_pipeline():
    print("Loading optimized dataset... (Lazy)")
    # Using dataset_optimized.parquet
    input_path = 'code/output/datasets/dataset_optimized.parquet'
    
    if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            # Fallback
            input_path = 'code/output/datasets/dataset_final.parquet'
            if not os.path.exists(input_path):
                print(f"Fallback not found: {input_path}")
                return

    try:
        lf = pl.scan_parquet(input_path)
    except Exception as e:
        print(f"Error scanning {input_path}: {e}")
        return

    # ---------------------------------------------------------
    # 1. Normalization Map (Separate Scan to avoid UDF issues)
    # ---------------------------------------------------------
    print("Collecting unique stations for normalization...")
    mapping_df = None
    nom_gare_col = 'PTCAR_LG_NM_NL'
    
    try:
        # Scan explicitly for unique stations
        unique_lf = pl.scan_parquet(input_path).select(pl.col(nom_gare_col).unique())
        unique_stations_df = unique_lf.collect()
        unique_stations = unique_stations_df.get_column(nom_gare_col).to_list()
        
        normalization_map = {}
        for station in unique_stations:
            if station is None:
                normalization_map[station] = None
                continue
            
            normalized = station.upper().strip()
            normalized = re.sub(r'\s+', ' ', normalized)
            normalized = ''.join(c for c in unicodedata.normalize('NFD', normalized) if unicodedata.category(c) != 'Mn')
            normalization_map[station] = normalized
            
        # Create mapping DataFrame
        mapping_df = pl.DataFrame({
            nom_gare_col: list(normalization_map.keys()),
            'nom_gare_norm': list(normalization_map.values())
        })
        
    except Exception as e:
        print(f"Station normalization map blocked: {e}")
        mapping_df = None

    # ---------------------------------------------------------
    # 2. Main Processing Chain
    # ---------------------------------------------------------
    
    # Train Type
    print("Extracting train type (Lazy expr)...")
    train_type_expr = (
        pl.when(pl.col('RELATION').str.contains(r'(?i)^IC[\s-]')).then(pl.lit('IC'))
        .when(pl.col('RELATION').str.contains(r'(?i)^S[\s\d]')).then(pl.lit('S'))
        .when(pl.col('RELATION').str.contains(r'(?i)^L[\s]')).then(pl.lit('L'))
        .when(pl.col('RELATION').str.contains(r'(?i)^P$')).then(pl.lit('P'))
        .when(pl.col('RELATION').str.contains(r'(?i)^CR[\s]')).then(pl.lit('CR'))
        .otherwise(pl.lit('UNKNOWN'))
    )

    lf = lf.with_columns(train_type_expr.alias('train_type'))
    
    # Datetime
    print("Handling datetime (Lazy)...")
    
    # Robust Date Construction using existing parsed columns
    # We rely on 'date_parsed' (Date type). 
    
    # Time Construction: Coalesce 'time_parsed' with fallback to ARRIVAL times
    # Note: 'time_parsed' likely comes from DEP time. If DEP is null (terminus?), we use ARR.
    time_expr = pl.coalesce([
        pl.col('time_parsed'),
        pl.col('PLANNED_TIME_ARR').str.strptime(pl.Time, "%H:%M:%S", strict=False),
        pl.col('REAL_TIME_ARR').str.strptime(pl.Time, "%H:%M:%S", strict=False),
        pl.col('REAL_TIME_DEP').str.strptime(pl.Time, "%H:%M:%S", strict=False)
    ])
    
    # Combine Date & Time
    datetime_expr = pl.col('date_parsed').dt.combine(time_expr)
    
    lf = lf.with_columns(datetime_expr.alias('datetime'))
    
    # Filter out any rows where datetime is still null (should be 0)
    lf = lf.filter(pl.col('datetime').is_not_null())

    # Cyclical Features
    print("Computing cyclical features (Lazy)...")
    
    # 1. Extract components
    lf = lf.with_columns([
        pl.col('datetime').dt.hour().cast(pl.Float64).alias('dt_hour'),
        pl.col('datetime').dt.minute().cast(pl.Float64).alias('dt_minute'),
        pl.col('datetime').dt.weekday().cast(pl.Int64).alias('dt_weekday'),
        pl.col('datetime').dt.month().cast(pl.Int64).alias('dt_month'),
        pl.col('datetime').dt.day().alias('jour_mois')
    ])
    
    # 2. Compute fractional hour
    lf = lf.with_columns(
        (pl.col('dt_hour') + pl.col('dt_minute') / 60.0).alias('hour_float')
    )
    
    # 3. Adjust Weekday/Month indices if needed (0-6, 0-11)
    lf = lf.with_columns(
        pl.when(pl.col('dt_weekday') == 7).then(0).otherwise(pl.col('dt_weekday')).alias('js_weekday'),
        (pl.col('dt_month') - 1).alias('js_month')
    )
    
    # 4. Compute Sine/Cosine
    # hour_float is 0..24, so period is 24. Formula: sin(2 * pi * h / 24)
    lf = lf.with_columns([
        ((2 * np.pi * pl.col('hour_float') / 24).sin()).round(4).cast(pl.Float32).alias('heure_prevue_sin'),
        ((2 * np.pi * pl.col('hour_float') / 24).cos()).round(4).cast(pl.Float32).alias('heure_prevue_cos'),
        ((2 * np.pi * pl.col('js_weekday') / 7).sin()).round(4).cast(pl.Float32).alias('jour_semaine_sin'),
        ((2 * np.pi * pl.col('js_weekday') / 7).cos()).round(4).cast(pl.Float32).alias('jour_semaine_cos'),
        ((2 * np.pi * pl.col('js_month') / 12).sin()).round(4).cast(pl.Float32).alias('mois_sin'),
        ((2 * np.pi * pl.col('js_month') / 12).cos()).round(4).cast(pl.Float32).alias('mois_cos'),
    ])
    
    # Join Normalization
    if mapping_df is not None:
        lf = lf.join(mapping_df.lazy(), on=nom_gare_col, how='left')
    else:
        # Fallback if map failed or cols missing
        try:
             lf = lf.with_columns(pl.col(nom_gare_col).alias('nom_gare_norm'))
        except:
             lf = lf.with_columns(pl.lit('UNKNOWN').alias('nom_gare_norm'))

    # group_id
    train_no_col = 'TRAIN_NO'
    lf = lf.with_columns(
        (pl.col(train_no_col).cast(pl.Utf8).fill_null('UNKNOWN') + '_' + pl.col('nom_gare_norm').fill_null('')).alias('group_id')
    )
    
    # Ligne principale
    line_cols = [pl.col('LINE_NO_DEP').cast(pl.Utf8), pl.col('LINE_NO_ARR').cast(pl.Utf8), pl.lit('UNKNOWN')]
    lf = lf.with_columns(
        pl.coalesce(line_cols).alias('ligne_principale')
    )
    
    # Key aliases
    lf = lf.with_columns([
        pl.col('RELATION').alias('relation'),
        pl.col(train_no_col).alias('numero_train'),
        pl.col('nom_gare_norm').alias('nom_gare')
    ])
    
    final_cols = [
        'group_id', 'relation', 'numero_train', 
        'train_type',
        'ligne_principale', 'nom_gare',
        'heure_prevue_sin', 'heure_prevue_cos',
        'jour_semaine_sin', 'jour_semaine_cos',
        'mois_sin', 'mois_cos',
        'jour_mois',
        'datetime',
        # Preserve target/delay columns
        'DELAY_ARR', 'target_delay_min'
    ]
    
    lf_final = lf.select([c for c in final_cols if c in lf.columns])
    
    result_path = 'code/output/datasets/dataset_moduleA_verified.parquet'
    
    print(f"Writing result to {result_path} (Streaming)...")
    lf_final.sink_parquet(result_path)
    print("Optimization Complete.")
    
    # Verify
    print("\n------------------------------------------------------------")
    print("VERIFICATION (Train 530, 2022-04-01)")
    print("------------------------------------------------------------")
    
    target_train_str = "530" 
    target_date = datetime.date(2022, 4, 1)
    
    # Try verification - scan_parquet is lazy again, so safe
    try:
        verify_df = pl.scan_parquet(result_path).filter(
            (pl.col('datetime').dt.date() == target_date) & 
            (pl.col('numero_train').cast(pl.Utf8) == target_train_str)
        ).collect()
        
        if verify_df.height > 0:
            row = verify_df.head(1).to_dict(as_series=False)
            dt_val = row['datetime'][0]
            sin_val = row['heure_prevue_sin'][0]
            train_type = row['train_type'][0]
             
            print(f"Train: {row['numero_train'][0]}")
            print(f"Date/Time: {dt_val}")
            print(f"Train Type: {train_type}")
            print(f"Heure sin (computed): {sin_val}")
            
            if dt_val:
                h = dt_val.hour + dt_val.minute / 60.0
                expected_sin = math.sin(2 * math.pi * h / 24)
                print(f"Expected sin({h}h): {expected_sin:.4f}")
                
               # 10h -> 0.5 Check
                if abs(h - 10.0) < 0.1:
                    if abs(sin_val - 0.5) < 0.01:
                         print("✅ 10h Check: Value is 0.5 as expected")
                    else:
                         print(f"❌ 10h Check: Value is {sin_val}, expected 0.5")
                elif abs(sin_val - expected_sin) < 0.001:
                    print("✅ MATCH: Sin value matches expected")
                else: 
                     print(f"❌ MISMATCH: {sin_val} vs {expected_sin}")
        else:
            print("Row not found for verification.")
            
    except Exception as e:
        print(f"Verification error: {e}")

if __name__ == "__main__":
    run_pipeline()
