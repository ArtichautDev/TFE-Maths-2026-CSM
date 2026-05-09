import os
import polars as pl
import numpy as np

INPUT_PATH = 'code/output/datasets/dataset_moduleD_verified.parquet'
OUTPUT_PATH = 'code/output/datasets/dataset_moduleE_verified.parquet'
TRAVAUX_PATH = os.environ.get('TRAVAUX_CSV', '/root/SNCB/test_data/SNCB_DATA/langetermijnplanning-tcr-spoornetwerk.csv')
GARES_PATH = 'data/gares.csv'

# Impact levels/types aligned to moduleF_travaux.js
IMPACT_LEVELS = {
    'TOTAL_CLOSURE': 3,
    'SINGLE_TRACK': 2,
    'SPEED_REDUCTION': 1,
    'MINOR': 0,
    'NONE': -1,
}

IMPACT_TYPES = {
    'TOTAL_CLOSURE': 'total_closure',
    'SINGLE_TRACK': 'single_track',
    'SPEED_REDUCTION': 'speed_reduction',
    'OTHER': 'other',
    'NONE': 'none',
}

def normalize_line(col):
    return pl.col(col).cast(pl.Utf8).str.strip_chars().str.replace(' ', '').str.to_uppercase()

def parse_impact_type(expr):
    expr = expr.fill_null('')
    lower = expr.str.to_lowercase()
    return (
        pl.when(lower.str.contains('total') | lower.str.contains('closure') | lower.str.contains('tlc'))
        .then(pl.lit(IMPACT_TYPES['TOTAL_CLOSURE']))
        .when(lower.str.contains('single') | lower.str.contains('unique') | lower.str.contains('enkel'))
        .then(pl.lit(IMPACT_TYPES['SINGLE_TRACK']))
        .when(lower.str.contains('speed') | lower.str.contains('vitesse'))
        .then(pl.lit(IMPACT_TYPES['SPEED_REDUCTION']))
        .when(lower == '')
        .then(pl.lit(IMPACT_TYPES['NONE']))
        .otherwise(pl.lit(IMPACT_TYPES['OTHER']))
    )

def impact_level_from_type(col):
    return (
        pl.when(col == IMPACT_TYPES['TOTAL_CLOSURE']).then(IMPACT_LEVELS['TOTAL_CLOSURE'])
        .when(col == IMPACT_TYPES['SINGLE_TRACK']).then(IMPACT_LEVELS['SINGLE_TRACK'])
        .when(col == IMPACT_TYPES['SPEED_REDUCTION']).then(IMPACT_LEVELS['SPEED_REDUCTION'])
        .otherwise(IMPACT_LEVELS['MINOR'])
    )

def normalize_station_name(col):
    return (
        pl.col(col).cast(pl.Utf8)
        .str.to_uppercase()
        .str.replace_all(r'[ÀÁÂÃÄÅ]', 'A')
        .str.replace_all(r'[ÈÉÊË]', 'E')
        .str.replace_all(r'[ÌÍÎÏ]', 'I')
        .str.replace_all(r'[ÒÓÔÕÖ]', 'O')
        .str.replace_all(r'[ÙÚÛÜ]', 'U')
        .str.replace_all(r'[Ý]', 'Y')
        .str.replace_all(r'[Ç]', 'C')
        .str.replace_all(r'[Ñ]', 'N')
        .str.replace_all(r'[^\w\s]', '')  # Remove punctuation AFTER unaccenting
        .str.replace_all(r'\s+', ' ')     # Collapse spaces
        .str.strip_chars()
    )

def load_station_coordinates():
    if not os.path.exists(GARES_PATH):
        print(f"Warning: {GARES_PATH} not found. Station coordinates will not be enriched.")
        return None
        
    try:
        # Load gares.csv
        df = pl.read_csv(GARES_PATH, separator=';', ignore_errors=True)
        
        # Geo Point format: "lat, lon"
        df = df.with_columns([
            pl.col('Geo Point').str.split(', ').list.get(0).cast(pl.Float32).alias('lat_new'),
            pl.col('Geo Point').str.split(', ').list.get(1).cast(pl.Float32).alias('lon_new')
        ]).filter(pl.col('lat_new').is_not_null() & pl.col('lon_new').is_not_null())
        
        # Normalize target columns
        name_cols = ['Nom FR complet', 'Nom NL complet', 'Nom symbolique', 'Code TAF/TAP', 'Nom FR court', 'Nom NL court']
        available_cols = [c for c in name_cols if c in df.columns]

        mappings = []
        for nc in available_cols:
            mappings.append(
                df.select([
                    normalize_station_name(nc).alias('nom_gare_norm'),
                    pl.col('lat_new'),
                    pl.col('lon_new')
                ]).filter(pl.col('nom_gare_norm').is_not_null())
            )
        
        if not mappings:
            return None
            
        full_map = pl.concat(mappings).unique(subset=['nom_gare_norm'], keep='first')
        
        # Manual patches for stubborn stations
        patches = pl.DataFrame({
            'nom_gare_norm': [
                'CHARLEROISUD', 'CHARLEROISUDQUAI01', 'CHARLEROISUDQUAI0103', 'CHARLEROISUDQUAI0405', 
                'MORTSELDEURNESTEENWEG', 'HASSELTAFLOSL35', 'HASSELTAFLOS L35', 'BAULERS', 'HAVERSINGARAGE', 
                'OLENSAS', 'PHILIPPEVILLE642', 'ANTWERPENLINKEROEVER', 'SCHAARBEEKPERRONVORMING', 'SCHAARBEEKPERRON VORMING', 
                'MONTZENN', 'BAASRODEWIJKSPOREN', 'SCHAARBEEKPERRON', 'SCHAARBEEKPERRON TWDNOS'
            ],
            'lat_new': [
                50.4122, 50.4122, 50.4122, 50.4122, 
                51.1717, 50.9304, 50.9304, 50.6074, 50.2372, 
                51.1444, 50.1982, 51.2194, 50.8789, 50.8789, 
                50.7067, 51.0185, 50.8789, 50.8789
            ],
            'lon_new': [
                4.4376, 4.4376, 4.4376, 4.4376, 
                4.4536, 5.3378, 5.3378, 4.3499, 5.2342, 
                4.8606, 4.5367, 4.3995, 4.3787, 4.3787, 
                5.9283, 4.1706, 4.3787, 4.3787
            ]
        }, schema={'nom_gare_norm': pl.Utf8, 'lat_new': pl.Float32, 'lon_new': pl.Float32})
        
        # Convert full_map to eager DataFrame if it's not already, or both to Lazy
        if isinstance(full_map, pl.LazyFrame):
            full_map = full_map.collect()
            
        full_map = pl.concat([full_map, patches], how="vertical").unique(subset=['nom_gare_norm'], keep='last')
        
        return full_map


        
    except Exception as e:
        print(f"Error loading station coordinates: {e}")
        return None

def load_travaux():
    if not os.path.exists(TRAVAUX_PATH):
        return None

    df = pl.read_csv(TRAVAUX_PATH, separator=';', ignore_errors=True)

    df = df.rename({
        'Date de Début': 'date_debut',
        'Date de Fin': 'date_fin',
        'Impact sur le Réseau': 'impact_reseau',
        'impact_code': 'impact_code',
        "Niveau de l'Impact": 'impact_niveau',
        'Ligne': 'ligne'
    })
    
    # Split geo_point_2d if exists
    if 'geo_point_2d' in df.columns:
        df = df.with_columns([
            pl.col('geo_point_2d').str.split(', ').list.get(0).cast(pl.Float32).alias('work_lat'),
            pl.col('geo_point_2d').str.split(', ').list.get(1).cast(pl.Float32).alias('work_lon')
        ])
    else:
        df = df.with_columns([
            pl.lit(None).cast(pl.Float32).alias('work_lat'),
            pl.lit(None).cast(pl.Float32).alias('work_lon')
        ])

    df = df.with_columns([
        pl.col('date_debut').str.strptime(pl.Date, strict=False, format='%Y-%m-%d'),
        pl.col('date_fin').str.strptime(pl.Date, strict=False, format='%Y-%m-%d'),
        normalize_line('ligne').alias('ligne_norm'),
    ])

    impact_expr = (
        pl.coalesce([pl.col('impact_reseau'), pl.col('impact_code'), pl.col('impact_niveau')])
        .cast(pl.Utf8)
    )
    df = df.with_columns(
        parse_impact_type(impact_expr).alias('impact_type')
    )
    df = df.with_columns(
        impact_level_from_type(pl.col('impact_type')).alias('impact_level')
    )

    df = df.with_columns(
        pl.date_ranges('date_debut', 'date_fin', interval='1d', closed='both').alias('date_range')
    ).explode('date_range').rename({'date_range': 'date'})

    df = df.with_columns(
        pl.col('date').dt.strftime('%Y-%m-%d')
    )

    return df.select(['ligne_norm', 'date', 'impact_type', 'impact_level', 'work_lat', 'work_lon'])

def run():
    print('Loading base dataset (Module D verified)...')
    base = pl.scan_parquet(INPUT_PATH)
    
    # Patch coordinates
    print(f"Loading station coordinates from {GARES_PATH}...")
    coords_map = load_station_coordinates()
    if coords_map is not None:
        print("Applying coordinate patch...")
        base = base.with_columns(
            normalize_station_name('nom_gare').alias('nom_gare_norm')
        )
        
        # We convert coords_map (DataFrame) to LazyFrame for join
        base = base.join(
            coords_map.lazy(), 
            on='nom_gare_norm', 
            how='left'
        )
        
        base = base.with_columns([
            pl.coalesce([pl.col('lat_new'), pl.col('stop_lat')]).alias('stop_lat'),
            pl.coalesce([pl.col('lon_new'), pl.col('stop_lon')]).alias('stop_lon')
        ]).drop(['lat_new', 'lon_new', 'nom_gare_norm'])
    
    travaux = load_travaux()
    
    if travaux is None:
        print('Travaux CSV not found; applying defaults.')
        lf = base.with_columns([
            pl.lit(0).cast(pl.Int8).alias('travaux_actif_ligne'),
            pl.lit(IMPACT_LEVELS['NONE']).cast(pl.Int8).alias('travaux_niveau_impact'),
            pl.lit(IMPACT_TYPES['NONE']).cast(pl.Utf8).alias('travaux_type_impact'),
            pl.lit(0).cast(pl.Int8).alias('travaux_nb_actifs_ligne'),
            pl.lit(-1).cast(pl.Float32).alias('travaux_distance_km'),
        ])
    else:
        print('Travaux CSV found; computing per-line/day impact...')
        
        # 1. Standard Aggregation for global line impact
        travaux_with_priority = travaux.with_columns(
            pl.when(pl.col('impact_type') == IMPACT_TYPES['TOTAL_CLOSURE']).then(3)
            .when(pl.col('impact_type') == IMPACT_TYPES['SINGLE_TRACK']).then(2)
            .when(pl.col('impact_type') == IMPACT_TYPES['SPEED_REDUCTION']).then(1)
            .otherwise(0).alias('impact_priority')
        )
        
        agg = travaux_with_priority.group_by(['ligne_norm', 'date']).agg([
            pl.len().alias('travaux_nb_actifs_ligne'),
            pl.max('impact_level').alias('travaux_niveau_impact'),
            pl.col('impact_type').sort_by('impact_priority', descending=True).first().alias('travaux_type_impact')
        ])
        
        # 2. Distance Calculation
        print("Calculating distances...")
        
        # We need to compute normalization on base line
        lf_base = base.with_columns([
            normalize_line('ligne_principale').alias('ligne_norm')
        ])
        
        # Unique (line, date, station) from trains to reduce join size
        # We assume stop_lat/stop_lon exist from Module D
        unique_stops = lf_base.select(['ligne_norm', 'date', 'stop_lat', 'stop_lon']).unique()
        
        # Join with travaux locations (lazy)
        travaux_locs = travaux.lazy().select(['ligne_norm', 'date', 'work_lat', 'work_lon'])
        
        # Join stops with works on the same line and date
        joined = unique_stops.join(travaux_locs, on=['ligne_norm', 'date'], how='left')
        
        # Filter where works exist
        joined_works = joined.filter(pl.col('work_lat').is_not_null())
        
        # Haversine calculation
        pid180 = np.pi / 180.0
        
        # Difference in radians
        dlat = (pl.col('work_lat') - pl.col('stop_lat')) * pid180
        dlon = (pl.col('work_lon') - pl.col('stop_lon')) * pid180
        
        # Latitudes in radians
        lat1 = pl.col('stop_lat') * pid180
        lat2 = pl.col('work_lat') * pid180
        
        # Trig
        a = (dlat / 2).sin().pow(2) + lat1.cos() * lat2.cos() * (dlon / 2).sin().pow(2)
        # c = 2 * atan2(sqrt(a), sqrt(1-a)) -> arcsin approximation
        dist_expr = 6371.0 * 2.0 * a.sqrt().arcsin()
        
        joined_works = joined_works.with_columns(dist_expr.alias('dist_km'))
        
        # Aggregate to find MINIMUM distance to a work for that stop/day
        min_dists = joined_works.group_by(['ligne_norm', 'date', 'stop_lat', 'stop_lon']).agg(
            pl.min('dist_km').alias('travaux_distance_km')
        )
        
        # Join back agg stats and distance
        # Join aggregations
        lf = lf_base.join(agg.lazy(), on=['ligne_norm', 'date'], how='left')
        
        # Join distances (on 4 keys)
        lf = lf.join(min_dists, on=['ligne_norm', 'date', 'stop_lat', 'stop_lon'], how='left')
        
        lf = lf.with_columns([
            pl.col('travaux_nb_actifs_ligne').fill_null(0).cast(pl.Int8),
            pl.col('travaux_niveau_impact').fill_null(IMPACT_LEVELS['NONE']).cast(pl.Int8),
            pl.col('travaux_type_impact').fill_null(IMPACT_TYPES['NONE']).cast(pl.Utf8),
            # Fill null distance with -1 (meaning no work on line, or no coordinates)
            pl.col('travaux_distance_km').fill_null(-1).cast(pl.Float32) 
        ])
        
        lf = lf.with_columns([
            (pl.col('travaux_nb_actifs_ligne') > 0).cast(pl.Int8).alias('travaux_actif_ligne')
        ])
        
        lf = lf.drop('ligne_norm')
    
    lf_final = lf.select(lf.columns)
    
    print(f'Writing Module E output to {OUTPUT_PATH}...')
    lf_final.sink_parquet(OUTPUT_PATH)
    
    print("Verifying Output...")
    # Cannot scan immediately as sink is lazy unless streaming? No sink_parquet executes.
    # But lazy sink_parquet finishes when it returns? Yes.
    
    df_verif = pl.scan_parquet(OUTPUT_PATH).filter(pl.col('travaux_distance_km') > 0).head(1).collect()
    if df_verif.height > 0:
        r = df_verif.row(0, named=True)
        print(f"Sample with distance > 0: {r['travaux_distance_km']} km at {r['nom_gare']}")
    else:
        print("No rows with calculated distance found (all -1 or 0).")

if __name__ == '__main__':
    run()
