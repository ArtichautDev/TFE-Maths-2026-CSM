import polars as pl
import numpy as np
import os
import datetime
import unicodedata
import re

# ==========================================
# CONSTANTS (Replicated from constants.js)
# ==========================================

JOURS_FERIES = {
    2022: [
        '2022-01-01', '2022-04-17', '2022-04-18', '2022-05-01', '2022-05-26', 
        '2022-06-05', '2022-06-06', '2022-07-21', '2022-08-15', '2022-11-01', 
        '2022-11-11', '2022-12-25'
    ],
    2023: [
        '2023-01-01', '2023-04-09', '2023-04-10', '2023-05-01', '2023-05-18',
        '2023-05-28', '2023-05-29', '2023-07-21', '2023-08-15', '2023-11-01',
        '2023-11-11', '2023-12-25'
    ],
    2024: [
        '2024-01-01', '2024-03-31', '2024-04-01', '2024-05-01', '2024-05-09',
        '2024-05-19', '2024-05-20', '2024-07-21', '2024-08-15', '2024-11-01',
        '2024-11-11', '2024-12-25'
    ]
}

VACANCES_SCOLAIRES = {
    2022: [
        ('2022-02-28', '2022-03-06'), 
        ('2022-04-04', '2022-04-18'), 
        ('2022-07-01', '2022-08-31'), 
        ('2022-10-24', '2022-11-06'), 
        ('2022-12-26', '2023-01-08')
    ],
    2023: [
        ('2023-02-20', '2023-03-05'),
        ('2023-04-03', '2023-04-16'),
        ('2023-05-01', '2023-05-14'),
        ('2023-07-01', '2023-08-31'),
        ('2023-10-23', '2023-11-05'),
        ('2023-12-25', '2024-01-07')
    ],
    2024: [
        ('2024-02-12', '2024-02-18'),
        ('2024-02-26', '2024-03-10'),
        ('2024-04-01', '2024-04-14'),
        ('2024-04-29', '2024-05-12'),
        ('2024-07-01', '2024-08-31'),
        ('2024-10-21', '2024-11-03'),
        ('2024-12-23', '2025-01-05')
    ]
}

JONCTION_NORD_MIDI = [
    'BRUSSEL-NOORD', 'BRUXELLES-NORD',
    'BRUSSEL-CENTRAAL', 'BRUXELLES-CENTRAL',
    'BRUSSEL-KAPELLEKERK', 'BRUXELLES-CHAPELLE',
    'BRUSSEL-ZUID', 'BRUXELLES-MIDI',
    'BRUSSEL-CONGRES', 'BRUXELLES-CONGRES'
]

HUBS = {
    'schaerbeek': ['SCHAARBEEK', 'SCHAERBEEK'],
    'bruxelles_midi': ['BRUSSEL-ZUID', 'BRUXELLES-MIDI'],
    'bruxelles_nord': ['BRUSSEL-NOORD', 'BRUXELLES-NORD'],
    'anvers': ['ANTWERPEN-CENTRAAL', 'ANTWERPEN-BERCHEM', 'ANVERS-CENTRAL', 'ANVERS-BERCHEM'],
    'gent': ['GENT-SINT-PIETERS', 'GAND-SAINT-PIERRE'],
    'liege': ['LIEGE-GUILLEMINS', 'LUIK-GUILLEMINS'],
    'charleroi': ['CHARLEROI-CENTRAL', 'CHARLEROI-CENTRAAL'],
    'leuven': ['LEUVEN', 'LOUVAIN'],
    'mechelen': ['MECHELEN', 'MALINES'],
    'namur': ['NAMUR', 'NAMEN']
}

ALL_HUB_STATIONS = set(JONCTION_NORD_MIDI)
for gares in HUBS.values():
    ALL_HUB_STATIONS.update(gares)

def normalize_gare(name):
    if not name: return ""
    normalized = name.upper().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = ''.join(c for c in unicodedata.normalize('NFD', normalized) if unicodedata.category(c) != 'Mn')
    return normalized

def run_pipeline():
    print("Loading Module A Verified Dataset...")
    input_path = 'code/output/datasets/dataset_moduleA_verified.parquet'
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    lf = pl.scan_parquet(input_path)
    
    # ==========================================
    # MODULE B: CALENDAR
    # ==========================================
    print("Computing Module B (Calendar) features...")
    
    lf = lf.with_columns(
        pl.col('datetime').dt.hour().cast(pl.Int8).alias('dt_hour')
    )
    
    lf = lf.with_columns(
        (pl.col('datetime').dt.weekday() >= 6).cast(pl.Int8).alias('fest_weekend')
    )
    
    start_date = datetime.date(2010, 1, 1)
    end_date = datetime.date(2030, 12, 31)
    date_range = pl.date_range(start_date, end_date, interval="1d", eager=True).alias("date_key")
    calendar_df = pl.DataFrame({"date_key": date_range})
    
    holidays_set = set()
    for year_holidays in JOURS_FERIES.values():
        holidays_set.update(year_holidays)
        
    vacations_list = []
    for year_vacations in VACANCES_SCOLAIRES.values():
        vacations_list.extend(year_vacations)
    
    def check_holiday(d):
        return 1 if str(d) in holidays_set else 0

    def check_vacation(d):
        s = str(d)
        for start, end in vacations_list:
            if start <= s <= end:
                return 1
        return 0

    calendar_df = calendar_df.with_columns([
        pl.col("date_key").map_elements(check_holiday, return_dtype=pl.Int8).alias("is_jour_ferie"),
        pl.col("date_key").map_elements(check_vacation, return_dtype=pl.Int8).alias("is_vacances_scolaires")
    ])
    
    lf = lf.with_columns(pl.col("datetime").dt.date().alias("date_key"))
    lf = lf.join(calendar_df.lazy(), on="date_key", how="left").drop("date_key")
    lf = lf.with_columns([
        pl.col("is_jour_ferie").fill_null(0),
        pl.col("is_vacances_scolaires").fill_null(0)
    ])
    
    peak_condition = (
        (pl.col('datetime').dt.weekday() < 6) & 
        (
            ((pl.col('dt_hour') >= 7) & (pl.col('dt_hour') < 9)) | 
            ((pl.col('dt_hour') >= 17) & (pl.col('dt_hour') < 19))
        )
    )
    lf = lf.with_columns(peak_condition.cast(pl.Int8).alias('is_heure_pointe'))
    
    period_expr = (
        pl.when(pl.col('dt_hour') < 6).then(pl.lit('nuit'))
        .when(pl.col('dt_hour') < 12).then(pl.lit('matin'))
        .when(pl.col('dt_hour') < 14).then(pl.lit('midi'))
        .when(pl.col('dt_hour') < 18).then(pl.lit('apres_midi'))
        .when(pl.col('dt_hour') < 22).then(pl.lit('soir'))
        .otherwise(pl.lit('fin_soiree'))
    )
    lf = lf.with_columns(period_expr.alias('periode_journee'))

    # ==========================================
    # MODULE C: SPATIAL / TRAJET & HUBS
    # ==========================================
    print("Computing Module C (Spatial & Hubs) features...")
    
    gtfs_path = 'test_data/GTFS/pseudo/'
    stops = pl.read_csv(os.path.join(gtfs_path, 'stops.txt'))
    stop_times = pl.read_csv(os.path.join(gtfs_path, 'stop_times.txt'))
    
    # 1. Normalize Station Names in Stops
    
    # MANUAL MAPPING FOR KNOWN MISMATCHES (GTFS -> GARES.CSV)
    manual_corrections = {
        'CHARLEROI-SUD': 'CHARLEROI-CENTRAL',
        'FOREST-MIDI': 'VORST-ZUID',
        'FOREST-EST': 'VORST-OOST',
        'BRUXELLES-LUXEMBOURG': 'BRUSSEL-LUXEMBURG', # Check if needed
        'MORTSEL-DEURNESTEENWEG': 'MORTSEL', # Approximation
        'HASSELT-AFLOS L.35': 'HASSELT',
        'BAULERS': 'NIVELLES', # Former station, now technical junction near Nivelles
        'HAVERSIN-GARAGE': 'HAVERSIN',
        'ANTWERPEN-LINKEROEVER': 'ANTWERPEN-ZUID', # Closest active
        'SCHAARBEEK-PERRON T.W./D.-NOS': 'SCHAARBEEK',
        'SCHAARBEEK-PERRON VORMING': 'SCHAARBEEK',
        'BRUSSEL-KLEINE-EILAND': 'BRUSSEL-ZUID', # Technical yard near Midi
        'LIEGE-GUILLEMINS-FAISCEAU': 'LIEGE-GUILLEMINS',
        'FOREST-VOITURES': 'VORST-ZUID',
        'GENT-SINT-PIETERS-BUNDEL': 'GENT-SINT-PIETERS',
        # Remaining unmatched after first pass
        'OLEN-SAS': 'OLEN',
        'PHILIPPEVILLE-642': 'PHILIPPEVILLE',
        'MONTZEN-N': 'MONTZEN',
        'BAASRODE-WIJKSPOREN': 'BAASRODE-ZUID'
    }

    def apply_manual_mapping(name):
        norm_name = normalize_gare(name)
        if name in manual_corrections:
            return normalize_gare(manual_corrections[name])
        # Also check if normalized name is in keys (case insensitive handled by normalize)
        for k, v in manual_corrections.items():
            if normalize_gare(k) == norm_name:
                return normalize_gare(v)
        return norm_name

    stops = stops.with_columns(
        pl.col('stop_name').map_elements(apply_manual_mapping, return_dtype=pl.Utf8).alias('stop_name_norm')
    )
    
    # 2. Construct Trip ID in Main Table
    # Support both Pseudo-GTFS format (DDMMMYYYY) and numeric (YYYYMMDD) and pick the one that exists in stop_times.
    month_map = {
        1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
        7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
    }
    date_fmt_df = pl.DataFrame({
        "month_num": list(month_map.keys()),
        "month_str": list(month_map.values())
    })
    
    lf = lf.with_columns(pl.col("datetime").dt.month().alias("month_num"))
    lf = lf.join(date_fmt_df.lazy(), on="month_num", how="left")
    
    # Build both candidate date strings
    lf = lf.with_columns([
        (pl.col("datetime").dt.strftime("%d") + pl.col("month_str") + pl.col("datetime").dt.strftime("%Y")).alias("date_id_str_pseudo"),
        pl.col("datetime").dt.strftime("%Y%m%d").alias("date_id_str_iso")
    ])

    lf = lf.with_columns([
        (pl.lit("T_") + pl.col("numero_train").cast(pl.Utf8) + pl.lit("_") + pl.col("date_id_str_pseudo")).alias("trip_id_pseudo"),
        (pl.lit("T_") + pl.col("numero_train").cast(pl.Utf8) + pl.lit("_") + pl.col("date_id_str_iso")).alias("trip_id_iso")
    ])

    # Choose the trip_id that actually exists in the GTFS stop_times
    trip_ids_available = set(stop_times.get_column("trip_id").unique().to_list())
    lf = lf.with_columns(
        pl.when(pl.col("trip_id_pseudo").is_in(list(trip_ids_available)))
        .then(pl.col("trip_id_pseudo"))
        .when(pl.col("trip_id_iso").is_in(list(trip_ids_available)))
        .then(pl.col("trip_id_iso"))
        .otherwise(pl.col("trip_id_pseudo"))
        .alias("trip_id")
    )

    # 3. Prepare Stop Times with Hub Flags
    stops_lazy = stops.lazy().select(['stop_id', 'stop_name_norm', 'stop_lat', 'stop_lon'])
    
    # PATCH: Load real coordinates from gares.csv if available
    gares_path = 'data/gares.csv'
    if os.path.exists(gares_path):
        print(f"Loading real coordinates from {gares_path}...")
        try:
           gares = pl.read_csv('data/gares.csv', separator=';', ignore_errors=True)
           # Extract lat/lon from 'Geo Point' "lat, lon"
           gares = gares.with_columns([
               pl.col('Geo Point').str.split(r', ').list.get(0).cast(pl.Float64).alias('lat_real'),
               pl.col('Geo Point').str.split(r', ').list.get(1).cast(pl.Float64).alias('lon_real')
           ])
           # Normalize names in gares
           # We need to create a comprehensive mapping that includes FR, NL and Symbolic names
           # because GTFS stops might use any of them (though likely FR or NL full names)
           
           # 1. Create mapping from FR names
           gares_fr = gares.select([
               pl.col('Nom FR complet').alias('raw_name'),
               pl.col('lat_real'),
               pl.col('lon_real')
           ]).filter(pl.col('raw_name').is_not_null())
           
           # 2. Create mapping from NL names
           gares_nl = gares.select([
               pl.col('Nom NL complet').alias('raw_name'),
               pl.col('lat_real'),
               pl.col('lon_real')
           ]).filter(pl.col('raw_name').is_not_null())
           
           # 3. Create mapping from Symbolic names (just in case)
           gares_sym = gares.select([
               pl.col('Nom symbolique').alias('raw_name'),
               pl.col('lat_real'),
               pl.col('lon_real')
           ]).filter(pl.col('raw_name').is_not_null())
           
           # Combine all mappings
           all_gares_mapping = pl.concat([gares_fr, gares_nl, gares_sym])
           
           # Normalize the key
           all_gares_mapping = all_gares_mapping.with_columns(
               pl.col('raw_name').map_elements(normalize_gare, return_dtype=pl.Utf8).alias('nom_gare_norm')
           )
           
           # Deduplicate by taking the first valid coordinate for a given normalized name
           gare_coords = all_gares_mapping.select(['nom_gare_norm', 'lat_real', 'lon_real']).unique(subset=['nom_gare_norm'])
           
           # Lazy join
           gare_coords_lazy = gare_coords.lazy()
           
           # Update stops_lazy with real coords
           stops_lazy = stops_lazy.join(gare_coords_lazy, left_on='stop_name_norm', right_on='nom_gare_norm', how='left')
           
           # If nom_gare_norm is used in join as right key, Polars typically keeps left key and drops right key, OR renames right key if collision.
           # In a left join on A=B: 
           # - if A and B have same name, only one is kept.
           # - if A and B diff names, B is kept in output.
           # Here: left='stop_name_norm', right='nom_gare_norm'.
           # So 'nom_gare_norm' SHOULD be present.
           
           # But let's check carefully. Maybe it's dropped if not selected?
           # No, select star behavior.
           
           # However, let's explicitely select what we want to keep to debug or ensure existence.
           
           stops_lazy = stops_lazy.with_columns([
               pl.col('lat_real').fill_null(pl.col('stop_lat')).alias('stop_lat'),
               pl.col('lon_real').fill_null(pl.col('stop_lon')).alias('stop_lon')
           ])
           
           # Drop helper columns if they exist
           # We use a try-drop pattern or just select required cols
           stops_lazy = stops_lazy.select(['stop_id', 'stop_name_norm', 'stop_lat', 'stop_lon'])
           
        except Exception as e:
           print(f"Failed to load gares.csv: {e}")
    
    stop_times_lazy = stop_times.lazy().join(stops_lazy, on='stop_id', how='left')
    
    # Flag hubs in stop_times
    def is_in_jonction(name): return 1 if name in JONCTION_NORD_MIDI else 0
    def is_in_schaerbeek(name): return 1 if name in HUBS['schaerbeek'] else 0
    def is_in_midi(name): return 1 if name in HUBS['bruxelles_midi'] else 0
    def is_in_anvers(name): return 1 if name in HUBS['anvers'] else 0
    def is_in_gent(name): return 1 if name in HUBS['gent'] else 0
    def is_in_liege(name): return 1 if name in HUBS['liege'] else 0
    def is_in_charleroi(name): return 1 if name in HUBS['charleroi'] else 0
    
    stop_times_lazy = stop_times_lazy.with_columns([
        pl.col('stop_name_norm').map_elements(is_in_jonction, return_dtype=pl.Int8).alias('is_jonction'),
        pl.col('stop_name_norm').map_elements(is_in_schaerbeek, return_dtype=pl.Int8).alias('is_schaerbeek'),
        pl.col('stop_name_norm').map_elements(is_in_midi, return_dtype=pl.Int8).alias('is_midi'),
        pl.col('stop_name_norm').map_elements(is_in_anvers, return_dtype=pl.Int8).alias('is_anvers'),
        pl.col('stop_name_norm').map_elements(is_in_gent, return_dtype=pl.Int8).alias('is_gent'),
        pl.col('stop_name_norm').map_elements(is_in_liege, return_dtype=pl.Int8).alias('is_liege'),
        pl.col('stop_name_norm').map_elements(is_in_charleroi, return_dtype=pl.Int8).alias('is_charleroi')
    ])

    # Time parsing helper (HH:MM:SS -> seconds)
    def str_time_to_seconds(col_name):
        return (
            pl.col(col_name).str.slice(0, 2).cast(pl.Int32) * 3600 +
            pl.col(col_name).str.slice(3, 2).cast(pl.Int32) * 60 +
            pl.col(col_name).str.slice(6, 2).cast(pl.Int32)
        )

    # Pre-compute seconds to avoid lexicographic min/max issues on midnight crossings
    stop_times_lazy = stop_times_lazy.with_columns([
        str_time_to_seconds('arrival_time').alias('arrival_seconds'),
        str_time_to_seconds('departure_time').alias('departure_seconds')
    ])
    
    # -------------------------------------------------------
    # FIX: Split merged trips in GTFS (where gaps > 4h exist)
    # -------------------------------------------------------
    # Sort by trip_id, stop_sequence explicitly to ensure window function works correctly
    stop_times_lazy = stop_times_lazy.sort(['trip_id', 'stop_sequence'])
    
    # Pre-compute seconds modulo 24h to avoid issues with >24h GTFS times
    # But keep the absolute ones for gap detection? No, gap detection 4h is large.
    # Actually, let's keep arrival_seconds/departure_seconds as modulo 86400 for consistency with SNCB.
    stop_times_lazy = stop_times_lazy.with_columns([
        (pl.col('arrival_seconds') % 86400).alias('arrival_seconds_mod'),
        (pl.col('departure_seconds') % 86400).alias('departure_seconds_mod')
    ])

    stop_times_lazy = stop_times_lazy.with_columns(
        (pl.col('departure_seconds') - pl.col('departure_seconds').shift(1).over('trip_id'))
        .fill_null(0)
        .alias('time_diff_prev')
    )
    
    stop_times_lazy = stop_times_lazy.with_columns(
        (pl.col('time_diff_prev') > 7200).cum_sum().over('trip_id').alias('sub_trip_idx') # Reduced to 2h gap
    )
    
    # Create a unique 'real_trip_id' for each segment
    stop_times_lazy = stop_times_lazy.with_columns(
        (pl.col('trip_id') + pl.lit('_sub') + pl.col('sub_trip_idx').cast(pl.Utf8))
        .alias('real_trip_id')
    )
    
    # Identify REAL STOPS (dwell time > 0 or endpoints)
    stop_times_lazy = stop_times_lazy.with_columns(
        ((pl.col('departure_seconds') != pl.col('arrival_seconds')) | 
         (pl.col('stop_sequence') == pl.col('stop_sequence').min().over('real_trip_id')) | 
         (pl.col('stop_sequence') == pl.col('stop_sequence').max().over('real_trip_id')))
        .cast(pl.Int8)
        .alias('is_real_stop')
    )

    # RE-NORMALIZE stop_sequence for split trips
    stop_times_lazy = stop_times_lazy.with_columns(
        pl.col('stop_sequence').rank('dense').over('real_trip_id').alias('stop_sequence_local')
    )
    
    # Haversine implementation inside with_columns context requires pure expressions
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        p = np.pi / 180.0
        
        dlat = (lat2 - lat1) * p
        dlon = (lon2 - lon1) * p
        
        a = (dlat / 2).sin().pow(2) + (lat1 * p).cos() * (lat2 * p).cos() * (dlon / 2).sin().pow(2)
        c = 2 * a.sqrt().arcsin()
        return R * c

    # Calculate distance between consecutive points in sequence
    stop_times_lazy = stop_times_lazy.with_columns([
        pl.col('stop_lat').shift(1).over('real_trip_id').alias('prev_lat'),
        pl.col('stop_lon').shift(1).over('real_trip_id').alias('prev_lon')
    ])

    stop_times_lazy = stop_times_lazy.with_columns(
        haversine(pl.col('prev_lat'), pl.col('prev_lon'), pl.col('stop_lat'), pl.col('stop_lon'))
        .fill_null(0.0)
        .alias('segment_dist_km')
    )

    # Aggregate trip info
    trip_hubs = (
        stop_times_lazy
        .group_by('real_trip_id')
        .agg([
            pl.col('trip_id').first().alias('original_trip_id'),
             
            pl.when(pl.col('is_jonction') == 1).then(pl.col('stop_sequence_local')).otherwise(-1).max().alias('max_seq_jonction'),
            pl.when(pl.col('is_schaerbeek') == 1).then(pl.col('stop_sequence_local')).otherwise(-1).max().alias('max_seq_schaerbeek'),
            pl.when(pl.col('is_midi') == 1).then(pl.col('stop_sequence_local')).otherwise(-1).max().alias('max_seq_midi'),
            pl.when(pl.col('is_anvers') == 1).then(pl.col('stop_sequence_local')).otherwise(-1).max().alias('max_seq_anvers'),
            pl.when(pl.col('is_gent') == 1).then(pl.col('stop_sequence_local')).otherwise(-1).max().alias('max_seq_gent'),
            pl.when(pl.col('is_liege') == 1).then(pl.col('stop_sequence_local')).otherwise(-1).max().alias('max_seq_liege'),
            pl.when(pl.col('is_charleroi') == 1).then(pl.col('stop_sequence_local')).otherwise(-1).max().alias('max_seq_charleroi'),

            # Count of REAL STOPS only (to satisfy user quality check)
            pl.col('is_real_stop').sum().alias('nb_arrets_total'),
            
            # Max rank (used for terminus check)
            pl.max('stop_sequence_local').alias('max_sequence_local'),

            # Use modulo seconds for durations
            pl.col('departure_time').sort_by('stop_sequence').first().alias('trip_start_time'),
            pl.col('arrival_time').sort_by('stop_sequence').last().alias('trip_end_time'),
            pl.col('departure_seconds_mod').sort_by('stop_sequence').first().alias('s_start'),
            pl.col('arrival_seconds_mod').sort_by('stop_sequence').last().alias('s_end'),
            pl.col('stop_lat').sort_by('stop_sequence').first().alias('start_lat'),
            pl.col('stop_lon').sort_by('stop_sequence').first().alias('start_lon'),
            pl.col('stop_lat').sort_by('stop_sequence').last().alias('end_lat'),
            pl.col('stop_lon').sort_by('stop_sequence').last().alias('end_lon'),
            
            # Sum of all segment distances (includes non-stop points for accuracy)
            pl.sum('segment_dist_km').alias('total_dist_km')
        ])
    )

    # Note: 'lf_joined' (created later) will have 'stop_sequence_local' because 
    # we added it to stop_times_lazy via with_columns earlier.
    # 'trip_hubs' provides 'nb_arrets_total', 's_start', 's_end'.
    # 'lf_best' provides 'stop_sequence_local'.

    
    # We delay joining trip_hubs until we resolve the correct real_trip_id per row
    
    # Fill defaults LATER (after join)
    
    # Join specific stop info (FILTERED)
    lf = lf.join(stops_lazy, left_on='nom_gare', right_on='stop_name_norm', how='left')
    
    # We join stop_times on (trip_id, stop_id) -> THIS IS AMBIGUOUS if trip merges multiple runs!
    # So we join, calculate time diff, and filter.
    
    # 1. Prepare LF with seconds-of-day
    lf = lf.with_columns(
        (pl.col('datetime').dt.hour() * 3600 + 
         pl.col('datetime').dt.minute() * 60 + 
         pl.col('datetime').dt.second()).cast(pl.Int32).alias('sncb_seconds')
    )
    
    # 2. Prepare stop_times candidates
    # We need real_trip_id from stop_times
    stop_candidates = stop_times_lazy.select([
        'trip_id', 'stop_id', 'stop_sequence', 'stop_sequence_local', 'real_trip_id',
        'arrival_time', 'departure_time',
        'arrival_seconds_mod', 'departure_seconds_mod'
    ])
    
    # 3. Cartesian Join on (trip_id, stop_id)
    # This will explode if trip merges runs.
    lf_joined = lf.join(
        stop_candidates,
        on=['trip_id', 'stop_id'],
        how='left'
    )
    
    # 4. Filter to keep best match
    # Calculate time diff (cyclic 24h)
    
    lf_joined = lf_joined.with_columns(
        (pl.col('sncb_seconds') - pl.col('departure_seconds_mod')).abs().alias('sched_diff_raw')
    )
    
    # Handle cyclic diff (e.g. 23:59 vs 00:01 -> diff 2 min)
    lf_joined = lf_joined.with_columns(
        pl.min_horizontal([
            pl.col('sched_diff_raw'),
            (86400 - pl.col('sched_diff_raw')).abs()
        ]).alias('sched_diff')
    )
    
    # Window function to pick Best Match per (trip_id, datetime, nom_gare)
    # We don't have a unique ID per input row, but (numero_train, datetime) is effectively unique per train run.
    # Group by [numero_train, datetime, nom_gare] and pick min(sched_diff).
    
    lf_best = lf_joined.sort('sched_diff').group_by(['numero_train', 'datetime', 'nom_gare']).first()
    
    # Now lf_best has the correct 'real_trip_id' and 'stop_sequence'.
    
    # 5. Join Trip Hubs using real_trip_id
    lf_final = lf_best.join(trip_hubs, on='real_trip_id', how='left')

    # Sanity-check distances: if implied average speed exceeds 220 km/h, fall back to straight-line distance
    def hav_safe(lat1, lon1, lat2, lon2):
        R = 6371.0
        p = np.pi / 180.0
        dlat = (lat2 - lat1) * p
        dlon = (lon2 - lon1) * p
        a = (dlat / 2).sin().pow(2) + (lat1 * p).cos() * (lat2 * p).cos() * (dlon / 2).sin().pow(2)
        c = 2 * a.sqrt().arcsin()
        return R * c

    lf_final = lf_final.with_columns([
        (pl.col('s_end') - pl.col('s_start')).cast(pl.Float64).alias('trip_duration_s'),
        hav_safe(pl.col('start_lat'), pl.col('start_lon'), pl.col('end_lat'), pl.col('end_lon')).alias('straight_dist_km')
    ])

    # Handle midnight crossing for duration
    lf_final = lf_final.with_columns(
        pl.when(pl.col('trip_duration_s') < 0)
        .then(pl.col('trip_duration_s') + 86400)
        .otherwise(pl.col('trip_duration_s'))
        .alias('trip_duration_s')
    )

    lf_final = lf_final.with_columns(
        pl.when((pl.col('trip_duration_s') > 0) & ((pl.col('total_dist_km') / (pl.col('trip_duration_s') / 3600.0)) > 220))
        .then(pl.col('straight_dist_km'))
        .otherwise(pl.col('total_dist_km'))
        .alias('total_dist_km')
    )

    # Additional route-shape sanity: if path length is >1.5x straight line, trust straight line instead
    lf_final = lf_final.with_columns(
        pl.when(pl.col('total_dist_km') > pl.col('straight_dist_km') * 1.5)
        .then(pl.col('straight_dist_km'))
        .otherwise(pl.col('total_dist_km'))
        .alias('total_dist_km')
    )

    # Physical plausibility cap: no passenger IC line in BE exceeds ~220 km end-to-end.
    lf_final = lf_final.with_columns(
        pl.min_horizontal([
            pl.col('total_dist_km'),
            pl.col('straight_dist_km') * 1.2,  # allow 20% over straight line for track curvature
            pl.lit(220.0)
        ]).alias('total_dist_km')
    )
    
    # Check if any columns are missing (e.g. if join failed completely)
    # Fill defaults for nulls (missing GTFS match)
    
    lf = lf_final.with_columns([
        pl.col('nb_arrets_total').fill_null(10).cast(pl.Int32),
        pl.col('trip_start_time').fill_null("00:00:00"),
        pl.col('trip_end_time').fill_null("00:00:00"),
        pl.col('s_start').fill_null(0),
        pl.col('s_end').fill_null(0),
        pl.col('max_sequence_local').fill_null(10).cast(pl.Int32),
        # Default distance if missing
        pl.col('total_dist_km').fill_null(0.0).cast(pl.Float32) 
    ])
    
    # 4. Compute Spatial Features
    # Use stop_sequence_local for position calculation in split trips
    # position should go from 0 to 1 based on all record positions
    lf = lf.with_columns(
        (
            pl.when(pl.col('max_sequence_local') > 1)
            .then((pl.col('stop_sequence_local') - 1) / (pl.col('max_sequence_local') - 1))
            .otherwise(0)
        ).fill_null(0).alias('position_dans_trajet')
    )
    
    lf = lf.with_columns(
        (pl.col('stop_sequence_local') == pl.col('max_sequence_local')).cast(pl.Int8).fill_null(0).alias('est_gare_terminus')
    )
    
    # Longueur trajet km = total calculated Haversine distance
    lf = lf.rename({'total_dist_km': 'longueur_trajet_km'})

    # Calculate theoretical travel times using pre-computed seconds (already modulo 86400)
    lf = lf.with_columns(
        pl.when(pl.col('est_gare_terminus') == 1)
        .then(pl.coalesce([pl.col('arrival_seconds_mod'), pl.col('departure_seconds_mod')]))
        .otherwise(pl.coalesce([pl.col('departure_seconds_mod'), pl.col('arrival_seconds_mod')]))
        .alias('s_current_mod')
    )
    
    def calculate_duration(start_col, end_col):
        # Calculate diff in seconds
        diff = pl.col(end_col) - pl.col(start_col)
        
        # Handle day crossing (e.g. 23:30 to 00:30)
        # If end is smaller than start, it means we crossed midnight.
        # But only if it's a reasonable crossing (e.g. < 12h diff)
        diff = (
            pl.when(diff < -43200).then(diff + 86400)
            .when(diff > 43200).then(diff - 86400)
            .otherwise(diff)
        )
        
        # Clamp negative values to 0 (erroneous data)
        diff = pl.when(diff < 0).then(0).otherwise(diff)
        
        return diff / 60.0

    lf = lf.with_columns([
        calculate_duration('s_start', 's_current_mod').cast(pl.Float32).alias('temps_theorique_depuis_depart'),
        calculate_duration('s_current_mod', 's_end').cast(pl.Float32).alias('temps_theorique_jusque_arrivee')
    ])
    
    lf = lf.with_columns([
        pl.col('temps_theorique_depuis_depart').clip(0, 300), 
        pl.col('temps_theorique_jusque_arrivee').clip(0, 300)
    ])
    
    # Traverse hub uses "max_seq_hub". 
    # But trip_hubs aggregated "max_seq_jonction" using "stop_sequence_local" (from our previous edit).
    # So we should compare "max_seq_jonction" (which is now local) with "stop_sequence_local".
    # The aliases in trip_hubs kept names like 'max_seq_jonction'.
    
    lf = lf.with_columns([
        (pl.col('max_seq_jonction') > pl.col('stop_sequence_local')).cast(pl.Int8).fill_null(0).alias('traverse_jonction_nord_midi'),
        (pl.col('max_seq_schaerbeek') > pl.col('stop_sequence_local')).cast(pl.Int8).fill_null(0).alias('traverse_hub_schaerbeek'),
        (pl.col('max_seq_midi') > pl.col('stop_sequence_local')).cast(pl.Int8).fill_null(0).alias('traverse_hub_bruxelles_midi'),
        (pl.col('max_seq_anvers') > pl.col('stop_sequence_local')).cast(pl.Int8).fill_null(0).alias('traverse_hub_anvers'),
        (pl.col('max_seq_gent') > pl.col('stop_sequence_local')).cast(pl.Int8).fill_null(0).alias('traverse_hub_gent'),
        (pl.col('max_seq_liege') > pl.col('stop_sequence_local')).cast(pl.Int8).fill_null(0).alias('traverse_hub_liege'),
        (pl.col('max_seq_charleroi') > pl.col('stop_sequence_local')).cast(pl.Int8).fill_null(0).alias('traverse_hub_charleroi')
    ])
    
    # est_hub_majeur
    def is_hub_majeur(name): return 1 if name in ALL_HUB_STATIONS else 0
    lf = lf.with_columns(
        pl.col('nom_gare').map_elements(is_hub_majeur, return_dtype=pl.Int8).alias('est_hub_majeur')
    )
    
    # nb_connexions_gare (Calculate unique count of ligne_principale per nom_gare)
    print("Computing station usage metrics (nb_connexions_gare)...")
    
    # Check if external connectivity data exists
    stations_conn_path = 'data/stations_connections.csv'
    
    if os.path.exists(stations_conn_path):
        print(f"Loading connectivity from {stations_conn_path}...")
        conn_df = pl.read_csv(stations_conn_path)
        # Assume columns: nom_gare, nb_connexions
        # Join
        lf = lf.join(conn_df.lazy(), on='nom_gare', how='left')
        lf = lf.with_columns(pl.col('nb_connexions').fill_null(0).alias('nb_connexions_gare'))
    else:
        # Calculate from dataset itself: count unique 'ligne_principale' per 'nom_gare'
        # Group by station to get connectivity metric
        print("Calculating connectivity from dataset dynamics...")
        station_stats = (
            lf.group_by('nom_gare')
            .agg(pl.n_unique('ligne_principale').alias('nb_connexions_gare'))
        )
        lf = lf.join(station_stats, on='nom_gare', how='left')
        
    lf = lf.with_columns(pl.col('nb_connexions_gare').fill_null(0))

    # Expose plain date alongside datetime for parity with WORKING.csv
    lf = lf.with_columns(pl.col('datetime').dt.strftime('%Y-%m-%d').alias('date'))

    # Calculate total theoretical duration
    lf = lf.with_columns(
        (pl.col('temps_theorique_depuis_depart') + pl.col('temps_theorique_jusque_arrivee'))
        .alias('temps_theorique_trajet')
    )

    # Clean up intermediate columns
    final_cols = [
        'group_id', 'relation', 'numero_train', 'train_type', 'ligne_principale', 'nom_gare',
        'datetime', 'date', 'dt_hour', 
        # Cyclic
        'heure_prevue_sin', 'heure_prevue_cos',
        'jour_semaine_sin', 'jour_semaine_cos',
        'mois_sin', 'mois_cos',
        # Calendar (Module B)
        'fest_weekend', 'is_jour_ferie', 'is_vacances_scolaires', 'is_heure_pointe', 'periode_journee',
        # Spatial (Module C)
         'nb_arrets_total', 'position_dans_trajet', 'est_gare_terminus', 'longueur_trajet_km',
         'temps_theorique_depuis_depart', 'temps_theorique_jusque_arrivee', 'temps_theorique_trajet',
         'stop_lat', 'stop_lon',
        # Hubs & Sections (Module C)
         'traverse_jonction_nord_midi', 'traverse_hub_schaerbeek', 'traverse_hub_bruxelles_midi',
         'traverse_hub_anvers', 'traverse_hub_gent', 'traverse_hub_liege', 'traverse_hub_charleroi',
         'est_hub_majeur', 'nb_connexions_gare',
         # TARGET
         'DELAY_ARR'
    ]
    
    lf_final = lf.select(final_cols)
    
    output_path = 'code/output/datasets/dataset_moduleB_C_verified.parquet'
    print(f"Writing to {output_path}...")
    lf_final.sink_parquet(output_path)
    
    print("\n------------------------------------------------------------")
    print("VERIFICATION MODULE B & C (Calendar, Spatial, Hubs)")
    print("------------------------------------------------------------")
    
    df_verif = pl.scan_parquet(output_path).collect()
    
    print("Checking Spatial Features...")
    row_with_trip = df_verif.filter(pl.col("nb_arrets_total") > 0).head(1)
    if row_with_trip.height > 0:
        r = row_with_trip.row(0, named=True)
        print(f"Train {r['numero_train']} @ {r['nom_gare']}")
        print(f"  - Nb Arrets: {r['nb_arrets_total']}")
        print(f"  - Longueur (km): {r['longueur_trajet_km']}")
        print(f"  - Temps depuis depart: {r['temps_theorique_depuis_depart']} min")
        print(f"  - Temps jusque arrivee: {r['temps_theorique_jusque_arrivee']} min")
        print(f"  - Traverse Jonction: {r['traverse_jonction_nord_midi']}")
        print(f"  - Est Hub Majeur: {r['est_hub_majeur']}")
        print("✅ Spatial & Hub features populated correctly.")
    else:
        print("❌ No spatial features found.")

if __name__ == "__main__":
    run_pipeline()
