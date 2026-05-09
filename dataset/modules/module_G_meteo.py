import os
import json
import re
import unicodedata
import polars as pl

INPUT_PATH = 'code/output/datasets/dataset_moduleE_verified.parquet'
OUTPUT_PATH = 'code/output/datasets/dataset_moduleG_verified.parquet'
METEO_PATH = os.environ.get('METEO_CSV', '/root/SNCB/test_data/weather/synop_2022_2025.csv')
MAPPING_PATH = os.environ.get('METEO_MAPPING', 'data/mappings/gare_station_meteo.json')
DEFAULT_STATION = '6447'  # Uccle (Node default)

EXTREME_CONDITIONS = {
    'WIND_SPEED_MS': 20.0,
    'WIND_PEAK_MS': 25.0,
    'PRECIPITATION_MM': 10.0,
    'TEMP_LOW': -5.0,
    'TEMP_HIGH': 35.0,
    'HUMIDITY_HIGH': 95.0,
    'CLOUDINESS_HIGH': 8.0,
}


def normalize_name(name: str):
    if not name:
        return None
    # Remove accents and keep alnum, space, dash
    n = unicodedata.normalize('NFD', str(name))
    n = ''.join(ch for ch in n if unicodedata.category(ch) != 'Mn')
    n = n.upper()
    n = re.sub(r'[^A-Z0-9\s-]', '', n)
    n = re.sub(r'\s+', ' ', n).strip()
    return n if n else None


def load_mapping(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
    mapping = data.get('mapping', {})
    # Keys already normalized in file
    return {k: v.get('station_code') for k, v in mapping.items()}


def build_meteo_lookup():
    if not os.path.exists(METEO_PATH):
        raise FileNotFoundError(f'Meteo file not found: {METEO_PATH}')

    meteo = pl.scan_csv(METEO_PATH, ignore_errors=True)

    meteo = meteo.with_columns([
        pl.col('code').cast(pl.Utf8).alias('station_code'),
        pl.col('timestamp').str.slice(0, 10).alias('date'),
        pl.col('timestamp').str.slice(11, 2).cast(pl.Int8).alias('hour'),
        pl.col('temp').cast(pl.Float32).alias('temperature'),
        pl.col('precip_quantity').cast(pl.Float32).alias('precipitations'),
        pl.col('wind_speed').cast(pl.Float32).alias('wind_speed'),
        pl.col('wind_peak_speed').cast(pl.Float32).alias('wind_peak'),
        pl.col('humidity_relative').cast(pl.Float32).alias('humidity'),
        pl.col('cloudiness').cast(pl.Float32).alias('cloudiness'),
    ]).select([
        'station_code', 'date', 'hour', 'temperature', 'precipitations',
        'wind_speed', 'wind_peak', 'humidity', 'cloudiness'
    ])

    # Expand hours with ±3h fallback and keep delta for priority
    frames = [
        meteo.with_columns([
            pl.lit(0).alias('delta'),
            pl.col('hour').alias('hour_target')
        ])
    ]
    for d in (1, 2, 3):
        frames.append(
            meteo.with_columns([
                pl.lit(d).alias('delta'),
                ((pl.col('hour') + d) % 24).alias('hour_target')
            ])
        )
        frames.append(
            meteo.with_columns([
                pl.lit(d).alias('delta'),
                ((pl.col('hour') - d + 24) % 24).alias('hour_target')
            ])
        )

    expanded = pl.concat(frames)

    grouped = expanded.group_by(['station_code', 'date', 'hour_target']).agg([
        pl.col('temperature').sort_by('delta').first().alias('temperature'),
        pl.col('precipitations').sort_by('delta').first().alias('precipitations'),
        pl.col('wind_speed').sort_by('delta').first().alias('wind_speed'),
        pl.col('wind_peak').sort_by('delta').first().alias('wind_peak'),
        pl.col('humidity').sort_by('delta').first().alias('humidity'),
        pl.col('cloudiness').sort_by('delta').first().alias('cloudiness'),
        pl.col('delta').min().alias('meteo_hour_delta')
    ])

    return grouped


def run():
    print('Loading base dataset (Module E verified)...')
    base = pl.scan_parquet(INPUT_PATH)
    base_cols = base.collect_schema().names()

    print('Loading gare->station mapping...')
    mapping_dict = load_mapping(MAPPING_PATH)
    mapping_df = pl.DataFrame({
        'gare_norm': list(mapping_dict.keys()),
        'station_code_map': list(mapping_dict.values())
    })

    print('Preparing meteo lookup...')
    meteo_grouped = build_meteo_lookup()

    # Build normalized gare and station code
    lf = base.with_columns([
        pl.col('nom_gare').map_elements(normalize_name, return_dtype=pl.Utf8).alias('gare_norm')
    ])

    lf = lf.join(mapping_df.lazy(), on='gare_norm', how='left')

    lf = lf.with_columns([
        pl.col('station_code_map').fill_null(DEFAULT_STATION).alias('station_code'),
        pl.col('datetime').dt.offset_by('-1d').dt.strftime('%Y-%m-%d').alias('date_j1'),
    ])

    # Main meteo join
    meteo_main = meteo_grouped.rename({
        'hour_target': 'hour_target_main',
        'temperature': 'temp_main',
        'precipitations': 'precip_main',
        'wind_speed': 'wind_main',
        'wind_peak': 'wind_peak_main',
        'humidity': 'hum_main',
        'cloudiness': 'cloud_main',
        'meteo_hour_delta': 'meteo_delta_main'
    })

    lf = lf.join(
        meteo_main,
        left_on=['station_code', 'date', 'dt_hour'],
        right_on=['station_code', 'date', 'hour_target_main'],
        how='left'
    )

    # Default station fallback (Uccle)
    meteo_default = meteo_grouped.filter(pl.col('station_code') == DEFAULT_STATION).select([
        pl.col('date'),
        pl.col('hour_target').alias('hour_target_def'),
        pl.col('temperature').alias('temp_def'),
        pl.col('precipitations').alias('precip_def'),
        pl.col('wind_speed').alias('wind_def'),
        pl.col('wind_peak').alias('wind_peak_def'),
        pl.col('humidity').alias('hum_def'),
        pl.col('cloudiness').alias('cloud_def'),
        pl.col('meteo_hour_delta').alias('meteo_delta_def')
    ])

    lf = lf.join(
        meteo_default,
        left_on=['date', 'dt_hour'],
        right_on=['date', 'hour_target_def'],
        how='left'
    )

    # J-1 joins
    meteo_main_j1 = meteo_grouped.rename({
        'hour_target': 'hour_target_j1',
        'temperature': 'temp_j1_main',
        'precipitations': 'precip_j1_main',
        'wind_speed': 'wind_j1_main',
        'wind_peak': 'wind_peak_j1_main',
        'humidity': 'hum_j1_main',
        'cloudiness': 'cloud_j1_main',
        'meteo_hour_delta': 'meteo_delta_j1_main'
    })

    lf = lf.join(
        meteo_main_j1,
        left_on=['station_code', 'date_j1', 'dt_hour'],
        right_on=['station_code', 'date', 'hour_target_j1'],
        how='left'
    )

    meteo_default_j1 = meteo_default.rename({
        'hour_target_def': 'hour_target_j1_def',
        'temp_def': 'temp_j1_def',
        'precip_def': 'precip_j1_def',
        'wind_def': 'wind_j1_def',
        'wind_peak_def': 'wind_peak_j1_def',
        'hum_def': 'hum_j1_def',
        'cloud_def': 'cloud_j1_def',
        'meteo_delta_def': 'meteo_delta_j1_def'
    })

    lf = lf.join(
        meteo_default_j1,
        left_on=['date_j1', 'dt_hour'],
        right_on=['date', 'hour_target_j1_def'],
        how='left'
    )

    # Final feature computations with fallback priority
    # 1. Main station (nearest)
    # 2. Default station (Uccle)
    # 3. Hardcoded safe defaults (to guarantee no nulls)
    
    use_main = (pl.col('temp_main').is_not_null()) & (pl.col('cloud_main').is_not_null())
    use_main_j1 = (pl.col('temp_j1_main').is_not_null()) & (pl.col('cloud_j1_main').is_not_null())

    # Safe Defaults
    # T=10.0, Precip=0.0, Wind=3.5 (modest breeze), WindPeak=5.0, Hum=80.0, Cloud=4.0
    
    temp = pl.when(use_main).then(pl.col('temp_main')).otherwise(pl.col('temp_def')).fill_null(10.0).cast(pl.Float32)
    precip = pl.when(use_main).then(pl.col('precip_main')).otherwise(pl.col('precip_def')).fill_null(0.0).cast(pl.Float32)
    wind = pl.when(use_main).then(pl.col('wind_main')).otherwise(pl.col('wind_def')).fill_null(3.5).cast(pl.Float32)
    wind_peak = pl.when(use_main).then(pl.col('wind_peak_main')).otherwise(pl.col('wind_peak_def')).fill_null(5.0).cast(pl.Float32)
    hum = pl.when(use_main).then(pl.col('hum_main')).otherwise(pl.col('hum_def')).fill_null(80.0).cast(pl.Float32)
    cloud = pl.when(use_main).then(pl.col('cloud_main')).otherwise(pl.col('cloud_def')).fill_null(4.0).cast(pl.Float32)

    temp_j1 = pl.when(use_main_j1).then(pl.col('temp_j1_main')).otherwise(pl.col('temp_j1_def')).fill_null(10.0).cast(pl.Float32)

    cond_expr = (
        (wind_peak > EXTREME_CONDITIONS['WIND_PEAK_MS']) |
        (wind > EXTREME_CONDITIONS['WIND_SPEED_MS']) |
        (precip > EXTREME_CONDITIONS['PRECIPITATION_MM']) |
        (temp < EXTREME_CONDITIONS['TEMP_LOW']) |
        (temp > EXTREME_CONDITIONS['TEMP_HIGH']) |
        ((cloud >= EXTREME_CONDITIONS['CLOUDINESS_HIGH']) & (hum >= EXTREME_CONDITIONS['HUMIDITY_HIGH']))
    ).fill_null(False).cast(pl.Int8)

    # Round to 1 decimal place. We keep as Float64 for cleaner representation (avoid 3.6 becoming 3.59999...)
    lf = lf.with_columns([
        temp.cast(pl.Float64).round(1).alias('meteo_temperature'),
        precip.cast(pl.Float64).round(1).alias('meteo_precipitations'),
        wind.cast(pl.Float64).round(1).alias('meteo_vent_vitesse'),
        cloud.cast(pl.Float64).round(0).alias('meteo_nebulosite'), # Cloudiness is usually integer octas (0-8) or %
        cond_expr.alias('meteo_conditions_extremes'),
        temp_j1.cast(pl.Float64).round(1).alias('meteo_observee_j_1'),
    ])

    # Drop helper columns
    drop_cols = [
        'gare_norm', 'station_code_map', 'station_code', 'date_j1',
        'hour_target_main', 'temp_main', 'precip_main', 'wind_main', 'wind_peak_main', 'hum_main', 'cloud_main', 'meteo_delta_main',
        'hour_target_def', 'temp_def', 'precip_def', 'wind_def', 'wind_peak_def', 'hum_def', 'cloud_def', 'meteo_delta_def',
        'hour_target_j1', 'temp_j1_main', 'precip_j1_main', 'wind_j1_main', 'wind_peak_j1_main', 'hum_j1_main', 'cloud_j1_main', 'meteo_delta_j1_main',
        'hour_target_j1_def', 'temp_j1_def', 'precip_j1_def', 'wind_j1_def', 'wind_peak_j1_def', 'hum_j1_def', 'cloud_j1_def', 'meteo_delta_j1_def'
    ]
    lf = lf.drop([c for c in drop_cols if c in lf.columns])

    keep_cols = base_cols + [
        'meteo_temperature', 'meteo_precipitations', 'meteo_vent_vitesse',
        'meteo_nebulosite', 'meteo_conditions_extremes', 'meteo_observee_j_1'
    ]
    lf_final = lf.select([c for c in keep_cols if c in lf.columns])

    print(f'Writing Module G output to {OUTPUT_PATH}...')
    lf_final.sink_parquet(OUTPUT_PATH)

    sample = pl.scan_parquet(OUTPUT_PATH).filter(pl.col('meteo_temperature').is_not_null()).head(1).collect()
    if sample.height:
        r = sample.row(0, named=True)
        print('Sample meteo features (Raw Values):')
        print(f"  meteo_temperature: {r.get('meteo_temperature')}")
        print(f"  meteo_precipitations: {r.get('meteo_precipitations')}")
        print(f"  meteo_vent_vitesse: {r.get('meteo_vent_vitesse')}")
        print(f"  meteo_nebulosite: {r.get('meteo_nebulosite')}")
        print(f"  meteo_conditions_extremes: {r.get('meteo_conditions_extremes')}")
        print(f"  meteo_observee_j_1: {r.get('meteo_observee_j_1')}")
        print('Sample meteo features (Formatted Output):')
        print(f"  meteo_temperature: {r.get('meteo_temperature'):.1f}")
        print(f"  meteo_precipitations: {r.get('meteo_precipitations'):.1f}")
    else:
        print('No meteo features found in sample; check joins/mapping.')


if __name__ == '__main__':
    run()
