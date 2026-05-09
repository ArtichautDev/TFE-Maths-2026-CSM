import json
import argparse
import os
import yaml
import polars as pl
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="Split dataset_final.parquet into train/val/test for TFT.")
    p.add_argument("--config", default="PyTorch/config.yaml", help="Path to YAML config")
    return p.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def split_exprs(train_end_str, val_end_str):
    # Dates are inclusive; datetimes assumed naive (Europe/Brussels in upstream)
    train_end = datetime.fromisoformat(train_end_str)
    val_end = datetime.fromisoformat(val_end_str)
    dt_col = pl.col("datetime")
    return (
        dt_col <= pl.lit(train_end),
        (dt_col > pl.lit(train_end)) & (dt_col <= pl.lit(val_end)),
        dt_col > pl.lit(val_end),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    source = cfg["source_parquet"]
    out_dir = cfg["output_dir"]
    target_col = cfg.get("target_col", "target")
    aux_target_col = cfg.get("aux_target_col")
    include = cfg.get("feature_include", []) or None
    exclude = set(cfg.get("feature_exclude", []))
    row_group_size = int(cfg.get("row_group_size", 200000))

    ensure_dir(out_dir)

    lf = pl.scan_parquet(source)

    # Select columns
    all_cols = lf.collect_schema().names()
    keep_cols = include if include else all_cols
    # Always keep target(s)
    for col in [target_col, aux_target_col]:
        if col and col not in keep_cols:
            keep_cols.append(col)
            
    # CRITIQUE: on a besoin des colonnes datetime et group_id pour le pré-traitement Polars.
    # On les retirera (ou on laissera PyTorch/Arrow les ignorer) dynamiquement plus tard.
    essential_polars_cols = ["datetime", "group_id"]
    for required_col in essential_polars_cols:
        if required_col not in keep_cols:
            keep_cols.append(required_col)
            
    # Exclusion décalée APRES encodage

    lf = lf.select([c for c in keep_cols if c in all_cols])
    
    # ---------------- RECALCUL DU TARGET EN MINUTES ----------------
    # Correction cruciale : DELAY_ARR est en secondes dans le fichier source, 
    # mais les features historiques (retard_j_x) sont en minutes.
    # On convertit DELAY_ARR en minutes puis on applique le clipping [-20 min, 240 min]
    print("Correction des unités : Conversion du target en MINUTES et clipping [-20, 240]...")
    lf = lf.with_columns(
        (pl.col("DELAY_ARR") / 60.0).clip(-20.0, 240.0).cast(pl.Float32).alias(target_col)
    )
    # -------------------------------------------------------------
    
    # ---------------- ENCODAGE CATEGORIEL ----------------
    cats_to_encode = cfg.get("categorical_to_encode", [])
    mappings_export = {}
    
    if cats_to_encode:
        print(f"Encodage categorical (Label Encoding pour PyTorch) de : {cats_to_encode}")
        
        for c_col in cats_to_encode:
            if c_col in all_cols:
                print(f" > Traitement de {c_col}...")
                unique_vals = pl.scan_parquet(source).select(c_col).drop_nulls().unique().collect()[c_col].to_list()
                mapping = {str(val): float(idx) for idx, val in enumerate(unique_vals)}
                
                # Sauvegarde pour le JSON d'export
                mappings_export[c_col] = {str(val): int(idx) for idx, val in enumerate(unique_vals)}
                
                lf = lf.with_columns(
                    pl.col(c_col).cast(pl.String).replace_strict(mapping, default=0.0).cast(pl.Float32).alias(f"{c_col}_idx")
                )
                
        # Sauvegarde des dictionnaires sur disque pour l'inférence en production !
        mapping_path = os.path.join(out_dir, "categorical_mappings.json")
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mappings_export, f, indent=4, ensure_ascii=False)
        print(f"✅ Dictionnaires d'encodage sauvegardés dans: {mapping_path}")
    # -------------------------------------------------------

    # APRES encodage (car l'encodage créé "c_col_idx", donc on peut jeter "c_col" original string)
    final_cols_to_keep = []
    for c in keep_cols:
        # On force la conservation de target_col et aux_target_col
        if c not in exclude or c in [target_col, aux_target_col]:
            final_cols_to_keep.append(c)
        elif c in essential_polars_cols:
            final_cols_to_keep.append(c)
            
    # Ajouter aussi les nouvelles colonnes d'index qu'on vient de créer au schéma à garder
    if cats_to_encode:
        for c_col in cats_to_encode:
            if c_col in all_cols:
                final_cols_to_keep.append(f"{c_col}_idx")

    lf = lf.select(final_cols_to_keep)

    # Intelligemment : Le TFT a besoin de séquences temporelles parfaites
    print("Sécurisation du dataset chronologique (time_idx + déduplication) pour le TFT...")
    
    # 1. Ajout de time_idx = jour absolu (epoch Unix) => identifiant temporel unique
    lf = lf.with_columns(
        time_idx=pl.col("datetime").dt.date().cast(pl.Int64)
    )
    
    # 2. On trie par group_id (identifiant de la série) et datetime
    lf = lf.sort(["group_id", "datetime"])
    
    # 3. On déduplique: une gare ne peut voir le MEME NUMERO DE TRAIN qu'une fois par jour
    # s'il y a 2 capteurs/logs, on garde la dernière valeur de la journée
    lf = lf.unique(subset=["group_id", "time_idx"], keep="last", maintain_order=True)

    # Build split filters
    train_expr, val_expr, test_expr = split_exprs(cfg["train_end"], cfg["val_end"])

    # ---------------- IMPUTATION & Z-SCORE SCALING STRICTEMENT SUR LE TRAIN ----------------
    print("Identification des colonnes continues pour le Scaling et Imputation...")
    final_schema = lf.collect_schema()
    continuous_cols = []
    
    # Règle : tout ce qui n'est pas metadata ou target, n'est pas string/catégorie, et qui ne termine pas par _idx
    exclusions_scaling = {target_col, aux_target_col, "time_idx", "group_id", "datetime", "date", "date_dt"}
    for col_name, dtype in final_schema.items():
        if col_name not in exclusions_scaling and not col_name.endswith("_idx"):
            if dtype.is_numeric():
                continuous_cols.append(col_name)

    print(f"Colonnes continues détectées ({len(continuous_cols)}) : {continuous_cols[:5]}...")

    # 1. Imputation avec 0.0 pour TOUTES les données en mode streaming (Lazy)
    lf = lf.with_columns([
        pl.col(c).fill_null(0.0) for c in continuous_cols
    ])

    print("Calcul des statistiques (Z-score) STRICTEMENT sur l'ensemble d'entraînement (Data Leakage prevention)...")
    # On isole temporairement un collect() hyper-optimisé juste sur la portion "train"
    train_stats = pl.scan_parquet(source).filter(train_expr)
    
    # On doit sélectionner et imputer ces colonnes avant d'en faire la moyenne
    # L'opération peut être lourde, on utilise pl.Expr avec collect() unique
    stats_expr = []
    for c in continuous_cols:
        col_imputed = pl.col(c).fill_null(0.0).cast(pl.Float64)
        stats_expr.append(col_imputed.mean().alias(f"{c}_avg"))
        stats_expr.append(col_imputed.std().alias(f"{c}_std"))
        
    stats_dict = train_stats.select(stats_expr).collect().to_dicts()[0]

    # Sauvegarde des paramètres dans un JSON pour l'inférence
    scaler_mappings = {}
    
    # Construction de l'étape de Z-Score Scaling directement dans le LazyFrame principal
    scaling_exprs = []
    for c in continuous_cols:
        mean_val = float(stats_dict[f"{c}_avg"]) if stats_dict[f"{c}_avg"] is not None else 0.0
        std_val = float(stats_dict[f"{c}_std"]) if stats_dict[f"{c}_std"] is not None else 1.0
        if std_val == 0.0:
            std_val = 1.0  # Prévention division par zéro
            
        scaler_mappings[c] = {"mean": mean_val, "std": std_val}
        
        # Scaling dynamique: (X - mean) / std
        scaling_exprs.append(
            ((pl.col(c) - pl.lit(mean_val)) / pl.lit(std_val)).cast(pl.Float32)
        )
    
    # On fusionne l'opération mathématique dans le LazyFrame global !
    lf = lf.with_columns(scaling_exprs)
    
    scaler_path = os.path.join(out_dir, "scaler_mappings.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_mappings, f, indent=4, ensure_ascii=False)
    print(f"✅ Dictionnaires des Scalers sauvegardés dans : {scaler_path}")
    # -----------------------------------------------------------------------------------------

    splits = {
        "train": train_expr,
        "val": val_expr,
        "test": test_expr,
    }

    for name, expr in splits.items():
        out_path = os.path.join(out_dir, f"{name}.parquet")
        print(f"Writing {name} -> {out_path}")
        lf.filter(expr).sink_parquet(
            out_path,
            compression="zstd",
            compression_level=8,
            statistics=True,
            row_group_size=row_group_size,
        )

    print("Done.")


if __name__ == "__main__":
    main()
