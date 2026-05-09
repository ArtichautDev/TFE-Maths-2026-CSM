#!/usr/bin/env python3
"""
Orchestrateur du pipeline de génération du dataset.

Lance les modules A → I dans l'ordre. Chaque module lit le parquet du module
précédent et écrit le sien dans OUTPUT_DIR.

Usage:
    python rebuild_all.py [--output-dir <path>]
"""
import subprocess
import sys
import time
import os
import argparse

MODULES_DIR = os.path.join(os.path.dirname(__file__), "modules")
LOG_FILE    = "rebuild_pipeline.log"

STEPS = [
    {
        "name":   "Module A — Temporal & Static",
        "script": "module_A_temporal.py",
        "desc":   "Features temporelles cycliques (heure, jour, mois sin/cos) + features statiques de base."
    },
    {
        "name":   "Module B/C — Calendar & Spatial",
        "script": "module_B_calendar_spatial.py",
        "desc":   "Jours fériés, vacances scolaires, données GTFS (hubs, connexions, position dans trajet)."
    },
    {
        "name":   "Module D — Density",
        "script": "module_D_density.py",
        "desc":   "Densité de trains prévue dans les hubs majeurs (Bruxelles, Anvers, Gand, Liège...)."
    },
    {
        "name":   "Module E — Travaux",
        "script": "module_E_travaux.py",
        "desc":   "Impact des travaux Infrabel : nombre actifs, niveau d'impact, distance, type."
    },
    {
        "name":   "Module G — Météo",
        "script": "module_G_meteo.py",
        "desc":   "Conditions météo SYNOP à la gare : température, précipitations, vent, conditions extrêmes."
    },
    {
        "name":   "Module H — Historique",
        "script": "module_H_historique.py",
        "desc":   "Stats de ponctualité historiques : retards J-1 à J-21, moyennes glissantes, fiabilité."
    },
    {
        "name":   "Module I — Derived",
        "script": "module_I_derived.py",
        "desc":   "Feature engineering final : scores dérivés, normalisation, nettoyage → dataset_final.parquet."
    },
]


def log(msg):
    ts   = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_step(step, output_dir):
    script_path = os.path.join(MODULES_DIR, step["script"])
    if not os.path.exists(script_path):
        log(f"ERREUR script introuvable : {script_path}")
        return False

    log(f"Démarrage {step['name']}…")
    log(f"  {step['desc']}")

    env = os.environ.copy()
    if output_dir:
        env["SNCB_OUTPUT_DIR"] = output_dir

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True, capture_output=True, text=True, env=env,
        )
        elapsed = time.time() - t0
        tail = "\n    ".join(result.stdout.strip().split("\n")[-10:])
        log(f"OK {step['name']} en {elapsed:.1f}s")
        log(f"  Dernière sortie :\n    {tail}")
        return True
    except subprocess.CalledProcessError as e:
        log(f"ECHEC {step['name']} en {time.time()-t0:.1f}s")
        log(f"  Erreur :\n{e.stderr[-2000:]}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None,
                        help="Répertoire de sortie pour les parquets intermédiaires")
    args = parser.parse_args()

    log("=== Pipeline SNCB — démarrage ===")
    for step in STEPS:
        if not run_step(step, args.output_dir):
            log("Pipeline interrompu.")
            sys.exit(1)
    log(f"=== Pipeline terminé ({len(STEPS)}/{len(STEPS)} modules) ===")


if __name__ == "__main__":
    main()
