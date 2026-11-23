#!/usr/bin/env bash
# Point d'entrée du pipeline SAGE-I.
#
#   ./run.sh check                        environnements, chemins, GPU
#   ./run.sh data [options]               pool d'entraînement 10k                       (01)
#   ./run.sh train  <modèle>              encodage du pool, cibles, régresseur, profils (02, 03)
#   ./run.sh encode <modèle>              encodage des bancs d'évaluation               (02)
#   ./run.sh eval   <modèle>              opérateurs du papier, tous budgets            (04)
#   ./run.sh model  <modèle>              train + encode + eval
#   ./run.sh baselines                    AGC sur ColBERT (le reste est commenté)       (04)
#   ./run.sh all                          tous les modèles, baselines puis report
#   ./run.sh report                       tableaux, contrôle, intervalles de confiance  (05)
#   ./run.sh seeds <modèle> <graine>...   variabilité selon la graine (ViDoRe v2)
#   ./run.sh cost  <modèle> [split]       coûts d'indexation et de recherche            (05)
#   ./run.sh smoke <modèle>               tout le pipeline sur quelques pages (test)
#
# Modèles : colpali colqwen colnomic jinav4 colbert. Variables (défaut) : MODELS (les cinq),
# BENCHES (vidore_v2 jinavdr), BUDGETS (05pct ... 30pct 64abs), KMEANS (auto),
# CLEANUP=1 efface les shards d'entraînement après le régresseur, CARBON=1 mesure le CO2.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source env.sh

MODELS=${MODELS:-"colpali colqwen colnomic jinav4 colbert"}
BENCHES=${BENCHES:-"vidore_v2 jinavdr"}
BUDGETS=${BUDGETS:-"05pct 10pct 15pct 20pct 25pct 30pct 64abs"}
KMEANS=${KMEANS:-auto}
read -ra BUDGET_LIST <<< "$BUDGETS"

py() { echo; echo ">>> python $*"; python "$@"; }
die() { echo "[!] $*" >&2; exit 1; }
need_model() { [[ " colpali colqwen colnomic jinav4 colbert " == *" ${1:-?} "* ]] || die "modèle inconnu '${1:-}'"; }
use_env() {
  local v=$VENV_COLPALI
  case $1 in colbert) v=$VENV_OMNI ;; jinav4) v=$VENV_JINA ;; esac
  if [ -f "$v/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$v/bin/activate"
  else
    echo "[!] venv introuvable : $v ; python utilisé : $(command -v python)" >&2
  fi
}

check() {
  echo "code           : $PWD"
  echo "SAGE_ARTIFACTS : $SAGE_ARTIFACTS"
  echo "SAGE_DATA      : $SAGE_DATA"
  if [ -d "$OMNI_COL_PRESS/.git" ]; then
    echo "omni-col-press : $(git -C "$OMNI_COL_PRESS" rev-parse --short HEAD) (attendu 4a55967)"
  else
    echo "omni-col-press : absent de $OMNI_COL_PRESS (nécessaire pour colbert et AGC)"
  fi
  for fam in colqwen jinav4 colbert; do
    (
      use_env "$fam"
      python - "$fam" <<'PY'
import importlib, sys
print(f"\n[venv {sys.argv[1]}] {sys.executable}")
for m in ("torch", "transformers", "colpali_engine", "sklearn", "scipy", "datasets", "qwen_vl_utils"):
    try:
        print(f"  {m:<15} {getattr(importlib.import_module(m), '__version__', 'ok')}")
    except Exception:
        print(f"  {m:<15} absent")
try:
    import torch
    print(f"  cuda           {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'non'}")
except ImportError:
    pass
PY
    ) || true
  done
  echo
  if [ -f "$SAGE_DATA/s00_top10k_pairs_indexed.parquet" ]; then
    ( use_env colqwen; python 01_data/build_train10k.py --out "$SAGE_DATA" --check ) || true
  else
    echo "pool 10k absent de $SAGE_DATA (./run.sh data)"
  fi
}

train() {
  use_env "$1"
  py 02_encode/encode_train.py --model "$1"
  py 03_salience/attribution.py --model "$1"
  py 03_salience/train_regressor.py --model "$1" ${CLEANUP:+--cleanup}
  py 03_salience/profiles.py --model "$1"
}

encode() {
  use_env "$1"
  for b in $BENCHES; do py 02_encode/encode_eval.py --model "$1" --bench "$b"; done
}

evaluate() {
  use_env "$1"
  for b in $BENCHES; do
    py 04_evaluate/evaluate.py --model "$1" --bench "$b" --methods paper \
       --budgets "${BUDGET_LIST[@]}" --kmeans "$KMEANS"
  done
}

baselines() {
  use_env colbert
  for b in $BENCHES; do
    py 04_evaluate/agc.py --bench "$b" --budgets "${BUDGET_LIST[@]}"
  done
  # Hors papier (baselines exploratoires) : décommenter pour les lancer.
  # for b in $BENCHES; do
  #   py 04_evaluate/evaluate.py --model colbert --bench "$b" --methods kmeans --budgets "${BUDGET_LIST[@]}" --kmeans "$KMEANS"
  # done
  # py 04_evaluate/evaluate.py --model colbert --methods ablation trainfree adaptive --budgets 05pct 64abs
  # py 04_evaluate/query_conditioned.py --model colbert --budgets 05pct 64abs
}

report() {
  use_env colqwen
  py 05_analysis/aggregate.py --out "$SAGE_REPORTS"
  py 05_analysis/check.py
  for m in $MODELS; do
    if [ -d "$SAGE_ARTIFACTS/$m/results/vidore_v2" ]; then
      py 05_analysis/significance.py --model "$m" --budgets 05pct 64abs --out "$SAGE_REPORTS/ic_$m.md"
    fi
  done
}

seeds() {
  local m=$1; shift
  [ $# -gt 0 ] || die "usage : ./run.sh seeds <modèle> <graine>..."
  use_env "$m"
  for s in "$@"; do
    py 03_salience/train_regressor.py --model "$m" --seed "$s"
    py 03_salience/profiles.py --model "$m" --seed "$s"
    py 04_evaluate/evaluate.py --model "$m" --methods paper --budgets "${BUDGET_LIST[@]}" --kmeans "$KMEANS" --seed "$s"
  done
  mkdir -p "$SAGE_REPORTS"
  py 05_analysis/seed_variance.py --model "$m" --seeds 42 "$@" | tee "$SAGE_REPORTS/seeds_$m.md"
}

cost() {
  local m=$1 split=${2:-econ}
  use_env "$m"
  py 05_analysis/cost.py --model "$m" --split "$split" --budget 64abs --out "$SAGE_REPORTS" ${CARBON:+--carbon}
  py 05_analysis/kmeans_backends.py --model "$m" --split "$split" --budgets 64abs 05pct
  if [ "$m" = colbert ]; then
    py 05_analysis/cost_encode.py --split "$split" --out "$SAGE_REPORTS" ${CARBON:+--carbon}
  fi
}

smoke() {
  local m=$1
  export SAGE_ARTIFACTS=${SMOKE_DIR:-${TMPDIR:-/tmp}/sage_smoke_${USER:-user}}
  export SAGE_REPORTS=$SAGE_ARTIFACTS/reports
  echo "artefacts de test : $SAGE_ARTIFACTS"
  use_env "$m"
  py 02_encode/encode_train.py --model "$m" --limit 64
  py 03_salience/attribution.py --model "$m"
  py 03_salience/train_regressor.py --model "$m"
  py 03_salience/profiles.py --model "$m"
  py 02_encode/encode_eval.py --model "$m" --bench vidore_v2 --splits econ --limit 60
  py 02_encode/encode_eval.py --model "$m" --bench jinavdr --splits slide --limit 60
  py 04_evaluate/evaluate.py --model "$m" --splits econ --budgets 05pct 64abs
  py 04_evaluate/evaluate.py --model "$m" --bench jinavdr --splits slide --budgets 05pct 64abs
  # hors papier : py 04_evaluate/evaluate.py --model "$m" --splits econ --methods all --budgets 05pct 64abs
  # hors papier : py 04_evaluate/query_conditioned.py --model "$m" --splits econ
  if [ "$m" = colbert ]; then py 04_evaluate/agc.py --splits econ --budgets 05pct 64abs; fi
  py 05_analysis/aggregate.py --out "$SAGE_REPORTS"
  py 05_analysis/significance.py --model "$m" --budgets 64abs
  py 05_analysis/check.py
}

cmd=${1:-help}
shift || true
case "$cmd" in
  check)     check ;;
  data)      py 01_data/build_train10k.py --out "$SAGE_DATA" "$@" ;;
  train)     need_model "$1"; train "$1" ;;
  encode)    need_model "$1"; encode "$1" ;;
  eval)      need_model "$1"; evaluate "$1" ;;
  model)     need_model "$1"; train "$1"; encode "$1"; evaluate "$1" ;;
  baselines) baselines ;;
  all)       for m in $MODELS; do need_model "$m"; train "$m"; encode "$m"; evaluate "$m"; done
             if [[ " $MODELS " == *" colbert "* ]]; then baselines; fi
             report ;;
  report)    report ;;
  seeds)     need_model "$1"; seeds "$@" ;;
  cost)      need_model "$1"; cost "$@" ;;
  smoke)     need_model "$1"; smoke "$1" ;;
  help|-h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//' ;;
  *)         die "commande inconnue '$cmd' (./run.sh help)" ;;
esac
