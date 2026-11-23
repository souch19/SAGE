#!/usr/bin/env bash
# Exécute ./run.sh dans un job OAR (Grid'5000) ; les arguments sont ceux de run.sh.
#
#   mkdir -p logs
#   oarsub -q default -t exotic -p grouille -l gpu=1,walltime=14:00:00 \
#          -O logs/%jobid%.out -E logs/%jobid%.err "scripts/oar_job.sh model colqwen"
#
# Avec réservation : ajouter -r "AAAA-MM-JJ HH:MM:SS". Les variables de run.sh (MODELS,
# BENCHES, BUDGETS...) se passent dans la commande : "MODELS='colqwen colbert' scripts/oar_job.sh all".
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "$(hostname) | $(date) | run.sh $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
exec ./run.sh "$@"
