#!/usr/bin/env bash
# À lancer sur la machine locale : rapatrie depuis Grid'5000 les résultats, régresseurs,
# profils et rapports (pas les embeddings), dans ./artifacts et ./reports.
#
#   G5K_USER=login REMOTE=chemin/du/code scripts/pull_results.sh
#
# REMOTE est relatif au home du frontal (défaut : sage/code) ; SITE (défaut : nancy) ;
# REMOTE_ARTIFACTS si $SAGE_ARTIFACTS pointe ailleurs que $REMOTE/artifacts sur le cluster.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
U=${G5K_USER:-$USER}
SITE=${SITE:-nancy}
REMOTE=${REMOTE:-sage/code}
SRC="$U@$SITE:${REMOTE_ARTIFACTS:-$REMOTE/artifacts}/"
SSH="ssh -J $U@access.grid5000.fr"

mkdir -p artifacts reports
rsync -avzm -e "$SSH" \
  --include='*/' --include='results*/**' --include='train/regressor*' --include='train/profiles64*' \
  --exclude='*' "$SRC" artifacts/
rsync -avz -e "$SSH" "$U@$SITE:$REMOTE/reports/" reports/ || true
echo "$(find artifacts -path '*/results*' -name '*.pkl' | wc -l) fichiers de résultats dans ./artifacts"
