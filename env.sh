# shellcheck shell=bash
# Chemins par défaut (run.sh le source). Toute variable déjà définie est conservée.

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SAGE_ARTIFACTS="${SAGE_ARTIFACTS:-$CODE/artifacts}"   # embeddings, régresseurs, résultats
export SAGE_DATA="${SAGE_DATA:-$HOME/STRIDE/data_10k}"       # pool d'entraînement 10k
export SAGE_REPORTS="${SAGE_REPORTS:-$CODE/reports}"         # tableaux et mesures de coût
export OMNI_COL_PRESS="${OMNI_COL_PRESS:-$HOME/omni-col-press}"

# un venv par famille d'encodeurs (versions de transformers incompatibles)
export VENV_COLPALI="${VENV_COLPALI:-$HOME/venv-mm}"         # colpali, colqwen, colnomic
export VENV_JINA="${VENV_JINA:-$HOME/venv-jina}"             # jinav4
export VENV_OMNI="${VENV_OMNI:-$HOME/venv-stride}"           # colbert, AGC (omni-col-press)
