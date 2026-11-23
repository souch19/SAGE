#!/usr/bin/env bash
# Range les résultats de l'ancienne disposition dans artifacts/{modèle}/results :
#   agc/results/...         -> colbert/results/.../agc.pkl
#   kmeans/results/...      -> colbert/results/.../kmeans.pkl
#   {modèle}_xp/results/... -> {modèle}/results/.../XP_wkmeans.pkl
# Les native.pkl de ces dossiers (recalculés, identiques) ne sont pas déplacés ; aucun fichier
# existant n'est écrasé. Les dossiers vidés peuvent ensuite être supprimés à la main.
#
#   SAGE_ARTIFACTS=~/STRIDE/artifacts scripts/migrate_artifacts.sh
set -eo pipefail
A=${SAGE_ARTIFACTS:?définir SAGE_ARTIFACTS}

move() {  # move <ancien dossier> <modèle>
  local src=$1/results dst=$A/$2/results n=0
  [ -d "$src" ] || return 0
  while IFS= read -r f; do
    if [ ! -e "$dst/$f" ]; then
      mkdir -p "$dst/$(dirname "$f")"
      mv "$src/$f" "$dst/$f"
      n=$((n + 1))
    fi
  done < <(cd "$src" && find . -name '*.pkl' ! -name native.pkl | sed 's|^\./||')
  echo "$1 -> $2 : $n fichiers"
}

move "$A/agc" colbert
move "$A/kmeans" colbert
for d in "$A"/*_xp; do
  [ -d "$d" ] && move "$d" "$(basename "${d%_xp}")"
done
