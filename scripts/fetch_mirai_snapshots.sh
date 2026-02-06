#!/usr/bin/env bash
set -euo pipefail

# repo and outputs
REPO="${REPO:-reginabarzilaygroup/Mirai}"
OUT="${OUT:-snapshots}"
TAG="${TAG:-}"                    # e.g., v0.14.1; if empty → latest
mkdir -p "$OUT"/callibrators

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1"; exit 1; }; }
need gh

download_from_release() {
  # patterns match the filenames used in README’s validate command
  #   -- mgh_mammo_MIRAI_Base_May20_2019.p
  #   -- mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020.p
  #   -- callibrators/MIRAI_FULL_PRED_RF.callibrator.p
  local tag_arg=()
  [[ -n "${TAG}" ]] && tag_arg=("${TAG}")
  gh release download "${tag_arg[@]}" -R "$REPO" -D "$OUT" \
    -p 'mgh_mammo_MIRAI_Base*' \
    -p 'mgh_mammo_cancer_MIRAI_Transformer*' \
    -p '*MIRAI_FULL_PRED_RF.callibrator.p'

  # normalize the callibrator location
  if [[ -f "$OUT"/MIRAI_FULL_PRED_RF.callibrator.p ]]; then
    mv "$OUT"/MIRAI_FULL_PRED_RF.callibrator.p "$OUT"/callibrators/
  fi
}

echo "→ Fetching Mirai snapshots from ${REPO} ${TAG:+(tag ${TAG})} into ${OUT}"
download_from_release

echo "final contents:"
find "$OUT" -maxdepth 2 -type f -printf '%P\n'