#!/usr/bin/env bash
# moves all tarballs under ChiMEC to a quarantine directory preserving tree structure
# set DRY_RUN=1 to preview without moving

set -euo pipefail

ROOT="/gpfs/data/huo-lab/Image/ChiMEC"
QUEUE="${ROOT}/_tarballs_delete_queue"
TS="$(date +%Y%m%d-%H%M%S)"
MANIFEST="${QUEUE}/moved_tarballs.${TS}.tsv"
DRY_RUN="${DRY_RUN:-0}"

echo "root:   $ROOT"
echo "queue:  $QUEUE"
echo "dryrun: $DRY_RUN"
mkdir -p "$QUEUE"

# count and size estimate (fast enough; excludes queue)
echo "scanning for tarballs..."
COUNT=$(find "$ROOT" -path "$QUEUE" -prune -o -type f \
  \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.tar.xz' -o -name '*.txz' \) \
  -print | wc -l)
echo "found $COUNT tarball(s)"

# optional size summary (may take a moment on many files)
find "$ROOT" -path "$QUEUE" -prune -o -type f \
  \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.tar.xz' -o -name '*.txz' \) \
  -print0 | du --files0-from=- -ch | tail -1 || true

# move, preserving relative dirs; write a manifest
echo -e "src_path\tdest_dir\tsize_bytes" > "$MANIFEST"
find "$ROOT" -path "$QUEUE" -prune -o -type f \
  \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.tar.xz' -o -name '*.txz' \) \
  -print0 |
while IFS= read -r -d '' f; do
  rel="${f#${ROOT}/}"                               # e.g. 13690980/2O42821-...tar.xz
  dest_dir="${QUEUE}/$(dirname "$rel")"
  mkdir -p -- "$dest_dir"
  if [[ "$DRY_RUN" = "1" ]]; then
    echo "[DRY RUN] mv -- \"$f\" \"$dest_dir/\""
  else
    # GNU stat; if that fails on your host, swap to: size=$(stat -f%z -- "$f")
    size=$(stat -c%s -- "$f" 2>/dev/null || stat -f%z -- "$f")
    echo -e "${f}\t${dest_dir}\t${size}" >> "$MANIFEST"
    mv -- "$f" "$dest_dir/"
  fi
done

echo "manifest: $MANIFEST"
echo "queue size now:"
du -sh "$QUEUE" || true
