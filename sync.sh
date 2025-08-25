#!/usr/bin/env bash
# parallel one-way mover: HIRO (SMB mount on macOS) → ChiMEC (remote GPFS) using rsync
# additions:
# - verbose discovery logs that work in dry-run and real modes
# - reports studies with only empty exam dirs; optionally prompts to delete them
# - each worker logs counts and bytes from its snapshot; prints what it will do in dry-run
# - preserves previous invariants: never delete study dirs automatically; delete exam dir only when empty

# EXAMPLE CONFIG
# export SRC_ROOT="/Volumes/16352A"
# export DST_SSH="annawoodard@cri-ksysappdsp3.cri.uchicago.edu"
# export DST_ROOT="/gpfs/data/huo-lab/Image/ChiMEC"
# export PARALLEL_JOBS=8            # tune: 1 process ≈ 1 MB/s; use 8–16 if VPN/CPU allows
# export QUIET_SECS=600             # delete only if file unchanged for ≥10 min since snapshot

# # optional: preview only
# # export DRY_RUN=1

# # run forever, rescanning every 120s for new exams; Ctrl-C to stop
# bash ./sync.sh

set -euo pipefail
SELF="$(cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")"
[ -x "$SELF" ] || { echo "script must be a file and executable: $SELF"; exit 1; }

### REQUIRED CONFIG (fail loudly if unset)
: "${SRC_ROOT:?set SRC_ROOT to your mounted HIRO root, e.g. /Volumes/16352A}"
: "${DST_SSH:?set DST_SSH to ssh target, e.g. annawoodard@cri-ksysappdsp3}"
: "${DST_ROOT:?set DST_ROOT to dest root on GPFS, e.g. /gpfs/data/huo-lab/Image/ChiMEC}"
: "${PARALLEL_JOBS:?set PARALLEL_JOBS to concurrency, e.g. 8}"
: "${QUIET_SECS:?set QUIET_SECS to seconds of quiescence before delete, e.g. 600}"

RSYNC_BIN="${RSYNC_BIN:-/opt/homebrew/bin/rsync}"   # brew rsync path on macOS; override if different
[ -x "$RSYNC_BIN" ] || { echo "rsync not found at $RSYNC_BIN"; exit 1; }

# optional knobs
DRY_RUN="${DRY_RUN:-0}"                              # 1 = simulate all actions
LOG_DIR="${LOG_DIR:-$HOME/hiro_parallel_logs}"       # per-exam logs go here
SCAN_INTERVAL="${SCAN_INTERVAL:-120}"                # seconds between scans
PROMPT_DELETE_EMPTY_STUDY="${PROMPT_DELETE_EMPTY_STUDY:-1}"  # 1 = ask before deleting empty study
AUTO_DELETE_EMPTY_STUDY="${AUTO_DELETE_EMPTY_STUDY:-0}"      # 1 = delete empty studies without prompt

mkdir -p "$LOG_DIR"

# rsync flags tuned for large files + resume + integrity
RSYNC_COMMON=(-aH --append-verify --partial --human-readable --info=stats2,progress2)
[ "$DRY_RUN" = "1" ] && RSYNC_COMMON+=(-n)

# macOS stat fields (portable across BSD/macOS)
# returns "mtime size" for a local file, or empty if missing
stat_local() {
  local f="$1"
  if [ -f "$f" ]; then
    stat -f "%m %z" -- "$f"
  else
    echo ""
  fi
}

# human-friendly bytes
hb() { awk 'function human(x){s="BKMGTPEZY";i=0;while (x>=1024 && i<length(s)) {x/=1024;i++} printf "%.2f%s", x, substr(s,i+1,1)} {print human($1)}'; }

# create an atomic lock dir; returns 0 if acquired, 1 if already locked
acquire_lock() {
  local exam="$1" lock="$exam/.xfer.lock"
  if mkdir "$lock" 2>/dev/null; then
    { echo "host=$(hostname)"; echo "pid=$$"; echo "ts=$(date -u +%FT%TZ)"; } > "$lock/meta"
    return 0
  else
    return 1
  fi
}

release_lock() {
  local exam="$1"
  rm -rf "$exam/.xfer.lock" 2>/dev/null || true
}

make_snapshot() {
  local exam="$1" snap="$2"
  : > "$snap"
  local n=0
  local total=0

  # walk files with find (portable on macOS); null-separated to handle spaces
  # for each file, write: relpath|mtime size
  while IFS= read -r -d '' f; do
    # compute relative path
    local rel="${f#$exam/}"
    # %m = epoch mtime, %z = size (macOS/BSD stat)
    local ms
    ms="$(stat -f "%m %z" -- "$f")" || ms=""
    [ -n "$ms" ] || continue
    printf '%s|%s\n' "$rel" "$ms" >> "$snap"
    n=$((n+1))
    total=$(( total + ${ms#* } ))
  done < <(find "$exam" -type f -print0)

  echo "$n" > "$snap.count"
  echo "$total" > "$snap.bytes"
}
# ensure destination exam directory exists remotely
ensure_dest() {
  local dst_exam="$1"
  ssh -o BatchMode=yes "$DST_SSH" "mkdir -p -- \"${dst_exam}\""
}

rsync_exam() {
  local exam="$1" study_id exam_id dst_exam
  study_id="$(basename "$(dirname "$exam")")"
  exam_id="$(basename "$exam")"
  dst_exam="${DST_ROOT%/}/$study_id/$exam_id"
  ensure_dest "$dst_exam"
  "$RSYNC_BIN" "${RSYNC_COMMON[@]}" "$exam/." "$DST_SSH:$dst_exam/"
}

# post-copy verification + selective delete of source files
prune_source_files() {
  local exam="$1" snap="$2" now ts age rel mtime size cur cur_m cur_s dst_path
  now="$(date +%s)"
  while IFS='|' read -r rel rest; do
    [ -n "$rel" ] || continue
    mtime="${rest% *}"
    size="${rest#* }"
    cur="$(stat_local "$exam/$rel")" || cur=""
    if [ -z "$cur" ]; then
      echo "already gone at source: $rel"
      continue
    fi
    cur_m="${cur% *}"
    cur_s="${cur#* }"
    age=$(( now - cur_m ))
    if [ "$cur_m" = "$mtime" ] && [ "$cur_s" = "$size" ] && [ "$age" -ge "$QUIET_SECS" ]; then
      dst_path="${DST_ROOT%/}/$(basename "$(dirname "$exam")")/$(basename "$exam")/$rel"
      if ssh -o BatchMode=yes "$DST_SSH" "test -f \"$dst_path\""; then
        # verify dest size >= source size
        if ssh -o BatchMode=yes "$DST_SSH" "python3 - <<'PY'\nimport os,sys\np=sys.argv[1]\ntry:\n print(os.stat(p).st_size)\nexcept FileNotFoundError:\n print(-1)\nPY\n\"$dst_path\"" | awk -v s="$cur_s" '{exit !($1>=s)}'; then
          if [ "$DRY_RUN" = "1" ]; then
            echo "would delete source file: $rel"
          else
            rm -f -- "$exam/$rel"
            echo "deleted source file: $rel"
          fi
        else
          echo "skip delete (dest smaller): $rel"
        fi
      else
        echo "skip delete (dest missing): $rel"
      fi
    else
      echo "skip delete (changed or not quiet): $rel"
    fi
  done < "$snap"
}

# remove exam dir if empty (never touch study_id)
prune_exam_dir_if_empty() {
  local exam="$1"
  if [ "$DRY_RUN" = "1" ]; then
    if [ -z "$(find "$exam" -mindepth 1 -type f -print -quit)" ]; then
      # if no files remain, check emptiness
      if [ -z "$(find "$exam" -mindepth 1 -print -quit)" ]; then
        echo "would rmdir exam: $exam"
      fi
    fi
  else
    rmdir "$exam" 2>/dev/null && echo "removed empty exam dir: $exam" || true
  fi
}

# returns 0 if a study dir contains only empty exam dirs, otherwise 1
study_all_exams_empty() {
  local study="$1"
  # any files anywhere under study? if yes, not empty
  if find "$study" -type f -print -quit | grep -q .; then
    return 1
  fi
  # any non-empty exam dir? if yes, not empty
  local exam
  for exam in "$study"/*; do
    [ -d "$exam" ] || continue
    if find "$exam" -mindepth 1 -print -quit | grep -q .; then
      return 1
    fi
  done
  return 0
}

maybe_prune_empty_studies() {
  local study
  for study in "$SRC_ROOT"/*; do
    [ -d "$study" ] || continue
    if study_all_exams_empty "$study"; then
      echo "found empty study dir (all exam dirs empty): $study"
      if [ "$DRY_RUN" = "1" ]; then
        echo "dry-run: would delete study dir $study (and its empty exams)"
        continue
      fi
      if [ "$AUTO_DELETE_EMPTY_STUDY" = "1" ]; then
        find "$study" -type d -empty -delete
        rmdir "$study" 2>/dev/null || true
        echo "deleted empty study dir: $study"
      elif [ "$PROMPT_DELETE_EMPTY_STUDY" = "1" ] && [ -t 0 ]; then
        read -r -p "delete empty study dir $study ? [y/N] " ans
        if [[ "$ans" =~ ^[Yy]$ ]]; then
          find "$study" -type d -empty -delete
          rmdir "$study" 2>/dev/null || true
          echo "deleted empty study dir: $study"
        else
          echo "kept study dir: $study"
        fi
      else
        echo "skipping deletion (prompt disabled). set AUTO_DELETE_EMPTY_STUDY=1 to delete automatically"
      fi
    fi
  done
}

transfer_exam() {
  local exam="$1"
  local study_id exam_id log snap nfiles nbytes
  study_id="$(basename "$(dirname "$exam")")"
  exam_id="$(basename "$exam")"
  log="$LOG_DIR/${study_id}_${exam_id}.log"
  snap="$exam/.xfer.snapshot"

  [ -d "$exam" ] || { echo "skip non-dir: $exam" | tee -a "$log"; return; }

  if ! acquire_lock "$exam"; then
    echo "locked, skipping: $exam" | tee -a "$log"
    return
  fi
  trap 'release_lock "$exam"' EXIT

  {
    echo "==== transfer start $(date -u +%FT%TZ) exam=$exam dry_run=$DRY_RUN ===="
    echo "snapshot..."
    make_snapshot "$exam" "$snap"
    nfiles="$(cat "$snap.count" 2>/dev/null || echo 0)"
    nbytes="$(cat "$snap.bytes" 2>/dev/null || echo 0)"
    echo "snapshot files=$nfiles bytes=$(printf '%s' "$nbytes" | hb)"

    if [ "$nfiles" -eq 0 ]; then
      echo "no files to copy in $exam"
    else
      echo "rsync to ${DST_SSH}:${DST_ROOT%/}/$study_id/$exam_id/"
      rsync_exam "$exam"
      echo "rsync complete"
    fi

    if [ "$DRY_RUN" = "1" ]; then
      echo "dry-run mode: skipping deletions"
    else
      echo "prune source files if unchanged and quiet≥${QUIET_SECS}s..."
      prune_source_files "$exam" "$snap"
      prune_exam_dir_if_empty "$exam"
    fi
    echo "==== transfer end $(date -u +%FT%TZ) exam=$exam ===="
  } 2>&1 | tee -a "$log"
}

# emit null-separated list of exam dirs 16352A/<study>/<exam>
list_exams_nullsep() {
  shopt -s nullglob
  local study exam
  for study in "$SRC_ROOT"/*; do
    [ -d "$study" ] || continue
    for exam in "$study"/*; do
      [ -d "$exam" ] || continue
      printf '%s\0' "$exam"
    done
  done
}

# discovery summary: prints counts and top examples without disturbing the null-separated listing
discovery_report() {
  local studies exams
  studies=$(find "$SRC_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | awk '{print $1}')
  exams=$(find "$SRC_ROOT" -mindepth 2 -maxdepth 2 -type d | wc -l | awk '{print $1}')
  echo "discovery: studies=$studies exams=$exams @ $(date -u +%FT%TZ)"
  echo "example studies:"
  find "$SRC_ROOT" -mindepth 1 -maxdepth 1 -type d | head -5 | sed 's/^/  - /'
}

dispatch_once() {
  local tmp n
  tmp="$(mktemp)"
  list_exams_nullsep > "$tmp"
  n=$(tr -cd '\0' < "$tmp" | wc -c | awk '{print $1}')
  echo "dispatch: found $n exam dirs to consider"
  if [ "$n" -gt 0 ]; then
    # invoke this script in 'worker' mode for each exam; avoids function export issues
    xargs -0 -n 1 -P "$PARALLEL_JOBS" -I{} "$SELF" worker "{}" < "$tmp" || true
  fi
  rm -f "$tmp"
}

main_loop() {
  while :; do
    echo "--- scan start ---"
    discovery_report
    dispatch_once
    maybe_prune_empty_studies
    echo "--- scan end; sleeping ${SCAN_INTERVAL}s ---"
    sleep "$SCAN_INTERVAL"
  done
}

if [ "${1:-}" = "worker" ]; then
  # shift off 'worker', remaining $1 is the exam path
  shift
  [ -n "${1:-}" ] || { echo "usage: $SELF worker /path/to/study/exam"; exit 2; }
  transfer_exam "$1"
  exit $?
fi

if [ "${RUN_ONCE:-0}" = "1" ]; then
  discovery_report
  dispatch_once
  maybe_prune_empty_studies
else
  main_loop
fi

