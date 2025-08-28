#!/usr/bin/env bash
# parallel one-way mover: HIRO (SMB mount on macOS) → ChiMEC (remote GPFS) using rsync
# robust + parallel + resumable + exam-level deletes (fast on slow SMB)
#
# semantics:
# - per-exam worker: 16352A/<study_id>/<exam_id>
# - rsync to GPFS, then verify exam-level completeness via a single rsync --dry-run
# - if dry-run shows no pending changes AND exam snapshot is quiet for ≥ QUIET_SECS, rm -rf the exam dir
# - NEVER delete study_id dirs automatically; optional prompt to prune studies with only empty exams
# - startup prompt to clear stale .xfer.lock markers, with a check for running workers
# - macOS bash 3.2 compatible (no globstar); uses BSD stat(1)
# - requires Homebrew rsync 3.x on the Mac
#
# usage (example):
#   export SRC_ROOT="/Volumes/16352A"
#   export DST_SSH="annawoodard@cri-ksysappdsp3.cri.uchicago.edu"
#   export DST_ROOT="/gpfs/data/huo-lab/Image/ChiMEC"
#   export PARALLEL_JOBS=8
#   export QUIET_SECS=600
#   # optional:
#   # export DRY_RUN=1                    # simulate everything
#   # export AUTO_UNLOCK_LOCKS=1          # auto-clear stale .xfer.lock
#   # export PROMPT_UNLOCK_LOCKS=1        # ask to clear stale locks (default 1 if interactive)
#   # export AUTO_DELETE_EMPTY_STUDY=0    # auto-delete empty studies (default 0)
#   # export PROMPT_DELETE_EMPTY_STUDY=1  # ask to delete empty studies (default 1)
#   # export AUTO_PROCEED=0               # ask to proceed each scan (default 0 = prompt if interactive)
#   ./sync.sh
#
# one-shot:
#   export RUN_ONCE=1; ./sync.sh

set -euo pipefail

### REQUIRED CONFIG
: "${SRC_ROOT:?set SRC_ROOT to your mounted HIRO root, e.g. /Volumes/16352A}"
: "${DST_SSH:?set DST_SSH to ssh target, e.g. annawoodard@cri-ksysappdsp3}"
: "${DST_ROOT:?set DST_ROOT to dest root on GPFS, e.g. /gpfs/data/huo-lab/Image/ChiMEC}"
: "${PARALLEL_JOBS:?set PARALLEL_JOBS to concurrency, e.g. 8}"
: "${QUIET_SECS:?set QUIET_SECS to seconds of quiescence before delete, e.g. 600}"

# options
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-$HOME/hiro_parallel_logs}"
SCAN_INTERVAL="${SCAN_INTERVAL:-120}"
PROMPT_DELETE_EMPTY_STUDY="${PROMPT_DELETE_EMPTY_STUDY:-1}"
AUTO_DELETE_EMPTY_STUDY="${AUTO_DELETE_EMPTY_STUDY:-0}"
# dotfile policy: never copy dotfiles to destination; treat recent dotfiles as active writes
EXCLUDE_DOTFILES="${EXCLUDE_DOTFILES:-1}"           # 1 = exclude dotfiles from sync/verification
DOTFILE_GRACE_SECS="${DOTFILE_GRACE_SECS:-1800}"    # do not remove exam if any dotfile newer than this (default 30m)

if [ -t 0 ]; then
  PROMPT_UNLOCK_LOCKS="${PROMPT_UNLOCK_LOCKS:-1}"
  AUTO_PROCEED="${AUTO_PROCEED:-0}"
else
  PROMPT_UNLOCK_LOCKS="${PROMPT_UNLOCK_LOCKS:-0}"
  AUTO_PROCEED="${AUTO_PROCEED:-1}"
fi
AUTO_UNLOCK_LOCKS="${AUTO_UNLOCK_LOCKS:-0}"

RSYNC_BIN="${RSYNC_BIN:-/opt/homebrew/bin/rsync}"
[ -x "$RSYNC_BIN" ] || { echo "rsync not found at $RSYNC_BIN"; exit 1; }
# exclude housekeeping, mac cruft, and (optionally) all dotfiles from rsync
RSYNC_FILTERS=(--exclude='.xfer.*' --exclude='*/.xfer.*' --exclude='.DS_Store' --exclude='._*')
if [ "$EXCLUDE_DOTFILES" = "1" ]; then
  RSYNC_FILTERS+=(--exclude='.*' --exclude='*/.*')
fi
mkdir -p "$LOG_DIR"

# self path for worker reinvocation
SELF="$(cd -- "$(dirname -- "$0")" && pwd)/$(basename -- "$0")"
[ -f "$SELF" ] || { echo "script must be a regular file: $SELF"; exit 1; }
[ -x "$SELF" ] || chmod +x "$SELF" || true

# rsync flags tuned for large files + resume + integrity
RSYNC_COMMON=(-aH --append-verify --partial --human-readable --info=stats2,progress2)
[ "$DRY_RUN" = "1" ] && RSYNC_COMMON+=(-n)

# BSD/macOS stat → "mtime size" for a local file; empty if missing
stat_local() {
  local f="$1"
  if [ -f "$f" ]; then
    stat -f "%m %z" -- "$f"
  else
    echo ""
  fi
}

# human bytes
hb() { awk 'function human(x){s="BKMGTPEZY";i=0;while(x>=1024&&i<length(s)){x/=1024;i++}printf "%.2f%s",x,substr(s,i+1,1)}{print human($1)}'; }

# snapshot an exam: emit list file + meta (nfiles, total_bytes, max_mtime)
make_snapshot() {
  local exam="$1" list="$2" meta="$3"
  : > "$list"
  local n=0 total=0 maxm=0
  while IFS= read -r -d '' f; do
    case "$f" in
      "$exam"/.xfer.*|"$exam"/.xfer.*/*) continue ;;
    esac
    # skip dotfiles entirely for snapshot
    local base="$(basename "$f")"
    [[ "$base" = .* ]] && continue
    local rel="${f#$exam/}"
    local ms; ms="$(stat -f "%m %z" -- "$f")" || ms=""
    [ -n "$ms" ] || continue
    local m="${ms% *}" s="${ms#* }"
    (( m > maxm )) && maxm="$m"
    printf '%s|%s\n' "$rel" "$ms" >> "$list"
    n=$((n+1)); total=$(( total + s ))
  done < <(find "$exam" -type f -print0)
  printf '%s\n%s\n%s\n' "$n" "$total" "$maxm" > "$meta"
}


ensure_dest() {
  local dst_exam="$1"
  ssh -o BatchMode=yes -o LogLevel=ERROR -T "$DST_SSH" "mkdir -p -- \"${dst_exam}\""
}

rsync_exam() {
  local exam="$1" study_id="$(basename "$(dirname "$exam")")" exam_id="$(basename "$exam")"
  local dst_exam="${DST_ROOT%/}/$study_id/$exam_id"
  ensure_dest "$dst_exam"
  "$RSYNC_BIN" "${RSYNC_COMMON[@]}" "${RSYNC_FILTERS[@]}" \
    "$exam/." "$DST_SSH:$dst_exam/"
}

# after rsync, check if *anything* would still be transferred for this exam
# returns 0 if exam is fully in sync (no pending changes), 1 otherwise
exam_in_sync() {
  local exam="$1"
  local study_id="$(basename "$(dirname "$exam")")"
  local exam_id="$(basename "$exam")"
  local dst_exam="${DST_ROOT%/}/$study_id/$exam_id"

  local RSV
  RSV="$("$RSYNC_BIN" -aHn --size-only --omit-dir-times -i \
          --out-format='%i %n' \
          -e 'ssh -q -o LogLevel=ERROR' \
          "${RSYNC_FILTERS[@]}" \
          "$exam/." "$DST_SSH:$dst_exam/" 2>/dev/null || true)"

  local CHG
  CHG="$(printf '%s\n' "$RSV" | grep -E '^[>chd].' || true)"

  if [ -n "$CHG" ]; then
    echo "exam_in_sync: pending changes for $exam (showing up to 20):"
    printf '%s\n' "$CHG" | head -20 | sed 's/^/  /'
    return 1
  fi
  return 0
}

exam_has_recent_dotfiles() {
  local exam="$1"
  local now="$(date +%s)"
  # any dotfile newer than DOTFILE_GRACE_SECS?
  local ts
  ts="$(find "$exam" -type f -name '.*' -exec stat -f '%m' {} + 2>/dev/null | sort -nr | head -1 || true)"
  [ -z "$ts" ] && return 1   # no dotfiles present → not recent
  [ $((now - ts)) -lt "$DOTFILE_GRACE_SECS" ] && return 0 || return 1
}

# after rsync + exam_in_sync + quiet period ⇒ remove entire exam dir
delete_exam_tree_if_safe() {
  local exam="$1" meta="$2"
  local nfiles total maxm now
  nfiles="$(sed -n '1p' "$meta" 2>/dev/null || echo 0)"
  total="$(sed -n '2p' "$meta" 2>/dev/null || echo 0)"
  maxm="$(sed -n '3p' "$meta" 2>/dev/null || echo 0)"
  now="$(date +%s)"

  # guard against active writes via non-dot files
  if [ $((now - maxm)) -lt "$QUIET_SECS" ]; then
    echo "exam not quiet long enough (non-dot files changed $(($now - maxm))s ago < ${QUIET_SECS}s) — keep for now"
    return
  fi

  # if any *recent* dotfiles exist, keep the exam (writer may still be active)
  if exam_has_recent_dotfiles "$exam"; then
    echo "recent dotfiles present (< ${DOTFILE_GRACE_SECS}s) — keep for now"
    return
  fi

  # all non-dot files are quiet; dotfiles either absent or older than grace
  if exam_in_sync "$exam"; then
    if [ "$DRY_RUN" = "1" ]; then
      echo "would rm -rf exam (in sync; quiet≥${QUIET_SECS}s; dotfiles old/absent): $exam"
    else
      rm -rf -- "$exam"
      echo "removed exam dir (in sync; quiet≥${QUIET_SECS}s; dotfiles old/absent): $exam"
    fi
  else
    echo "exam not fully in sync — keep for next pass"
  fi
}

# study helpers
study_all_exams_empty() {
  local study="$1"
  if find "$study" -type f -print -quit | grep -q .; then return 1; fi
  local exam
  for exam in "$study"/*; do
    [ -d "$exam" ] || continue
    if find "$exam" -mindepth 1 -print -quit | grep -q .; then return 1; fi
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
        echo "dry-run: would delete study dir $study"
        continue
      fi
      if [ "$AUTO_DELETE_EMPTY_STUDY" = "1" ]; then
        find "$study" -type d -empty -delete
        rmdir "$study" 2>/dev/null || true
        echo "deleted empty study dir: $study"
      elif [ "$PROMPT_DELETE_EMPTY_STUDY" = "1" ] && [ -t 0 ]; then
        read -r -p "delete empty study dir $study ? [y/N] " ans || ans="n"
        [[ "$ans" =~ ^[Yy]$ ]] && { find "$study" -type d -empty -delete; rmdir "$study" 2>/dev/null || true; echo "deleted empty study dir: $study"; } || echo "kept study dir: $study"
      else
        echo "skipping deletion (prompt disabled). set AUTO_DELETE_EMPTY_STUDY=1 to delete automatically"
      fi
    fi
  done
}

# safer lock clear with running-worker check
maybe_clear_stale_locks() {
  local pids
  pids=$(ps -o pid=,ppid=,command= -u "$USER" | grep -F "$SELF worker" | grep -v grep || true)
  if [ -n "$pids" ]; then
    echo "WARNING: worker processes are still running:"
    echo "$pids" | sed 's/^/  /'
    echo "Do NOT clear locks while workers are active."
    echo "To stop them, run: kill <pid>    # for the PIDs listed above"
    return
  fi

  local locks
  locks=$(find "$SRC_ROOT" -type d -name '.xfer.lock' 2>/dev/null | wc -l | awk '{print $1}')
  if [ "$locks" -eq 0 ]; then
    echo "startup: no lock dirs found"
    return
  fi
  echo "startup: found $locks lock dirs:"
  find "$SRC_ROOT" -type d -name '.xfer.lock' -print | sed 's/^/  - /'

  [ "$DRY_RUN" = "1" ] && { echo "dry-run: would offer to clear locks; leaving in place"; return; }

  if [ "$AUTO_UNLOCK_LOCKS" = "1" ]; then
    find "$SRC_ROOT" -type d -name '.xfer.lock' -exec rm -rf {} +
    echo "startup: cleared all lock dirs automatically"
  elif [ "$PROMPT_UNLOCK_LOCKS" = "1" ] && [ -t 0 ]; then
    read -r -p "Clear the stale .xfer.lock markers listed above (this ONLY unlocks, NO exam data will be deleted)? [y/N] " ans || ans="n"
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      find "$SRC_ROOT" -type d -name '.xfer.lock' -exec rm -rf {} +
      echo "startup: cleared lock dirs"
    else
      echo "startup: keeping lock dirs"
    fi
  else
    echo "startup: not clearing locks (prompt disabled)"
  fi
}

transfer_exam() {
  local exam="$1"
  local study_id exam_id log list meta lockdir
  study_id="$(basename "$(dirname "$exam")")"
  exam_id="$(basename "$exam")"
  log="$LOG_DIR/${study_id}_${exam_id}.log"
  list="$exam/.xfer.snapshot"
  meta="$exam/.xfer.snapshot.meta"
  lockdir="$exam/.xfer.lock"

  [ -d "$exam" ] || { echo "skip non-dir: $exam" | tee -a "$log"; return; }

  # acquire lock
  if mkdir "$lockdir" 2>/dev/null; then
    { echo "host=$(hostname)"; echo "pid=$$"; echo "ts=$(date -u +%FT%TZ)"; } > "$lockdir/meta" || true
  else
    echo "locked, skipping: $exam" | tee -a "$log"
    return
  fi
  trap "rm -rf '$lockdir' 2>/dev/null || true" EXIT

  {
    echo "==== transfer start $(date -u +%FT%TZ) exam=$exam dry_run=$DRY_RUN ===="
    echo "snapshot..."
    make_snapshot "$exam" "$list" "$meta"
    local nfiles total maxm
    nfiles="$(sed -n '1p' "$meta" 2>/dev/null || echo 0)"
    total="$(sed -n '2p' "$meta" 2>/dev/null || echo 0)"
    maxm="$(sed -n '3p' "$meta" 2>/dev/null || echo 0)"
    echo "snapshot files=$nfiles bytes=$(printf '%s' "$total" | hb) max_mtime=$maxm"

    if [ "$nfiles" -eq 0 ]; then
      echo "no files to copy in $exam"
    else
      echo "rsync → ${DST_SSH}:${DST_ROOT%/}/$study_id/$exam_id/"
      rsync_exam "$exam"
      echo "rsync complete"
    fi

    if [ "$DRY_RUN" = "1" ]; then
      echo "dry-run: skipping deletions"
    else
      echo "exam verification (rsync dry-run) and quiet-period check…"
      delete_exam_tree_if_safe "$exam" "$meta"
    fi
    echo "==== transfer end $(date -u +%FT%TZ) exam=$exam ===="
  } 2>&1 | tee -a "$log"
}

# enumerate exam dirs (null-separated)
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

discovery_report_and_prompt() {
  local studies exams
  studies=$(find "$SRC_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | awk '{print $1}')
  exams=$(find "$SRC_ROOT" -mindepth 2 -maxdepth 2 -type d | wc -l | awk '{print $1}')
  echo "discovery: studies=$studies exams=$exams @ $(date -u +%FT%TZ)"
  echo "examples:"
  find "$SRC_ROOT" -mindepth 1 -maxdepth 1 -type d | head -5 | sed 's/^/  - /'

  # recent landing detector: derive exam_touched from *files* only (dirs are noisy on SMB)
  local RECENT="${RECENT_MINUTES:-15}"
  # count of files modified in last RECENT minutes
  local recent_files
  recent_files=$(find "$SRC_ROOT" -type f -mmin -"$RECENT" -print -quit | wc -l | awk '{print $1}')
  # number of unique study/exam pairs that have at least one recent file
  # path looks like .../<study>/<exam>/... ; extract those two components
  local recent_exams
  recent_exams=$(
    find "$SRC_ROOT" -type f -mmin -"$RECENT" -print 2>/dev/null \
    | awk -F/ 'NF>=3 {print $(NF-2) "/" $(NF-1)}' \
    | sort -u | wc -l | awk '{print $1}'
  )
  echo "recent activity (last ${RECENT}m): files=${recent_files} exams_with_new_files=${recent_exams}"

  # proceed gate
  if [ "$AUTO_PROCEED" = "1" ]; then
    return
  fi
  if [ -t 0 ]; then
    read -r -p "proceed with this scan and dispatch workers? [y/N] " ans || ans="n"
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "scan canceled by user"; return 1; }
  fi
}

dispatch_once() {
  local tmp n
  tmp="$(mktemp)"
  list_exams_nullsep > "$tmp"
  n=$(tr -cd '\0' < "$tmp" | wc -c | awk '{print $1}')
  echo "dispatch: found $n exam dirs to consider"
  if [ "$n" -gt 0 ]; then
    xargs -0 -n 1 -P "$PARALLEL_JOBS" -I{} "$SELF" worker "{}" < "$tmp" || true
  fi
  rm -f "$tmp"
}

startup_banner() {
  echo "=== hiro→chimec mover start $(date -u +%FT%TZ) ==="
  echo "SRC_ROOT=$SRC_ROOT"
  echo "DST=$DST_SSH:$DST_ROOT"
  echo "PARALLEL_JOBS=$PARALLEL_JOBS QUIET_SECS=$QUIET_SECS DRY_RUN=$DRY_RUN"
}

main_loop() {
  startup_banner
  maybe_clear_stale_locks
  while :; do
    echo "--- scan start ---"
    discovery_report_and_prompt || { echo "--- scan skipped ---"; sleep "$SCAN_INTERVAL"; continue; }
    dispatch_once
    maybe_prune_empty_studies
    echo "--- scan end; sleeping ${SCAN_INTERVAL}s ---"
    sleep "$SCAN_INTERVAL"
  done
}

run_once() {
  startup_banner
  maybe_clear_stale_locks
  echo "--- scan start (one-shot) ---"
  discovery_report_and_prompt || { echo "--- canceled ---"; exit 0; }
  dispatch_once
  maybe_prune_empty_studies
  echo "--- done ---"
}

### worker entry
if [ "${1:-}" = "worker" ]; then
  shift
  [ -n "${1:-}" ] || { echo "usage: $SELF worker /path/to/study/exam"; exit 2; }
  transfer_exam "$1"
  exit $?
fi

### controller (with global logging)
# global log goes in $LOG_DIR/global-YYYYMMDD-HHMMSS.log
ts="$(date +%Y%m%d-%H%M%S)"
GLOBAL_LOG="$LOG_DIR/global-$ts.log"
exec > >(tee -a "$GLOBAL_LOG") 2>&1

if [ "${RUN_ONCE:-0}" = "1" ]; then
  run_once
else
  main_loop
fi
