#!/usr/bin/env bash
# Run the paired SR -> full-H production trainings in one fail-fast process.
# Intended entry point:
#   tmux new-session -d -s maceh_train 'exec bash workflows/training/run_sr_then_full.sh'

set -Eeuo pipefail
umask 022

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
WORKSPACE="$(readlink -f -- "${MACEH_DATA_ROOT:?set MACEH_DATA_ROOT to the dataset workspace}")"
TRAINING_ROOT="$(readlink -f -- "${MACEH_RUNS_ROOT:?set MACEH_RUNS_ROOT to the generated-run root}")"
RUN_ID="${MACEH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CONTROL_DIR="$TRAINING_ROOT/paired_sessions/$RUN_ID"
SNAPSHOT_ROOT="$CONTROL_DIR/source"
SESSION_LOG="$CONTROL_DIR/session.log"
STATUS_FILE="$CONTROL_DIR/status.env"
LOCK_FILE="$TRAINING_ROOT/.sr_then_full.lock"
CURRENT_STAGE="initializing"
FINAL_STATE=""
AWAKE_PID=""

die() {
    echo "ERROR: $*" >&2
    return 1
}

write_status() {
    local state="$1"
    local stage="$2"
    local message="${3:-}"
    local tmp="$STATUS_FILE.tmp.$$"
    {
        printf 'state=%s\n' "$state"
        printf 'stage=%s\n' "$stage"
        printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'run_id=%s\n' "$RUN_ID"
        printf 'message=%s\n' "$message"
        printf 'session_log=%s\n' "$SESSION_LOG"
    } > "$tmp"
    mv -f -- "$tmp" "$STATUS_FILE"
}

cleanup() {
    local rc=$?
    trap - EXIT
    if [[ -n "$AWAKE_PID" ]]; then
        kill "$AWAKE_PID" 2>/dev/null || true
        wait "$AWAKE_PID" 2>/dev/null || true
    fi
    if [[ "$FINAL_STATE" != "complete" ]]; then
        write_status "failed" "$CURRENT_STAGE" "driver exit $rc"
        echo
        echo "PAIRED TRAINING FAILED during: $CURRENT_STAGE (exit $rc)"
    fi
    exit "$rc"
}
trap cleanup EXIT

[[ -x "$PYTHON" ]] || die "CUDA Python is not executable: $PYTHON"
[[ -d "$WORKSPACE" ]] || die "workspace is missing: $WORKSPACE"
[[ -d "$TRAINING_ROOT" ]] || die "training root is missing: $TRAINING_ROOT"

mkdir -p -- "$CONTROL_DIR"
exec > >(tee -a "$SESSION_LOG") 2>&1

exec 9>"$LOCK_FILE"
flock -n 9 || die "another SR/full driver holds $LOCK_FILE"

if [[ -e "$TRAINING_ROOT/run_sr" || -e "$TRAINING_ROOT/run_full" ]]; then
    die "production run_sr or run_full already exists; refusing to mix runs"
fi
if [[ -e "$TRAINING_ROOT/graphs_full" ]]; then
    die "graphs_full already exists; refusing to trust an unverified stale cache"
fi

fs_type="$(df --output=fstype "$TRAINING_ROOT" | tail -n 1 | xargs)"
case "$fs_type" in
    9p|drvfs|fuseblk)
        die "training root is on $fs_type, not WSL native storage"
        ;;
esac
free_bytes="$(df -B1 --output=avail "$TRAINING_ROOT" | tail -n 1 | xargs)"
(( free_bytes >= 100 * 1024 * 1024 * 1024 )) || \
    die "less than 100 GiB is free under $TRAINING_ROOT"

write_status "running" "$CURRENT_STAGE" "creating immutable source snapshot"
echo "Paired training launch: $RUN_ID"
echo "Repository:    $REPO_ROOT"
echo "Workspace:     $WORKSPACE"
echo "Training root: $TRAINING_ROOT"
echo "Filesystem:    $fs_type"
echo "Session log:   $SESSION_LOG"

# Both multi-day runs execute from this one source/config snapshot.  Edits to
# the working tree while SR trains therefore cannot change the full-H baseline.
mkdir -p -- "$SNAPSHOT_ROOT"
rsync -a \
    --exclude='/.git/' \
    --exclude='/data' \
    --exclude='/runs' \
    --exclude='/.pytest_cache/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    "$REPO_ROOT/" "$SNAPSHOT_ROOT/"
ln -s -- "$WORKSPACE" "$SNAPSHOT_ROOT/data"
ln -s -- "$TRAINING_ROOT" "$SNAPSHOT_ROOT/runs"
(
    cd -- "$SNAPSHOT_ROOT"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$CONTROL_DIR/source.SHA256SUMS"

export PYTHONUNBUFFERED=1
export CUDA_MODULE_LOADING=LAZY
export MACEH_DATA_ROOT="$WORKSPACE"
export MACEH_RUNS_ROOT="$TRAINING_ROOT"
cd -- "$SNAPSHOT_ROOT"

# Windows currently sleeps automatically after one hour on AC.  This helper
# holds a temporary system-required request and is always stopped by cleanup().
keep_awake_windows="$(wslpath -w "$REPO_ROOT/workflows/training/keep_awake.ps1")"
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass \
    -File "$keep_awake_windows" > "$CONTROL_DIR/keep_awake.log" 2>&1 &
AWAKE_PID=$!
sleep 2
kill -0 "$AWAKE_PID" 2>/dev/null || {
    sed -n '1,120p' "$CONTROL_DIR/keep_awake.log" >&2 || true
    die "Windows keep-awake helper did not stay running"
}
echo "Windows keep-awake helper active (PID $AWAKE_PID)."

run_stage() {
    local name="$1"
    shift
    CURRENT_STAGE="$name"
    write_status "running" "$CURRENT_STAGE"
    echo
    echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) :: $CURRENT_STAGE ====="
    "$@"
}

check_view_target() {
    local target="$1"
    VIEW_TARGET="$target" "$PYTHON" - <<'PY'
import glob
import os

target = os.environ["VIEW_TARGET"]
workspace = os.environ["MACEH_DATA_ROOT"]
training_root = os.environ["MACEH_RUNS_ROOT"]
links = sorted(glob.glob(os.path.join(
    training_root, "data_trainval", "*", "hamiltonians.h5")))
if len(links) != 367:
    raise SystemExit(f"view has {len(links)} Hamiltonian links; expected 367")
wrong = []
for link in links:
    sid = os.path.basename(os.path.dirname(link))
    got = os.path.realpath(link)
    want = os.path.realpath(os.path.join(
        workspace, "main", sid, f"hamiltonians_{target}.h5"))
    if got != want:
        wrong.append((sid, got, want))
if wrong:
    sample = "\n".join(f"  {sid}: {got} != {want}"
                       for sid, got, want in wrong[:10])
    raise SystemExit(
        f"{len(wrong)} view links do not resolve to {target} labels:\n{sample}")
print(f"view target OK: all 367 links resolve to hamiltonians_{target}.h5")
PY
}

validate_completed_run() {
    local target="$1"
    local base="$TRAINING_ROOT/run_$target"
    local suffix="_$target"
    local -a runs=()
    mapfile -t runs < <(find "$base" -mindepth 1 -maxdepth 1 \
        -type d -name "*$suffix" -print | sort)
    [[ ${#runs[@]} -eq 1 ]] || \
        die "expected exactly one $target run under $base; found ${#runs[@]}"
    local run_dir="${runs[0]}"
    local result="$run_dir/result.txt"
    local required
    for required in result.txt best_model.pkl model.pkl test_report.txt test_result.h5; do
        [[ -s "$run_dir/$required" ]] || \
            die "$target run is incomplete: $run_dir/$required is missing or empty"
    done
    grep -Fq 'Training finished.' "$result" || \
        die "$target result log has no normal training completion marker"
    grep -Fq 'Test finished, cost' "$result" || \
        die "$target result log has no completed-test marker"
    if grep -Fq 'KeyboardInterrupt' "$result"; then
        die "$target was interrupted; the next target will not start"
    fi
    RUN_DIR="$run_dir" "$PYTHON" - <<'PY'
import math
import os
import torch

run_dir = os.environ["RUN_DIR"]
checkpoint = torch.load(os.path.join(run_dir, "best_model.pkl"),
                        map_location="cpu", weights_only=False)
loss = float(checkpoint["val_loss"])
if not math.isfinite(loss):
    raise SystemExit(f"best checkpoint has non-finite val_loss: {loss}")
if not checkpoint.get("state_dict"):
    raise SystemExit("best checkpoint has no model state")
print(f"completed run OK: {run_dir}")
print(f"best checkpoint: epoch {checkpoint['epoch']}, val_loss {loss:.8e}")
PY
    printf '%s\n' "$run_dir" > "$CONTROL_DIR/${target}_run_dir.txt"
}

CONFIG="$SNAPSHOT_ROOT/provenance/config.resolved.yaml"

run_stage "verify frozen provenance" \
    "$PYTHON" -m workflows.training.freeze_provenance --verify
run_stage "verify training controls" \
    "$PYTHON" -m workflows.training.check_training_controls
run_stage "verify paired configs and frozen splits" \
    "$PYTHON" -m workflows.training.check_split_wiring --configs-only

run_stage "publish SR labels" \
    "$PYTHON" -m workflows.mgo_dataset export-target --target sr \
    --config "$CONFIG" --workspace "$WORKSPACE"
run_stage "organize SR loader views" \
    "$PYTHON" -m workflows.mgo_dataset organize --config "$CONFIG" --workspace "$WORKSPACE"
run_stage "build combined SR train-validation view" \
    "$PYTHON" -m workflows.training.make_trainval_view \
    --workspace "$WORKSPACE" --training-root "$TRAINING_ROOT"
run_stage "verify all SR view labels" check_view_target sr
run_stage "verify SR loader membership" \
    "$PYTHON" -m workflows.training.check_split_wiring \
    --workspace "$WORKSPACE" --training-root "$TRAINING_ROOT"
run_stage "verify frozen SR graph cache" \
    "$PYTHON" -m workflows.training.freeze_cache --verify --target sr \
    --workspace "$WORKSPACE" --training-root "$TRAINING_ROOT"

run_stage "train SR" \
    "$PYTHON" -u -m maceh train workflows/training/train_sr.ini
run_stage "validate completed SR run" validate_completed_run sr

# The full-H cache is built only after the workspace, loader views, and all
# 367 direct label resolutions have been switched and checked.
run_stage "publish full-H labels" \
    "$PYTHON" -m workflows.mgo_dataset export-target --target full \
    --config "$CONFIG" --workspace "$WORKSPACE"
run_stage "organize full-H loader views" \
    "$PYTHON" -m workflows.mgo_dataset organize --config "$CONFIG" --workspace "$WORKSPACE"
run_stage "build combined full-H train-validation view" \
    "$PYTHON" -m workflows.training.make_trainval_view \
    --workspace "$WORKSPACE" --training-root "$TRAINING_ROOT"
run_stage "verify all full-H view labels" check_view_target full
run_stage "verify full-H loader membership" \
    "$PYTHON" -m workflows.training.check_split_wiring \
    --workspace "$WORKSPACE" --training-root "$TRAINING_ROOT"
run_stage "build and preflight full-H graph cache" \
    "$PYTHON" -m workflows.training.smoke_production --target full --skip-train \
    --workspace "$WORKSPACE" --training-root "$TRAINING_ROOT"
run_stage "freeze full-H graph cache provenance" \
    "$PYTHON" -m workflows.training.freeze_cache --target full \
    --workspace "$WORKSPACE" --training-root "$TRAINING_ROOT"
run_stage "verify frozen full-H graph cache" \
    "$PYTHON" -m workflows.training.freeze_cache --verify --target full \
    --workspace "$WORKSPACE" --training-root "$TRAINING_ROOT"

run_stage "train full-H" \
    "$PYTHON" -u -m maceh train workflows/training/train_full.ini
run_stage "validate completed full-H run" validate_completed_run full

CURRENT_STAGE="complete"
FINAL_STATE="complete"
write_status "complete" "$CURRENT_STAGE" "SR and full-H both completed and validated"
echo
echo "PAIRED TRAINING COMPLETE"
echo "SR:     $(<"$CONTROL_DIR/sr_run_dir.txt")"
echo "full-H: $(<"$CONTROL_DIR/full_run_dir.txt")"
