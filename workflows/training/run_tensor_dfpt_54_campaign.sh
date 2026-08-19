#!/usr/bin/env bash
# Run the gated 17-snapshot 54-atom Born-charge/dielectric prototype.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runs_root=${MACEH_RUNS_ROOT:?set MACEH_RUNS_ROOT to the generated-run root}
campaign_root=${MGO_TENSOR_DFPT_ROOT:-"$runs_root/tensor_dfpt_54_fast"}
python_bin=${PYTHON:-python}
runner="$repo_root/workflows/training/run_tensor_dfpt_54.sh"
benchmark=snapshot_000386

fast_snapshots=(
    snapshot_000386
    snapshot_000011
    snapshot_000033
    snapshot_000063
    snapshot_000175
    snapshot_000244
    snapshot_000275
    snapshot_000312
    snapshot_000353
    snapshot_000354
    snapshot_000028
    snapshot_000327
    snapshot_000080
    snapshot_000229
    snapshot_000295
    snapshot_000333
    snapshot_000334
)

mkdir -p "$campaign_root"
campaign_log="$campaign_root/campaign.log"

log() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" \
        | tee -a "$campaign_log"
}

is_collected() {
    local profile=$1
    local snapshot=$2
    local directory="$campaign_root/$profile/$snapshot"
    [[ -s "$directory/quality.json" \
       && -s "$directory/born_effective_charges.npy" \
       && -s "$directory/dielectric_infinity.npy" ]] \
       && rg -q 'JOB DONE\.' "$directory/ph.out"
}

run_one() {
    local profile=$1
    local snapshot=$2
    if is_collected "$profile" "$snapshot"; then
        log "already complete: $profile/$snapshot"
        return 0
    fi
    # A completed QE process can leave several GiB in the WSL page cache for a
    # short time.  Wait for the runner's 18-GiB safety gate instead of letting
    # one transient low-memory reading terminate the whole campaign.
    while true; do
        available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
        if (( available_kib >= 18 * 1024 * 1024 )); then
            break
        fi
        log "waiting for 18 GiB available memory before $profile/$snapshot (now $((available_kib / 1024)) MiB)"
        sleep 60
    done
    log "launching: $profile/$snapshot"
    attempt=1
    while ! bash "$runner" "$profile" "$snapshot"; do
        if (( attempt >= 3 )); then
            log "failed after $attempt launch attempts: $profile/$snapshot"
            return 1
        fi
        log "launch attempt $attempt failed for $profile/$snapshot; retrying in 60 seconds"
        attempt=$((attempt + 1))
        sleep 60
    done
}

for profile in fast anchor; do
    if [[ ! -s "$campaign_root/manifest_${profile}.json" ]]; then
        "$python_bin" -m workflows.training.tensor_dfpt_54 \
            --output "$campaign_root" prepare --profile "$profile"
    fi
done

if pgrep -x pw.x >/dev/null || pgrep -x ph.x >/dev/null; then
    log "refusing campaign launch while another Quantum ESPRESSO job is active"
    exit 1
fi

# First prove that the cheap 2x2x2 label is close enough to a 3x3x3 anchor.
run_one fast "$benchmark"
run_one anchor "$benchmark"
log "comparing fast and anchor tensors for $benchmark"
compare_status=0
"$python_bin" -m workflows.training.tensor_dfpt_54 \
    --output "$campaign_root" compare "$benchmark" \
    | tee "$campaign_root/profile_comparison_${benchmark}.log" \
    || compare_status=$?
if (( compare_status != 0 )); then
    if [[ ${MGO_TENSOR_ACCEPT_FAST:-0} == 1 ]]; then
        log "WARNING: user-authorized fast-profile override; comparison gate returned $compare_status"
    else
        log "quality gate failed; remaining fast snapshots were not launched"
        exit "$compare_status"
    fi
fi

log "quality gate accepted or explicitly overridden; launching the remaining fast prototype sequentially"
for snapshot in "${fast_snapshots[@]:1}"; do
    run_one fast "$snapshot"
done
log "all ${#fast_snapshots[@]} fast prototype snapshots are complete"
