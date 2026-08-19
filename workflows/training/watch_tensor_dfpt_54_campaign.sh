#!/usr/bin/env bash
# Keep the resumable tensor campaign alive and restore AC power policy on success.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runs_root=${MACEH_RUNS_ROOT:?set MACEH_RUNS_ROOT to the generated-run root}
campaign_root=${MGO_TENSOR_DFPT_ROOT:-"$runs_root/tensor_dfpt_54_fast"}
expected=17
log_file="$campaign_root/watchdog.log"
python_bin=${PYTHON:-python}

log() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" \
        >>"$log_file"
}

completed_count() {
    find -L "$campaign_root/fast" -mindepth 2 -maxdepth 2 \
        -name quality.json -type f -printf '.\n' 2>/dev/null | wc -l
}

launch_campaign() {
    tmux new-session -d -s mgo_tensor54_campaign \
        "cd '$repo_root' && export MGO_TENSOR_ACCEPT_FAST=1 && exec bash workflows/training/run_tensor_dfpt_54_campaign.sh >> '$campaign_root/campaign_session.log' 2>&1"
    log "relaunched resumable campaign"
}

log "watchdog started"
while true; do
    completed=$(completed_count)
    if (( completed >= expected )); then
        log "all $expected quality files present; running full tensor audit"
        if "$python_bin" -m workflows.training.tensor_dfpt_54 \
                --output "$campaign_root" audit --profile fast \
                >"$campaign_root/audit_fast.log" 2>&1; then
            log "full tensor audit passed; restoring original AC power timers"
            powercfg.exe /change standby-timeout-ac 60
            powercfg.exe /change hibernate-timeout-ac 120
            powercfg.exe /setactive SCHEME_CURRENT
            log "original AC power timers restored; watchdog complete"
            exit 0
        fi
        log "full tensor audit failed; power timers remain disabled"
    fi

    if ! tmux has-session -t mgo_tensor54_campaign 2>/dev/null; then
        log "campaign absent with $completed/$expected complete"
        if pgrep -x pw.x >/dev/null || pgrep -x ph.x >/dev/null; then
            log "QE still active without campaign tmux; waiting"
        else
            launch_campaign
        fi
    fi
    sleep 60
done
