#!/usr/bin/env bash
# Run one prepared 54-atom dielectric/Born-charge calculation safely.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
profile=${1:-fast}
snapshot=${2:-snapshot_000386}
runs_root=${MACEH_RUNS_ROOT:?set MACEH_RUNS_ROOT to the generated-run root}
campaign_root=${MGO_TENSOR_DFPT_ROOT:-"$runs_root/tensor_dfpt_54_fast"}
run_dir="$campaign_root/$profile/$snapshot"

qe_root=${QE_ROOT:?set QE_ROOT to the Quantum ESPRESSO installation}
mpi="$qe_root/bin/mpirun"
pw="$qe_root/bin/pw.x"
ph="$qe_root/bin/ph.x"
python_bin=${PYTHON:-python}
mpi_ranks=${MGO_TENSOR_MPI_RANKS:-16}
npools=${MGO_TENSOR_NPOOLS:-4}
omp_threads=${MGO_TENSOR_OMP_THREADS:-1}

if (( mpi_ranks < 1 || npools < 1 || omp_threads < 1 \
      || mpi_ranks % npools != 0 )); then
    echo "invalid parallel layout: ranks=$mpi_ranks pools=$npools " \
         "threads=$omp_threads" >&2
    exit 1
fi
if (( mpi_ranks * omp_threads > 16 )); then
    echo "parallel layout exceeds the 16 physical-core safety limit" >&2
    exit 1
fi

for required in "$run_dir/pw.in" "$run_dir/ph.in" "$run_dir/source.json" \
                "$mpi" "$pw" "$ph" "$python_bin"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required path: $required" >&2
        exit 1
    fi
done

if pgrep -x pw.x >/dev/null || pgrep -x ph.x >/dev/null; then
    echo "another Quantum ESPRESSO pw.x/ph.x process is already active" >&2
    exit 1
fi

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < 18 * 1024 * 1024 )); then
    echo "less than 18 GiB WSL memory is available; refusing a 54-atom launch" >&2
    exit 1
fi

exec 9>"$campaign_root/.run.lock"
if ! flock -n 9; then
    echo "another tensor-DFPT campaign process holds $campaign_root/.run.lock" >&2
    exit 1
fi

export OMP_NUM_THREADS=$omp_threads
export OMP_DYNAMIC=FALSE
# Let Open MPI own process affinity.  Setting OMP_PLACES without a per-rank
# place list under WSL pins every independent rank to the first physical core.
export OMP_PROC_BIND=FALSE
unset OMP_PLACES
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMPI_MCA_btl=^openib
export OMPI_MCA_btl_vader_single_copy_mechanism=none

mkdir -p "$run_dir/out"

keep_awake_pid=
monitor_pid=
cleanup() {
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
    if [[ -n "$keep_awake_pid" ]]; then kill "$keep_awake_pid" 2>/dev/null || true; fi
    # Do not return while a helper still inherits descriptor 9: the next
    # sequential snapshot would otherwise see this run's flock momentarily.
    if [[ -n "$monitor_pid" ]]; then wait "$monitor_pid" 2>/dev/null || true; fi
    if [[ -n "$keep_awake_pid" ]]; then wait "$keep_awake_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

if command -v powershell.exe >/dev/null 2>&1; then
    keep_awake_script=$(wslpath -w "$repo_root/workflows/training/keep_awake.ps1")
    powershell.exe -NoProfile -ExecutionPolicy Bypass \
        -File "$keep_awake_script" \
        >"$run_dir/keep_awake.log" 2>&1 9>&- &
    keep_awake_pid=$!
fi

monitor_resources() {
    while true; do
        {
            date -u '+%Y-%m-%dT%H:%M:%SZ'
            awk '/MemTotal:|MemAvailable:|SwapTotal:|SwapFree:/ {print}' /proc/meminfo
            ps -C pw.x -C ph.x -o pid=,etime=,rss=,pcpu=,comm= || true
        } >>"$run_dir/resources.log"
        sleep 60
    done
}
monitor_resources 9>&- &
monitor_pid=$!

run_qe() {
    local executable=$1
    local input=$2
    local output=$3
    local timing=$4
    (
        cd "$run_dir"
        /usr/bin/time -v -o "$timing" \
            "$mpi" -n "$mpi_ranks" --map-by core --bind-to core \
            "$executable" -nk "$npools" -in "$input" >"$output" 2>&1
    )
}

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') starting $profile/$snapshot" \
    | tee -a "$run_dir/campaign.log"
echo "parallel layout: $mpi_ranks MPI x $omp_threads OpenMP; " \
     "$npools k-pools" | tee -a "$run_dir/campaign.log"

if ! rg -q 'JOB DONE\.' "$run_dir/pw.out" 2>/dev/null; then
    run_qe "$pw" pw.in pw.out pw.time
fi
if ! rg -qi 'convergence has been achieved' "$run_dir/pw.out" \
        || ! rg -q 'JOB DONE\.' "$run_dir/pw.out"; then
    echo "pw.x did not converge cleanly; ph.x was not started" >&2
    exit 1
fi

if ! rg -q 'JOB DONE\.' "$run_dir/ph.out" 2>/dev/null; then
    run_qe "$ph" ph.in ph.out ph.time
fi
if ! rg -q 'JOB DONE\.' "$run_dir/ph.out"; then
    echo "ph.x did not finish cleanly" >&2
    exit 1
fi

cd "$repo_root"
"$python_bin" -m workflows.training.tensor_dfpt_54 --output "$campaign_root" \
    collect --profile "$profile" "$snapshot" \
    >"$run_dir/collect.log" 2>&1
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') completed $profile/$snapshot" \
    | tee -a "$run_dir/campaign.log"
