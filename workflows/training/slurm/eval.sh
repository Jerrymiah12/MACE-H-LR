#!/bin/bash
#SBATCH -J MACEH
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu=3850
#SBATCH --cpus-per-task=1
#SBATCH --output=%j.out
#SBATCH --partition=compute
#SBATCH --account=su007-rjm

module purge
module load GCC/13.2.0 CUDA/11.8.0

python_path="${PYTHON:-python}"
runs_root="${MACEH_RUNS_ROOT:?set MACEH_RUNS_ROOT to the generated-run root}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$repo_root"
"${python_path}" -m maceh eval ./configs/eval.ini | tee -a "$runs_root/log_eval.txt"
