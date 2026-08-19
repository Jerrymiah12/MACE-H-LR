#!/bin/bash
#SBATCH -J DeepH-E3
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --output=%j.out
#SBATCH --account=su007-rjm-gpu

module purge
module load GCC/13.2.0 CUDA/11.8.0

python_path="${PYTHON:-python}"
runs_root="${MACEH_RUNS_ROOT:?set MACEH_RUNS_ROOT to the generated-run root}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

cd "$repo_root"
"${python_path}" -m maceh train ./configs/train.ini | tee -a "$runs_root/log_train.txt"
