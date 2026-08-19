"""Run prepared EPC-reference ABACUS calculations with bounded parallelism."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time


from maceh.data.io.abacus import parse_running_scf


DEFAULT_DFT_BIN = os.environ.get('MACEH_DFT_BIN', '')


def finished(folder):
    log = os.path.join(folder, 'OUT.MgO', 'running_scf.log')
    if not os.path.isfile(log):
        return False
    try:
        return bool(parse_running_scf(log)['converged'])
    except (OSError, ValueError):
        return False


def run_one(folder, mpi_ranks, dft_bin):
    if finished(folder):
        return os.path.basename(folder), 0.0, 'already complete'
    mpirun = os.path.join(dft_bin, 'mpirun')
    abacus = os.path.join(dft_bin, 'abacus')
    if not (os.path.isfile(mpirun) and os.path.isfile(abacus)):
        raise FileNotFoundError(f'ABACUS environment missing below {dft_bin}')
    command = [mpirun, '-n', str(mpi_ranks), '--oversubscribe',
               '--mca', 'btl', '^openib', abacus]
    environment = os.environ.copy()
    environment['OMP_NUM_THREADS'] = '1'
    environment['OPENBLAS_NUM_THREADS'] = '1'
    begin = time.time()
    with open(os.path.join(folder, 'run.out'), 'w') as output:
        result = subprocess.run(command, cwd=folder, env=environment,
                                stdout=output, stderr=subprocess.STDOUT)
    elapsed = time.time() - begin
    if result.returncode != 0:
        return os.path.basename(folder), elapsed, \
            f'ABACUS exit code {result.returncode}'
    if not finished(folder):
        return os.path.basename(folder), elapsed, 'SCF did not converge'
    return os.path.basename(folder), elapsed, 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/epc/dft_reference')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--mpi-ranks', type=int, default=4)
    parser.add_argument('--dft-bin', default=DEFAULT_DFT_BIN)
    args = parser.parse_args()
    if args.jobs < 1 or args.mpi_ranks < 1:
        raise SystemExit('--jobs and --mpi-ranks must be positive')
    root = os.path.abspath(args.root)
    with open(os.path.join(root, 'manifest.json')) as handle:
        manifest = json.load(handle)
    folders = [os.path.join(root, item['name'])
               for item in manifest['calculations']]
    print(f'Running {len(folders)} calculations: {args.jobs} concurrent jobs, '
          f'{args.mpi_ranks} MPI ranks each')
    failures = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs) as executor:
        future_to_folder = {
            executor.submit(run_one, folder, args.mpi_ranks,
                            os.path.abspath(args.dft_bin)): folder
            for folder in folders}
        for future in concurrent.futures.as_completed(future_to_folder):
            name, elapsed, status = future.result()
            print(f'  {name:12s} {elapsed:7.1f} s  {status}', flush=True)
            if status not in ('complete', 'already complete'):
                failures.append((name, status))
    if failures:
        raise SystemExit(f'{len(failures)} DFT calculations failed: {failures}')
    print('All DFT reference calculations converged.')


if __name__ == '__main__':
    main()
