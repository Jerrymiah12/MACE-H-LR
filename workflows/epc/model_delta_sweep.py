"""Measure finite-difference-step convergence for one EPC derivative group."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch


from maceh.epc.derivative import build_supercell_graph, finite_difference_pair
from maceh.epc.run import (atom_norb_from_model, load_model_contexts,
                           make_predict_fn)
from maceh.epc.supercell import build_supercell, load_structure
from maceh.parse_configs import EPCConfig


def difference(a, b):
    sq = 0.0
    max_abs = 0.0
    for key in set(a) | set(b):
        av, bv = a.get(key), b.get(key)
        value = av - bv if av is not None and bv is not None \
            else av if bv is None else -bv
        sq += float(np.square(np.abs(value)).sum())
        max_abs = max(max_abs, float(np.abs(value).max()))
    return np.sqrt(sq), max_abs


def norm(a):
    return np.sqrt(sum(float(np.square(np.abs(value)).sum())
                       for value in a.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--atom', type=int, default=0)
    parser.add_argument('--direction', choices=('x', 'y', 'z'), default='x')
    parser.add_argument('--deltas', nargs='+', type=float,
                        default=(0.01, 0.005, 0.0025, 0.001, 0.0005,
                                 0.0002, 0.0001, 0.00005, 0.00002))
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    deltas = sorted(set(args.deltas), reverse=True)
    if not deltas or min(deltas) <= 0:
        raise SystemExit('all deltas must be positive')

    config = EPCConfig(args.config)
    torch.set_default_dtype(config.torch_dtype)
    structure = load_structure(config.structure_dir)
    contexts = load_model_contexts(config)
    kernel = contexts[0][0]
    norb = atom_norb_from_model(kernel.dataset_info, structure.numbers)
    supercell, smap = build_supercell(structure, config.q_grid)
    data = build_supercell_graph(
        supercell, config.radius + 2 * max(deltas),
        torch.get_default_dtype())
    data.x = kernel.dataset_info.Z_to_index[data.x]
    predict = make_predict_fn(contexts, data, config)
    positions0 = data.pos.clone()
    if config.analytic_lr_workspace:
        from maceh.epc.lr_correction import make_lr_corrected_predict_fn
        predict = make_lr_corrected_predict_fn(
            predict, positions0, supercell,
            config.analytic_lr_workspace,
            config.analytic_lr_overlap_dir,
            config.analytic_lr_config)
    if config.gauge_overlap_dir:
        from maceh.epc.lr_correction import make_gauge_fixed_predict_fn
        predict = make_gauge_fixed_predict_fn(
            predict, supercell, config.gauge_overlap_dir)

    alpha = 'xyz'.index(args.direction)
    derivatives = []
    begin = time.time()
    for delta in deltas:
        derivative = finite_difference_pair(
            predict, positions0, smap, norb, delta,
            args.atom, alpha, grad_threshold=0.0)
        derivatives.append(derivative)
        print(f'delta={delta:.8g} norm={norm(derivative):.8e}', flush=True)
    rows = []
    for delta, coarse, fine_delta, fine in zip(
            deltas[:-1], derivatives[:-1], deltas[1:], derivatives[1:]):
        l2, max_abs = difference(coarse, fine)
        fine_norm = norm(fine)
        fine_peak = max(float(np.abs(value).max()) for value in fine.values())
        row = {'delta': delta, 'next_delta': fine_delta,
               'relative_l2_difference': l2 / fine_norm,
               'max_abs_difference_eV_per_A': max_abs,
               'max_difference_over_peak': max_abs / fine_peak}
        rows.append(row)
        print('  %.8g -> %.8g: rel_L2=%.6g max/peak=%.6g' %
              (delta, fine_delta, row['relative_l2_difference'],
               row['max_difference_over_peak']))
    report = {
        'config': os.path.abspath(args.config),
        'atom': args.atom,
        'direction': args.direction,
        'elapsed_seconds': time.time() - begin,
        'deltas': deltas,
        'comparisons': rows,
    }
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.config)),
        os.path.splitext(os.path.basename(args.config))[0] +
        '_delta_sweep.json')
    with open(output, 'w') as handle:
        json.dump(report, handle, indent=1)
        handle.write('\n')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
