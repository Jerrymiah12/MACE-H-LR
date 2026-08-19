"""Compare the SR-target (+ fixed LR) and direct Full-H EPC against DFT."""
import argparse
import hashlib
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from maceh.analysis.figures import save_figure_formats



SR_COLOR = '#0072B2'
FULL_COLOR = '#D55E00'
DFT_COLOR = '#222222'
GRID_COLOR = '#D9D9D9'


def load_epc(path):
    with h5py.File(path, 'r') as handle:
        data = {
            'g': handle['g_real'][:] + 1j * handle['g_imag'][:],
            'kpoints': handle['kpoints'][:],
            'qpoints': handle['qpoints'][:],
            'atom_indices': handle['atom_indices'][:],
            'atomic_numbers': handle['atomic_numbers'][:],
            'lattice': handle['lattice'][:],
            'positions': handle['positions'][:],
            'delta': float(handle['finite_difference_delta'][()]),
            'attrs': {key: value.item() if isinstance(value, np.generic)
                      else value for key, value in handle.attrs.items()},
        }
    return data


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_grids(reference, candidates):
    for name, candidate in candidates.items():
        if candidate['g'].shape != reference['g'].shape:
            raise ValueError(f'{name}: EPC tensor shape differs from DFT')
        for field in ('kpoints', 'qpoints', 'atom_indices', 'atomic_numbers',
                      'lattice', 'positions'):
            if not np.allclose(candidate[field], reference[field],
                               rtol=0, atol=1e-12):
                raise ValueError(f'{name}: {field} differs from DFT')


def metric(prediction, truth):
    error = prediction - truth
    truth_norm = float(np.linalg.norm(truth))
    prediction_norm = float(np.linalg.norm(prediction))
    inner = float(np.vdot(truth.ravel(), prediction.ravel()).real)
    return {
        'relative_l2': float(np.linalg.norm(error) / truth_norm),
        'complex_mae_eV_per_A': float(np.mean(np.abs(error))),
        'complex_rmse_eV_per_A': float(np.sqrt(np.mean(np.abs(error) ** 2))),
        'max_abs_error_eV_per_A': float(np.max(np.abs(error))),
        'cosine_similarity': inner / (truth_norm * prediction_norm),
        'norm_ratio_to_dft': prediction_norm / truth_norm,
        'dft_norm': truth_norm,
        'prediction_norm': prediction_norm,
        'n_complex_elements': int(truth.size),
    }


def hermiticity_metric(data):
    """Check g(k,q)^dagger = g(k+q,-q) on the stored periodic grids."""
    g = data['g']
    kpoints = data['kpoints']
    qpoints = data['qpoints']
    worst = 0.0
    for ik, kpoint in enumerate(kpoints):
        for iq, qpoint in enumerate(qpoints):
            target_k = np.mod(kpoint + qpoint, 1.0)
            target_q = np.mod(-qpoint, 1.0)
            k_distance = np.linalg.norm(
                np.mod(kpoints - target_k + 0.5, 1.0) - 0.5, axis=1)
            q_distance = np.linalg.norm(
                np.mod(qpoints - target_q + 0.5, 1.0) - 0.5, axis=1)
            ik_target = int(np.argmin(k_distance))
            iq_target = int(np.argmin(q_distance))
            if k_distance[ik_target] > 1e-12 or q_distance[iq_target] > 1e-12:
                raise ValueError('stored k/q grids are not closed under Hermiticity')
            difference = (g[ik, iq].conj().swapaxes(-1, -2)
                          - g[ik_target, iq_target])
            worst = max(worst, float(np.max(np.abs(difference))))
    peak = float(np.max(np.abs(g)))
    return {
        'max_abs_eV_per_A': worst,
        'fraction_of_tensor_peak': worst / peak,
    }


def breakdown(prediction, truth, qpoints):
    by_q = []
    for iq, qpoint in enumerate(qpoints):
        row = metric(prediction[:, iq], truth[:, iq])
        row['q_index'] = iq
        row['qpoint'] = qpoint.tolist()
        by_q.append(row)
    by_component = []
    for atom, element in enumerate(('Mg', 'O')):
        for alpha, direction in enumerate('xyz'):
            row = metric(prediction[:, :, atom, alpha],
                         truth[:, :, atom, alpha])
            row['atom_index'] = atom
            row['element'] = element
            row['direction'] = direction
            by_component.append(row)
    return by_q, by_component


def style(ax):
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.7)
    ax.tick_params(direction='in', top=True, right=True)


def save(fig, plots_dir, stem):
    for path in save_figure_formats(fig, plots_dir, stem,
                                    bbox_inches='tight'):
        print(f'wrote {path}')
    plt.close(fig)


def q_labels(qpoints):
    def number(value):
        if abs(value) < 1e-12:
            return '0'
        if abs(value - 0.5) < 1e-12:
            return '½'
        return f'{value:g}'
    return ['(' + ','.join(number(value) for value in point) + ')'
            for point in qpoints]


def plot_q_comparison(results, reference, plots_dir, dft_convergence):
    qpoints = reference['qpoints']
    labels = q_labels(qpoints)
    xpos = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    for shift, name, color, label in (
            (-width / 2, 'sr', SR_COLOR, 'SR-target (+ fixed LR)'),
            (width / 2, 'full', FULL_COLOR, 'Direct full-H')):
        errors = 100 * np.asarray(
            [row['relative_l2'] for row in results[name]['by_q']])
        axes[0].bar(xpos + shift, errors, width, color=color, label=label)
        ratios = np.asarray(
            [row['norm_ratio_to_dft'] for row in results[name]['by_q']])
        axes[1].plot(xpos, ratios, 'o-', color=color, lw=1.8,
                     ms=5, label=label)
    axes[0].axhline(100 * dft_convergence, color=DFT_COLOR, ls='--', lw=1,
                    label='DFT δ-step change')
    axes[0].set_ylabel('Relative L2 error (%)')
    axes[0].set_title('EPC error against DFT')
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(1, color=DFT_COLOR, ls='--', lw=1)
    axes[1].set_ylabel('‖g(model)‖ / ‖g(DFT)‖')
    axes[1].set_title('Coupling-strength ratio')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(xpos, labels, rotation=35, ha='right')
        ax.set_xlabel('q point (primitive reciprocal coordinates)')
        style(ax)
    fig.suptitle('2×2×2-grid Cartesian-AO electron–phonon coupling')
    fig.tight_layout()
    save(fig, plots_dir, 'epc_01_q_resolved_comparison')


def plot_parity(results, reference, plots_dir):
    truth_complex = reference['g'].ravel()
    truth = np.concatenate((truth_complex.real, truth_complex.imag))
    all_values = [truth]
    predictions = {}
    for name in ('sr', 'full'):
        flat = results[name]['g'].ravel()
        predictions[name] = np.concatenate((flat.real, flat.imag))
        all_values.append(predictions[name])
    limit = float(np.quantile(np.abs(np.concatenate(all_values)), 0.9995))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), sharex=True, sharey=True)
    for ax, name, cmap, title in (
            (axes[0], 'sr', 'Blues', 'SR-target (+ fixed LR)'),
            (axes[1], 'full', 'Oranges', 'Direct full-H')):
        pred = predictions[name]
        shown = (np.abs(truth) <= limit) & (np.abs(pred) <= limit)
        hb = ax.hexbin(truth[shown], pred[shown], gridsize=90, bins='log',
                       mincnt=1, extent=(-limit, limit, -limit, limit),
                       cmap=cmap)
        ax.plot([-limit, limit], [-limit, limit], '--', color='black', lw=1)
        summary = results[name]['overall']
        ax.text(0.04, 0.96,
                f"relative L2 = {100*summary['relative_l2']:.2f}%\n"
                f"MAE = {summary['complex_mae_eV_per_A']:.3f} eV/Å\n"
                f"cosine = {summary['cosine_similarity']:.4f}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox={'facecolor': 'white', 'edgecolor': 'none',
                      'alpha': 0.9, 'pad': 4})
        ax.set_title(title)
        ax.set_xlabel('DFT EPC component (eV/Å)')
        ax.set_aspect('equal', adjustable='box')
        style(ax)
        fig.colorbar(hb, ax=ax, label='component count')
    axes[0].set_ylabel('Predicted EPC component (eV/Å)')
    fig.suptitle('Real and imaginary Cartesian-AO EPC parity\n'
                 '(axes cover 99.95% of components)')
    fig.tight_layout()
    save(fig, plots_dir, 'epc_02_dft_parity')


def annotate(ax, matrix, fmt, white):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = 'white' if white(value) else 'black'
            ax.text(j, i, format(value, fmt), ha='center', va='center',
                    fontsize=7.2, color=color)


def plot_component_heatmap(results, reference, plots_dir):
    rows = ['Mg-x', 'Mg-y', 'Mg-z', 'O-x', 'O-y', 'O-z']
    columns = q_labels(reference['qpoints'])
    matrices = {}
    for name in ('sr', 'full'):
        values = np.asarray([row['relative_l2']
                             for row in results[name]['by_component_q']])
        matrices[name] = 100 * values.reshape(6, len(columns))
    ratio = matrices['full'] / matrices['sr']
    positive = np.concatenate((matrices['sr'].ravel(),
                               matrices['full'].ravel()))
    norm = LogNorm(vmin=max(float(positive.min()), 1e-4),
                   vmax=float(positive.max()))
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))
    for ax, name, cmap, title in (
            (axes[0], 'sr', 'Blues', 'SR-target (+ fixed LR) error (%)'),
            (axes[1], 'full', 'Oranges', 'Direct full-H error (%)')):
        image = ax.imshow(matrices[name], cmap=cmap, norm=norm, aspect='auto')
        cutoff = np.sqrt(norm.vmin * norm.vmax)
        annotate(ax, matrices[name], '.1f', lambda value: value > cutoff)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
    ratio_max = max(2.0, float(np.max(ratio)))
    image = axes[2].imshow(ratio, cmap='RdBu_r', vmin=0, vmax=ratio_max,
                           aspect='auto')
    annotate(axes[2], ratio, '.1f',
             lambda value: value < 0.25 or value > 0.8 * ratio_max)
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title('Error ratio: Full / SR\n(>1 favors SR-target)')
    for ax in axes:
        ax.set_xticks(range(len(columns)), columns, rotation=35, ha='right')
        ax.set_yticks(range(len(rows)), rows)
        ax.set_xlabel('q point')
    axes[0].set_ylabel('Displaced primitive atom / direction')
    fig.suptitle('Component-resolved EPC error', y=1.01)
    fig.tight_layout()
    save(fig, plots_dir, 'epc_03_component_error_heatmap')


def plot_delta_convergence(full_paths, sr_path, dft_convergence,
                           plots_dir):
    series = {'full': [], 'sr': []}
    for path in full_paths:
        with open(path) as handle:
            data = json.load(handle)
        series['full'].extend(data['comparisons'])
    with open(sr_path) as handle:
        series['sr'].extend(json.load(handle)['comparisons'])
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for name, color, label in (
            ('sr', SR_COLOR, 'SR-target (+ fixed LR)'),
            ('full', FULL_COLOR, 'Direct full-H')):
        dedup = {float(row['delta']): row for row in series[name]}
        xs = np.asarray(sorted(dedup))
        ys = np.asarray([dedup[x]['relative_l2_difference'] for x in xs])
        ax.loglog(xs, ys, 'o-', color=color, lw=1.8, label=label)
    ax.scatter([0.005], [dft_convergence], marker='D', s=55,
               color=DFT_COLOR, label='DFT: 0.005 vs 0.0025 Å', zorder=4)
    ax.axvline(5e-6, color='#777777', ls=':', lw=1,
               label='model production step')
    ax.set_xlabel('Coarser finite-difference step δ (Å)')
    ax.set_ylabel('Relative L2 change to next smaller δ')
    ax.set_title('Finite-difference convergence')
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    save(fig, plots_dir, 'epc_04_finite_difference_convergence')


def component_q_rows(prediction, truth, qpoints):
    rows = []
    for atom, element in enumerate(('Mg', 'O')):
        for alpha, direction in enumerate('xyz'):
            for iq, qpoint in enumerate(qpoints):
                row = metric(prediction[:, iq, atom, alpha],
                             truth[:, iq, atom, alpha])
                row.update({'atom_index': atom, 'element': element,
                            'direction': direction, 'q_index': iq,
                            'qpoint': qpoint.tolist()})
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', default=(
        'runs/epc/actual/structure_primitive/epc_cartesian_actual.h5'))
    parser.add_argument('--actual-coarse', default=(
        'runs/epc/actual_d005/structure_primitive/epc_cartesian_actual.h5'))
    parser.add_argument('--sr', default=(
        'runs/epc/sr/structure_primitive/epc_cartesian_pred.h5'))
    parser.add_argument('--full', default=(
        'runs/epc/full/structure_primitive/epc_cartesian_pred.h5'))
    parser.add_argument('--plots-dir', default='plots')
    parser.add_argument('--metrics', default='workflows/epc/comparison_metrics.json')
    args = parser.parse_args()
    paths = {name: os.path.abspath(getattr(args, name))
             for name in ('actual', 'actual_coarse', 'sr', 'full')}
    data = {name: load_epc(path) for name, path in paths.items()}
    validate_grids(data['actual'], {name: data[name]
                                    for name in ('actual_coarse', 'sr', 'full')})
    truth = data['actual']['g']
    dft_convergence_metrics = metric(data['actual_coarse']['g'], truth)
    dft_convergence = dft_convergence_metrics['relative_l2']
    results = {}
    for name in ('sr', 'full'):
        overall = metric(data[name]['g'], truth)
        by_q, by_component = breakdown(
            data[name]['g'], truth, data['actual']['qpoints'])
        results[name] = {
            'g': data[name]['g'], 'overall': overall,
            'by_q': by_q, 'by_component': by_component,
            'by_component_q': component_q_rows(
                data[name]['g'], truth, data['actual']['qpoints']),
        }

    os.makedirs(args.plots_dir, exist_ok=True)
    plot_q_comparison(results, data['actual'], args.plots_dir,
                      dft_convergence)
    plot_parity(results, data['actual'], args.plots_dir)
    plot_component_heatmap(results, data['actual'], args.plots_dir)
    plot_delta_convergence(
        ('workflows/epc/full_delta_sweep_gauge_fixed.json',
         'workflows/epc/full_delta_sweep_fine.json'),
        'workflows/epc/sr_delta_sweep.json', dft_convergence, args.plots_dir)

    asr = {}
    for name in ('actual', 'actual_coarse', 'sr', 'full'):
        gamma = data[name]['g'][:, 0]
        asr[name] = {
            'max_abs_eV_per_A': float(np.max(np.abs(gamma.sum(axis=1)))),
            'fraction_of_tensor_peak': float(
                np.max(np.abs(gamma.sum(axis=1)))
                / np.max(np.abs(data[name]['g']))),
        }
    hermiticity = {name: hermiticity_metric(data[name])
                   for name in ('actual', 'actual_coarse', 'sr', 'full')}
    sr_error = results['sr']['overall']['relative_l2']
    full_error = results['full']['overall']['relative_l2']
    output = {
        'definition': {
            'quantity': 'Cartesian AO EPC before phonon/band contraction',
            'units': 'eV/Angstrom',
            'q_grid': [2, 2, 2], 'k_grid': [2, 2, 2],
            'energy_gauge': 'H - <H,S0>/<S0,S0> S0',
            'sr_reconstruction': 'H_SR(pred) + H_LR(analytic)',
            'model_delta_angstrom': data['sr']['delta'],
            'dft_delta_angstrom': data['actual']['delta'],
        },
        'paths': paths,
        'sha256': {name: sha256(path) for name, path in paths.items()},
        'dft_delta_convergence': {
            'coarse_delta_angstrom': data['actual_coarse']['delta'],
            'fine_delta_angstrom': data['actual']['delta'],
            'relative_l2_change': dft_convergence,
            'metrics': dft_convergence_metrics,
        },
        'headline': {
            'sr_relative_l2': sr_error,
            'full_relative_l2': full_error,
            'relative_l2_reduction_full_to_sr': 1 - sr_error / full_error,
        },
        'models': {
            name: {key: value for key, value in results[name].items()
                   if key != 'g'} for name in ('sr', 'full')},
        'acoustic_sum_rule': asr,
        'hermiticity': hermiticity,
    }
    with open(args.metrics, 'w') as handle:
        json.dump(output, handle, indent=1)
        handle.write('\n')
    print(f'wrote {args.metrics}')


if __name__ == '__main__':
    main()
