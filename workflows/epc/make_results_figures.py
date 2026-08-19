"""Create the six main MgO EPC comparison figures and their analysis report."""
import argparse
import itertools
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from maceh.analysis.figures import save_figure_formats



SR_COLOR = '#0072B2'
FULL_COLOR = '#D55E00'
DFT_COLOR = '#222222'
GRID_COLOR = '#D9D9D9'
DIRECTION_COLORS = ('#4477AA', '#228833', '#AA3377')
NEIGHBOR_SHIFTS = np.asarray(
    list(itertools.product(range(-2, 3), repeat=3)), dtype=float)


def load_epc(path):
    with h5py.File(path, 'r') as handle:
        return {
            'g': handle['g_real'][:] + 1j * handle['g_imag'][:],
            'kpoints': handle['kpoints'][:],
            'qpoints': handle['qpoints'][:],
            'atom_indices': handle['atom_indices'][:],
            'orbital_indices': handle['orbital_indices'][:],
            'atomic_numbers': handle['atomic_numbers'][:],
            'lattice': handle['lattice'][:],
            'positions': handle['positions'][:],
            'supercell_matrix': handle['supercell_matrix'][:],
        }


def validate_inputs(reference, candidates):
    for name, candidate in candidates.items():
        if candidate['g'].shape != reference['g'].shape:
            raise ValueError(f'{name}: EPC tensor shape does not match DFT')
        for key in ('kpoints', 'qpoints', 'atom_indices', 'orbital_indices',
                    'atomic_numbers', 'lattice', 'positions',
                    'supercell_matrix'):
            if not np.allclose(candidate[key], reference[key], rtol=0,
                               atol=1e-12):
                raise ValueError(f'{name}: {key} does not match DFT')


def complex_metrics(prediction, truth):
    error = prediction - truth
    norm_truth = float(np.linalg.norm(truth))
    norm_prediction = float(np.linalg.norm(prediction))
    return {
        'relative_l2': float(np.linalg.norm(error) / norm_truth),
        'complex_mae_eV_per_A': float(np.mean(np.abs(error))),
        'complex_rmse_eV_per_A': float(
            np.sqrt(np.mean(np.abs(error) ** 2))),
        'cosine_similarity': float(
            np.vdot(truth.ravel(), prediction.ravel()).real
            / (norm_truth * norm_prediction)),
        'norm_ratio_to_dft': norm_prediction / norm_truth,
    }


def style(ax, log_grid=False):
    ax.set_axisbelow(True)
    ax.grid(True, which='both' if log_grid else 'major', color=GRID_COLOR,
            linewidth=0.7, alpha=0.7)
    ax.tick_params(direction='in')


def save_figure(fig, output_dir, stem):
    paths = save_figure_formats(fig, output_dir, stem, bbox_inches='tight')
    for path in paths:
        print(f'wrote {path}')
    plt.close(fig)
    return paths


def q_labels(qpoints):
    def number(value):
        if abs(value) < 1e-12:
            return '0'
        if abs(value - 0.5) < 1e-12:
            return '½'
        return f'{value:g}'
    return ['(' + ','.join(number(value) for value in point) + ')'
            for point in qpoints]


def plot_complex_parity(reference, predictions, summaries, output_dir):
    truth_complex = reference['g'].ravel()
    truth = np.concatenate((truth_complex.real, truth_complex.imag))
    flat = {}
    values = [truth]
    for name in ('sr', 'full'):
        current = predictions[name]['g'].ravel()
        flat[name] = np.concatenate((current.real, current.imag))
        values.append(flat[name])
    limit = float(np.quantile(np.abs(np.concatenate(values)), 0.9995))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), sharex=True,
                             sharey=True)
    for ax, name, cmap, title in (
            (axes[0], 'sr', 'Blues', 'SR-target (+ fixed LR)'),
            (axes[1], 'full', 'Oranges', 'Direct Full-H')):
        prediction = flat[name]
        shown = ((np.abs(truth) <= limit)
                 & (np.abs(prediction) <= limit))
        density = ax.hexbin(
            truth[shown], prediction[shown], gridsize=90, bins='log',
            mincnt=1, extent=(-limit, limit, -limit, limit), cmap=cmap)
        ax.plot([-limit, limit], [-limit, limit], '--', color=DFT_COLOR,
                linewidth=1.1)
        result = summaries[name]
        ax.text(
            0.04, 0.96,
            f"relative L2 = {100 * result['relative_l2']:.2f}%\n"
            f"complex MAE = {result['complex_mae_eV_per_A']:.3f} eV/Å\n"
            f"cosine = {result['cosine_similarity']:.4f}",
            transform=ax.transAxes, va='top', fontsize=9,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.9,
                  'pad': 4})
        ax.set_title(title)
        ax.set_xlabel('Actual DFT EPC component (eV/Å)')
        ax.set_aspect('equal', adjustable='box')
        style(ax)
        fig.colorbar(density, ax=ax, label='component count')
    axes[0].set_ylabel('Predicted EPC component (eV/Å)')
    fig.suptitle('Actual vs predicted Cartesian-AO EPC parity\n'
                 '(real and imaginary parts; common 99.95% plotting range)')
    fig.tight_layout()
    save_figure(fig, output_dir,
                'epc_results_01_actual_predicted_parity')


def component_mae(reference, predictions):
    output = {}
    for name in ('sr', 'full'):
        output[name] = [float(np.mean(np.abs(
            predictions[name]['g'][:, :, :, alpha]
            - reference['g'][:, :, :, alpha]))) for alpha in range(3)]
    return output


def plot_component_mae(values, output_dir):
    xpos = np.arange(3)
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for shift, name, color, label in (
            (-width / 2, 'sr', SR_COLOR, 'SR-target (+ fixed LR)'),
            (width / 2, 'full', FULL_COLOR, 'Direct Full-H')):
        bars = ax.bar(xpos + shift, values[name], width, color=color,
                      label=label)
        ax.bar_label(bars, labels=[f'{value:.3f}' for value in values[name]],
                     padding=3, fontsize=8)
    ax.set_xticks(xpos, ('x', 'y', 'z'))
    ax.set_xlabel('Displacement direction α')
    ax.set_ylabel('Complex MAE (eV/Å)')
    ax.set_title('Cartesian component EPC error')
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.16 * max(values['full']))
    style(ax)
    fig.tight_layout()
    save_figure(fig, output_dir,
                'epc_results_02_cartesian_component_mae')


def magnitude_metrics(prediction, truth, floor):
    truth_magnitude = np.abs(truth).ravel()
    prediction_magnitude = np.abs(prediction).ravel()
    shown = ((truth_magnitude >= floor)
             | (prediction_magnitude >= floor))
    truth_power = float(np.square(truth_magnitude).sum())
    return {
        'magnitude_mae_eV_per_A': float(np.mean(np.abs(
            prediction_magnitude - truth_magnitude))),
        'selected_magnitude_mae_eV_per_A': float(np.mean(np.abs(
            prediction_magnitude[shown] - truth_magnitude[shown]))),
        'shown_fraction': float(np.mean(shown)),
        'dft_power_fraction_shown': float(
            np.square(truth_magnitude[shown]).sum() / truth_power),
        'shown_count': int(shown.sum()),
        'total_count': int(shown.size),
    }


def plot_magnitude_parity(reference, predictions, metrics, floor, output_dir):
    truth = np.abs(reference['g']).ravel()
    maximum = max(float(np.max(truth)),
                  *(float(np.max(np.abs(predictions[name]['g'])))
                    for name in ('sr', 'full')))
    low = np.log10(floor)
    high = np.log10(maximum * 1.08)
    ticks = np.arange(np.ceil(low), np.floor(high) + 1, 2, dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharex=True,
                             sharey=True)
    for ax, name, cmap, title in (
            (axes[0], 'sr', 'Blues', 'SR-target (+ fixed LR)'),
            (axes[1], 'full', 'Oranges', 'Direct Full-H')):
        prediction = np.abs(predictions[name]['g']).ravel()
        shown = (truth >= floor) | (prediction >= floor)
        x = np.log10(np.maximum(truth[shown], floor))
        y = np.log10(np.maximum(prediction[shown], floor))
        density = ax.hexbin(x, y, gridsize=85, bins='log', mincnt=1,
                            extent=(low, high, low, high), cmap=cmap)
        ax.plot([low, high], [low, high], '--', color=DFT_COLOR,
                linewidth=1.1)
        result = metrics[name]
        ax.text(
            0.04, 0.96,
            f"magnitude MAE = {result['magnitude_mae_eV_per_A']:.3f} eV/Å\n"
            f"shown = {100 * result['shown_fraction']:.1f}% of components\n"
            f"DFT power retained = {100 * result['dft_power_fraction_shown']:.4f}%",
            transform=ax.transAxes, va='top', fontsize=8.5,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.9,
                  'pad': 4})
        ax.set_title(title)
        ax.set_xlabel('Actual |g| (eV/Å)')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(ticks, [rf'$10^{{{tick}}}$' for tick in ticks])
        ax.set_yticks(ticks, [rf'$10^{{{tick}}}$' for tick in ticks])
        style(ax)
        fig.colorbar(density, ax=ax, label='component count')
    axes[0].set_ylabel('Predicted |g| (eV/Å)')
    fig.suptitle('Actual vs predicted EPC magnitude\n'
                 f'(values below {floor:g} eV/Å shown at the plotting floor)')
    fig.tight_layout()
    save_figure(fig, output_dir, 'epc_results_03_magnitude_parity')


def _atom_slices(orbital_indices):
    slices = []
    for atom in np.unique(orbital_indices):
        indices = np.flatnonzero(orbital_indices == atom)
        if not np.array_equal(indices,
                              np.arange(indices[0], indices[-1] + 1)):
            raise ValueError('orbitals belonging to an atom are not contiguous')
        slices.append(slice(int(indices[0]), int(indices[-1]) + 1))
    return slices


def nearest_periodic_distance(vector, lattice):
    """Shortest length of ``vector`` modulo a possibly skewed lattice."""
    fractional = np.asarray(vector, dtype=float) @ np.linalg.inv(lattice)
    centered = fractional - np.round(fractional)
    candidates = (centered[None, :] - NEIGHBOR_SHIFTS) @ lattice
    return float(np.min(np.linalg.norm(candidates, axis=1)))


def _distance_rows(sums, bin_width):
    rows = []
    for index in sorted(sums):
        source = sums[index]
        count = source['count']
        sr_mae = source['sr_abs_error_sum'] / count
        full_mae = source['full_abs_error_sum'] / count
        rows.append({
            'lower_A': index * bin_width,
            'upper_A': (index + 1) * bin_width,
            'center_A': (index + 0.5) * bin_width,
            'ao_element_count': count,
            'atom_block_count': source['block_count'],
            'sr_mae_eV_per_A': sr_mae,
            'full_mae_eV_per_A': full_mae,
            'dft_mean_abs_eV_per_A': source['dft_abs_sum'] / count,
            'full_error_removed_by_sr_fraction': (
                (full_mae - sr_mae) / full_mae),
        })
    return rows


def distance_binned_errors(paths, pair_bin_width=1.0,
                           perturbation_bin_width=0.5,
                           occupancy_threshold=1e-12):
    """MAE of real-space dH blocks under two complementary distances.

    ``pair_separation`` is the unwrapped bra--ket AO-center distance and tests
    errors on long hopping blocks. ``perturbation_midpoint`` is the minimum
    periodic distance from the displaced atom to the AO-pair midpoint and tests
    how far the perturbation propagates. The latter cannot be interpreted as an
    isolated distance beyond the displacement supercell's image-free radius.

    A dense stored (p,R) matrix contains structural padding for atom pairs that
    do not have a graph block. An atom-pair submatrix is counted only when at
    least one of DFT/SR/Full has a value above ``occupancy_threshold``; once a
    block is present, all of its AO elements (including exact zeros) count.
    """
    widths = {
        'pair_separation': pair_bin_width,
        'perturbation_midpoint': perturbation_bin_width,
    }
    sums = {name: {} for name in widths}
    with h5py.File(paths['actual'], 'r') as actual, \
            h5py.File(paths['sr'], 'r') as sr, \
            h5py.File(paths['full'], 'r') as full:
        lattice = actual['lattice'][:]
        positions = actual['positions'][:]
        supercell_lattice = actual['supercell_matrix'][:] @ lattice
        orbital_indices = actual['orbital_indices'][:]
        atom_slices = _atom_slices(orbital_indices)
        for other in (sr, full):
            for field in ('lattice', 'positions', 'orbital_indices'):
                if not np.allclose(other[field][:], actual[field][:], rtol=0,
                                   atol=1e-12):
                    raise ValueError(f'dH files disagree in {field}')

        for kappa in actual['dH']:
            for direction in ('x', 'y', 'z'):
                groups = {name: handle[f'dH/{kappa}/{direction}']
                          for name, handle in (
                              ('actual', actual), ('sr', sr), ('full', full))}
                key_set = set(groups['actual'])
                if set(groups['sr']) != key_set or set(groups['full']) != key_set:
                    raise ValueError(
                        f'dH key sets differ for atom {kappa} {direction}')
                for key in sorted(key_set):
                    labels = json.loads(key)
                    if len(labels) != 6:
                        raise ValueError(f'invalid dH key: {key}')
                    displaced_cell = np.asarray(labels[:3], dtype=float)
                    lattice_r = np.asarray(labels[3:], dtype=float)
                    matrices = {name: group[key][:]
                                for name, group in groups.items()}
                    for i, row_slice in enumerate(atom_slices):
                        for j, col_slice in enumerate(atom_slices):
                            block_slice = (row_slice, col_slice)
                            blocks = {name: matrix[block_slice]
                                      for name, matrix in matrices.items()}
                            peak = max(float(np.max(np.abs(block)))
                                       for block in blocks.values())
                            if peak < occupancy_threshold:
                                continue
                            bra = positions[i]
                            ket = positions[j] + lattice_r @ lattice
                            displaced = (positions[int(kappa)]
                                         + displaced_cell @ lattice)
                            distances = {
                                'pair_separation': float(np.linalg.norm(
                                    ket - bra)),
                                'perturbation_midpoint': (
                                    nearest_periodic_distance(
                                        0.5 * (bra + ket) - displaced,
                                        supercell_lattice)),
                            }
                            count = blocks['actual'].size
                            sr_error = float(np.sum(np.abs(
                                blocks['sr'] - blocks['actual'])))
                            full_error = float(np.sum(np.abs(
                                blocks['full'] - blocks['actual'])))
                            dft_abs = float(np.sum(np.abs(blocks['actual'])))
                            for name, distance in distances.items():
                                index = int(np.floor(
                                    distance / widths[name] + 1e-10))
                                row = sums[name].setdefault(index, {
                                    'count': 0, 'block_count': 0,
                                    'sr_abs_error_sum': 0.0,
                                    'full_abs_error_sum': 0.0,
                                    'dft_abs_sum': 0.0,
                                })
                                row['count'] += count
                                row['block_count'] += 1
                                row['sr_abs_error_sum'] += sr_error
                                row['full_abs_error_sum'] += full_error
                                row['dft_abs_sum'] += dft_abs

    return {name: _distance_rows(sums[name], widths[name])
            for name in widths}


def aggregate_distance_rows(rows, minimum=0.0, maximum=np.inf):
    chosen = [row for row in rows
              if row['center_A'] >= minimum and row['center_A'] < maximum]
    count = sum(row['ao_element_count'] for row in chosen)
    if count == 0:
        raise ValueError('distance range contains no AO elements')
    sr = sum(row['sr_mae_eV_per_A'] * row['ao_element_count']
             for row in chosen) / count
    full = sum(row['full_mae_eV_per_A'] * row['ao_element_count']
               for row in chosen) / count
    return {
        'minimum_A': minimum,
        'maximum_A': None if not np.isfinite(maximum) else maximum,
        'ao_element_count': count,
        'sr_mae_eV_per_A': sr,
        'full_mae_eV_per_A': full,
        'full_error_removed_by_sr_fraction': (full - sr) / full,
    }


def _plot_distance_mae_panel(ax, rows, title, xlabel, shade_start,
                             shade_label, boundary_line=False):
    x = np.asarray([row['center_A'] for row in rows])
    sr = np.asarray([row['sr_mae_eV_per_A'] for row in rows])
    full = np.asarray([row['full_mae_eV_per_A'] for row in rows])
    upper = max(row['upper_A'] for row in rows)
    ax.semilogy(x, sr, 'o-', color=SR_COLOR, linewidth=2,
                label='SR-target (+ fixed LR)')
    ax.semilogy(x, full, 's-', color=FULL_COLOR, linewidth=2,
                label='Direct Full-H')
    ax.axvspan(shade_start, upper, color='#777777', alpha=0.08,
               label=shade_label)
    if boundary_line:
        ax.axvline(shade_start, color='#777777', linestyle=':', linewidth=1)
    ax.set_xlim(0, upper)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.text(0.98, 0.05, shade_label, transform=ax.transAxes, ha='right',
            va='bottom', fontsize=7.5, color='#555555')
    style(ax, log_grid=True)


def plot_distance_mae(distance_rows, image_free_radius, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), sharey=True)
    _plot_distance_mae_panel(
        axes[0], distance_rows['perturbation_midpoint'],
        'Perturbation propagation',
        'Displaced atom → AO-pair midpoint (Å)', image_free_radius,
        f'periodic-image regime (>{image_free_radius:.2f} Å)', True)
    _plot_distance_mae_panel(
        axes[1], distance_rows['pair_separation'],
        'Long-hopping blocks', 'Bra–ket AO-center separation (Å)', 7.0,
        'far AO pairs (≥7 Å)')
    axes[0].set_ylabel('Real-space dH/dτ complex MAE (eV/Å)')
    handles, labels = axes[0].get_legend_handles_labels()
    method_handles, method_labels = handles[:2], labels[:2]
    fig.legend(method_handles, method_labels, loc='upper center', ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle('Real-space Cartesian EPC error vs distance', y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_dir, 'epc_results_04_mae_vs_ao_distance')


def _plot_improvement_panel(ax, rows, title, xlabel, shade_start,
                            boundary_line=False):
    x = np.asarray([row['center_A'] for row in rows])
    improvement = 100 * np.asarray([
        row['full_error_removed_by_sr_fraction'] for row in rows])
    colors = np.where(improvement >= 0, SR_COLOR, FULL_COLOR)
    width = 0.82 * (rows[0]['upper_A'] - rows[0]['lower_A'])
    upper = max(row['upper_A'] for row in rows)
    bars = ax.bar(x, improvement, width=width, color=colors,
                  edgecolor='white', linewidth=0.6)
    ax.axhline(0, color=DFT_COLOR, linewidth=1)
    ax.axvspan(shade_start, upper, color='#777777', alpha=0.08)
    if boundary_line:
        ax.axvline(shade_start, color='#777777', linestyle=':', linewidth=1)
    for bar, value in zip(bars, improvement):
        ax.text(bar.get_x() + bar.get_width() / 2,
                value + (1.8 if value >= 0 else -1.8), f'{value:.0f}%',
                ha='center', va='bottom' if value >= 0 else 'top',
                fontsize=7.5)
    ax.set_xlim(0, upper)
    ax.set_ylim(-10, 106)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    style(ax)


def plot_distance_improvement(distance_rows, image_free_radius, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    _plot_improvement_panel(
        axes[0], distance_rows['perturbation_midpoint'],
        'Perturbation propagation',
        'Displaced atom → AO-pair midpoint (Å)', image_free_radius, True)
    _plot_improvement_panel(
        axes[1], distance_rows['pair_separation'],
        'Long-hopping blocks', 'Bra–ket AO-center separation (Å)', 7.0)
    axes[0].set_ylabel('Full-H error reduction with SR-target model (%)')
    fig.suptitle('Distance-resolved checkpoint comparison\n'
                 '(positive values favor SR-target + fixed LR)')
    fig.tight_layout()
    save_figure(fig, output_dir,
                'epc_results_05_full_error_removed_vs_distance')


def weighted_quantile(values, weights, quantiles):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    valid = (np.isfinite(values) & np.isfinite(weights) & (weights > 0))
    if not valid.any():
        raise ValueError('weighted quantile has no positive finite weights')
    order = np.argsort(values[valid])
    sorted_values = values[valid][order]
    sorted_weights = weights[valid][order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, sorted_values)


def angular_data(prediction, truth):
    """Signed Hermitian angle between complex Cartesian EPC 3-vectors."""
    truth_vectors = np.moveaxis(truth, 3, -1)
    prediction_vectors = np.moveaxis(prediction, 3, -1)

    def calculate(reference, estimate):
        reference = reference.reshape(-1, 3)
        estimate = estimate.reshape(-1, 3)
        ref_norm = np.linalg.norm(reference, axis=1)
        pred_norm = np.linalg.norm(estimate, axis=1)
        weights = np.square(ref_norm)
        denominator = ref_norm * pred_norm
        cosine = np.zeros_like(denominator)
        valid = denominator > 0
        cosine[valid] = (np.real(np.sum(
            np.conj(reference[valid]) * estimate[valid], axis=1))
            / denominator[valid])
        cosine = np.clip(cosine, -1.0, 1.0)
        angles = np.degrees(np.arccos(cosine))
        return angles, weights, cosine

    angles, weights, cosine = calculate(
        truth_vectors, prediction_vectors)
    valid = weights > 0
    quantiles = weighted_quantile(
        angles[valid], weights[valid], (0.5, 0.9, 0.99))
    by_q = []
    for iq in range(truth.shape[1]):
        q_angles, q_weights, q_cosine = calculate(
            truth_vectors[:, iq], prediction_vectors[:, iq])
        q_valid = q_weights > 0
        by_q.append({
            'q_index': iq,
            'power_weighted_mean_angle_deg': float(np.sum(
                q_weights[q_valid] * q_angles[q_valid])
                / np.sum(q_weights[q_valid])),
            'power_weighted_mean_cosine': float(np.sum(
                q_weights[q_valid] * q_cosine[q_valid])
                / np.sum(q_weights[q_valid])),
        })
    return {
        'angles': angles[valid],
        'weights': weights[valid],
        'summary': {
            'power_weighted_mean_angle_deg': float(np.sum(
                weights[valid] * angles[valid]) / np.sum(weights[valid])),
            'power_weighted_mean_cosine': float(np.sum(
                weights[valid] * cosine[valid]) / np.sum(weights[valid])),
            'power_weighted_median_angle_deg': float(quantiles[0]),
            'power_weighted_90pct_angle_deg': float(quantiles[1]),
            'power_weighted_99pct_angle_deg': float(quantiles[2]),
            'vector_count': int(valid.sum()),
        },
        'by_q': by_q,
    }


def plot_angular_accuracy(angular, qpoints, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    for name, color, label in (
            ('sr', SR_COLOR, 'SR-target (+ fixed LR)'),
            ('full', FULL_COLOR, 'Direct Full-H')):
        angles = angular[name]['angles']
        weights = angular[name]['weights']
        order = np.argsort(angles)
        sorted_angles = angles[order]
        cumulative = 100 * np.cumsum(weights[order]) / np.sum(weights)
        keep = np.unique(np.linspace(
            0, len(sorted_angles) - 1, min(3000, len(sorted_angles)),
            dtype=int))
        axes[0].plot(sorted_angles[keep], cumulative[keep], color=color,
                     linewidth=2, label=label)
        by_q = angular[name]['by_q']
        values = [row['power_weighted_mean_angle_deg'] for row in by_q]
        shift = -0.18 if name == 'sr' else 0.18
        axes[1].bar(np.arange(len(qpoints)) + shift, values, width=0.36,
                    color=color, label=label)

    q99 = max(angular[name]['summary']['power_weighted_99pct_angle_deg']
              for name in ('sr', 'full'))
    axes[0].set_xlim(0, min(180, max(25, 1.12 * q99)))
    axes[0].set_ylim(0, 100.5)
    axes[0].set_xlabel('Cartesian-vector angular error (degrees)')
    axes[0].set_ylabel('Cumulative DFT coupling power (%)')
    axes[0].set_title('Power-weighted angular-error distribution')
    axes[0].legend(frameon=False)
    axes[1].set_xticks(range(len(qpoints)), q_labels(qpoints), rotation=35,
                       ha='right')
    axes[1].set_xlabel('q point')
    axes[1].set_ylabel('Power-weighted mean angle (degrees)')
    axes[1].set_title('Direction error by q point')
    axes[1].legend(frameon=False, fontsize=8.5)
    for ax in axes:
        style(ax)
    fig.suptitle('Cartesian EPC direction accuracy\n'
                 r'$\theta=\cos^{-1}[\mathrm{Re}(g_{DFT}^{\dagger}g_{pred})/'
                 r'(\|g_{DFT}\|\|g_{pred}\|)]$')
    fig.tight_layout()
    save_figure(fig, output_dir,
                'epc_results_06_angular_direction_accuracy')


def write_markdown(path, report):
    component = report['cartesian_component_mae_eV_per_A']
    angular = report['angular_direction_accuracy']
    summary = report['distance_summary']
    far = summary['pair_far_ge_7A']
    perturbation_inner = summary['perturbation_inside_image_free_bins']
    perturbation_periodic = summary['perturbation_periodic_image_bins']
    pair_rows = report['distance_bins']['pair_separation']
    perturbation_rows = report['distance_bins']['perturbation_midpoint']
    pair_lines = '\n'.join(
        f"| {row['lower_A']:.0f}–{row['upper_A']:.0f} | "
        f"{row['sr_mae_eV_per_A']:.4g} | "
        f"{row['full_mae_eV_per_A']:.4g} | "
        f"{100 * row['full_error_removed_by_sr_fraction']:.1f}% |"
        for row in pair_rows)
    perturbation_lines = '\n'.join(
        f"| {row['lower_A']:.1f}–{row['upper_A']:.1f} | "
        f"{row['sr_mae_eV_per_A']:.4g} | "
        f"{row['full_mae_eV_per_A']:.4g} | "
        f"{100 * row['full_error_removed_by_sr_fraction']:.1f}% |"
        for row in perturbation_rows)
    text = f"""# EPC results-figure analysis

## Main result

Across the full Cartesian-AO tensor, the SR-target checkpoint with the fixed
analytic LR wrapper enabled has
**{100 * report['overall']['sr']['relative_l2']:.2f}%** relative L2 error,
versus **{100 * report['overall']['full']['relative_l2']:.2f}%** for direct
Full-H. Its relative L2 error is
**{100 * report['overall']['full_error_removed_by_sr_fraction']:.2f}% lower**
than Full-H's.

This is a checkpoint comparison, not an isolated analytic-LR effect. The
controlled A/B decomposition in `EPC_PIPELINE_DECOMPOSITION.md` shows that
adding the fixed LR term to this SR checkpoint changes relative L2 error by
only 0.000078 percentage points on the 2×2×2 grid. The EPC advantage is already
present in the SR-only checkpoint.

The result is not driven by one Cartesian axis:

| direction | SR MAE (eV/Å) | Full-H MAE (eV/Å) |
|---|---:|---:|
| x | {component['sr'][0]:.3f} | {component['full'][0]:.3f} |
| y | {component['sr'][1]:.3f} | {component['full'][1]:.3f} |
| z | {component['sr'][2]:.3f} | {component['full'][2]:.3f} |

## Long-range behavior

The distance figures report two complementary quantities from the stored
real-space Cartesian `dH/dtau` blocks before the k/q Fourier transform:

1. Minimum periodic distance from the displaced atom to the bra–ket AO-pair
   midpoint. This measures perturbation propagation. The 2×2×2 displacement
   supercell has an image-free radius of
   **{report['definition']['image_free_radius_A']:.2f} Å**, so distances beyond
   it mix the perturbation with its periodic images and are shaded accordingly.
2. Unwrapped bra–ket separation `|r_j + R - r_i|`. This measures errors on
   long-hopping blocks, not propagation distance.

Every occupied atom-pair block contributes all of its AO matrix elements,
including exact zeros. Within the whole bins inside the image-free radius, SR
has **{100 * perturbation_inner['full_error_removed_by_sr_fraction']:.1f}% lower**
error than Full-H. In the periodic-image bins its error is
**{100 * perturbation_periodic['full_error_removed_by_sr_fraction']:.1f}% lower**,
but that latter percentage is a supercell-periodic diagnostic rather than an
isolated-distance result.

For AO pairs at or beyond 7 Å, the SR MAE is
**{far['sr_mae_eV_per_A']:.4g} eV/Å**, compared with
**{far['full_mae_eV_per_A']:.4g} eV/Å** for Full-H, a
**{100 * far['full_error_removed_by_sr_fraction']:.1f}% lower** error for the
SR-target pipeline. The A/B decomposition shows this difference cannot be
causally assigned to the analytic LR addition on this grid. It is also not
proof that the tail is quantitatively accurate: beyond about 8 Å the mean DFT
block magnitude is smaller than either model's MAE. The SR-target checkpoint
mostly suppresses the much larger spurious Full-H tail.

### Perturbation-to-pair-midpoint distance

| distance (Å) | SR MAE | Full-H MAE | Full-H error removed by SR |
|---:|---:|---:|---:|
{perturbation_lines}

### Bra–ket AO separation

| distance (Å) | SR MAE | Full-H MAE | Full-H error removed by SR |
|---:|---:|---:|---:|
{pair_lines}

## Cartesian direction accuracy

For each fixed `(k,q,kappa,i,j)`, the three complex x/y/z values form one
Cartesian vector. The angle uses the real Hermitian inner product and is
weighted by DFT coupling power `||g_DFT||^2`, so symmetry-zero elements do not
dominate the statistic.

| method | mean angle | median angle | 90% power angle | mean cosine |
|---|---:|---:|---:|---:|
| SR-target (+ fixed LR) | {angular['sr']['power_weighted_mean_angle_deg']:.2f}° | {angular['sr']['power_weighted_median_angle_deg']:.2f}° | {angular['sr']['power_weighted_90pct_angle_deg']:.2f}° | {angular['sr']['power_weighted_mean_cosine']:.4f} |
| Direct Full-H | {angular['full']['power_weighted_mean_angle_deg']:.2f}° | {angular['full']['power_weighted_median_angle_deg']:.2f}° | {angular['full']['power_weighted_90pct_angle_deg']:.2f}° | {angular['full']['power_weighted_mean_cosine']:.4f} |

## Figure files

1. `epc_results_01_actual_predicted_parity` — actual-vs-predicted complex EPC.
2. `epc_results_02_cartesian_component_mae` — x/y/z complex MAE.
3. `epc_results_03_magnitude_parity` — actual-vs-predicted `|g|`.
4. `epc_results_04_mae_vs_ao_distance` — real-space EPC MAE vs perturbation and AO-pair distances.
5. `epc_results_05_full_error_removed_vs_distance` — percent of Full-H error removed by SR under both distances.
6. `epc_results_06_angular_direction_accuracy` — Cartesian-vector angular error.

Every figure is available as PNG and PDF in this directory. Numerical values
and definitions are in `epc_results_analysis.json`.

## Scope

These are Cartesian AO Hamiltonian derivatives on the 2×2×2 k/q grid. They are
not yet band- and phonon-mode-resolved EPCs and do not include the downstream
phonon eigenvectors, mass/frequency factors, electronic eigenvectors, or
`dS/dtau` contribution.
"""
    with open(path, 'w') as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', default=(
        'runs/epc/actual/structure_primitive/epc_cartesian_actual.h5'))
    parser.add_argument('--sr', default=(
        'runs/epc/sr/structure_primitive/epc_cartesian_pred.h5'))
    parser.add_argument('--full', default=(
        'runs/epc/full/structure_primitive/epc_cartesian_pred.h5'))
    parser.add_argument('--output-dir', default='plots')
    parser.add_argument('--magnitude-floor', type=float, default=1e-6)
    parser.add_argument('--pair-distance-bin-width', type=float, default=1.0)
    parser.add_argument('--perturbation-distance-bin-width', type=float,
                        default=0.5)
    args = parser.parse_args()
    if (args.magnitude_floor <= 0 or args.pair_distance_bin_width <= 0
            or args.perturbation_distance_bin_width <= 0):
        raise SystemExit('floors and bin widths must be positive')

    paths = {name: os.path.abspath(getattr(args, name))
             for name in ('actual', 'sr', 'full')}
    data = {name: load_epc(path) for name, path in paths.items()}
    validate_inputs(data['actual'], {'sr': data['sr'], 'full': data['full']})
    predictions = {'sr': data['sr'], 'full': data['full']}
    summaries = {name: complex_metrics(data[name]['g'], data['actual']['g'])
                 for name in ('sr', 'full')}
    components = component_mae(data['actual'], predictions)
    magnitudes = {name: magnitude_metrics(
        data[name]['g'], data['actual']['g'], args.magnitude_floor)
        for name in ('sr', 'full')}
    distance_rows = distance_binned_errors(
        paths, pair_bin_width=args.pair_distance_bin_width,
        perturbation_bin_width=args.perturbation_distance_bin_width)
    pair_near = aggregate_distance_rows(
        distance_rows['pair_separation'], maximum=7.0)
    pair_far = aggregate_distance_rows(
        distance_rows['pair_separation'], minimum=7.0)
    supercell_lattice = (data['actual']['supercell_matrix']
                         @ data['actual']['lattice'])
    inverse_supercell = np.linalg.inv(supercell_lattice)
    image_free_radius = 0.5 * min(
        1.0 / np.linalg.norm(inverse_supercell[:, axis])
        for axis in range(3))
    # The image-free radius is 2.474 A and the bin boundary is 2.5 A.
    # Summarize whole bins rather than splitting one after it was accumulated.
    perturbation_boundary = (
        np.ceil(image_free_radius / args.perturbation_distance_bin_width)
        * args.perturbation_distance_bin_width)
    perturbation_inner = aggregate_distance_rows(
        distance_rows['perturbation_midpoint'],
        maximum=perturbation_boundary)
    perturbation_periodic = aggregate_distance_rows(
        distance_rows['perturbation_midpoint'],
        minimum=perturbation_boundary)
    angular = {name: angular_data(data[name]['g'], data['actual']['g'])
               for name in ('sr', 'full')}

    os.makedirs(args.output_dir, exist_ok=True)
    plot_complex_parity(data['actual'], predictions, summaries,
                        args.output_dir)
    plot_component_mae(components, args.output_dir)
    plot_magnitude_parity(data['actual'], predictions, magnitudes,
                          args.magnitude_floor, args.output_dir)
    plot_distance_mae(distance_rows, image_free_radius, args.output_dir)
    plot_distance_improvement(
        distance_rows, image_free_radius, args.output_dir)
    plot_angular_accuracy(angular, data['actual']['qpoints'], args.output_dir)

    report = {
        'definition': {
            'quantity': 'Cartesian AO EPC dH/dtau',
            'units': 'eV/Angstrom',
            'tensor_shape': list(data['actual']['g'].shape),
            'distance_metrics': {
                'pair_separation': (
                    '|r_j + R - r_i| between bra and ket AO centers'),
                'perturbation_midpoint': (
                    'minimum periodic distance from displaced atom to '
                    'the bra-ket AO-pair midpoint'),
            },
            'distance_source': 'real-space dH groups before Fourier transform',
            'pair_distance_bin_width_A': args.pair_distance_bin_width,
            'perturbation_distance_bin_width_A': (
                args.perturbation_distance_bin_width),
            'image_free_radius_A': image_free_radius,
            'angular_error': (
                'acos(Re(g_DFT^dagger g_pred) / '
                '(norm(g_DFT) norm(g_pred))) over complex Cartesian xyz vectors'),
            'angular_weight': 'norm(g_DFT)^2',
            'magnitude_plot_floor_eV_per_A': args.magnitude_floor,
        },
        'paths': paths,
        'overall': {
            'sr': summaries['sr'],
            'full': summaries['full'],
            'full_error_removed_by_sr_fraction': (
                1 - summaries['sr']['relative_l2']
                / summaries['full']['relative_l2']),
        },
        'cartesian_component_mae_eV_per_A': components,
        'magnitude': magnitudes,
        'distance_bins': distance_rows,
        'distance_summary': {
            'pair_near_lt_7A': pair_near,
            'pair_far_ge_7A': pair_far,
            'perturbation_inside_image_free_bins': perturbation_inner,
            'perturbation_periodic_image_bins': perturbation_periodic,
            'perturbation_bin_boundary_A': perturbation_boundary,
        },
        'angular_direction_accuracy': {
            name: {
                **angular[name]['summary'],
                'by_q': angular[name]['by_q'],
            } for name in ('sr', 'full')
        },
    }
    json_path = os.path.join(args.output_dir, 'epc_results_analysis.json')
    with open(json_path, 'w') as handle:
        json.dump(report, handle, indent=1)
        handle.write('\n')
    print(f'wrote {json_path}')
    markdown_path = os.path.join(
        args.output_dir, 'EPC_RESULTS_ANALYSIS.md')
    write_markdown(markdown_path, report)
    print(f'wrote {markdown_path}')


if __name__ == '__main__':
    main()
