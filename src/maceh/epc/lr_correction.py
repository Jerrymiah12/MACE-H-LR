"""Analytic long-range reconstruction during SR-model EPC inference.

Material-agnostic: species, Born charges, and the dielectric tensor all come
from the reference directory at call time (``element.dat``,
``species_order.json``), so nothing here is specific to MgO even though MgO is
the only system it has been exercised on.

The trained SR network predicts ``H_SR``.  A physically comparable derivative
must finite-difference ``H_SR + H_LR``, not ``H_SR`` alone.  At the equilibrium
geometry used for EPC, ``V_LR(0) = 0`` and

    d[V_LR(u) S(u)] / du |0 = dV_LR/du |0 S(0),

so the equilibrium DFT overlap gives the exact first-order LR correction; an
overlap predictor is unnecessary.  This module deliberately supports only a
supercell whose equilibrium geometry exactly matches the supplied overlap
snapshot, and fails closed otherwise.
"""
import json
import os

import numpy as np

from maceh.config import load_config
from maceh.data.io.blocks import parse_key, read_blocks
from maceh.data.structures import reciprocal, remove_uniform_translation
from maceh.response.long_range import (assemble_lr_hamiltonian,
                                       evaluate_potential, gmax_squared,
                                       lr_coefficients, reciprocal_set)
from maceh.response.provenance import require_current_lr_definition


def _load_snapshot_geometry(folder):
    cell = np.loadtxt(os.path.join(folder, 'lat.dat')).T
    positions = np.loadtxt(os.path.join(folder, 'site_positions.dat')).T
    numbers = np.loadtxt(os.path.join(folder, 'element.dat'), dtype=int)
    return cell, positions, np.atleast_1d(numbers)


def _target_to_source(target_positions, target_numbers,
                      source_positions, source_numbers, tol=1e-7):
    """Map target atom indices to an ordering-equivalent source geometry."""
    mapping = []
    used = set()
    for position, number in zip(target_positions, target_numbers):
        candidates = [i for i, (other, z) in enumerate(
            zip(source_positions, source_numbers))
            if i not in used and int(z) == int(number)
            and np.linalg.norm(other - position) <= tol]
        if len(candidates) != 1:
            raise ValueError(
                'analytic-LR overlap geometry does not uniquely match the '
                'EPC displacement supercell')
        mapping.append(candidates[0])
        used.add(candidates[0])
    return np.asarray(mapping, dtype=int)


def _reindex_blocks_source_to_target(blocks, target_to_source):
    source_to_target = np.empty_like(target_to_source)
    source_to_target[target_to_source] = np.arange(len(target_to_source))
    out = {}
    for key, value in blocks.items():
        r0, r1, r2, source_i, source_j = parse_key(key)
        target_i = int(source_to_target[source_i - 1]) + 1
        target_j = int(source_to_target[source_j - 1]) + 1
        new_key = str([r0, r1, r2, target_i, target_j])
        if new_key in out:
            raise ValueError(f'duplicate overlap key after reindexing: {new_key}')
        out[new_key] = value
    return out


def load_reindexed_equilibrium_overlap(overlap_dir, sc_struct):
    """Load an equilibrium overlap and map its atom order to ``sc_struct``."""
    source_cell, source_positions, source_numbers = \
        _load_snapshot_geometry(overlap_dir)
    reference_positions = np.asarray(sc_struct.positions, dtype=np.float64)
    reference_numbers = np.asarray(sc_struct.numbers, dtype=int)
    if not np.allclose(source_cell, sc_struct.lattice, atol=1e-7):
        raise ValueError('equilibrium overlap cell differs from EPC supercell')
    target_to_source = _target_to_source(
        reference_positions, reference_numbers,
        source_positions, source_numbers)
    return _reindex_blocks_source_to_target(
        read_blocks(os.path.join(overlap_dir, 'overlaps.h5')),
        target_to_source)


def project_hamiltonian_gauge(blocks, overlaps):
    """Return ``H - <H,S>/<S,S> S`` and the removed scalar coefficient."""
    denominator = sum(float(np.square(value).sum())
                      for value in overlaps.values())
    if denominator <= 0:
        raise ValueError('equilibrium overlap has zero norm')
    numerator = sum(float(np.sum(blocks[key] * value))
                    for key, value in overlaps.items() if key in blocks)
    coefficient = numerator / denominator
    projected = {key: np.asarray(value).copy()
                 for key, value in blocks.items()}
    missing = set(overlaps) - set(projected)
    if missing:
        raise ValueError(
            f'equilibrium overlap has {len(missing)} blocks absent from the '
            'EPC graph; increase [data] radius')
    for key, overlap in overlaps.items():
        projected[key] -= coefficient * overlap
    return projected, coefficient


def reconstruct_total_hamiltonian(sr_blocks, lr_blocks):
    """Return exactly ``H_SR + H_LR`` and enforce the reconstruction identity."""
    missing = set(lr_blocks) - set(sr_blocks)
    if missing:
        raise ValueError(
            f'analytic LR has {len(missing)} overlap blocks absent from the '
            'EPC graph; increase [data] radius')
    reported = {key: np.asarray(value).copy()
                for key, value in sr_blocks.items()}
    for key, value in lr_blocks.items():
        reported[key] = reported[key] + value
    for key, value in lr_blocks.items():
        if not np.array_equal(reported[key], sr_blocks[key] + value):
            raise AssertionError('SR + analytic-LR reconstruction identity failed')
    return reported


def make_gauge_fixed_predict_fn(base_predict_fn, sc_struct, overlap_dir):
    """Project every predicted H into one equilibrium-overlap energy gauge."""
    overlaps = load_reindexed_equilibrium_overlap(
        os.path.abspath(overlap_dir), sc_struct)

    def predict_fn(positions):
        projected, _ = project_hamiltonian_gauge(
            base_predict_fn(positions), overlaps)
        return projected

    for name in ('lr_provenance', 'tensor_provenance'):
        if hasattr(base_predict_fn, name):
            setattr(predict_fn, name, getattr(base_predict_fn, name))
    return predict_fn


def make_lr_corrected_predict_fn(base_predict_fn, positions0, sc_struct,
                                 workspace, overlap_dir, config_path,
                                 tensor_source='reference',
                                 tensor_mode='equilibrium_frozen'):
    """Wrap an H_SR predictor so every evaluation returns H_SR + H_LR."""
    if tensor_source not in {'reference', 'model'}:
        raise ValueError('tensor_source must be reference or model')
    if tensor_mode not in {'equilibrium_frozen', 'geometry_dependent'}:
        raise ValueError('tensor_mode must be equilibrium_frozen or geometry_dependent')
    workspace = os.path.abspath(workspace)
    overlap_dir = os.path.abspath(overlap_dir)
    cfg = load_config(config_path)
    require_current_lr_definition(cfg, workspace)

    reference_positions = np.asarray(sc_struct.positions, dtype=np.float64)
    reference_numbers = np.asarray(sc_struct.numbers, dtype=int)
    overlaps = load_reindexed_equilibrium_overlap(overlap_dir, sc_struct)

    ref_dir = os.path.join(workspace, 'reference')
    reference_born = np.load(os.path.join(ref_dir, 'born_effective_charges.npy'))
    reference_eps = np.load(os.path.join(ref_dir, 'dielectric_infinity.npy'))
    with open(os.path.join(ref_dir, 'species_order.json')) as handle:
        species_order = json.load(handle)
    with open(os.path.join(ref_dir, 'atomic_numbers.npy'), 'rb') as handle:
        atomic_numbers = np.load(handle)
    z_to_basis = {int(z): i for i, z in enumerate(atomic_numbers)}
    if len(species_order) != len(atomic_numbers):
        raise ValueError('LR reference species metadata is inconsistent')
    try:
        basis_index = np.asarray([z_to_basis[int(z)]
                                  for z in reference_numbers], dtype=int)
    except KeyError as exc:
        raise ValueError(f'element Z={exc.args[0]} absent from LR reference') \
            from exc

    lam = float(cfg['lr']['ewald_lambda'])
    tolerance = float(cfg['lr']['reciprocal_tolerance'])
    gmax_sq = gmax_squared(lam, tolerance)
    volume = abs(float(np.linalg.det(sc_struct.lattice)))
    positions0_np = positions0.detach().cpu().numpy()
    if not np.allclose(positions0_np, reference_positions, atol=1e-7):
        raise ValueError('EPC graph equilibrium positions differ from LR reference')

    def model_tensors():
        tensors = getattr(base_predict_fn, 'last_tensors', None)
        if tensors is None:
            raise ValueError(
                'model tensor source requested, but checkpoint has no Born/'
                'dielectric heads')
        born = np.asarray(tensors['born'], dtype=np.float64)
        epsilon = np.asarray(tensors['epsilon'], dtype=np.float64)
        if born.shape != (len(reference_numbers), 3, 3):
            raise ValueError(f'predicted Born shape is {born.shape}')
        if epsilon.shape != (1, 3, 3):
            raise ValueError(f'predicted epsilon shape is {epsilon.shape}')
        epsilon = epsilon[0]
        if not np.isfinite(born).all() or not np.isfinite(epsilon).all():
            raise ValueError('predicted LR tensors contain non-finite values')
        if np.max(np.abs(born.sum(axis=0))) > 1.0e-5:
            raise ValueError('predicted Born tensors violate the acoustic sum rule')
        if not np.allclose(epsilon, epsilon.T, atol=1.0e-8):
            raise ValueError('predicted dielectric tensor is not symmetric')
        if np.linalg.eigvalsh(epsilon).min() <= 0.0:
            raise ValueError('predicted dielectric tensor is not positive definite')
        return born, epsilon

    if tensor_source == 'model':
        # One equilibrium call both validates the checkpoint and defines the
        # reciprocal truncation consistently for the whole finite-difference sweep.
        base_predict_fn(positions0)
        frozen_born, frozen_eps = model_tensors()
        provenance = dict(getattr(base_predict_fn, 'tensor_provenance', {}))
    else:
        frozen_born = reference_born[basis_index]
        frozen_eps = reference_eps
        provenance = {
            'source': 'reference',
            'born_path': os.path.join(ref_dir, 'born_effective_charges.npy'),
            'epsilon_path': os.path.join(ref_dir, 'dielectric_infinity.npy'),
        }
    _, g_cart = reciprocal_set(
        reciprocal(sc_struct.lattice), frozen_eps, gmax_sq)

    def predict_fn(positions):
        sr_blocks = base_predict_fn(positions)
        if tensor_source == 'model' and tensor_mode == 'geometry_dependent':
            born, eps = model_tensors()
        else:
            born, eps = frozen_born, frozen_eps
        current = positions.detach().cpu().numpy()
        displacement = remove_uniform_translation(current - reference_positions)
        dipoles = np.einsum('nab,nb->na', born, displacement)
        coefficients = lr_coefficients(
            g_cart, dipoles, reference_positions, eps, lam, volume)
        potential = np.real(evaluate_potential(
            g_cart, coefficients, current))
        lr_blocks = assemble_lr_hamiltonian(overlaps, potential)
        return reconstruct_total_hamiltonian(sr_blocks, lr_blocks)

    predict_fn.lr_provenance = {
        **provenance,
        'tensor_source': tensor_source,
        'tensor_mode': tensor_mode,
        'analytic_lr_reconstruction': True,
        'direct_full_h_head': False,
    }
    return predict_fn
