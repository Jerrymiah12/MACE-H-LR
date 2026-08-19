"""Convert matched ABACUS +/-delta calculations into an actual EPC tensor."""
import argparse
import json
import os
import sys

import numpy as np


from maceh.epc.build_tensor import write_epc_cartesian_h5
from maceh.epc.derivative import DerivativeData
from maceh.epc.lr_correction import (
    load_reindexed_equilibrium_overlap, project_hamiltonian_gauge)
from maceh.epc.supercell import (build_supercell, fold_key, load_structure,
                                 uniform_grid)
from maceh.data.io.abacus import parse_csr, parse_running_scf, parse_stru
from maceh.config import load_config
from maceh.response.constants import RY_TO_EV
from maceh.data.io.blocks import matrices_to_blocks, parse_key


DIRECTIONS = 'xyz'


def _species_to_cell_mapping(structure, source_equilibrium_positions,
                             source_numbers):
    """Return source->target index map by species and equilibrium position."""
    out = np.empty(len(source_numbers), dtype=int)
    used = set()
    for source_index, (position, number) in enumerate(
            zip(source_equilibrium_positions, source_numbers)):
        candidates = [i for i, (target_position, target_number) in enumerate(
            zip(structure.positions, structure.numbers))
            if i not in used and int(target_number) == int(number)
            and np.linalg.norm(target_position - position) <= 1e-7]
        if len(candidates) != 1:
            raise ValueError('DFT and EPC supercell atom orders cannot be mapped')
        out[source_index] = candidates[0]
        used.add(candidates[0])
    return out


def _unwrap_shifts(cell, positions):
    frac = np.asarray(positions) @ np.linalg.inv(cell)
    near = np.abs(frac - np.round(frac)) < 1e-9
    frac = np.where(near, np.round(frac), frac)
    return np.floor(frac).astype(int)


def _rekey_to_common_gauge(blocks, source_to_target, wrap_shifts):
    """Map species-major atoms and ABACUS-wrapped R labels to fixed labels."""
    out = {}
    for key, value in blocks.items():
        r0, r1, r2, source_i1, source_j1 = parse_key(key)
        source_i, source_j = source_i1 - 1, source_j1 - 1
        shift = (np.asarray([r0, r1, r2], dtype=int)
                 + wrap_shifts[source_i] - wrap_shifts[source_j])
        target_i = int(source_to_target[source_i]) + 1
        target_j = int(source_to_target[source_j]) + 1
        new_key = str([int(shift[0]), int(shift[1]), int(shift[2]),
                       target_i, target_j])
        if new_key in out:
            raise ValueError(f'duplicate block after gauge/order mapping: {new_key}')
        out[new_key] = value
    return out


def load_hamiltonian(folder, cfg, source_to_target, equilibrium_overlaps):
    out_dir = os.path.join(folder, 'OUT.MgO')
    scf = parse_running_scf(os.path.join(out_dir, 'running_scf.log'))
    if not scf['converged']:
        raise ValueError(f'{folder}: SCF is not converged')
    cell, displaced_positions, species = parse_stru(
        os.path.join(folder, 'STRU'))
    dim, csr = parse_csr(os.path.join(
        out_dir, cfg['abacus']['csr_h_filename']))
    blocks = matrices_to_blocks(csr, dim, cfg, species, RY_TO_EV)
    shifts = _unwrap_shifts(cell, displaced_positions)
    blocks = _rekey_to_common_gauge(blocks, source_to_target, shifts)
    # Include overlap-only sparse keys as zero H before projecting. This makes
    # the operation exactly H - <H,S>/<S,S>S over the shared fixed basis.
    for key, overlap in equilibrium_overlaps.items():
        blocks.setdefault(key, np.zeros_like(overlap))
    projected, coefficient = project_hamiltonian_gauge(
        blocks, equilibrium_overlaps)
    return projected, coefficient, scf


def finite_difference_group(plus, minus, delta, smap, norb_cumsum):
    norb = int(norb_cumsum[-1])
    groups = {}
    occupied = {}
    for key in sorted(set(plus) | set(minus)):
        a, b = plus.get(key), minus.get(key)
        derivative = ((a if a is not None else np.zeros_like(b))
                      - (b if b is not None else np.zeros_like(a))) / (2 * delta)
        r0, r1, r2, i1, j1 = parse_key(key)
        p, lattice_r, i, j = fold_key(
            [r0, r1, r2, i1, j1], smap)
        group_key = (p, lattice_r)
        if group_key not in groups:
            groups[group_key] = np.zeros((norb, norb), dtype=np.float64)
            occupied[group_key] = set()
        atom_pair = (i, j)
        if atom_pair in occupied[group_key]:
            raise ValueError(
                f'duplicate folded DFT atom block {atom_pair} in {group_key}')
        occupied[group_key].add(atom_pair)
        groups[group_key][norb_cumsum[i]:norb_cumsum[i + 1],
                          norb_cumsum[j]:norb_cumsum[j + 1]] = derivative
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/epc/dft_reference')
    parser.add_argument('--output', default='runs/epc/actual')
    parser.add_argument('--config', default='provenance/config.resolved.yaml')
    parser.add_argument('--structure', default='workflows/epc/structure_primitive')
    parser.add_argument('--overlap-dir',
                        default='data/pilot/snapshot_000001')
    parser.add_argument('--k-grid', nargs=3, type=int, default=(2, 2, 2))
    args = parser.parse_args()

    cfg = load_config(args.config)
    with open(os.path.join(args.root, 'manifest.json')) as handle:
        manifest = json.load(handle)
    delta = float(manifest['delta_angstrom'])
    primitive = load_structure(args.structure)
    supercell, smap = build_supercell(primitive, (2, 2, 2))
    equilibrium_overlaps = load_reindexed_equilibrium_overlap(
        args.overlap_dir, supercell)

    # The ABACUS decks are species-major. Recover their equilibrium order from
    # the plus deck minus its recorded displacement, then map to EPC cell-major.
    first = os.path.join(args.root, manifest['calculations'][0]['name'])
    source_cell, source_displaced, source_species = parse_stru(
        os.path.join(first, 'STRU'))
    displacement = np.load(os.path.join(first, 'displacements.npy'))
    source_equilibrium = source_displaced - displacement
    z_map = {'Mg': 12, 'O': 8}
    source_numbers = np.asarray([z_map[name] for name in source_species])
    if not np.allclose(source_cell, supercell.lattice, atol=1e-7):
        raise ValueError('DFT and EPC supercells differ')
    source_to_target = _species_to_cell_mapping(
        supercell, source_equilibrium, source_numbers)

    norb_cumsum = np.asarray([0, 15, 28], dtype=int)
    blocks = {}
    gauge_coefficients = {}
    scf_records = {}
    for kappa, element in enumerate(('Mg', 'O')):
        for alpha, direction in enumerate(DIRECTIONS):
            pair = {}
            for sign_name in ('plus', 'minus'):
                name = f'{element}_{direction}_{sign_name}'
                folder = os.path.join(args.root, name)
                pair[sign_name], coefficient, scf = load_hamiltonian(
                    folder, cfg, source_to_target, equilibrium_overlaps)
                gauge_coefficients[name] = coefficient
                scf_records[name] = scf
            blocks[(kappa, alpha)] = finite_difference_group(
                pair['plus'], pair['minus'], delta, smap, norb_cumsum)

    derivative = DerivativeData(
        n_grid=(2, 2, 2), n_uc_atoms=2, delta=delta,
        norb_cumsum=norb_cumsum, blocks=blocks)
    out_dir = os.path.join(args.output, 'structure_primitive')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'epc_cartesian_actual.h5')
    write_epc_cartesian_h5(
        out_path, primitive, derivative,
        kpts=uniform_grid(tuple(args.k_grid)),
        qpts=uniform_grid((2, 2, 2)),
        attrs={
            'units': 'g in eV/Angstrom; k, q fractional; lattice, positions in Angstrom',
            'source': 'ABACUS 3.7.4 central finite difference',
            'delta': delta,
            'energy_gauge': 'H - <H,S0>/<S0,S0> S0 using equilibrium DFT overlap',
            'atom_order_conversion': 'ABACUS species-major to EPC cell-major',
            'r_label_gauge': 'ABACUS wrapped positions mapped to fixed unwrapped labels',
        }, save_derivatives=True)
    report = {
        'output': os.path.abspath(out_path),
        'delta_angstrom': delta,
        'gauge_coefficients_eV': gauge_coefficients,
        'scf': scf_records,
        'source_to_target_atom_index': source_to_target.tolist(),
        'nbytes_derivatives': derivative.nbytes(),
    }
    with open(os.path.join(out_dir, 'collection_report.json'), 'w') as handle:
        json.dump(report, handle, indent=1)
        handle.write('\n')
    print(f'Actual DFT EPC written to {out_path}')


if __name__ == '__main__':
    main()
