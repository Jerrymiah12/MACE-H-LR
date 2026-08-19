"""Prepare a continuous one-atom ABACUS displacement scan for MgO response."""
import argparse
import json
import os
import sys

import numpy as np


from maceh.data.io.abacus import write_input, write_kpt, write_stru
from maceh.config import atomic_write_text, load_config
from workflows.mgo_dataset.snapshot import load_reference
from maceh.data.structures import make_supercell


DIRECTIONS = 'xyz'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='provenance/config.resolved.yaml')
    parser.add_argument('--workspace', default='run')
    parser.add_argument('--output', default='data/epc/response_scan_dft')
    parser.add_argument('--element', choices=('Mg', 'O'), default='Mg')
    parser.add_argument('--direction', choices=tuple(DIRECTIONS), default='x')
    parser.add_argument('--minimum', type=float, default=-0.03)
    parser.add_argument('--maximum', type=float, default=0.03)
    parser.add_argument('--points', type=int, default=25)
    args = parser.parse_args()
    if (not np.isfinite(args.minimum) or not np.isfinite(args.maximum)
            or args.minimum >= args.maximum):
        raise SystemExit('--minimum/--maximum must be finite and increasing')
    if args.points < 5 or args.points % 2 == 0:
        raise SystemExit('--points must be an odd integer of at least 5')

    cfg = load_config(args.config)
    reference = load_reference(args.workspace)
    supercell = make_supercell(reference['prim_cell'], reference['frac'],
                               reference['species'], 2)
    kappa = reference['species'].index(args.element)
    alpha = DIRECTIONS.index(args.direction)
    candidates = np.flatnonzero(
        (supercell.basis_index == kappa)
        & np.all(supercell.cell_index == 0, axis=1))
    if len(candidates) != 1:
        raise RuntimeError(f'cannot locate home-cell {args.element} atom')
    atom_index = int(candidates[0])
    root = os.path.abspath(args.output)
    os.makedirs(root, exist_ok=True)
    calculations = []
    deltas = np.linspace(args.minimum, args.maximum, args.points)
    if not np.any(np.isclose(deltas, 0.0, rtol=0, atol=1e-15)):
        raise RuntimeError('scan grid must contain zero displacement')

    for index, delta in enumerate(deltas):
        delta = float(delta)
        if abs(delta) < 1e-15:
            delta = 0.0
        name = f'point_{index:03d}'
        folder = os.path.join(root, name)
        os.makedirs(folder, exist_ok=True)
        displacement = np.zeros_like(supercell.cart)
        displacement[atom_index, alpha] = delta
        metadata = {
            'name': name,
            'element': args.element,
            'primitive_atom_index': kappa,
            'species_major_supercell_atom_index': atom_index,
            'cartesian_direction': args.direction,
            'cartesian_direction_index': alpha,
            'signed_displacement_angstrom': delta,
            'supercell': [2, 2, 2],
            'atom_order': 'species-major, cell-minor',
            'kmesh': list(cfg['abacus']['kmesh_supercell']['pilot']),
        }
        marker = os.path.join(folder, 'metadata.json')
        if os.path.isfile(marker):
            with open(marker) as handle:
                old = json.load(handle)
            if old != metadata:
                raise SystemExit(
                    f'{folder} already contains a different scan point')
        write_input(os.path.join(folder, 'INPUT'), cfg, out_mat_hs2=1)
        write_kpt(os.path.join(folder, 'KPT'), metadata['kmesh'])
        write_stru(os.path.join(folder, 'STRU'), supercell.cell,
                   supercell.cart + displacement, supercell.species, cfg)
        np.save(os.path.join(folder, 'displacements.npy'), displacement)
        atomic_write_text(marker, json.dumps(metadata, indent=1) + '\n')
        calculations.append(metadata)

    manifest = {
        'purpose': 'continuous DFT Hamiltonian response scan',
        'element': args.element,
        'primitive_atom_index': kappa,
        'direction': args.direction,
        'direction_index': alpha,
        'minimum_angstrom': args.minimum,
        'maximum_angstrom': args.maximum,
        'point_count': args.points,
        'step_angstrom': float(deltas[1] - deltas[0]),
        'calculations': calculations,
        'count': len(calculations),
    }
    atomic_write_text(os.path.join(root, 'manifest.json'),
                      json.dumps(manifest, indent=1) + '\n')
    print(f'Prepared {len(calculations)} scan calculations in {root}')
    print(f'{args.element}-{args.direction}: {args.minimum:g} to '
          f'{args.maximum:g} A, step {deltas[1] - deltas[0]:g} A')


if __name__ == '__main__':
    main()
