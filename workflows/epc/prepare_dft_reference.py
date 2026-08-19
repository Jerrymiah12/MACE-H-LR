"""Prepare matched +/-delta ABACUS calculations for a true MgO EPC reference."""
import argparse
import json
import os
import sys

import numpy as np


from maceh.data.io.abacus import write_input, write_kpt, write_stru
from maceh.config import atomic_write_text, load_config
from workflows.mgo_dataset.snapshot import load_reference
from maceh.data.structures import make_supercell


DIRECTIONS = "xyz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='provenance/config.resolved.yaml')
    parser.add_argument('--workspace', default='run')
    parser.add_argument('--output', default='data/epc/dft_reference')
    parser.add_argument('--delta', type=float, default=0.0025)
    args = parser.parse_args()
    if not np.isfinite(args.delta) or args.delta <= 0:
        raise SystemExit('--delta must be positive and finite')

    cfg = load_config(args.config)
    reference = load_reference(args.workspace)
    supercell = make_supercell(reference['prim_cell'], reference['frac'],
                               reference['species'], 2)
    root = os.path.abspath(args.output)
    os.makedirs(root, exist_ok=True)
    calculations = []

    for kappa, element in enumerate(reference['species']):
        candidates = np.flatnonzero(
            (supercell.basis_index == kappa)
            & np.all(supercell.cell_index == 0, axis=1))
        if len(candidates) != 1:
            raise RuntimeError(f'cannot locate home-cell {element} atom')
        atom_index = int(candidates[0])
        for alpha, direction in enumerate(DIRECTIONS):
            for sign, sign_name in ((1, 'plus'), (-1, 'minus')):
                name = f'{element}_{direction}_{sign_name}'
                folder = os.path.join(root, name)
                os.makedirs(folder, exist_ok=True)
                displacement = np.zeros_like(supercell.cart)
                displacement[atom_index, alpha] = sign * args.delta
                metadata = {
                    'name': name,
                    'element': element,
                    'primitive_atom_index': kappa,
                    'species_major_supercell_atom_index': atom_index,
                    'cartesian_direction': direction,
                    'cartesian_direction_index': alpha,
                    'sign': sign,
                    'delta_angstrom': args.delta,
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
                            f'{folder} already contains a different calculation')
                write_input(os.path.join(folder, 'INPUT'), cfg,
                            out_mat_hs2=1)
                write_kpt(os.path.join(folder, 'KPT'), metadata['kmesh'])
                write_stru(os.path.join(folder, 'STRU'), supercell.cell,
                           supercell.cart + displacement,
                           supercell.species, cfg)
                np.save(os.path.join(folder, 'displacements.npy'), displacement)
                atomic_write_text(marker, json.dumps(metadata, indent=1) + '\n')
                calculations.append(metadata)

    manifest = {
        'purpose': 'central-finite-difference DFT Cartesian-AO EPC reference',
        'delta_angstrom': args.delta,
        'calculations': calculations,
        'count': len(calculations),
    }
    atomic_write_text(os.path.join(root, 'manifest.json'),
                      json.dumps(manifest, indent=1) + '\n')
    print(f'Prepared {len(calculations)} ABACUS calculations in {root}')


if __name__ == '__main__':
    main()
