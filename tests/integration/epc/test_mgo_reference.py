from types import SimpleNamespace

import numpy as np
import pytest

from workflows.epc.collect_dft_epc import (
    _rekey_to_common_gauge,
    _species_to_cell_mapping,
)
from workflows.epc.compare_epc import hermiticity_metric
from maceh.epc.lr_correction import (project_hamiltonian_gauge,
                                     reconstruct_total_hamiltonian)


def test_species_mapping_recovers_reordered_atoms():
    structure = SimpleNamespace(
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        numbers=np.array([12, 8]),
    )
    source_positions = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    source_numbers = np.array([8, 12])
    assert np.array_equal(
        _species_to_cell_mapping(structure, source_positions, source_numbers),
        [1, 0],
    )


def test_rekey_maps_atom_order_and_unwrapped_position_gauge():
    blocks = {'[2, 3, 4, 1, 2]': np.array([[7.0]])}
    source_to_target = np.array([1, 0])
    wrap_shifts = np.array([[1, 0, 0], [0, -1, 0]])
    got = _rekey_to_common_gauge(
        blocks, source_to_target, wrap_shifts)
    assert list(got) == ['[3, 4, 4, 2, 1]']
    assert got['[3, 4, 4, 2, 1]'][0, 0] == pytest.approx(7.0)


def test_energy_gauge_projection_is_overlap_orthogonal():
    overlaps = {'a': np.array([[1.0]]), 'b': np.array([[2.0]])}
    blocks = {'a': np.array([[4.0]]), 'b': np.array([[3.0]])}
    projected, coefficient = project_hamiltonian_gauge(blocks, overlaps)
    assert coefficient == pytest.approx(2.0)
    inner = sum(float(np.sum(projected[key] * value))
                for key, value in overlaps.items())
    assert inner == pytest.approx(0.0)
    assert projected['a'][0, 0] == pytest.approx(2.0)
    assert projected['b'][0, 0] == pytest.approx(-1.0)


def test_reported_hamiltonian_is_exact_sr_plus_analytic_lr():
    sr = {'a': np.array([[1.25, -2.0]]), 'untouched': np.array([[4.0]])}
    lr = {'a': np.array([[0.5, 3.0]])}
    reported = reconstruct_total_hamiltonian(sr, lr)
    assert np.array_equal(reported['a'], sr['a'] + lr['a'])
    assert np.array_equal(reported['untouched'], sr['untouched'])
    assert reported['a'] is not sr['a']


def test_reconstruction_rejects_lr_block_outside_sr_graph():
    with pytest.raises(ValueError, match='increase \\[data\\] radius'):
        reconstruct_total_hamiltonian(
            {'a': np.zeros((1, 1))}, {'missing': np.ones((1, 1))})


def test_hermiticity_metric_detects_residual():
    data = {
        'g': np.array([[[[[[2.0 + 0.0j]]]]]]),
        'kpoints': np.zeros((1, 3)),
        'qpoints': np.zeros((1, 3)),
    }
    assert hermiticity_metric(data)['max_abs_eV_per_A'] == 0.0
    data['g'][0, 0, 0, 0, 0, 0] = 2.0 + 0.25j
    assert hermiticity_metric(data)['max_abs_eV_per_A'] == pytest.approx(0.5)
