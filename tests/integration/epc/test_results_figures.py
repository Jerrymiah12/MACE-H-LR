import numpy as np
import pytest

from workflows.epc.make_results_figures import (
    aggregate_distance_rows,
    angular_data,
    nearest_periodic_distance,
    weighted_quantile,
)


def test_nearest_periodic_distance_handles_skewed_lattice():
    lattice = np.array([[0.0, 4.0, 4.0],
                        [4.0, 0.0, 4.0],
                        [4.0, 4.0, 0.0]])
    assert nearest_periodic_distance(lattice[0], lattice) == pytest.approx(0)
    vector = 2 * lattice[0] + np.array([1.0, 0.0, 0.0])
    assert nearest_periodic_distance(vector, lattice) == pytest.approx(1.0)


def test_weighted_quantile_tracks_coupling_power():
    values = np.array([1.0, 10.0, 100.0])
    weights = np.array([9.0, 1.0, 0.0])
    median, pct90 = weighted_quantile(values, weights, (0.5, 0.9))
    assert median == pytest.approx(1.0)
    assert pct90 == pytest.approx(1.0)


def test_angular_data_identical_and_antiparallel_vectors():
    truth = np.zeros((1, 1, 1, 3, 1, 1), dtype=np.complex128)
    truth[0, 0, 0, :, 0, 0] = [1.0, 2.0j, 3.0]
    same = angular_data(truth.copy(), truth)
    opposite = angular_data(-truth, truth)
    assert same['summary']['power_weighted_mean_angle_deg'] \
        == pytest.approx(0.0)
    assert opposite['summary']['power_weighted_mean_angle_deg'] \
        == pytest.approx(180.0)
    assert same['summary']['power_weighted_mean_cosine'] \
        == pytest.approx(1.0)
    assert opposite['summary']['power_weighted_mean_cosine'] \
        == pytest.approx(-1.0)


def test_aggregate_distance_rows_weights_by_ao_element_count():
    rows = [
        {'center_A': 0.5, 'ao_element_count': 1,
         'sr_mae_eV_per_A': 1.0, 'full_mae_eV_per_A': 2.0},
        {'center_A': 1.5, 'ao_element_count': 3,
         'sr_mae_eV_per_A': 3.0, 'full_mae_eV_per_A': 6.0},
    ]
    result = aggregate_distance_rows(rows)
    assert result['sr_mae_eV_per_A'] == pytest.approx(2.5)
    assert result['full_mae_eV_per_A'] == pytest.approx(5.0)
    assert result['full_error_removed_by_sr_fraction'] == pytest.approx(0.5)
