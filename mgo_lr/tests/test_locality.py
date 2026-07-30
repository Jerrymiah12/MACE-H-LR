import json
import os

import numpy as np

from mgo_lr import convert, locality, validate
from mgo_lr.tests.test_convert import Args
from mgo_lr.tests.test_validate import ladder_workspace


def test_block_distance():
    cell = 10.0 * np.eye(3)
    cart = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert abs(locality.block_distance("[0, 0, 0, 1, 2]", cart, cell) - 1.0) < 1e-12
    assert abs(locality.block_distance("[1, 0, 0, 1, 1]", cart, cell) - 10.0) < 1e-12
    assert abs(locality.block_distance("[-1, 0, 0, 1, 2]", cart, cell) - 9.0) < 1e-12


def test_tail_fractions_ordering():
    cell = 10.0 * np.eye(3)
    cart = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    blocks = {"[0, 0, 0, 1, 1]": np.array([[1.0]]),      # d = 0
              "[0, 0, 0, 1, 2]": np.array([[0.1]])}      # d = 4
    f = locality.tail_fractions(blocks, cart, cell, [1.0, 5.0])
    total = 1.0 + 0.01
    assert abs(f[0] - 0.01 / total) < 1e-12
    assert f[1] == 0.0
    b = locality.binned_norms(blocks, cart, cell, 1.0)
    assert b[0]["count"] == 1 and abs(b[0]["max"] - 1.0) < 1e-12


def test_odd_response_perfect_match():
    k = "[0, 0, 0, 1, 1]"
    h_p = {k: np.array([[2.0]])}
    h_m = {k: np.array([[-2.0]])}
    out = locality.odd_response(h_p, h_m, {k: np.array([[2.0]])}, 1e-12)
    assert abs(out["cos_theta"] - 1.0) < 1e-9
    assert abs(out["r_lr"] - 1.0) < 1e-9
    out2 = locality.odd_response(h_p, h_m, {k: np.array([[-2.0]])}, 1e-12)
    assert abs(out2["cos_theta"] + 1.0) < 1e-9


def test_locality_equal_tails_is_not_a_pass():
    # P1 regression: identical SR and full tails = NO localization improvement,
    # yet `s <= f + 1e-12` accepted it.  Equality must not PASS.
    f_full = [0.9, 0.7, 0.5, 0.3]
    assert locality.long_range_localizes(f_full, list(f_full), 1e-6, 0.05) is False


def test_locality_measurable_improvement_passes():
    f_full = [0.9, 0.7, 0.5, 0.3]
    f_sr = [0.9, 0.7, 0.20, 0.10]          # >5% lower in the long-distance half
    assert locality.long_range_localizes(f_full, f_sr, 1e-6, 0.05) is True


def test_locality_all_zero_long_range_is_not_a_pass():
    # radii beyond the largest nonzero block: both zero, no evidence -> not PASS
    f_full = [0.5, 0.2, 0.0, 0.0]
    assert locality.long_range_localizes(f_full, list(f_full), 1e-6, 0.05) is False


def test_controlled_q_comparisons_average_each_shell():
    metas = {
        "small-plus": {"wavevector_family_id": "qfam", "q_magnitude": 0.2,
                       "polarization_class": "longitudinal"},
        "small-minus": {"wavevector_family_id": "qfam", "q_magnitude": 0.2,
                        "polarization_class": "longitudinal"},
        "medium": {"wavevector_family_id": "qfam", "q_magnitude": 0.4,
                   "polarization_class": "longitudinal"},
        "large": {"wavevector_family_id": "qfam", "q_magnitude": 0.6,
                  "polarization_class": "longitudinal"},
    }
    norms = {"small-plus": 5.0, "small-minus": 3.0,
             "medium": 2.0, "large": 1.0}
    comparisons = locality.controlled_q_comparisons(metas, norms)
    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert [shell["mean_lr_norm"] for shell in comparison["shells"]] == \
        [4.0, 2.0, 1.0]
    assert comparison["small_q_has_stronger_lr"] is True


def test_locality_report_stage(tmp_path):
    ws, cfg, store = ladder_workspace(tmp_path)
    assert validate.validate_stage(cfg, ws, Args()) == 0
    assert locality.locality_report_stage(cfg, ws, Args()) == 0
    rep = json.load(open(os.path.join(
        ws, "generation_logs", "locality", "locality_pilot.json")))
    assert rep["n_snapshots"] == 4
    t = rep["tail"]
    assert len(t["radii"]) == len(t["F_full"]) == len(t["F_lr"]) == len(t["F_sr"])
    assert isinstance(t["f_sr_below_f_full"], bool)
    assert all(0.0 <= x <= 1.0 for x in t["F_full"])
    # first radii tail fractions are monotonically non-increasing
    assert all(a >= b - 1e-12 for a, b in zip(t["F_full"], t["F_full"][1:]))
    amps = sorted(round(e["amplitude"], 6) for e in rep["odd_response"])
    assert amps == [0.01, 0.02]               # one entry per ± pair
    for e in rep["odd_response"]:
        assert -1.0 - 1e-9 <= e["cos_theta"] <= 1.0 + 1e-9
        assert e["r_lr"] >= 0.0
    fam = rep["families"]["fam-test"]
    assert len(fam["members"]) == 4
    assert "mean_lr_norm_by_class" in fam


def _bin(r_lo, reduction, count=100, dh_full=1.0):
    return {"r_lo": r_lo, "r_hi": r_lo + 1.0, "count": count,
            "dh_full": dh_full, "dh_sr": dh_full * (1.0 - reduction),
            "reduction": reduction}


def test_farfield_gate_requires_every_far_bin_to_improve():
    bins = [_bin(0.0, 0.0), _bin(4.0, 0.06), _bin(5.0, 0.09)]
    ok, qualifying = locality.farfield_gate(bins, 4.0, 1e-6, 20, 0.05)
    assert ok and len(qualifying) == 2        # near bins are not judged
    # one far bin below threshold fails the whole gate
    ok, _ = locality.farfield_gate(bins + [_bin(6.0, 0.01)], 4.0, 1e-6, 20, 0.05)
    assert not ok


def test_farfield_gate_ignores_noise_and_thin_bins():
    # a bin at the SCF noise floor would otherwise contribute a meaningless
    # (often negative) reduction; a near-empty bin is not evidence either
    bins = [_bin(4.0, 0.09),
            _bin(5.0, -1.4, dh_full=1e-9),     # noise
            _bin(6.0, -1.4, count=3)]          # too few blocks
    ok, qualifying = locality.farfield_gate(bins, 4.0, 1e-6, 20, 0.05)
    assert ok and [b["r_lo"] for b in qualifying] == [4.0]


def test_farfield_gate_without_evidence_is_not_a_pass():
    assert locality.farfield_gate([], 4.0, 1e-6, 20, 0.05)[0] is False
    # every candidate bin excluded -> still no pass
    assert locality.farfield_gate([_bin(1.0, 0.9)], 4.0, 1e-6, 20, 0.05)[0] is False


def test_farfield_sensitivity_scores_a_perfect_lr_term(tmp_path):
    """If H^LR reproduces the whole response, the residual vanishes and the
    reduction is 1; blocks are matched by geometry, not by R label."""
    cell = np.eye(3) * 40.0
    # atom 0 is the displaced one; the measured block joins atoms 2 and 3,
    # both ~5 A away from it, so the bin reports far-field response
    cart = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    key = "[0, 0, 0, 2, 3]"
    ref = {key: np.zeros((1, 1))}
    resp = np.array([[0.25]])
    probe = {key: resp}
    bins, unmatched = locality.farfield_sensitivity(
        ref, cart, probe, {key: resp}, {}, cart, cell,
        atom=0, bin_width=1.0)
    assert unmatched == 0
    assert len(bins) == 1 and bins[0]["r_lo"] == 5.0
    assert bins[0]["reduction"] == 1.0
    # with no LR term at all the response is untouched
    bins0, _ = locality.farfield_sensitivity(
        ref, cart, probe, {}, {}, cart, cell, atom=0, bin_width=1.0)
    assert bins0[0]["reduction"] == 0.0


def test_locality_report_empty_set(tmp_path, capsys):
    ws, cfg, store = ladder_workspace(tmp_path)
    # nothing validated yet -> report nothing, still exit 0
    assert locality.locality_report_stage(cfg, ws, Args()) == 0
    assert "no validated" in capsys.readouterr().out
