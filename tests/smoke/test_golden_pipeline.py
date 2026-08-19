import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from maceh.epc.build_tensor import compute_epc_cartesian
from workflows.mgo_dataset.dfpt import apply_asr, parse_ph_output

PH_OUT = """
 Dielectric constant in cartesian axis
 ( 3.135573418 0.0 0.0 )
 ( 0.0 3.135573418 0.0 )
 ( 0.0 0.0 3.135573418 )
 Effective charges (d Force / dE) in cartesian axis without asr
 atom 1 Mg
 Ex ( 1.97120 0.0 0.0 )
 Ey ( 0.0 1.97120 0.0 )
 Ez ( 0.0 0.0 1.97120 )
 atom 2 O
 Ex ( -1.95320 0.0 0.0 )
 Ey ( 0.0 -1.95320 0.0 )
 Ez ( 0.0 0.0 -1.95320 )
"""


def _golden():
    return json.loads((Path(__file__).with_name("golden.json")).read_text())


def test_synthetic_response_to_epc_golden_values():
    """Fast numerical canary spanning DFPT parsing, ASR, scoring, and EPC."""
    eps, born, _ = parse_ph_output(PH_OUT)
    born = apply_asr(born)

    # A deterministic held-out SR error in eV, reported in meV.
    truth = np.array([0.0, 0.002])
    prediction = np.zeros_like(truth)
    sr_mae = float(np.mean(np.abs(prediction - truth)) * 1000.0)

    blocks = {(0, 0): {
            ((0, 0, 0), (0, 0, 0)): np.array([[1.0]]),
            ((1, 0, 0), (2, 0, 0)): np.array([[0.5]]),
        }, (0, 1): {}, (0, 2): {}}
    derivatives = SimpleNamespace(
        n_grid=(2, 1, 1), n_uc_atoms=1, delta=0.01,
        norb_cumsum=np.array([0, 1]), norb_tot=1,
        pairs=lambda: blocks.keys(),
        group=lambda kappa, alpha: blocks.get((kappa, alpha), {}),
    )
    epc = compute_epc_cartesian(
        derivatives, np.zeros((1, 3)), np.zeros((1, 3)))
    values = {
        "sr_test_mae_meV": sr_mae,
        "epc_xx_meV_per_ang": float(epc["g"][0, 0, 0, 0, 0, 0].real * 1000.0),
        "born_Mg_xx": float(born[0, 0, 0]),
        "eps_inf_xx": float(eps[0, 0]),
    }
    for name, expected in _golden().items():
        assert np.isclose(values[name], expected["value"],
                          rtol=expected["rtol"], atol=0.0), (name, values[name])
