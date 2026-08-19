"""Identity checks for analytic long-range labels used during inference."""

import os

import yaml

from maceh.config import sha256_file
from maceh.response.long_range import gmax_squared

REFERENCE_DEFINITION_FILES = (
    "reference_cell.npy",
    "reference_positions.npy",
    "atomic_numbers.npy",
    "species_order.json",
    "born_effective_charges.npy",
    "dielectric_infinity.npy",
)


def reference_fingerprints(workspace):
    """Content hashes for every artifact that changes physical LR labels."""
    ref_dir = os.path.join(workspace, "reference")
    missing = [name for name in REFERENCE_DEFINITION_FILES
               if not os.path.isfile(os.path.join(ref_dir, name))]
    if missing:
        raise FileNotFoundError(
            f"LR reference artifacts missing from {ref_dir}: {missing}")
    return {name: sha256_file(os.path.join(ref_dir, name))
            for name in REFERENCE_DEFINITION_FILES}


def lr_definition(cfg, gmax_sq, reference_hashes):
    """Return the cell-invariant identity of an LR label definition."""
    return {"ewald_lambda": float(cfg["lr"]["ewald_lambda"]),
            "reciprocal_cutoff": float(gmax_sq),
            "reciprocal_tolerance": float(cfg["lr"]["reciprocal_tolerance"]),
            "reciprocal_set": {"inversion_symmetric": True,
                               "excludes_G_zero": True,
                               "cutoff_type": "dielectric_ellipsoid"},
            "imaginary_tolerance": float(cfg["lr"]["imaginary_tolerance"]),
            "gauge": "G_zero_equals_zero",
            "sign_convention": "electron_potential_energy",
            "phase_convention": "reference_positions",
            "reference_artifacts_sha256": dict(sorted(reference_hashes.items()))}


def expected_lr_definition(cfg, workspace):
    gmax_sq = gmax_squared(cfg["lr"]["ewald_lambda"],
                           cfg["lr"]["reciprocal_tolerance"])
    return lr_definition(cfg, gmax_sq, reference_fingerprints(workspace))


def require_current_lr_definition(cfg, workspace):
    """Fail if published labels no longer match config/reference artifacts."""
    path = os.path.join(workspace, "metadata.yaml")
    if not os.path.isfile(path):
        raise SystemExit(
            "metadata.yaml is missing; run lr-process and validate first")
    with open(path) as handle:
        stored = (yaml.safe_load(handle) or {}).get("lr_definition")
    expected = expected_lr_definition(cfg, workspace)
    if stored != expected:
        raise SystemExit(
            "workspace lr_definition does not match the current LR config and "
            "reference artifacts; rerun in a clean workspace or restore the "
            "original references")
    return stored
