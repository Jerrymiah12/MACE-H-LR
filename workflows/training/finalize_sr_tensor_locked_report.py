"""Add the two post-selection EPC comparisons to a locked tensor report once."""
import argparse
import json
import os
import sys

import numpy as np


from workflows.epc.compare_epc import load_epc, metric, sha256, validate_grids


def checked_mode(data, expected):
    attrs = data["attrs"]
    source = attrs.get("analytic_lr_tensor_source")
    mode = attrs.get("analytic_lr_tensor_mode")
    if source != "model" or mode != expected:
        raise ValueError(
            f"EPC provenance is {source}/{mode}, expected model/{expected}")
    provenance = json.loads(attrs["analytic_lr_tensor_provenance"])
    if (provenance.get("tensor_source") != "model"
            or provenance.get("tensor_mode") != expected
            or provenance.get("direct_full_h_head") is not False):
        raise ValueError(f"incomplete predicted-tensor EPC provenance: {provenance}")
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--actual", required=True)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--geometry", required=True)
    args = parser.parse_args()

    with open(args.report) as handle:
        report = json.load(handle)
    if "epc" in report:
        raise SystemExit("locked report already contains EPC results")

    actual = load_epc(args.actual)
    frozen = load_epc(args.frozen)
    geometry = load_epc(args.geometry)
    validate_grids(actual, {"equilibrium_frozen": frozen,
                            "geometry_dependent": geometry})
    frozen_provenance = checked_mode(frozen, "equilibrium_frozen")
    geometry_provenance = checked_mode(geometry, "geometry_dependent")
    report["epc"] = {
        "quantity": "Cartesian AO EPC before phonon/band contraction",
        "units": "eV/Angstrom",
        "actual": {"path": os.path.abspath(args.actual),
                   "sha256": sha256(args.actual)},
        "equilibrium_frozen": {
            "path": os.path.abspath(args.frozen),
            "sha256": sha256(args.frozen),
            "metrics": metric(frozen["g"], actual["g"]),
            "provenance": frozen_provenance,
        },
        "geometry_dependent": {
            "path": os.path.abspath(args.geometry),
            "sha256": sha256(args.geometry),
            "metrics": metric(geometry["g"], actual["g"]),
            "provenance": geometry_provenance,
        },
        "geometry_minus_frozen": metric(geometry["g"], frozen["g"]),
        "max_abs_geometry_minus_frozen": float(np.max(np.abs(
            geometry["g"] - frozen["g"]))),
    }
    temporary = args.report + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, args.report)
    print(json.dumps(report["epc"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
