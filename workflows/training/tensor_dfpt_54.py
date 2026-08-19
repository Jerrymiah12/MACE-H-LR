#!/usr/bin/env python3
"""Prepare, collect, and inspect the fast 54-atom MgO tensor-label campaign.

The existing 400-snapshot Hamiltonian data are never modified.  This tool
writes a separate, partially labelled dataset used to supervise geometry-
dependent Born-charge and electronic-dielectric heads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

from maceh.data.io.abacus import parse_stru
from maceh.config import atomic_write_text
from maceh.paths import data_root, runs_root
from workflows.mgo_dataset.dfpt import apply_asr, parse_ph_output, write_ph_input


# A deliberately small first campaign.  Train labels span the major pattern
# classes; validation and test labels remain held out from tensor-head fitting.
FAST_SELECTION = (
    # split, snapshot, role
    ("train", "snapshot_000386", "benchmark; near-equilibrium random"),
    ("train", "snapshot_000011", "single-q transverse, low-q, 0.01 A"),
    ("train", "snapshot_000033", "single-q longitudinal, high amplitude"),
    ("train", "snapshot_000063", "single-q transverse, high amplitude"),
    ("train", "snapshot_000175", "mixed low-q longitudinal"),
    ("train", "snapshot_000244", "mixed low-q transverse"),
    ("train", "snapshot_000275", "random-local, low amplitude"),
    ("train", "snapshot_000312", "random-local, high amplitude"),
    ("train", "snapshot_000353", "positive sign-pair member"),
    ("train", "snapshot_000354", "negative sign-pair member"),
    ("validation", "snapshot_000028", "held-out single-q transverse"),
    ("validation", "snapshot_000327", "held-out random-local"),
    ("test", "snapshot_000080", "held-out high-q single-q"),
    ("test", "snapshot_000229", "held-out high-q mixed mode"),
    ("test", "snapshot_000295", "held-out random-local high amplitude"),
    ("test", "snapshot_000333", "held-out positive sign-pair member"),
    ("test", "snapshot_000334", "held-out negative sign-pair member"),
)

PROFILES = {
    # 2^3 in the 3x3x3 supercell corresponds to a 6^3 primitive-cell density.
    # It is accepted only after comparison with the convergence-anchor profile.
    "fast": {"ecutwfc": 80.0, "kmesh": [2, 2, 2],
             "conv_thr": 1.0e-10, "tr2_ph": 1.0e-12},
    # Matches the k density used for the existing 54-atom Hamiltonians.
    "anchor": {"ecutwfc": 80.0, "kmesh": [3, 3, 3],
               "conv_thr": 1.0e-11, "tr2_ph": 1.0e-13},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_output() -> Path:
    return runs_root() / "tensor_dfpt_54_fast"


def _load_qe_settings(workspace: Path) -> dict:
    path = workspace / "reference" / "dft_settings.yaml"
    if not path.is_file():
        raise SystemExit(f"missing resolved DFT settings: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    qe = dict(data.get("qe") or {})
    required = ("pseudo_dir", "pseudopotentials")
    missing = [key for key in required if key not in qe]
    if missing:
        raise SystemExit(f"{path}: missing QE fields {missing}")
    for filename in qe["pseudopotentials"].values():
        candidate = Path(qe["pseudo_dir"]) / filename
        if not candidate.is_file():
            raise SystemExit(f"missing QE pseudopotential: {candidate}")
    return qe


def _write_pw_input(path: Path, qe: dict, profile: dict, cell: np.ndarray,
                    cart: np.ndarray, species: list[str]) -> None:
    frac = np.asarray(cart, float) @ np.linalg.inv(np.asarray(cell, float))
    ordered = [s for s in ("Mg", "O") for _ in range(species.count(s))]
    if list(species) != ordered:
        raise ValueError("54-atom source must retain Mg-major/O-major ordering")
    lines = [
        "&CONTROL",
        "  calculation = 'scf'",
        "  restart_mode = 'from_scratch'",
        "  prefix = 'mgo'",
        "  outdir = './out'",
        f"  pseudo_dir = '{qe['pseudo_dir']}'",
        "  disk_io = 'low'",
        "  tprnfor = .true.",
        "  tstress = .true.",
        "/",
        "&SYSTEM",
        "  ibrav = 0",
        f"  nat = {len(species)}",
        "  ntyp = 2",
        f"  ecutwfc = {profile['ecutwfc']}",
        "  occupations = 'fixed'",
        "/",
        "&ELECTRONS",
        f"  conv_thr = {profile['conv_thr']:.12g}",
        "  electron_maxstep = 120",
        "  mixing_beta = 0.35",
        "  diagonalization = 'david'",
        "/",
        "ATOMIC_SPECIES",
        f"Mg 24.305 {qe['pseudopotentials']['Mg']}",
        f"O 15.999 {qe['pseudopotentials']['O']}",
        "CELL_PARAMETERS angstrom",
    ]
    for row in np.asarray(cell, float):
        lines.append(" ".join(f"{value:.12f}" for value in row))
    lines.append("ATOMIC_POSITIONS crystal")
    for label, row in zip(species, frac):
        lines.append(f"{label}  " + " ".join(f"{value:.12f}" for value in row))
    kx, ky, kz = profile["kmesh"]
    lines.extend(["K_POINTS automatic", f"{kx} {ky} {kz} 0 0 0"])
    atomic_write_text(str(path), "\n".join(lines) + "\n")


def _write_one(workspace: Path, output: Path, profile_name: str,
               split: str, sid: str, role: str) -> dict:
    profile = dict(PROFILES[profile_name])
    source = workspace / "main" / sid
    stru = source / "STRU"
    metadata_path = source / "displacement_metadata.json"
    if not stru.is_file() or not metadata_path.is_file():
        raise SystemExit(f"incomplete source snapshot: {source}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("split_hint") != split:
        raise SystemExit(
            f"{sid}: requested split {split}, source says "
            f"{metadata.get('split_hint')}")
    cell, cart, species = parse_stru(str(stru))
    if len(species) != 54:
        raise SystemExit(f"{sid}: expected 54 atoms, found {len(species)}")

    destination = output / profile_name / sid
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".mgo_tensor_dfpt_54"
    if any(destination.iterdir()) and not marker.exists():
        raise SystemExit(f"refusing to modify unowned directory: {destination}")
    marker.touch(exist_ok=True)
    _write_pw_input(destination / "pw.in", _load_qe_settings(workspace),
                    profile, cell, cart, species)
    ph_cfg = {"qe": {"tr2_ph": profile["tr2_ph"]}}
    write_ph_input(str(destination / "ph.in"), ph_cfg,
                   trans=False, zeu=True)
    source_record = {
        "snapshot_id": sid,
        "split": split,
        "role": role,
        "source_stru": str(stru.resolve()),
        "source_stru_sha256": _sha256(stru),
        "source_metadata_sha256": _sha256(metadata_path),
        "nat": len(species),
        "species": species,
        "cell_angstrom": np.asarray(cell).tolist(),
        "profile": profile_name,
        "settings": profile,
        "pattern": metadata,
    }
    atomic_write_text(str(destination / "source.json"),
                      json.dumps(source_record, indent=2) + "\n")
    return source_record


def prepare(args) -> int:
    workspace, output = Path(args.workspace).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for split, sid, role in FAST_SELECTION:
        entries.append(_write_one(workspace, output, args.profile,
                                  split, sid, role))
    manifest = {
        "schema_version": 1,
        "purpose": "partially labelled 54-atom Zstar/epsilon fast prototype",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workspace": str(workspace),
        "output": str(output),
        "profile": args.profile,
        "benchmark_snapshot": FAST_SELECTION[0][1],
        "entries": [{key: item[key] for key in
                     ("snapshot_id", "split", "role", "profile", "settings")}
                    for item in entries],
        "acceptance_gate": {
            "compare_snapshot": FAST_SELECTION[0][1],
            "fast_profile": "fast",
            "anchor_profile": "anchor",
            "required_before_campaign": True,
            "note": "Do not treat 2x2x2 tensor labels as production labels "
                    "until the same geometry agrees with 3x3x3.",
        },
    }
    atomic_write_text(str(output / f"manifest_{args.profile}.json"),
                      json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(entries)} 54-atom {args.profile} inputs under {output}")
    print(f"Benchmark: {output / args.profile / FAST_SELECTION[0][1]}")
    return 0


def _job_done(path: Path) -> bool:
    return path.is_file() and "JOB DONE." in path.read_text(errors="replace")


def _pw_converged(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(errors="replace").lower()
    return "convergence has been achieved" in text and "job done." in text


def collect_one(directory: Path) -> dict:
    source = json.loads((directory / "source.json").read_text())
    ph_out = directory / "ph.out"
    if not _pw_converged(directory / "pw.out"):
        raise SystemExit(f"{directory}: pw.x is not converged and complete")
    if not _job_done(ph_out):
        raise SystemExit(f"{directory}: ph.x is not complete")
    eps_raw, born_raw, labels = parse_ph_output(
        ph_out.read_text(errors="replace"))
    expected = source["species"]
    if labels != expected:
        raise SystemExit(
            f"{directory}: ph.x atom labels do not match source ordering")
    if born_raw.shape != (54, 3, 3) or eps_raw.shape != (3, 3):
        raise SystemExit(
            f"{directory}: bad tensor shapes born={born_raw.shape}, "
            f"epsilon={eps_raw.shape}")
    if not np.all(np.isfinite(born_raw)) or not np.all(np.isfinite(eps_raw)):
        raise SystemExit(f"{directory}: non-finite tensor values")
    eps = 0.5 * (eps_raw + eps_raw.T)
    eigenvalues = np.linalg.eigvalsh(eps)
    if eigenvalues.min() <= 0.0:
        raise SystemExit(f"{directory}: epsilon is not positive definite")
    born = apply_asr(born_raw)
    mg = born[:27]
    oxygen = born[27:]
    if np.trace(mg.mean(axis=0)) <= 0 or np.trace(oxygen.mean(axis=0)) >= 0:
        raise SystemExit(f"{directory}: species-mean Born-charge signs are wrong")
    np.save(directory / "born_effective_charges_raw.npy", born_raw)
    np.save(directory / "born_effective_charges.npy", born)
    np.save(directory / "dielectric_infinity_raw.npy", eps_raw)
    np.save(directory / "dielectric_infinity.npy", eps)
    quality = {
        "snapshot_id": source["snapshot_id"],
        "profile": source["profile"],
        "pw_complete": True,
        "ph_complete": True,
        "raw_asr_max_abs": float(np.abs(born_raw.sum(axis=0)).max()),
        "corrected_asr_max_abs": float(np.abs(born.sum(axis=0)).max()),
        "epsilon_antisymmetric_max_abs": float(
            np.abs(eps_raw - eps_raw.T).max()),
        "epsilon_eigenvalues": eigenvalues.tolist(),
        "epsilon_trace_over_3": float(np.trace(eps) / 3.0),
        "mg_mean_trace_over_3": float(np.trace(mg.mean(axis=0)) / 3.0),
        "o_mean_trace_over_3": float(np.trace(oxygen.mean(axis=0)) / 3.0),
        "born_raw_sha256": _sha256(ph_out),
        "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_text(str(directory / "quality.json"),
                      json.dumps(quality, indent=2) + "\n")
    print(json.dumps(quality, indent=2))
    return quality


def collect(args) -> int:
    directory = Path(args.output).resolve() / args.profile / args.snapshot
    collect_one(directory)
    return 0


def _elapsed_from_log(path: Path) -> str:
    if not path.is_file():
        return "-"
    match = re.search(r"^\s*Elapsed \(wall clock\) time.*\):\s*(\S+)\s*$",
                      path.read_text(errors="replace"), re.MULTILINE)
    return match.group(1).strip() if match else "running/unknown"


def status(args) -> int:
    root = Path(args.output).resolve() / args.profile
    manifest_path = Path(args.output).resolve() / f"manifest_{args.profile}.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}; run prepare first")
    manifest = json.loads(manifest_path.read_text())
    print(f"{'snapshot':20s} {'split':10s} {'PW':11s} {'PH':11s} {'PW wall':12s} {'PH wall':12s}")
    for item in manifest["entries"]:
        sid = item["snapshot_id"]
        directory = root / sid
        pw = "done" if _pw_converged(directory / "pw.out") else (
            "running" if (directory / "pw.out").exists() else "pending")
        ph = "done" if _job_done(directory / "ph.out") else (
            "running" if (directory / "ph.out").exists() else "pending")
        print(f"{sid:20s} {item['split']:10s} {pw:11s} {ph:11s} "
              f"{_elapsed_from_log(directory / 'pw.time'):12s} "
              f"{_elapsed_from_log(directory / 'ph.time'):12s}")
    return 0


def compare(args) -> int:
    output = Path(args.output).resolve()
    fast = output / "fast" / args.snapshot
    anchor = output / "anchor" / args.snapshot
    needed = [fast / "born_effective_charges.npy",
              fast / "dielectric_infinity.npy",
              anchor / "born_effective_charges.npy",
              anchor / "dielectric_infinity.npy"]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise SystemExit("profile comparison needs collected tensors:\n  "
                         + "\n  ".join(missing))
    born_fast, born_anchor = np.load(needed[0]), np.load(needed[2])
    eps_fast, eps_anchor = np.load(needed[1]), np.load(needed[3])
    born_delta = born_fast - born_anchor
    eps_delta = eps_fast - eps_anchor
    report = {
        "snapshot_id": args.snapshot,
        "reference_profile": "anchor",
        "candidate_profile": "fast",
        "born_mae_e": float(np.abs(born_delta).mean()),
        "born_max_abs_e": float(np.abs(born_delta).max()),
        "born_relative_l2": float(
            np.linalg.norm(born_delta) / np.linalg.norm(born_anchor)),
        "epsilon_mae": float(np.abs(eps_delta).mean()),
        "epsilon_max_abs": float(np.abs(eps_delta).max()),
        "epsilon_relative_l2": float(
            np.linalg.norm(eps_delta) / np.linalg.norm(eps_anchor)),
        "acceptance_thresholds": {
            "born_mae_e": 0.01,
            "born_max_abs_e": 0.05,
            "epsilon_relative_l2": 0.005,
        },
    }
    threshold = report["acceptance_thresholds"]
    report["fast_profile_accepted"] = bool(
        report["born_mae_e"] <= threshold["born_mae_e"]
        and report["born_max_abs_e"] <= threshold["born_max_abs_e"]
        and report["epsilon_relative_l2"]
        <= threshold["epsilon_relative_l2"])
    report["note"] = (
        "These are prototype gates. Final tolerances must also be supported "
        "by tensor-head validation and downstream EPC sensitivity.")
    destination = output / f"profile_comparison_{args.snapshot}.json"
    atomic_write_text(str(destination), json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["fast_profile_accepted"] else 2


def audit(args) -> int:
    """Prove that every manifest entry has a complete, valid tensor label."""
    output = Path(args.output).resolve()
    manifest_path = output / f"manifest_{args.profile}.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    failures = []
    reports = []
    for item in manifest["entries"]:
        sid = item["snapshot_id"]
        directory = output / args.profile / sid
        try:
            source = json.loads((directory / "source.json").read_text())
            quality = json.loads((directory / "quality.json").read_text())
            born = np.load(directory / "born_effective_charges.npy")
            eps = np.load(directory / "dielectric_infinity.npy")
            if source["snapshot_id"] != sid or quality["snapshot_id"] != sid:
                raise ValueError("snapshot identity mismatch")
            if source["profile"] != args.profile or quality["profile"] != args.profile:
                raise ValueError("profile identity mismatch")
            if not _pw_converged(directory / "pw.out"):
                raise ValueError("pw.x is not converged and complete")
            if not _job_done(directory / "ph.out"):
                raise ValueError("ph.x is not complete")
            if born.shape != (54, 3, 3) or eps.shape != (3, 3):
                raise ValueError(f"bad tensor shapes born={born.shape}, eps={eps.shape}")
            if not np.all(np.isfinite(born)) or not np.all(np.isfinite(eps)):
                raise ValueError("non-finite tensor values")
            asr = float(np.abs(born.sum(axis=0)).max())
            symmetry = float(np.abs(eps - eps.T).max())
            eps_min = float(np.linalg.eigvalsh(eps).min())
            mg_mean = float(np.trace(born[:27].mean(axis=0)) / 3.0)
            o_mean = float(np.trace(born[27:].mean(axis=0)) / 3.0)
            if asr > 1.0e-10:
                raise ValueError(f"ASR residual {asr:.3e} exceeds 1e-10")
            if symmetry > 1.0e-10:
                raise ValueError(f"epsilon asymmetry {symmetry:.3e} exceeds 1e-10")
            if eps_min <= 0.0:
                raise ValueError(f"epsilon minimum eigenvalue is {eps_min:.6g}")
            if mg_mean <= 0.0 or o_mean >= 0.0:
                raise ValueError("species-mean Born-charge signs are wrong")
            reports.append({
                "snapshot_id": sid,
                "asr_max_abs": asr,
                "epsilon_symmetry_max_abs": symmetry,
                "epsilon_min_eigenvalue": eps_min,
                "mg_mean_trace_over_3": mg_mean,
                "o_mean_trace_over_3": o_mean,
            })
        except Exception as exc:
            failures.append({"snapshot_id": sid, "error": str(exc)})
    report = {
        "profile": args.profile,
        "expected": len(manifest["entries"]),
        "verified": len(reports),
        "complete": not failures and len(reports) == len(manifest["entries"]),
        "failures": failures,
        "snapshots": reports,
        "audited_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    destination = output / f"audit_{args.profile}.json"
    atomic_write_text(str(destination), json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--output", default=str(_default_output()))
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--profile", choices=sorted(PROFILES), default="fast")
    prep.set_defaults(func=prepare)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("snapshot")
    collect_parser.add_argument("--profile", choices=sorted(PROFILES),
                                default="fast")
    collect_parser.set_defaults(func=collect)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--profile", choices=sorted(PROFILES),
                               default="fast")
    status_parser.set_defaults(func=status)
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("snapshot")
    compare_parser.set_defaults(func=compare)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--profile", choices=sorted(PROFILES),
                              default="fast")
    audit_parser.set_defaults(func=audit)
    args = parser.parse_args(argv)
    args.workspace = str(data_root(args.workspace))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
