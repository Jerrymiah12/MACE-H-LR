"""Collect aligned DFT, SR-target, analytic-LR, and Full-H scan curves.

The output is a flat, block-indexed HDF5 cache.  Every Hamiltonian is mapped to
the same cell-major atom order, DeepH orbital convention, sparse block labels,
and equilibrium-overlap energy gauge before it is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import torch


from workflows.epc.collect_dft_epc import (_species_to_cell_mapping, load_hamiltonian)
from maceh.epc.derivative import build_supercell_graph
from maceh.epc.lr_correction import (
    load_reindexed_equilibrium_overlap, make_gauge_fixed_predict_fn,
    make_lr_corrected_predict_fn)
from maceh.epc.run import load_model_contexts, make_predict_fn
from maceh.epc.supercell import build_supercell, load_structure
from maceh.parse_configs import EPCConfig
from maceh.data.io.abacus import parse_stru
from maceh.config import load_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_blocks(blocks: dict, keys: list[str], shapes: list[tuple[int, int]],
                   offsets: np.ndarray) -> np.ndarray:
    row = np.zeros(int(offsets[-1]), dtype=np.float64)
    for index, (key, shape) in enumerate(zip(keys, shapes)):
        value = blocks.get(key)
        if value is None:
            continue
        value = np.asarray(value, dtype=np.float64)
        if value.shape != shape:
            raise ValueError(f"{key}: block shape {value.shape} != {shape}")
        row[offsets[index]:offsets[index + 1]] = value.ravel()
    return row


def model_curves(config_path: Path, deltas: np.ndarray, maximum_delta: float,
                 expected_geometry=None) -> tuple[list[dict], object, object]:
    config = EPCConfig(str(config_path))
    torch.set_default_dtype(config.torch_dtype)
    primitive = load_structure(config.structure_dir)
    contexts = load_model_contexts(config)
    kernel = contexts[0][0]
    supercell, _ = build_supercell(primitive, config.q_grid)
    graph = build_supercell_graph(
        supercell, config.radius + 2.0 * maximum_delta,
        torch.get_default_dtype())
    graph.x = kernel.dataset_info.Z_to_index[graph.x]
    if not torch.all(graph.x >= 0):
        raise ValueError("scan structure contains an element absent from model")
    base_predictor = make_predict_fn(contexts, graph, config)
    positions0 = graph.pos.clone()
    if expected_geometry is not None:
        if not np.allclose(supercell.lattice, expected_geometry.lattice,
                           atol=1e-12):
            raise ValueError("model scan supercell lattice changed")
        if not np.allclose(positions0.numpy(), expected_geometry.positions,
                           atol=1e-12):
            raise ValueError("model scan supercell positions changed")
    predictor = base_predictor
    if config.analytic_lr_workspace:
        predictor = make_lr_corrected_predict_fn(
            predictor, positions0, supercell,
            config.analytic_lr_workspace,
            config.analytic_lr_overlap_dir,
            config.analytic_lr_config)
    if config.gauge_overlap_dir:
        predictor = make_gauge_fixed_predict_fn(
            predictor, supercell, config.gauge_overlap_dir)

    curves = []
    begin = time.time()
    for index, delta in enumerate(deltas):
        positions = positions0.clone()
        positions[0, 0] += float(delta)
        curves.append(predictor(positions))
        print(f"  {config_path.name}: {index + 1:2d}/{len(deltas)} "
              f"delta={delta:+.5f} A", flush=True)
    print(f"  {config_path.name}: {len(deltas)} predictions in "
          f"{time.time() - begin:.1f} s")
    del predictor, base_predictor, contexts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return curves, supercell, positions0


def analytic_lr_curves(positions0, supercell, deltas: np.ndarray,
                       workspace: Path, overlap_dir: Path,
                       config_path: Path) -> list[dict]:
    overlaps = load_reindexed_equilibrium_overlap(str(overlap_dir), supercell)

    def zero_predictor(_positions):
        return {key: np.zeros_like(value) for key, value in overlaps.items()}

    predictor = make_lr_corrected_predict_fn(
        zero_predictor, positions0, supercell, str(workspace),
        str(overlap_dir), str(config_path))
    predictor = make_gauge_fixed_predict_fn(
        predictor, supercell, str(overlap_dir))
    curves = []
    for delta in deltas:
        positions = positions0.clone()
        positions[0, 0] += float(delta)
        curves.append(predictor(positions))
    return curves


def dft_context(scan_root: Path, primitive, overlap_dir: Path):
    with (scan_root / "manifest.json").open() as handle:
        manifest = json.load(handle)
    supercell, _ = build_supercell(primitive, (2, 2, 2))
    overlaps = load_reindexed_equilibrium_overlap(str(overlap_dir), supercell)
    first = scan_root / manifest["calculations"][0]["name"]
    source_cell, displaced, species = parse_stru(str(first / "STRU"))
    displacement = np.load(first / "displacements.npy")
    equilibrium = displaced - displacement
    number_map = {"Mg": 12, "O": 8}
    source_numbers = np.asarray([number_map[name] for name in species])
    if not np.allclose(source_cell, supercell.lattice, atol=1e-7):
        raise ValueError("DFT and model scan supercells differ")
    source_to_target = _species_to_cell_mapping(
        supercell, equilibrium, source_numbers)
    return manifest, supercell, overlaps, source_to_target


def create_dataset(handle, name: str, npoints: int, nelements: int):
    return handle.create_dataset(
        name, shape=(npoints, nelements), dtype=np.float64,
        chunks=(1, min(nelements, 1 << 18)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scan-root", default="data/epc/response_scan_dft")
    parser.add_argument("--sr-config", default="workflows/epc/sr.ini")
    parser.add_argument("--full-config", default="workflows/epc/full.ini")
    parser.add_argument("--workspace", default="data")
    parser.add_argument("--overlap-dir", default="data/pilot/snapshot_000001")
    parser.add_argument("--lr-config", default="provenance/config.resolved.yaml")
    parser.add_argument("--structure", default="workflows/epc/structure_primitive")
    parser.add_argument("--output", default="runs/epc/response_scan_curves.h5")
    parser.add_argument("--report", default="workflows/epc/response_scan_collection.json")
    args = parser.parse_args()

    scan_root = Path(args.scan_root).resolve()
    sr_config = Path(args.sr_config).resolve()
    full_config = Path(args.full_config).resolve()
    workspace = Path(args.workspace).resolve()
    overlap_dir = Path(args.overlap_dir).resolve()
    lr_config = Path(args.lr_config).resolve()
    primitive = load_structure(str(Path(args.structure).resolve()))
    manifest, dft_supercell, equilibrium_overlaps, source_to_target = \
        dft_context(scan_root, primitive, overlap_dir)
    deltas = np.asarray([
        item["signed_displacement_angstrom"]
        for item in manifest["calculations"]], dtype=np.float64)
    if not np.all(np.diff(deltas) > 0):
        raise ValueError("scan displacements are not strictly increasing")
    maximum_delta = float(np.max(np.abs(deltas)))

    print("Evaluating SR-target + fixed analytic LR")
    sr_curves, model_supercell, positions0 = model_curves(
        sr_config, deltas, maximum_delta, expected_geometry=dft_supercell)
    keys = sorted(sr_curves[0])
    if any(set(curve) != set(keys) for curve in sr_curves):
        raise ValueError("SR model block-key set changes across scan")
    shapes = [tuple(np.asarray(sr_curves[0][key]).shape) for key in keys]
    offsets = np.concatenate((
        np.array([0], dtype=np.int64),
        np.cumsum([int(np.prod(shape)) for shape in shapes], dtype=np.int64)))
    nelements = int(offsets[-1])
    print(f"Common model representation: {len(keys):,} blocks, "
          f"{nelements:,} matrix elements per point")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=output.parent, prefix=output.name + ".", suffix=".tmp")
    os.close(fd)
    os.unlink(tmp_name)
    coverage = []
    gauge_coefficients = []
    try:
        with h5py.File(tmp_name, "w") as handle:
            string_dtype = h5py.string_dtype("utf-8")
            handle.create_dataset("block_keys", data=np.asarray(keys, dtype=object),
                                  dtype=string_dtype)
            handle["block_shapes"] = np.asarray(shapes, dtype=np.int16)
            handle["block_offsets"] = offsets
            handle["displacements_angstrom"] = deltas
            handle["supercell_lattice_angstrom"] = model_supercell.lattice
            handle["equilibrium_positions_angstrom"] = model_supercell.positions
            handle["atomic_numbers"] = model_supercell.numbers
            handle.attrs["atom_displaced_cell_major"] = 0
            handle.attrs["direction"] = "x"
            handle.attrs["energy_gauge"] = (
                "H - <H,S0>/<S0,S0> S0, common equilibrium overlap")
            handle.attrs["sr_case"] = (
                "independently trained SR-target checkpoint + fixed analytic LR")
            handle.attrs["full_case"] = (
                "independently trained direct total-Hamiltonian checkpoint")

            sr_dataset = create_dataset(
                handle, "hamiltonian/sr_fixed_lr", len(deltas), nelements)
            for index, blocks in enumerate(sr_curves):
                sr_dataset[index] = flatten_blocks(blocks, keys, shapes, offsets)
            del sr_curves

            print("Evaluating standalone fixed analytic LR")
            lr_curves = analytic_lr_curves(
                positions0, model_supercell, deltas, workspace, overlap_dir,
                lr_config)
            lr_dataset = create_dataset(
                handle, "hamiltonian/analytic_lr", len(deltas), nelements)
            for index, blocks in enumerate(lr_curves):
                lr_dataset[index] = flatten_blocks(blocks, keys, shapes, offsets)
            del lr_curves

            print("Evaluating direct Full-H")
            full_curves, _, _ = model_curves(
                full_config, deltas, maximum_delta,
                expected_geometry=dft_supercell)
            if any(set(curve) != set(keys) for curve in full_curves):
                full_keys = set.intersection(*(set(curve)
                                               for curve in full_curves))
                raise ValueError(
                    f"Full/SR graph-key mismatch: SR {len(keys)}, "
                    f"Full common {len(full_keys)}")
            full_dataset = create_dataset(
                handle, "hamiltonian/full_direct", len(deltas), nelements)
            for index, blocks in enumerate(full_curves):
                full_dataset[index] = flatten_blocks(
                    blocks, keys, shapes, offsets)
            del full_curves

            print("Parsing and aligning 25 DFT Hamiltonians")
            cfg = load_config(str(lr_config))
            dft_dataset = create_dataset(
                handle, "hamiltonian/dft", len(deltas), nelements)
            model_key_set = set(keys)
            for index, item in enumerate(manifest["calculations"]):
                folder = scan_root / item["name"]
                blocks, coefficient, scf = load_hamiltonian(
                    str(folder), cfg, source_to_target,
                    equilibrium_overlaps)
                dft_dataset[index] = flatten_blocks(
                    blocks, keys, shapes, offsets)
                present = model_key_set & set(blocks)
                missing_model = set(blocks) - model_key_set
                present_norm_sq = sum(float(np.square(blocks[key]).sum())
                                      for key in present)
                missing_norm_sq = sum(float(np.square(blocks[key]).sum())
                                      for key in missing_model)
                coverage.append({
                    "point": item["name"],
                    "dft_block_count": len(blocks),
                    "model_block_count": len(keys),
                    "common_block_count": len(present),
                    "dft_blocks_absent_from_model": len(missing_model),
                    "absent_dft_norm_fraction": (
                        np.sqrt(missing_norm_sq)
                        / max(np.sqrt(present_norm_sq + missing_norm_sq),
                              1e-300)),
                    "scf_iterations": scf.get("niter"),
                })
                gauge_coefficients.append(float(coefficient))
                print(f"  DFT: {index + 1:2d}/{len(deltas)} "
                      f"{item['name']} common={len(present):,}", flush=True)
            handle["dft_gauge_coefficients_eV"] = np.asarray(
                gauge_coefficients)
            handle.attrs["scan_manifest_sha256"] = sha256(
                scan_root / "manifest.json")
            handle.attrs["born_effective_charges_sha256"] = sha256(
                workspace / "reference" / "born_effective_charges.npy")
            handle.attrs["dielectric_infinity_sha256"] = sha256(
                workspace / "reference" / "dielectric_infinity.npy")
        os.replace(tmp_name, output)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise

    report = {
        "output": str(output),
        "output_size_bytes": output.stat().st_size,
        "point_count": len(deltas),
        "displacements_angstrom": deltas.tolist(),
        "block_count": len(keys),
        "matrix_elements_per_point": nelements,
        "coverage": coverage,
        "maximum_absent_dft_norm_fraction": max(
            row["absent_dft_norm_fraction"] for row in coverage),
        "dft_gauge_coefficient_range_eV": [
            min(gauge_coefficients), max(gauge_coefficients)],
        "source_to_target_atom_index": source_to_target.tolist(),
        "provenance": {
            "scan_manifest": str((scan_root / "manifest.json").resolve()),
            "scan_manifest_sha256": sha256(scan_root / "manifest.json"),
            "sr_config": str(sr_config),
            "full_config": str(full_config),
            "lr_config": str(lr_config),
        },
    }
    report_path = Path(args.report)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output} ({output.stat().st_size / 1024 ** 3:.2f} GiB)")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
