from typing import Union, Dict, Tuple, List
import os
import time
import tqdm
import hashlib
import json

from pymatgen.core.structure import Structure
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from pathos.multiprocessing import ProcessingPool as Pool

from ..graph import get_graph, load_orbital_types
from ..utils import process_targets
from ..e3modules import e3TensorDecomp


TENSOR_SPLIT_CODES = {"unlabelled": 0, "train": 1, "validation": 2,
                      "test": 3}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_path(manifest_path, value):
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(os.path.dirname(manifest_path), value))


def _load_tensor_entry(manifest_path, manifest, entry):
    """Load and audit one campaign entry without touching the graph cache."""
    sid = entry["snapshot_id"]
    split = entry["split"]
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"{sid}: invalid tensor split {split!r}")
    root = _resolve_manifest_path(manifest_path, manifest["output"])
    directory = os.path.join(root, entry.get("profile", manifest["profile"]), sid)
    source_path = os.path.join(directory, "source.json")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"{sid}: missing tensor source record {source_path}")
    with open(source_path) as handle:
        source = json.load(handle)
    if source.get("snapshot_id") != sid or source.get("split") != split:
        raise ValueError(f"{sid}: manifest/source snapshot or split mismatch")
    source_stru = source.get("source_stru")
    if not source_stru or not os.path.isfile(source_stru):
        raise FileNotFoundError(f"{sid}: source STRU is unavailable: {source_stru}")
    if _sha256(source_stru) != source.get("source_stru_sha256"):
        raise ValueError(f"{sid}: source STRU SHA-256 mismatch")

    born_path = os.path.join(directory, "born_effective_charges.npy")
    epsilon_path = os.path.join(directory, "dielectric_infinity.npy")
    born = np.load(born_path, allow_pickle=False)
    epsilon = np.load(epsilon_path, allow_pickle=False)
    nat = int(source["nat"])
    if born.shape != (nat, 3, 3):
        raise ValueError(f"{sid}: Born shape {born.shape}, expected {(nat, 3, 3)}")
    if epsilon.shape != (3, 3):
        raise ValueError(f"{sid}: epsilon shape {epsilon.shape}, expected (3, 3)")
    if not np.isfinite(born).all() or not np.isfinite(epsilon).all():
        raise ValueError(f"{sid}: tensor label contains a non-finite value")
    asr = float(np.abs(born.sum(axis=0)).max())
    symmetry = float(np.abs(epsilon - epsilon.T).max())
    minimum_eigenvalue = float(np.linalg.eigvalsh(epsilon).min())
    if asr > 1.0e-8:
        raise ValueError(f"{sid}: Born ASR residual {asr:.3e} exceeds 1e-8")
    if symmetry > 1.0e-8:
        raise ValueError(f"{sid}: dielectric asymmetry {symmetry:.3e} exceeds 1e-8")
    if minimum_eigenvalue <= 0.0:
        raise ValueError(f"{sid}: dielectric is not positive definite")
    species = np.asarray(source.get("species") or [])
    if len(species) != nat:
        raise ValueError(f"{sid}: source species count does not equal nat")
    if "Mg" in species and "O" in species:
        mg_charge = float(np.trace(born[species == "Mg"].mean(axis=0)) / 3.0)
        o_charge = float(np.trace(born[species == "O"].mean(axis=0)) / 3.0)
        if mg_charge <= 0.0 or o_charge >= 0.0:
            raise ValueError(f"{sid}: Mg/O Born-charge signs are inconsistent")
    return {
        "id": sid, "split": split, "source": source,
        "born": born, "epsilon": epsilon,
        "born_path": born_path, "epsilon_path": epsilon_path,
    }

class AijData(InMemoryDataset):
    def __init__(self, raw_data_dir: str, graph_dir: str, target: str,
                 dataset_name : str, multiprocessing: bool, radius: float, max_num_nbr: int, edge_Aij: bool,
                 default_dtype_torch, nums: int = None, inference:bool = False, only_ij: bool = False, load_graph=True):
        """
        :param raw_data_dir: 原始数据目录, 允许存在嵌套
when interface == 'h5',
raw_data_dir
├── 00
│     ├──<target>s.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── 01
│     ├──<target>s.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── 02
│     ├──<target>s.h5
│     ├──element.dat
│     ├──orbital_types.dat
│     ├──site_positions.dat
│     ├──lat.dat
│     └──info.json
├── ...
        :param graph_dir: 存储图的目录
        :param multiprocessing: 多进程生成图
        :param radius: 生成图的截止半径
        :param max_num_nbr: 生成图限制最大近邻数, 为 0 时不限制
        :param edge_Aij: 图的边是否一一对应 Aij, 如果是为 False 则存在两套图的连接, 一套用于节点更新, 一套用于记录 Aij
        :param default_dtype_torch: 浮点数数据类型
        """
        self.raw_data_dir = raw_data_dir
        assert dataset_name.find('-') == -1, '"-" can not be included in the dataset name'
        create_from_DFT = radius < 0
        radius_info = 'rFromDFT' if create_from_DFT else f'{radius}r{max_num_nbr}mn'
        if target == 'hamiltonian':
            graph_file_name = f'HGraph-{dataset_name}-{radius_info}-edge{"" if edge_Aij else "!"}=Aij{"-undrct" if only_ij else ""}.pkl' # undrct = undirected
        elif target == 'density_matrix':
            graph_file_name = f'DMGraph-{dataset_name}-{radius_info}-{edge_Aij}edge{"-undrct" if only_ij else ""}.pkl'
        else:
            raise ValueError('Unknown prediction target: {}'.format(target))
        self.data_file = os.path.join(graph_dir, graph_file_name)
        os.makedirs(graph_dir, exist_ok=True)
        self.data, self.slices = None, None
        self.target = target
        self.target_file_name = 'overlaps.h5' if inference else f'{self.target}s.h5'
        self.dataset_name = dataset_name
        self.multiprocessing = multiprocessing
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.create_from_DFT = create_from_DFT
        self.edge_Aij = edge_Aij
        self.default_dtype_torch = default_dtype_torch

        self.nums = nums
        self.inference = inference
        self.only_ij = only_ij
        self.transform = None
        self.__indices__ = None
        self.__data_list__ = None
        self._indices = None
        self._data_list = None

        print(f'Graph data file: {graph_file_name}')
        if os.path.exists(self.data_file):
            print('Use existing graph data file')
        else:
            assert raw_data_dir, 'Required graph does not exist, or graph filename cannot be correctly identified'
            print('Process new data file......')
            self.process()
        if load_graph:
            begin = time.time()
            # torch >= 2.6 flipped torch.load's weights_only default to True,
            # which cannot unpickle the torch_geometric classes in a graph
            # cache (DataEdgeAttr, DataTensorAttr, GlobalStorage) and fails
            # with an UnpicklingError.  self.data_file is always written by
            # self.process() on this machine, so it is a trusted local file,
            # not untrusted input -- weights_only=False is safe here and is
            # preferable to pinning torch or maintaining a global allowlist.
            # Graph caches are multi-gigabyte tensors.  Loading every storage
            # eagerly makes a second resident copy before ``set_mask`` has a
            # chance to discard ``Aij`` and can exhaust a memory-constrained
            # WSL VM.  The cache is written with torch's zipfile serializer,
            # so mmap keeps its storages file-backed while preserving normal
            # tensor semantics.  Pages are faulted in as individual graphs
            # are prepared and can be reclaimed by the kernel immediately.
            loaded_data = torch.load(
                self.data_file, map_location='cpu', weights_only=False,
                mmap=True)
            self.data, self.slices, self.info = loaded_data
            print(f'Finish loading the processed {len(self)} structures (spinful: {self.info["spinful"]}, '
                f'the number of atomic types: {len(self.info["index_to_Z"])}), cost {time.time() - begin:.2f} seconds')
    
    @classmethod
    def from_existing_graph(cls, existing_graph_dir, default_dtype_torch):
        assert os.path.isfile(existing_graph_dir), f'Required graph {existing_graph_dir} does not exist'
        save_graph_dir = os.path.dirname(existing_graph_dir)
        existing_graph_dir = os.path.basename(existing_graph_dir)
        assert existing_graph_dir[-4:] == '.pkl', 'graph filename extension should be .pkl'
        options = existing_graph_dir.rstrip('.pkl').split('-')
        if options[0][0] == 'H':
            target = 'hamiltonian'
        elif options[0][0:2] == 'DM':
            target = 'density_matrix'
        else:
            raise ValueError(f'Cannot identify graph file {existing_graph_dir}')
        dataset_name = options[1]
        if options[2] == 'rFromDFT':
            cutoff_radius = -1
        else:
            cutoff_radius = float(options[2].split('r')[0])
        only_ij = options[-1] == 'undrct'
        return cls(
            raw_data_dir=None, 
            graph_dir=save_graph_dir, 
            target=target, 
            dataset_name=dataset_name, 
            multiprocessing=False, 
            radius=cutoff_radius, 
            max_num_nbr=0,               #todo
            edge_Aij=True,               #todo
            default_dtype_torch=default_dtype_torch,
            only_ij=only_ij,
            )

    def process_worker(self, folder, **kwargs):
        stru_id = os.path.split(folder)[-1]

        
        site_positions = np.loadtxt(os.path.join(folder, 'site_positions.dat')).T
        elements = np.loadtxt(os.path.join(folder, 'element.dat'))
        if len(elements.shape) == 0:
            elements = elements[None]
            site_positions = site_positions[None, :]
        structure = Structure(np.loadtxt(os.path.join(folder, 'lat.dat')).T,
                              elements,
                              site_positions,
                              coords_are_cartesian=True,
                              to_unit_cell=False)

        cart_coords = torch.tensor(structure.cart_coords, dtype=self.default_dtype_torch)
        frac_coords = torch.tensor(structure.frac_coords, dtype=self.default_dtype_torch)
        numbers = torch.tensor(structure.atomic_numbers)
        structure.lattice.matrix.setflags(write=True)
        lattice = torch.tensor(structure.lattice.matrix, dtype=self.default_dtype_torch)
        return get_graph(cart_coords, frac_coords, numbers, stru_id, r=self.radius, max_num_nbr=self.max_num_nbr,
                         edge_Aij=self.edge_Aij, lattice=lattice, default_dtype_torch=self.default_dtype_torch,
                         data_folder=folder, target_file_name=self.target_file_name, inference=self.inference, 
                         only_ij=self.only_ij, create_from_DFT=self.create_from_DFT, **kwargs)

    def process(self):
        begin = time.time()
        folder_list = []
        print(f'Looking for preprocessed data under: {self.raw_data_dir}')
        for root, dirs, files in os.walk(self.raw_data_dir):
            if {'element.dat', 'orbital_types.dat', 'lat.dat', 'site_positions.dat'}.issubset(files):
                # if self.target_file_name in files:
                folder_list.append(root)
        folder_list = folder_list[: self.nums]
        assert len(folder_list) != 0, "Can not find any structure"
        print('Found %d structures, have cost %d seconds' % (len(folder_list), time.time() - begin))

        begin = time.time()
        if self.multiprocessing:
            print('Use multiprocessing')
            with Pool() as pool:
                data_list = list(tqdm.tqdm(pool.imap(self.process_worker, folder_list), total=len(folder_list)))
        else:
            data_list = [self.process_worker(folder) for folder in tqdm.tqdm(folder_list)]
        print('Finish processing %d structures, have cost %d seconds' % (len(data_list), time.time() - begin))
        index_to_Z, Z_to_index = self.element_statistics(data_list)

        spinful = data_list[0].spinful
        for d in data_list:
            assert spinful == d.spinful
            
        _, orbital_types = load_orbital_types(path=os.path.join(folder_list[0], 'orbital_types.dat'),
                                           return_orbital_types=True) 
        elements = np.loadtxt(os.path.join(folder_list[0], 'element.dat'))
        orbital_types_new = []
        for i in range(len(index_to_Z)):
            orbital_types_new.append(orbital_types[np.where(elements == index_to_Z[i].numpy())[0][0]])
        #TODO 数据集包含不同元素

        begin = time.time()
        data, slices = self.collate(data_list)
        torch.save((data, slices, dict(spinful=spinful, index_to_Z=index_to_Z, Z_to_index=Z_to_index, orbital_types=orbital_types_new)), self.data_file)
        print('Finished saving %d structures to save_graph_dir, have cost %d seconds' % (len(data_list), time.time() - begin))

    def element_statistics(self, data_list):
        # TODO 没有处理数据集包括不同元素组成的情况
        index_to_Z, inverse_indices = torch.unique(data_list[0].x, sorted=True, return_inverse=True)
        Z_to_index = torch.full((100,), -1, dtype=torch.int64)
        Z_to_index[index_to_Z] = torch.arange(len(index_to_Z))

        for data in data_list:
            data.x = Z_to_index[data.x]

        return index_to_Z, Z_to_index

    def attach_tensor_labels(self, manifest_path, train_ids=None,
                             validation_ids=None, test_ids=None,
                             coordinate_tolerance=1.0e-6):
        """Attach small Born/epsilon sidecars to an existing graph cache.

        The manifest is the tensor-DFPT campaign manifest produced by
        ``training/tensor_dfpt_54.py``.  Test entries are fully audited but are
        intentionally allowed to be absent from a train+validation graph cache.
        """
        manifest_path = os.path.abspath(manifest_path)
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        entries = manifest.get("entries") or []
        if not entries:
            raise ValueError(f"{manifest_path}: tensor manifest has no entries")
        ids = [item.get("snapshot_id") for item in entries]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{manifest_path}: duplicate snapshot IDs")

        declared = {
            "train": set(train_ids or []),
            "validation": set(validation_ids or []),
            "test": set(test_ids or []),
        }
        manifest_splits = {
            name: {item["snapshot_id"] for item in entries
                   if item.get("split") == name}
            for name in declared
        }
        if any(declared.values()):
            for name in declared:
                if declared[name] != manifest_splits[name]:
                    raise ValueError(
                        f"tensor {name} IDs differ from manifest: "
                        f"missing={sorted(manifest_splits[name] - declared[name])}, "
                        f"extra={sorted(declared[name] - manifest_splits[name])}")
        split_sets = list(manifest_splits.values())
        if any(split_sets[i] & split_sets[j] for i in range(3)
               for j in range(i + 1, 3)):
            raise ValueError("tensor train/validation/test splits overlap")

        audited = {item["snapshot_id"]:
                   _load_tensor_entry(manifest_path, manifest, item)
                   for item in entries}
        graph_ids = list(self._data.stru_id)
        graph_index = {sid: i for i, sid in enumerate(graph_ids)}
        missing_non_test = ((manifest_splits["train"] |
                             manifest_splits["validation"]) - set(graph_ids))
        if missing_non_test:
            raise ValueError("labelled train/validation snapshots absent from graph: "
                             + ", ".join(sorted(missing_non_test)))
        leaked_test = manifest_splits["test"] & set(graph_ids)
        if leaked_test:
            raise ValueError("locked tensor-test snapshots are present in the "
                             "training graph cache: " + ", ".join(sorted(leaked_test)))

        node_slices = self.slices["x"]
        born_target = torch.zeros(
            (int(node_slices[-1]), 3, 3), dtype=self.default_dtype_torch)
        epsilon_target = torch.zeros(
            (len(self), 3, 3), dtype=self.default_dtype_torch)
        has_born = torch.zeros(len(self), dtype=torch.bool)
        has_epsilon = torch.zeros(len(self), dtype=torch.bool)
        split_code = torch.zeros(len(self), dtype=torch.int8)

        # Parse the exact source STRU used for DFPT.  This proves coordinate
        # provenance and ordering against graph positions without rebuilding it.
        from .io.abacus import parse_stru
        from pymatgen.core import Element
        train_records = []
        for sid, record in audited.items():
            if sid not in graph_index:
                continue  # locked test: audited above, never attached/sampled
            index = graph_index[sid]
            graph = self.get(index)
            cell, positions, species = parse_stru(record["source"]["source_stru"])
            expected_z = np.asarray([Element(symbol).Z for symbol in species])
            graph_z = self.info["index_to_Z"][graph.x].cpu().numpy()
            if len(graph.x) != len(record["born"]):
                raise ValueError(f"{sid}: graph/label atom-count mismatch")
            if not np.array_equal(graph_z, expected_z):
                raise ValueError(f"{sid}: graph and DFPT atom ordering differ")
            coordinate_delta = graph.pos.cpu().numpy() - positions
            fractional_delta = coordinate_delta @ np.linalg.inv(cell)
            if not np.allclose(fractional_delta, np.rint(fractional_delta),
                               atol=coordinate_tolerance, rtol=0.0):
                raise ValueError(
                    f"{sid}: graph and DFPT coordinates differ beyond "
                    "periodic lattice translations")
            graph_cell = graph.lattice.squeeze(0).cpu().numpy()
            if not np.allclose(graph_cell, cell, atol=coordinate_tolerance,
                               rtol=0.0):
                raise ValueError(f"{sid}: graph and DFPT cells differ")
            begin, end = int(node_slices[index]), int(node_slices[index + 1])
            born_target[begin:end] = torch.as_tensor(
                record["born"], dtype=self.default_dtype_torch)
            epsilon_target[index] = torch.as_tensor(
                record["epsilon"], dtype=self.default_dtype_torch)
            has_born[index] = True
            has_epsilon[index] = True
            split_code[index] = TENSOR_SPLIT_CODES[record["split"]]
            if record["split"] == "train":
                train_records.append((record, graph.x.cpu().numpy()))

        if not train_records:
            raise ValueError("tensor manifest has no attached training labels")

        # Fixed residual baselines and componentwise scales are derived from
        # tensor-training labels only; validation and test cannot influence them.
        n_species = len(self.info["index_to_Z"])
        species_values = [[] for _ in range(n_species)]
        epsilon_values = []
        for record, species_indices in train_records:
            for species_index in range(n_species):
                species_values[species_index].append(
                    record["born"][species_indices == species_index])
            epsilon_values.append(record["epsilon"])
        born_mean = np.stack([
            np.concatenate(values, axis=0).mean(axis=0)
            for values in species_values
        ])
        born_baseline = np.stack([
            np.eye(3) * (np.trace(value) / 3.0) for value in born_mean
        ])
        epsilon_mean = np.stack(epsilon_values).mean(axis=0)
        epsilon_baseline = np.eye(3) * (np.trace(epsilon_mean) / 3.0)
        born_residuals = np.concatenate([
            record["born"] - born_baseline[species_indices]
            for record, species_indices in train_records
        ], axis=0)
        epsilon_residuals = np.stack(epsilon_values) - epsilon_baseline
        born_scale = np.maximum(born_residuals.std(axis=0), 1.0e-8)
        epsilon_scale = np.maximum(epsilon_residuals.std(axis=0), 1.0e-8)
        constant_born_baseline = np.stack([
            record["born"] for record, _ in train_records
        ]).mean(axis=0)
        constant_baseline_metrics = {}
        model_baseline_metrics = {}
        for split in ("train", "validation", "test"):
            born_errors, epsilon_errors = [], []
            model_born_errors, model_epsilon_errors = [], []
            for record in audited.values():
                if record["split"] != split:
                    continue
                if record["born"].shape != constant_born_baseline.shape:
                    raise ValueError(
                        'constant Born baseline requires a common atom ordering')
                predicted_born = constant_born_baseline
                born_errors.append(predicted_born - record["born"])
                source_z = np.asarray([
                    Element(symbol).Z for symbol in record["source"]["species"]
                ])
                source_species = np.asarray([
                    int(self.info["Z_to_index"][int(z)]) for z in source_z
                ])
                model_born = born_baseline[source_species]
                model_born = model_born - model_born.mean(axis=0)
                model_born_errors.append(model_born - record["born"])
                epsilon_errors.append(epsilon_mean - record["epsilon"])
                model_epsilon_errors.append(
                    epsilon_baseline - record["epsilon"])
            born_error = np.concatenate(born_errors, axis=0)
            model_born_error = np.concatenate(model_born_errors, axis=0)
            epsilon_error = np.stack(epsilon_errors)
            model_epsilon_error = np.stack(model_epsilon_errors)
            constant_baseline_metrics[split] = {
                "born_mae": float(np.abs(born_error).mean()),
                "born_rmse": float(np.sqrt(np.square(born_error).mean())),
                "epsilon_mae": float(np.abs(epsilon_error).mean()),
                "epsilon_rmse": float(np.sqrt(np.square(epsilon_error).mean())),
            }
            model_baseline_metrics[split] = {
                "born_mae": float(np.abs(model_born_error).mean()),
                "born_rmse": float(np.sqrt(
                    np.square(model_born_error).mean())),
                "epsilon_mae": float(np.abs(model_epsilon_error).mean()),
                "epsilon_rmse": float(np.sqrt(
                    np.square(model_epsilon_error).mean())),
            }

        self._data.born_target = born_target
        self._data.epsilon_target = epsilon_target
        self._data.has_born_label = has_born
        self._data.has_epsilon_label = has_epsilon
        self._data.tensor_split = split_code
        graph_slices = torch.arange(len(self) + 1, dtype=torch.long)
        self.slices["born_target"] = node_slices.clone()
        for key in ("epsilon_target", "has_born_label",
                    "has_epsilon_label", "tensor_split"):
            self.slices[key] = graph_slices.clone()
        self.tensor_label_info = {
            "manifest_path": manifest_path,
            "manifest_sha256": _sha256(manifest_path),
            "splits": {key: sorted(value) for key, value in manifest_splits.items()},
            "attached": sorted(set(audited) & set(graph_ids)),
            "audited_test": sorted(manifest_splits["test"]),
            "born_baseline": torch.as_tensor(
                born_baseline, dtype=self.default_dtype_torch),
            "epsilon_baseline": torch.as_tensor(
                epsilon_baseline, dtype=self.default_dtype_torch),
            "born_scale": torch.as_tensor(
                born_scale, dtype=self.default_dtype_torch),
            "epsilon_scale": torch.as_tensor(
                epsilon_scale, dtype=self.default_dtype_torch),
            "constant_baseline_metrics": constant_baseline_metrics,
            "model_baseline_metrics": model_baseline_metrics,
        }
        # ``get`` populated per-graph cache entries during the provenance
        # audit; clear them so subsequent reads expose the newly attached data.
        self.__data_list__ = None
        self._data_list = None
        return self.tensor_label_info

    def set_mask(self, targets, del_Aij=True, convert_to_net=False):
        begin = time.time()
        print("\nSetting mask for dataset...")
        assert self._indices is None, \
            "set_mask must be called on the complete AijData dataset"
        
        spinful = self.info['spinful']
        
        dtype = torch.get_default_dtype()
        if spinful:
            if dtype == torch.float32:
                dtype = torch.complex64
            elif dtype == torch.float64:
                dtype = torch.complex128
            else:
                raise ValueError(f'Unsupported dtype: {dtype}')
        
        equivariant_blocks, out_js_list, out_slices = process_targets(self.info['orbital_types'], self.info["index_to_Z"], targets)
        if convert_to_net:
            construct_kernel = e3TensorDecomp(None, out_js_list, torch.get_default_dtype(), spinful=spinful, if_sort=True) # todo: dtype
        
        atom_num_orbital = [sum(map(lambda x: 2 * x + 1,atom_orbital_types)) for atom_orbital_types in self.info['orbital_types']]

        # ``self`` is an InMemoryDataset whose raw Aij tensor can be many GiB.
        # The previous implementation accumulated one label/mask pair per
        # graph and then ``collate`` allocated another complete pair.  At the
        # collation point the raw Aij plus both output copies were resident at
        # once (over 24 GiB for the MgO production cache), triggering WSL's
        # OOM killer.  Allocate the final collated buffers once and fill each
        # graph's edge slice in place instead.
        if self.slices is None:
            edge_slices = torch.tensor(
                [0, self._data.Aij.shape[0]], dtype=torch.long)
        else:
            edge_slices = self.slices['Aij'].clone()
        num_edges = int(edge_slices[-1])
        out_dim = int(out_slices[-1])
        raw_tail_shape = (4, out_dim) if spinful else (out_dim,)
        if convert_to_net:
            dummy = torch.zeros((1, *raw_tail_shape), dtype=dtype)
            converted_dummy = construct_kernel.get_net_out(dummy)
            output_tail_shape = tuple(converted_dummy.shape[1:])
            labels = torch.empty(
                (num_edges, *output_tail_shape),
                dtype=converted_dummy.dtype)
            masks = torch.empty(
                (num_edges, *output_tail_shape), dtype=torch.int8)
        else:
            labels = torch.zeros(
                (num_edges, *raw_tail_shape), dtype=dtype)
            masks = torch.zeros(
                (num_edges, *raw_tail_shape), dtype=torch.int8)

        for graph_index in range(len(self)):
            data = self.get(graph_index)
            assert data.spinful == spinful
            if data.Aij is not None:
                if not torch.all(data.Aij_mask):
                    raise NotImplementedError("Not yet have support for graph radius including Aij without calculation")

            # label of each edge is a vector which is each target H block flattened and concatenated
            edge_begin = int(edge_slices[graph_index])
            edge_end = int(edge_slices[graph_index + 1])
            assert edge_end - edge_begin == data.num_edges
            if convert_to_net:
                label = torch.zeros(
                    (data.num_edges, *raw_tail_shape), dtype=dtype)
                mask = torch.zeros(
                    (data.num_edges, *raw_tail_shape), dtype=torch.int8)
            else:
                label = labels[edge_begin:edge_end]
                mask = masks[edge_begin:edge_end]

            atomic_number_edge_i = data.x[data.edge_index[0]]
            atomic_number_edge_j = data.x[data.edge_index[1]]

            for index_out, equivariant_block in enumerate(equivariant_blocks):
                for N_M_str, block_slice in equivariant_block.items():
                    condition_atomic_number_i, condition_atomic_number_j = map(lambda x: self.info["Z_to_index"][int(x)], N_M_str.split())
                    condition_slice_i = slice(block_slice[0], block_slice[1])
                    condition_slice_j = slice(block_slice[2], block_slice[3])
                    if spinful:
                        condition_slice_i_ds = slice(atom_num_orbital[condition_atomic_number_i] + block_slice[0],
                                                      atom_num_orbital[condition_atomic_number_i] + block_slice[1]) # ds = down spin
                        condition_slice_j_ds = slice(atom_num_orbital[condition_atomic_number_j] + block_slice[2],
                                                     atom_num_orbital[condition_atomic_number_j] + block_slice[3])
                    if data.Aij is not None:
                        out_slice = slice(out_slices[index_out], out_slices[index_out + 1])
                        condition_index = torch.where(
                            (atomic_number_edge_i == condition_atomic_number_i)
                            & (atomic_number_edge_j == condition_atomic_number_j)
                        )
                        if spinful:
                            # noncollinear spin block order:
                            # 0(uu) 1(ud)
                            # 2(du) 3(dd)
                            label[condition_index[0], 0, out_slice] += data.Aij[:, condition_slice_i, condition_slice_j].reshape(data.num_edges, -1)[condition_index]
                            label[condition_index[0], 1, out_slice] += data.Aij[:, condition_slice_i, condition_slice_j_ds].reshape(data.num_edges, -1)[condition_index]
                            label[condition_index[0], 2, out_slice] += data.Aij[:, condition_slice_i_ds, condition_slice_j].reshape(data.num_edges, -1)[condition_index]
                            label[condition_index[0], 3, out_slice] += data.Aij[:, condition_slice_i_ds, condition_slice_j_ds].reshape(data.num_edges, -1)[condition_index]
                            mask[condition_index[0], :, out_slice] += 1
                        else:
                            label[condition_index[0], out_slice] += data.Aij[:, condition_slice_i, condition_slice_j].reshape(data.num_edges, -1)[condition_index]
                            mask[condition_index[0], out_slice] += 1
            if data.Aij is not None:
                assert torch.all((mask == 1) | (mask == 0)), 'Some blocks are required to predict multiple times'
                if convert_to_net:
                    converted_label = construct_kernel.get_net_out(label)
                    assert converted_label.shape[1:] == output_tail_shape
                    labels[edge_begin:edge_end].copy_(converted_label)
                    if spinful:
                        converted_mask = construct_kernel.convert_mask(
                            mask.bool())
                    else:
                        # Preserve the existing non-spinful behavior: the
                        # transform changes values/order but not mask layout.
                        converted_mask = mask.bool()
                    assert converted_mask.shape[1:] == output_tail_shape
                    masks[edge_begin:edge_end].copy_(converted_mask)

        self.__indices__ = None
        self.__data_list__ = None
        self._indices = None
        self._data_list = None
        # Drop the final per-graph view before removing the mmap-backed Aij
        # storage, otherwise that view keeps the entire storage alive.
        data = None

        self._data.label = labels
        # Every entry was checked to be the int8 byte 0 or 1 above.  int8 and
        # bool have the same item size, so a dtype view is exact and avoids a
        # transient second 1.9 GiB mask allocation on the production cache.
        self._data.mask = masks.view(torch.bool)
        if self.slices is not None:
            self.slices['label'] = edge_slices
            self.slices['mask'] = edge_slices.clone()

        if del_Aij:
            del self._data.Aij_mask
            del self._data.Aij
            if self.slices is not None:
                self.slices.pop('Aij_mask', None)
                self.slices.pop('Aij', None)
        print(f"Finished setting mask for dataset, cost {time.time() - begin:.2f} seconds")

        return out_js_list, out_slices
