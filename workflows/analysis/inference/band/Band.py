import os
import h5py
import json
import argparse
import warnings
import numpy as np
import spglib, seekpath
from typing import Callable, Union
from mendeleev import element
from scipy.optimize import minimize_scalar
from scipy.special import erf

import matplotlib.pyplot as plt
from maceh.analysis.figures import save_figure
from matplotlib.lines import Line2D
import seaborn as sns

from ase import Atoms
from ase.units import kB

# Optional: specify the path to your Julia project or environment,
# such that the environment variable does not need to be specified for each terminal session
# os.environ["JULIA_PROJECT"] = "/path/to/project"

import julia

julia.install()

from julia.api import Julia
from julia import Main

jl = Julia(compiled_modules=False)
Main.eval("using Distributed")


def fermi_dirac(
    E: Union[float, np.ndarray], mu: float, T: float, exp_cutoff: float = 100.0
):
    if T == 0:
        return np.heaviside(mu - E, 0.0)
    else:
        fd = np.zeros_like(E)
        boltzmann_factor = (E - mu) / (kB * T)
        mask = boltzmann_factor < exp_cutoff
        fd[mask] = 1 / (1 + np.exp(boltzmann_factor[mask]))
        return fd


def fermi_dirac_integral(
    E: Union[float, np.ndarray], mu: float, T: float, exp_cutoff: float = 100.0
):
    fdi = np.zeros_like(E)
    if T == 0:
        fdi[np.where(E - mu <= 0)] = E[np.where(E - mu <= 0)] - mu
        fdi[np.where(E - mu > 0)] = 0
        return fdi
    else:
        boltzmann_factor = -1 * (E - mu) / (kB * T)
        overflow_mask = boltzmann_factor > exp_cutoff
        fdi[~overflow_mask] = (
            -kB * T * np.log(1 + np.exp(boltzmann_factor[~overflow_mask]))
        )
        fdi[overflow_mask] = E[overflow_mask] - mu
        return fdi


def gaussian_smearing(E: Union[float, np.ndarray], mu: float, T: float):
    if T == 0:
        return np.heaviside(mu - E, 0.0)
    else:
        boltzmann_factor = (E - mu) / (kB * T)
        return 0.5 * (1 - erf(boltzmann_factor))


def get_atoms(data_dir: str) -> Atoms:
    numbers_path = os.path.join(data_dir, "element.dat")
    numbers = np.loadtxt(numbers_path).astype(int)
    positions_path = os.path.join(data_dir, "site_positions.dat")
    positions = np.loadtxt(positions_path).astype(float).T
    cell_path = os.path.join(data_dir, "lat.dat")
    cell = np.loadtxt(cell_path).astype(float).T
    box = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=(1, 1, 1))
    return box


def get_R(data_dir: str) -> np.ndarray:
    R_list_path = os.path.join(data_dir, "R_list.dat")
    R_list = np.loadtxt(R_list_path).astype(int)
    return R_list


def get_kpath(atoms: Atoms, crystal_type: str = "2D") -> np.ndarray:
    cell = (atoms.cell, atoms.get_scaled_positions(), atoms.get_atomic_numbers())

    # symmetry_dataset = spglib.get_symmetry_dataset(cell)
    # path = seekpath.get_path(cell)
    # path['path']
    # path['point_coords']

    explicit_path = seekpath.get_explicit_k_path(cell, reference_distance=0.005)

    x = explicit_path["explicit_kpoints_linearcoord"]
    labels = explicit_path["explicit_kpoints_labels"]
    kpoints_rel = explicit_path["explicit_kpoints_rel"]
    # kpoints_abs = explicit_path['explicit_kpoints_abs']
    # rlat = explicit_path['reciprocal_primitive_lattice']

    # explicit_path['primitive_lattice']
    # explicit_path['explicit_segments']
    # explicit_path['path']

    if crystal_type.lower() == "2D":
        kpoint_dict = explicit_path["point_coords"]
        mask = np.full(len(labels), True)
        for idx, (path, segments) in enumerate(
            zip(explicit_path["path"], explicit_path["explicit_segments"])
        ):
            if kpoint_dict[path[0]][2] != 0 or kpoint_dict[path[1]][2] != 0:
                if (
                    idx > 0
                    and explicit_path["path"][idx - 1][1] == path[0]
                    and mask[segments[0] - 1] == True
                ):
                    mask[segments[0] + 1 : segments[1]] = False
                else:
                    mask[segments[0] : segments[1]] = False
        x = x[mask]
        labels = np.array(labels)[mask].tolist()
        kpoints_rel = kpoints_rel[mask]

    return {"x": x, "labels": labels, "kpoints_rel": kpoints_rel}


def Hermitian(matrix: dict) -> dict:
    output = {}
    for key in matrix.keys():
        key_self_adjoint = str(
            [-i for i in eval(key)[:3]] + [eval(key)[4], eval(key)[3]]
        )
        output[key] = np.array(
            (matrix[key] + np.matrix(matrix[key_self_adjoint]).getH()) / 2
        )
    return output


def get_matrices(data_dir: str, force_hermiticity=True) -> dict:

    matrices = {}

    hamiltonians_path = os.path.join(data_dir, "hamiltonians.h5")
    if os.path.exists(hamiltonians_path):
        with h5py.File(hamiltonians_path, "r") as f:
            hamiltonians = dict([(key, np.array(value)) for key, value in f.items()])
        if force_hermiticity == True:
            hamiltonians = Hermitian(hamiltonians)
        matrices["hamiltonians"] = hamiltonians

    overlaps_path = os.path.join(data_dir, "overlaps.h5")
    if os.path.exists(overlaps_path):
        with h5py.File(overlaps_path, "r") as f:
            overlaps = dict([(key, np.array(value)) for key, value in f.items()])
        if force_hermiticity == True:
            overlaps = Hermitian(overlaps)
        matrices["overlaps"] = overlaps

    hamiltonians_pred_path = os.path.join(data_dir, "hamiltonians_pred.h5")
    if os.path.exists(hamiltonians_pred_path):
        with h5py.File(hamiltonians_pred_path, "r") as f:
            hamiltonians_pred = dict(
                [(key, np.array(value)) for key, value in f.items()]
            )
        if force_hermiticity == True:
            hamiltonians_pred = Hermitian(hamiltonians_pred)
        matrices["hamiltonians_pred"] = hamiltonians_pred

    return matrices


def matrices_conversion(
    matrices: dict, R_list: np.ndarray, spinful: bool = False
) -> dict:

    natoms = np.array([eval(i) for i in list(matrices["overlaps"].keys())])[:, 3].max()
    # norbs = list(matrices['overlaps'].values())[0].shape[0]

    rs_matrices = {}
    for name, matrix in matrices.items():

        norbs_list = []
        for i in range(natoms):
            norbs_list.append(matrix[str([0, 0, 0, i + 1, i + 1])].shape[0])
        norbs_cumsum_list = np.concatenate([[0], np.cumsum(norbs_list)], axis=-1)

        if name == "overlaps" and spinful:
            norbs_list = [i * 2 for i in norbs_list]
            norbs_cumsum_list = [i * 2 for i in norbs_cumsum_list]

        rs_matrix = {}
        for R in R_list:
            rs_matrix[tuple(R)] = np.zeros(
                (sum(norbs_list), sum(norbs_list)),
                dtype=list(matrix.values())[-1].dtype,
            )
        for idx, block in matrix.items():
            idx = eval(idx)
            images = tuple(idx[:3])
            i_idx = idx[3]
            j_idx = idx[4]
            # rs_matrix[images][(i_idx-1)*norbs: i_idx*norbs, (j_idx-1)*norbs: j_idx*norbs] = block

            if name == "overlaps" and spinful:
                rs_matrix[images][
                    norbs_cumsum_list[(i_idx - 1)] : norbs_cumsum_list[i_idx],
                    norbs_cumsum_list[(j_idx - 1)] : norbs_cumsum_list[j_idx],
                ] = np.block(
                    [[block, np.zeros_like(block)], [np.zeros_like(block), block]]
                )
            else:
                rs_matrix[images][
                    norbs_cumsum_list[(i_idx - 1)] : norbs_cumsum_list[i_idx],
                    norbs_cumsum_list[(j_idx - 1)] : norbs_cumsum_list[j_idx],
                ] = block

            # rs_matrix[images][norbs_cumsum_list[(i_idx-1)]: norbs_cumsum_list[i_idx], norbs_cumsum_list[(j_idx-1)]: norbs_cumsum_list[j_idx]] = block

        rs_matrix = np.stack([rs_matrix[tuple(R)] for R in R_list], axis=0).transpose()
        rs_matrices[name] = rs_matrix

    return rs_matrices


def get_atoms_rs_matrices_R_KPath(
    data_dir: str, crystal_type: str = "2D", spinful: bool = False
) -> tuple:
    box = get_atoms(data_dir)
    R_list = get_R(data_dir)
    KPath = get_kpath(box, crystal_type)
    matrices = get_matrices(data_dir)
    rs_matrices = matrices_conversion(matrices, R_list, spinful)
    R_list = R_list.transpose()
    return box, rs_matrices, R_list, KPath


# from scipy.linalg import eigh
# H_real.sum(axis=2).shape
# vals, vecs = eigh(H_real.sum(axis=2), S_real.sum(axis=2))

# hr=np.sum(list(rs_matrices['hamiltonians'].values()), axis=0)
# sr=np.sum(list(rs_matrices['overlaps'].values()), axis=0)
# Main.hr=hr
# Main.sr=sr
# Main.eval("""
# valsj, vecsj = eigen(hr, sr)
# """)


def get_kpoints(
    atoms: Atoms, kpr: float = 0.012, kpg: str = None, spinful: bool = False
) -> tuple:
    # kpoints_density is in the unit of 2*PI /Angstrom as in https://vaspkit.com/tutorials.html#generate-kpoints

    if kpg is None:
        reciprocal_cell = 2 * np.pi * atoms.cell.reciprocal()
        recip_lengths = np.linalg.norm(reciprocal_cell, axis=1)
        kpts = [
            max(1, int(np.ceil(length / (kpr * 2 * np.pi)))) for length in recip_lengths
        ]
    else:
        kpts = [int(kp) for kp in kpg.split()]

    # # Determine the space group and get symmetry operations
    # cell = (atoms.cell, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    # dataset = spglib.get_symmetry_dataset(cell)
    # rotations = dataset['rotations']
    # translations = dataset['translations']
    # kpoints = monkhorst_pack(kpts)

    cell = (atoms.cell, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    mapping, grid = spglib.get_ir_reciprocal_mesh(
        kpts, cell, is_time_reversal=(not spinful)
    )
    maps, counts = np.unique(mapping, return_counts=True)
    kpoints = grid[maps] / np.array(kpts)
    weights = counts / counts.sum()

    return [i for i in kpoints], weights.tolist()


def get_occupations(
    chemical_potential: float,
    all_energies: np.ndarray,
    temperature: float,
    spinful: bool = False,
    smearing_function: str = "fermi_dirac",
):
    """
    Occupations of energies at a given temperature and chemical potential.
    """
    if smearing_function.lower() == "fermi_dirac":
        smearing_function = fermi_dirac
    elif smearing_function.lower() == "gaussian":
        smearing_function = gaussian_smearing
    degeneracy = 1 if spinful else 2
    return degeneracy * smearing_function(
        E=all_energies, mu=chemical_potential, T=temperature
    )


def get_chemical_potential(
    total_electrons: int,
    all_energies: np.ndarray,
    all_weights: np.ndarray,
    temperature: float,
    spinful: bool = False,
    smearing_function: str = "fermi_dirac",
):
    """
    Get chemical potential by optimisation based on the target number of electrons.
    """

    def _chemical_potential_loss_function(
        total_electrons: int,
        all_energies: np.ndarray,
        all_weights: np.ndarray,
        temperature: float,
        spinful: bool = False,
        smearing_function: str = "fermi_dirac",
    ):
        """
        Construct loss function used in optimising chemical potential.
        """

        def loss_function(chemical_potential: float):
            occupations = get_occupations(
                chemical_potential=chemical_potential,
                all_energies=all_energies,
                temperature=temperature,
                spinful=spinful,
                smearing_function=smearing_function,
            )
            return np.abs(total_electrons - np.sum(all_weights * occupations))

        return loss_function

    loss_function = _chemical_potential_loss_function(
        total_electrons=total_electrons,
        all_energies=all_energies,
        all_weights=all_weights,
        temperature=temperature,
        spinful=spinful,
        smearing_function=smearing_function,
    )
    res = minimize_scalar(loss_function)
    return res.x


def calculate_fermi_level(
    eigenvalues_kpoints: list,
    kpoint_weights: list,
    total_electrons: int,
    spinful: bool = False,
    temperature: float = 0.0,
    smearing_function: str = "fermi_dirac",
) -> float:
    """
    Calculate the Fermi level in a semiconductor.

    Parameters are the same as before.

    Returns:
    - fermi_level: The calculated Fermi level in eV.
    """

    # Flatten the eigenvalues and associate K-point weights
    all_energies = []
    all_weights = []

    for eigenvalues, weight in zip(eigenvalues_kpoints, kpoint_weights):
        for energy in eigenvalues.real:
            all_energies.append(energy)
            all_weights.append(weight)

    # Convert to numpy arrays
    all_energies = np.array(all_energies)
    all_weights = np.array(all_weights)

    degeneracy = 1 if spinful else 2

    if temperature == 0.0:
        # Sort energies and corresponding weights
        sorted_indices = np.argsort(all_energies)
        sorted_energies = all_energies[sorted_indices]
        sorted_weights = all_weights[sorted_indices]

        # Calculate cumulative electron count
        cumulative_electrons = np.cumsum(degeneracy * sorted_weights)

        # Find the index where cumulative electrons exceed total electrons
        idx = np.searchsorted(cumulative_electrons, total_electrons, side="left")

        # Determine the valence band maximum and conduction band minimum
        E_V = sorted_energies[idx] if idx > 0 else sorted_energies[0]
        E_C = (
            sorted_energies[idx + 1]
            if idx < len(sorted_energies)
            else sorted_energies[-1]
        )

        # Place the Fermi level in the middle of the band gap
        fermi_level = (E_V + E_C) / 2

    else:
        # This optimisation-based approach might not place the chemical potential in the middle
        # of the band gap for semiconductors at low temperature
        fermi_level = get_chemical_potential(
            total_electrons=total_electrons,
            all_energies=all_energies,
            all_weights=all_weights,
            temperature=temperature,
            spinful=spinful,
            smearing_function=smearing_function,
        )

    return fermi_level


def get_fermi_level(
    atoms: Atoms,
    H: np.ndarray,
    S: np.ndarray,
    R_list: list,
    total_electrons: int = None,
    k_point_spacing: float = 0.012,
    k_point_grid: str = None,
    spinful: bool = False,
    temperature: float = 0.0,
    smearing_function: str = "fermi_dirac",
) -> float:
    kpoints, weights = get_kpoints(atoms, k_point_spacing, k_point_grid, spinful)
    _, energies = Main.diagnalize_HS(H, S, R_list, kpoints)
    fermi_level = calculate_fermi_level(
        eigenvalues_kpoints=energies,
        kpoint_weights=weights,
        total_electrons=total_electrons,
        spinful=spinful,
        temperature=temperature,
        smearing_function=smearing_function,
    )
    return fermi_level, energies


def compute_eigenvalue_error(
    energies_true: np.ndarray,
    energies_pred: np.ndarray,
    fermi_level_true: float,
    fermi_level_pred: float,
    atoms: Atoms,
    k_point_spacing: float = 0.012,
    k_point_grid: str = None,
    spinful: bool = False,
    temperature: float = 0.0,
    smearing_function: str = "fermi_dirac",
):

    # Turn eigenvalue lists into arrays and take the real value
    eigenvalues_true = np.real(np.stack(energies_true))
    eigenvalues_pred = np.real(np.stack(energies_pred))

    # Get the same k-point grid which was used for computing DOS in order to get the weights
    kpoints, weights = get_kpoints(atoms, k_point_spacing, k_point_grid, spinful)
    weights = np.array(weights)
    assert len(kpoints) == len(energies_true)
    assert len(kpoints) == len(energies_pred)

    # Compute occupation numbers for true eigenvalues
    occ_array = get_occupations(
        chemical_potential=fermi_level_true,
        all_energies=eigenvalues_true,
        temperature=temperature,
        spinful=spinful,
        smearing_function=smearing_function,
    )
    weighted_occ_array = np.einsum("k,ki->ki", weights, occ_array)

    # Compute normalisation constant (number of electrons in the unit cell)
    Nel_cell = np.sum(weighted_occ_array)

    # Compute MAE of shifted eigenvalues
    eigenvalue_diff = np.abs(
        (eigenvalues_true - fermi_level_true) - (eigenvalues_pred - fermi_level_pred)
    )

    # Compute normalised smeared eigenvalue error
    eigenvalue_error = np.sum(eigenvalue_diff * weighted_occ_array) / Nel_cell

    return eigenvalue_error


def get_electronic_entropy(
    energies: np.ndarray,
    fermi_level: float,
    atoms: Atoms,
    k_point_spacing: float = 0.012,
    k_point_grid: str = None,
    spinful: bool = False,
    temperature: float = 0.0,
    smearing_function: str = "fermi_dirac",
):
    """
    Get electronic entropy at a given temperature from computed eigenvalues on a k-point grid
    """
    if temperature == 0:
        return 0.0

    if smearing_function.lower() != "fermi_dirac":
        warnings.warn(
            "Warning: using unphysical smearing function for electronic entropy.",
            UserWarning,
        )

    # Turn eigenvalue lists into arrays and take the real value
    eigenvalues = np.real(np.stack(energies))

    # Get the same k-point grid which was used for computing DOS in order to get the weights
    kpoints, weights = get_kpoints(atoms, k_point_spacing, k_point_grid, spinful)
    assert len(kpoints) == len(energies)

    # Compute fermi dirac function and fermi dirac integral
    degeneracy = 1 if spinful else 2
    fd = degeneracy * fermi_dirac(eigenvalues, fermi_level, temperature)
    fdi = degeneracy * fermi_dirac_integral(eigenvalues, fermi_level, temperature)

    # Sum over states for each kpoint
    contribution_kpoint = np.sum((eigenvalues - fermi_level) * fd - fdi, axis=1)

    return np.sum(np.array(weights) * contribution_kpoint) / (
        temperature * atoms.cell.volume
    )


def compute_electronic_entropy_error(
    energies_true: np.ndarray,
    energies_pred: np.ndarray,
    fermi_level_true: float,
    fermi_level_pred: float,
    atoms: Atoms,
    k_point_spacing: float = 0.012,
    k_point_grid: str = None,
    spinful: bool = False,
    temperature: float = 0.0,
    smearing_function: str = "fermi_dirac",
):

    el_entropy_true = get_electronic_entropy(
        energies=energies_true,
        fermi_level=fermi_level_true,
        atoms=atoms,
        k_point_spacing=k_point_spacing,
        k_point_grid=k_point_grid,
        spinful=spinful,
        temperature=temperature,
        smearing_function=smearing_function,
    )
    el_entropy_pred = get_electronic_entropy(
        energies=energies_pred,
        fermi_level=fermi_level_pred,
        atoms=atoms,
        k_point_spacing=k_point_spacing,
        k_point_grid=k_point_grid,
        spinful=spinful,
        temperature=temperature,
        smearing_function=smearing_function,
    )

    return np.abs(el_entropy_true - el_entropy_pred)


def get_total_electrons(atoms: Atoms, is_all_ele: bool = False):
    elems, counts = np.unique(atoms.get_chemical_symbols(), return_counts=True)
    # The number of valence electrons in pseudopotential methods is ambiguous and problematic, be very careful!
    nelep = np.array(
        [
            element(ele).protons if is_all_ele else element(ele).nvalence()
            for ele in elems
        ]
    )
    total_electrons = (nelep * counts).sum()
    return total_electrons


def get_nele_kpath(atoms: Atoms, SI_path: str):
    symbols = atoms.get_chemical_symbols()
    with open(SI_path, "r") as f:
        SI = json.load(f)
    nele = sum([SI["occupied_valence_electrons"][symbol] for symbol in symbols])

    x = []
    labels = []
    kpoints_rel = []
    for path in SI["k_data"]:
        kpr = np.stack(
            [
                np.linspace(path[1], path[4], path[0]),
                np.linspace(path[2], path[5], path[0]),
                np.linspace(path[3], path[5], path[0]),
            ],
            axis=-1,
        )
        label = [path[-2]] + [""] * (kpr.shape[0] - 2) + [path[-1]]
        xr = np.concatenate(
            [[0.0], np.linalg.norm(kpr[1:] - kpr[:-1], ord=2, axis=-1).cumsum()], axis=0
        )
        if len(labels) != 0:
            if labels[-1] == label[0]:
                x.append(x[-1][-1] + xr[1:])
                labels.extend(label[1:])
                kpoints_rel.append(kpr[1:])
            else:
                x.append(x[-1][-1] + xr)
                labels.extend(label)
                kpoints_rel.append(kpr)
        else:
            x.append(xr)
            labels.extend(label)
            kpoints_rel.append(kpr)

    x = np.concatenate(x, axis=-1)
    kpoints_rel = np.concatenate(kpoints_rel, axis=0)

    return nele, {"x": x, "labels": labels, "kpoints_rel": kpoints_rel}


def get_xticks_labels(KPath: dict):
    xticks = []
    xtick_labels = []
    for xtick, xtick_label in zip(KPath["x"], KPath["labels"]):
        if xtick_label != "":
            if xtick_label.upper() == "GAMMA":
                xtick_label = r"$\Gamma$"
            xticks.append(xtick)
            xtick_labels.append(xtick_label)
    indices = np.where(np.array(xticks[:-1]) == np.array(xticks[1:]))[0]
    for z, idx in enumerate(indices):
        label = "|".join([xtick_labels[idx - z], xtick_labels[idx + 1 - z]])
        xtick_labels[idx - z] = label
        del (xticks[idx + 1 - z], xtick_labels[idx + 1 - z])
    return xticks, xtick_labels


def plot_band(
    KPath: dict,
    bands: list,
    fermi_level: float,
    dir: str,
    ground_truth: bool = True,
    d_min: float = -3,
    d_max: float = 3,
) -> None:
    # fermi_level = -3.825442624593951
    colors = sns.color_palette("bright", 10)
    plt.figure(figsize=(8, 6))
    bands = np.stack(bands, axis=0)
    for band in bands.transpose():
        plt.plot(KPath["x"], band.real, linewidth=2, color=colors[0])

    # xticks = []
    # xtick_labels = []
    # for xtick, xtick_label in zip(KPath['x'], KPath['labels']):
    #     if xtick_label != '':
    #         if xtick_label == 'GAMMA':
    #             xtick_label = r'$\Gamma$'
    #         xticks.append(xtick)
    #         xtick_labels.append(xtick_label)

    xticks, xtick_labels = get_xticks_labels(KPath)

    plt.xlim([min(xticks), max(xticks)])
    plt.xticks(xticks, labels=xtick_labels, fontsize=20)
    ax = plt.gca()
    for xtick in xticks:
        ax.axvline(x=xtick, color="grey", linestyle="--", linewidth=1.5, zorder=1)
    ax.axhline(y=fermi_level, color="grey", linestyle="--", linewidth=1.5, zorder=1)
    plt.ylim([fermi_level + d_min, fermi_level + d_max])
    plt.yticks(
        np.arange(np.ceil(plt.ylim()[0]), plt.ylim()[1] + 0.001, 1.0), fontsize=20
    )
    plt.xlabel("K-path", fontsize=20)
    plt.ylabel("Energy (eV)", fontsize=20)
    ax.tick_params(direction="in", length=8, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    path = (
        os.path.join(dir, "band_true.png")
        if ground_truth
        else os.path.join(dir, "band_pred.png")
    )
    save_figure(plt.gcf(), path, format="png", dpi=800)
    plt.show()
    return


def plot_band_compare(
    KPath: dict,
    bands_true: list,
    bands_pred: list,
    fermi_level: float,
    dir: str,
    d_min: float = -3,
    d_max: float = 3,
) -> None:
    # fermi_level = -3.825442624593951
    colors = sns.color_palette("bright", 10)
    plt.figure(figsize=(8, 6))
    bands_true = np.stack(bands_true, axis=0)
    for band in bands_true.transpose():
        plt.plot(KPath["x"], band.real, linewidth=2, color=colors[0])
    bands_pred = np.stack(bands_pred, axis=0)
    for band in bands_pred.transpose():
        plt.plot(
            KPath["x"], band.real, linewidth=2, linestyle="dashed", color=colors[1]
        )

    # xticks = []
    # xtick_labels = []
    # for xtick, xtick_label in zip(KPath['x'], KPath['labels']):
    #     if xtick_label != '':
    #         if xtick_label == 'GAMMA':
    #             xtick_label = r'$\Gamma$'
    #         xticks.append(xtick)
    #         xtick_labels.append(xtick_label)

    xticks, xtick_labels = get_xticks_labels(KPath)

    plt.xlim([min(xticks), max(xticks)])
    plt.xticks(xticks, labels=xtick_labels, fontsize=20)
    ax = plt.gca()
    for xtick in xticks:
        ax.axvline(x=xtick, color="grey", linestyle="--", linewidth=1.5, zorder=1)
    ax.axhline(y=fermi_level, color="grey", linestyle="--", linewidth=1.5, zorder=1)
    plt.ylim([fermi_level + d_min, fermi_level + d_max])
    plt.yticks(
        np.arange(np.ceil(plt.ylim()[0]), plt.ylim()[1] + 0.001, 1.0), fontsize=20
    )
    plt.xlabel("K-path", fontsize=20)
    plt.ylabel("Energy (eV)", fontsize=20)
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=2, label="DFT"),
        Line2D([0], [0], color=colors[1], lw=2, linestyle="dashed", label="Predicted"),
    ]
    plt.legend(handles=legend_elements, fontsize=15)
    ax.tick_params(direction="in", length=8, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    path = os.path.join(dir, "band_compare.png")
    save_figure(plt.gcf(), path, format="png", dpi=800)
    plt.show()
    return


def get_DOS(
    atoms: Atoms,
    H: np.ndarray,
    S: np.ndarray,
    R_list: list,
    fermi_level: float,
    d_min: float = -3,
    d_max: float = 3,
    k_point_spacing: float = 0.012,
    k_point_grid: str = None,
    spinful: bool = False,
    energies: list = None,
) -> float:

    kpoints, weights = get_kpoints(atoms, k_point_spacing, k_point_grid, spinful)
    if energies is None or len(energies) == len(kpoints):
        _, energies = Main.diagnalize_HS(H, S, R_list, kpoints)

    num_bins = 500
    energy_bins = np.linspace(fermi_level + d_min, fermi_level + d_max, num_bins)
    bin_width = energy_bins[1] - energy_bins[0]

    sigma = 0.01  # Broadening width
    energy_centers = (energy_bins[:-1] + energy_bins[1:]) / 2
    dos_smoothed = np.zeros_like(energy_centers)
    for eigenvalues, weight in zip(energies, weights):
        for E_i in eigenvalues.real:
            dos_smoothed += (
                np.exp(-((energy_centers - E_i) ** 2) / (2 * sigma**2))
                / (np.sqrt(2 * np.pi) * sigma)
                * weight
            )
            ## weight and bin_width are used for normalization

    degeneracy = 1 if spinful else 2
    dos_smoothed *= degeneracy

    return energy_centers, dos_smoothed


def plot_DOS(
    energy_centers: np.ndarray,
    dos_smoothed: np.ndarray,
    fermi_level: float,
    dir: str,
    ground_truth: bool = True,
) -> None:

    colors = sns.color_palette("bright", 10)
    plt.figure(figsize=(8, 6))
    plt.plot(energy_centers, dos_smoothed, linewidth=2, color=colors[0])
    plt.xlim([energy_centers.min(), energy_centers.max()])
    plt.xticks(
        np.arange(np.ceil(plt.xlim()[0]), plt.xlim()[1] + 0.001, 1.0), fontsize=20
    )
    ax = plt.gca()
    ax.axvline(x=fermi_level, color="grey", linestyle="--", linewidth=1.5, zorder=1)
    plt.xlabel("Energy (eV)", fontsize=20)
    plt.ylabel(r"DOS (eV$^{-1}$)", fontsize=20)
    plt.ylim([0, dos_smoothed.max() * 1.2])
    ax.tick_params(direction="in", length=8, width=1)
    plt.yticks(fontsize=20)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    path = (
        os.path.join(dir, "DOS_true.png")
        if ground_truth
        else os.path.join(dir, "DOS_pred.png")
    )
    save_figure(plt.gcf(), path, format="png", dpi=800)
    plt.show()
    return


def plot_DOS_compare(
    energy_centers: np.ndarray,
    dos_smoothed_true: np.ndarray,
    dos_smoothed_pred: np.ndarray,
    fermi_level: float,
    dir: str,
) -> None:

    colors = sns.color_palette("bright", 10)
    plt.figure(figsize=(8, 6))
    plt.plot(
        energy_centers, dos_smoothed_true, linewidth=2, color=colors[0], label="DFT"
    )
    plt.plot(
        energy_centers,
        dos_smoothed_pred,
        linewidth=2,
        linestyle="dashed",
        color=colors[1],
        label="Predicted",
    )
    plt.xlim([energy_centers.min(), energy_centers.max()])
    plt.xticks(
        np.arange(np.ceil(plt.xlim()[0]), plt.xlim()[1] + 0.001, 1.0), fontsize=20
    )
    ax = plt.gca()
    ax.axvline(x=fermi_level, color="grey", linestyle="--", linewidth=1.5, zorder=1)
    plt.xlabel("Energy (eV)", fontsize=20)
    plt.ylabel(r"DOS (eV$^{-1}$)", fontsize=20)
    plt.ylim([0, dos_smoothed_true.max() * 1.2])
    ax.tick_params(direction="in", length=8, width=1)
    plt.yticks(fontsize=20)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    legend_properties = {"family": "Times New Roman", "size": 20}
    plt.legend(prop=legend_properties)
    path = os.path.join(dir, "DOS_compare.png")
    save_figure(plt.gcf(), path, format="png", dpi=800)
    plt.show()
    return


def get_args(args: list = None):
    """Read in the command line parameters"""
    parser = argparse.ArgumentParser("Arguements for processing bands")
    parser.add_argument(
        "-np",
        "--n_processes",
        dest="n_processes",
        default=16,
        type=int,
        help="Number of processes used in diagonalisation (parallelisation over k-points).",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        dest="data_dir",
        default=os.getcwd(),
        type=str,
        help="The directory for *.h5, *.dat and info.json.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        default=os.getcwd(),
        type=str,
        help="The output directory.",
    )
    parser.add_argument(
        "-c",
        "--crystal_type",
        dest="crystal_type",
        default="2D",
        type=str,
        choices=["2D", "3D"],
        help="Crystal type, 2D or 3D.",
    )
    parser.add_argument(
        "-all",
        "--is_all_ele",
        dest="is_all_ele",
        default="False",
        type=str,
        choices=["True", "False"],
        help="If the calculation use an all electron method.",
    )
    parser.add_argument(
        "-dmin",
        "--d_min",
        dest="d_min",
        default=-3.0,
        type=float,
        help="The ymin with respect to fermi level for plotting, negative value means lower than fermi level.",
    )
    parser.add_argument(
        "-dmax",
        "--d_max",
        dest="d_max",
        default=3.0,
        type=float,
        help="The ymax with respect to fermi level for plotting.",
    )
    parser.add_argument(
        "-kpr",
        "--k_point_spacing",
        dest="k_point_spacing",
        default=0.012,
        type=float,
        help=(
            "k-point spacing in units (2 pi Å)^{-1} for fermi level and DOS calculations. "
            "Ignored if k_point_grid is supplied."
        ),
    )
    parser.add_argument(
        "-kpg",
        "--k_point_grid",
        dest="k_point_grid",
        default=None,
        type=str,
        help='Number of k-points along each reciprocal lattice vector in the order of xyz, e.g. "5 5 1".',
    )
    parser.add_argument(
        "-T",
        "--temperature",
        dest="temperature",
        default=0.0,
        type=float,
        help="Temperature for computing the chemical potential.",
    )
    parser.add_argument(
        "-smear",
        "--smearing_function",
        dest="smearing_function",
        default="fermi_dirac",
        type=str,
        choices=["fermi_dirac", "gaussian"],
        help="Smearing function used for computing the chemical potential if temperature is not zero.",
    )
    parser.add_argument(
        "-seigv",
        "--save_eigenvalues",
        dest="save_eigenvalues",
        default=True,
        type=bool,
        help="Whether to save eigenvalues computed on a k-point grid. True by default.",
    )
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return {
        "n_processes": args.n_processes,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "crystal_type": args.crystal_type,
        "is_all_ele": eval(args.is_all_ele),
        "d_min": args.d_min,
        "d_max": args.d_max,
        "k_point_spacing": args.k_point_spacing,
        "k_point_grid": args.k_point_grid,
        "temperature": args.temperature,
        "smearing_function": args.smearing_function,
        "save_eigenvalues": args.save_eigenvalues,
    }


def main(args: list = None) -> None:
    args = get_args(args)
    n_processes = args["n_processes"]
    data_dir = args["data_dir"]
    output_dir = args["output_dir"]
    crystal_type = args["crystal_type"]
    is_all_ele = args["is_all_ele"]
    d_min = args["d_min"]
    d_max = args["d_max"]
    k_point_spacing = args["k_point_spacing"]
    k_point_grid = args["k_point_grid"]
    temperature = args["temperature"]
    smearing_function = args["smearing_function"]
    save_eigenvalues = args["save_eigenvalues"]

    Main.eval(f"addprocs({n_processes})")
    Main.include(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Band.jl"))

    with open(os.path.join(data_dir, "info.json"), "r") as f:
        info = json.load(f)
    spinful = info["isspinful"]
    atoms, rs_matrices, R_list, KPath = get_atoms_rs_matrices_R_KPath(
        data_dir, crystal_type, spinful
    )

    total_electrons = get_total_electrons(atoms, is_all_ele)

    if os.path.exists(os.path.join(data_dir, "SI.json")):
        total_electrons, KPath = get_nele_kpath(
            atoms, os.path.join(data_dir, "SI.json")
        )

    S_real = rs_matrices["overlaps"]
    KPoints = [i for i in KPath["kpoints_rel"]]

    if "hamiltonians" in rs_matrices.keys():
        H_real = rs_matrices["hamiltonians"]
        k_points, bands_true = Main.diagnalize_HS(H_real, S_real, R_list, KPoints)

        # if 'fermi_level' in info.keys():
        #     fermi_level_true = info['fermi_level']
        #     energy_centers_true, dos_smoothed_true = get_DOS(atoms, H_real, S_real, R_list, fermi_level_true, d_min, d_max,
        #                                                      k_point_spacing, k_point_grid, spinful, None)
        # else:
        #     fermi_level_true, energies_true = get_fermi_level(atoms, H_real, S_real, R_list, total_electrons, k_point_spacing, k_point_grid,
        #                                                       spinful, temperature, smearing_function)
        #     energy_centers_true, dos_smoothed_true = get_DOS(atoms, H_real, S_real, R_list, fermi_level_true, d_min, d_max,
        #                                                      k_point_spacing, k_point_grid, spinful, energies_true)

        fermi_level_true, energies_true = get_fermi_level(
            atoms=atoms,
            H=H_real,
            S=S_real,
            R_list=R_list,
            total_electrons=total_electrons,
            k_point_spacing=k_point_spacing,
            k_point_grid=k_point_grid,
            spinful=spinful,
            temperature=temperature,
            smearing_function=smearing_function,
        )
        energy_centers_true, dos_smoothed_true = get_DOS(
            atoms=atoms,
            H=H_real,
            S=S_real,
            R_list=R_list,
            fermi_level=fermi_level_true,
            d_min=d_min,
            d_max=d_max,
            k_point_spacing=k_point_spacing,
            k_point_grid=k_point_grid,
            spinful=spinful,
            energies=energies_true,
        )

        plot_band(
            KPath=KPath,
            bands=bands_true,
            fermi_level=fermi_level_true,
            dir=output_dir,
            ground_truth=True,
            d_min=d_min,
            d_max=d_max,
        )
        plot_DOS(
            energy_centers=energy_centers_true,
            dos_smoothed=dos_smoothed_true,
            fermi_level=fermi_level_true,
            dir=output_dir,
            ground_truth=True,
        )
        np.savez(
            os.path.join(output_dir, "band_true.npz"),
            KPath=KPath,
            bands_true=bands_true,
            fermi_level_true=fermi_level_true,
        )
        np.savez(
            os.path.join(output_dir, "DOS_true.npz"),
            energy_centers_true=energy_centers_true,
            dos_smoothed_true=dos_smoothed_true,
        )
        if save_eigenvalues:
            k_points, weights = get_kpoints(
                atoms, k_point_spacing, k_point_grid, spinful
            )
            np.savez(
                os.path.join(output_dir, "eigenvals_true.npz"),
                k_points=k_points,
                weights=weights,
                energies_true=energies_true,
                fermi_level_true=fermi_level_true,
            )

    if "hamiltonians_pred" in rs_matrices.keys():
        H_pred = rs_matrices["hamiltonians_pred"]
        k_points, bands_pred = Main.diagnalize_HS(H_pred, S_real, R_list, KPoints)

        # if 'fermi_level' in info.keys():
        #     fermi_level_pred = info['fermi_level']
        #     energy_centers_pred, dos_smoothed_pred = get_DOS(atoms, H_pred, S_real, R_list, fermi_level_pred, d_min, d_max,
        #                                                      k_point_spacing, k_point_grid, spinful, None)
        # else:
        #     fermi_level_pred, energies_pred = get_fermi_level(atoms, H_pred, S_real, R_list, total_electrons, k_point_spacing, k_point_grid,
        #                                                       spinful, temperature, smearing_function)
        #     energy_centers_pred, dos_smoothed_pred = get_DOS(atoms, H_pred, S_real, R_list, fermi_level_pred, d_min, d_max,
        #                                                      k_point_spacing, k_point_grid, spinful, energies_pred)

        fermi_level_pred, energies_pred = get_fermi_level(
            atoms=atoms,
            H=H_pred,
            S=S_real,
            R_list=R_list,
            total_electrons=total_electrons,
            k_point_spacing=k_point_spacing,
            k_point_grid=k_point_grid,
            spinful=spinful,
            temperature=temperature,
            smearing_function=smearing_function,
        )
        energy_centers_pred, dos_smoothed_pred = get_DOS(
            atoms=atoms,
            H=H_pred,
            S=S_real,
            R_list=R_list,
            fermi_level=fermi_level_pred,
            d_min=d_min,
            d_max=d_max,
            k_point_spacing=k_point_spacing,
            k_point_grid=k_point_grid,
            spinful=spinful,
            energies=energies_pred,
        )

        plot_band(
            KPath=KPath,
            bands=bands_pred,
            fermi_level=fermi_level_pred,
            dir=output_dir,
            ground_truth=False,
            d_min=d_min,
            d_max=d_max,
        )
        plot_DOS(
            energy_centers=energy_centers_pred,
            dos_smoothed=dos_smoothed_pred,
            fermi_level=fermi_level_pred,
            dir=output_dir,
            ground_truth=False,
        )
        np.savez(
            os.path.join(output_dir, "band_pred.npz"),
            KPath=KPath,
            bands_pred=bands_pred,
            fermi_level_pred=fermi_level_pred,
        )
        np.savez(
            os.path.join(output_dir, "DOS_pred.npz"),
            energy_centers_pred=energy_centers_pred,
            dos_smoothed_pred=dos_smoothed_pred,
        )
        if save_eigenvalues:
            k_points, weights = get_kpoints(
                atoms, k_point_spacing, k_point_grid, spinful
            )
            np.savez(
                os.path.join(output_dir, "eigenvals_pred.npz"),
                k_points=k_points,
                weights=weights,
                energies_pred=energies_pred,
                fermi_level_pred=fermi_level_pred,
            )

    if (
        "hamiltonians" in rs_matrices.keys()
        and "hamiltonians_pred" in rs_matrices.keys()
    ):
        plot_band_compare(
            KPath=KPath,
            bands_true=bands_true,
            bands_pred=bands_pred,
            fermi_level=fermi_level_true,
            dir=output_dir,
            d_min=d_min,
            d_max=d_max,
        )
        plot_DOS_compare(
            energy_centers=energy_centers_true,
            dos_smoothed_true=dos_smoothed_true,
            dos_smoothed_pred=dos_smoothed_pred,
            fermi_level=fermi_level_true,
            dir=output_dir,
        )

        error_metrics = {}
        eigenvalue_error = compute_eigenvalue_error(
            energies_true=energies_true,
            energies_pred=energies_pred,
            fermi_level_true=fermi_level_true,
            fermi_level_pred=fermi_level_pred,
            atoms=atoms,
            k_point_spacing=k_point_spacing,
            k_point_grid=k_point_grid,
            spinful=spinful,
            temperature=temperature,
            smearing_function=smearing_function,
        )
        el_entropy_error = compute_electronic_entropy_error(
            energies_true=energies_true,
            energies_pred=energies_pred,
            fermi_level_true=fermi_level_true,
            fermi_level_pred=fermi_level_pred,
            atoms=atoms,
            k_point_spacing=k_point_spacing,
            k_point_grid=k_point_grid,
            spinful=spinful,
            temperature=temperature,
            smearing_function=smearing_function,
        )
        error_metrics["eigenvalue_error"] = {"value": eigenvalue_error, "units": "eV"}
        error_metrics["electronic_entropy_error"] = {
            "value": el_entropy_error,
            "units": r"eV K^{-1} Ang^{-3}",
        }

        with open(os.path.join(output_dir, "error_metrics.json"), "w") as f:
            json.dump(error_metrics, f)

    # Remove workers in case `main()` is called multiple times
    Main.eval(f"rmprocs(workers())")

    return


if __name__ == "__main__":
    main()