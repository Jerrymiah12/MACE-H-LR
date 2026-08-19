import torch
import os
from e3nn import o3
from e3nn.io import CartesianTensor
from torch_geometric.utils import scatter
from .macehmodules import MACE, NodeDegreeExpansion, EdgeUpdate
from typing import Any, Callable, Dict, List, Optional, Type, Tuple
from torch_geometric.data import Batch
from .e3modules import SeparateWeightTensorProduct, e3LayerNorm
from .statistics import shift_scale_out


# TODO: rmax 需要根据数据集求得?
# deleted keyword: 
class Net(torch.nn.Module):
    def __init__(self, num_species, irreps_edge_init, irreps_sh,
            irreps_post_node, irreps_mid_edge, irreps_post_edge, irreps_out_edge, 
            num_blocks, r_max, use_sc=True, edge_upd=True, num_basis=128, 
            act={1: torch.nn.functional.silu, -1: torch.tanh},
            act_gates={1: torch.sigmoid, -1: torch.tanh},
            num_bessel=8, num_polynomial_cutoff=5, max_ell=3, hidden_irreps='128x0e+128x1o',
            avg_num_neighbors=None, atomic_numbers=None, correlation=3, radial_MLP: Optional[List[int]] = None, mace_norm: str = 'e3LayerNorm',
            expand_nonlin: bool = True, expand_norm: str = 'e3LayerNorm', basis_func: str = 'Bessel', num_gaussian: int = 128,
            shift_scale: bool = True):
        
        super(Net, self).__init__()

        irreps_edge_init = o3.Irreps(irreps_edge_init)
        assert irreps_edge_init == o3.Irreps(f'{irreps_edge_init.dim}x0e')
        irreps_sh =o3.Irreps(irreps_sh)
        irreps_post_node=o3.Irreps(irreps_post_node)
        irreps_mid_edge=o3.Irreps(irreps_mid_edge)
        irreps_post_edge=o3.Irreps(irreps_post_edge)
        irreps_out_edge=o3.Irreps(irreps_out_edge)
        hidden_irreps=o3.Irreps(hidden_irreps)

        self.register_buffer("atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))

        self.num_species = num_species

        if max_ell is None:
            max_ell = irreps_sh.lmax

        self.mace = MACE(r_max=r_max, num_bessel=num_bessel, num_polynomial_cutoff=num_polynomial_cutoff, max_ell=max_ell, num_interactions=num_blocks,
                         num_species=num_species, hidden_irreps=hidden_irreps, avg_num_neighbors=avg_num_neighbors, correlation=correlation,
                         radial_MLP=radial_MLP, mace_norm=mace_norm, basis_func=basis_func, num_gaussian=num_gaussian)

        self.expansion = NodeDegreeExpansion(hidden_irreps=hidden_irreps, max_ell=max_ell, irreps_post_node=irreps_post_node,
                                            num_interactions=num_blocks, num_species=num_species, use_sc=use_sc, expand_nonlin=expand_nonlin,
                                            expand_norm=expand_norm)

        node_irreps_out_list = [block.irreps_out for block in self.expansion.node_order_expansion_blocks]

        self.edge_update = EdgeUpdate(num_species=num_species, irreps_edge_init=irreps_edge_init, irreps_sh=irreps_sh,
                                      node_irreps_out_list=node_irreps_out_list, irreps_mid_edge=irreps_mid_edge,
                                      irreps_post_edge=irreps_post_edge, irreps_out_edge=irreps_out_edge, r_max=r_max,
                                      use_sc=use_sc, num_basis=num_basis, edge_upd=edge_upd,
                                      act=act, act_gates=act_gates)
        
        self.shift_scale = shift_scale
        if shift_scale:
            self.register_buffer('mean_tensor', torch.zeros(num_species, num_species, irreps_out_edge.dim))
            self.register_buffer('std_tensor', torch.ones(num_species, num_species, irreps_out_edge.dim))
        
    def forward_features(self, data: Batch):
        """Return the final shared node representation and SR edge output."""

        edge_attr = data["edge_attr"][:, [0, 2, 3, 1]] # (y, z, x) order

        node_one_hot = torch.nn.functional.one_hot(data.x, num_classes=self.num_species).type(torch.get_default_dtype())
        
        node_feats_hidden_list, node_feats_up = self.mace(edge_attr, data.edge_index, node_one_hot, data.batch)

        post_node_feats_list = self.expansion(node_feats_hidden_list, node_feats_up, node_one_hot, data.batch)

        _, edge_fea = self.edge_update(post_node_feats_list, edge_attr, data.edge_index, data.x, data.batch)

        if self.shift_scale:
            edge_fea = shift_scale_out(edge_fea, self.mean_tensor, self.std_tensor, data.x, data.edge_index) #, self.edge_update.irreps_out_edge)

        return post_node_feats_list[-1], edge_fea

    def forward(self, data: Batch):
        # Keep the historical two-tuple interface and state-dict layout intact.
        _, edge_fea = self.forward_features(data)
        return None, edge_fea
    
    def __repr__(self):
        info = '===== MACEH model structure: ====='
        info += f'\nusing spherical harmonics in MACE: {self.mace.spherical_harmonics.irreps_out}'
        info += f'\nusing spherical harmonics in EdgeUpdate: {self.edge_update.irreps_sh}'
        for index, (nupd, eupd) in enumerate(zip(self.expansion.node_order_expansion_blocks, self.edge_update.edge_update_blocks)):
            info += f'\n=== layer {index} ==='
            info += f'\nnode update: ({self.mace.interactions[0].hidden_irreps} -> {nupd.irreps_out})'
            if eupd is not None:
                info += f'\nedge update: ({eupd.irreps_in_edge} -> {eupd.irreps_out})'
        info += '\n=== output ==='
        info += f'\noutput edge: ({self.edge_update.irreps_out_edge})'
        
        return info
    
    def analyze_tp(self, path):
        from .analysis.figures import save_figure  # debug-only; keeps matplotlib off the model import path
        os.makedirs(path, exist_ok=True)
        for index, (ninter, nexpand, eupd) in enumerate(zip(self.mace.interactions, self.expansion.node_order_expansion_blocks,
                                                    self.edge_update.edge_update_blocks)):
            fig, ax = ninter.conv_tp.visualize()
            save_figure(fig, os.path.join(path, f'node_update_{index}.png'))
            fig.clf()
            fig, ax = nexpand.tp.tp.visualize()
            save_figure(fig, os.path.join(path, f'node_expand_{index}.png'))
            fig.clf()
            fig, ax = eupd.conv.tp.tp.visualize()
            save_figure(fig, os.path.join(path, f'edge_update_{index}.png'))
            fig.clf()


class EquivariantTensorReadout(torch.nn.Module):
    """Equivariant linear/quadratic readout in irreducible coordinates."""

    def __init__(self, irreps_in, cartesian_formula):
        super().__init__()
        self.cartesian = CartesianTensor(cartesian_formula)
        self.linear = o3.Linear(irreps_in, self.cartesian, biases=True)
        self.quadratic = o3.FullyConnectedTensorProduct(
            irreps_in, irreps_in, self.cartesian,
            internal_weights=True, shared_weights=True)
        # New heads initially predict exactly the fixed residual baseline.
        for parameter in self.parameters():
            torch.nn.init.zeros_(parameter)
        rtp = self.cartesian.reduced_tensor_products(torch.zeros(1))
        self.register_buffer(
            "change_of_basis",
            rtp.change_of_basis.flatten(-len(self.cartesian.indices)))

    def forward(self, features):
        irreducible = self.linear(features) + self.quadratic(features, features)
        return (irreducible @ self.change_of_basis).reshape(
            *irreducible.shape[:-1], 3, 3)


class BornChargeHead(torch.nn.Module):
    """Per-atom full rank-two tensor (0e + 1e + 2e), projected to ASR."""

    def __init__(self, irreps_in, baseline):
        super().__init__()
        baseline = torch.as_tensor(baseline, dtype=torch.get_default_dtype())
        if baseline.ndim != 3 or baseline.shape[-2:] != (3, 3):
            raise ValueError("Born baseline must have shape (num_species, 3, 3)")
        isotropic = torch.eye(3, dtype=baseline.dtype).unsqueeze(0) * \
            (torch.diagonal(baseline, dim1=-2, dim2=-1).mean(-1)
             .reshape(-1, 1, 1))
        if not torch.allclose(baseline, isotropic, atol=1.0e-6, rtol=0.0):
            raise ValueError('Born baseline must be rotationally isotropic')
        self.register_buffer("baseline", baseline.clone())
        self.readout = EquivariantTensorReadout(irreps_in, "ij")

    def forward(self, features, species, batch):
        residual = self.readout(features)
        # The historical Hamiltonian model feeds MACE vectors in (y,z,x)
        # order. Convert the tensor readout back to public Cartesian (x,y,z).
        order = [2, 0, 1]
        residual = residual[..., order, :][..., :, order]
        born = self.baseline[species] + residual
        number_of_graphs = int(batch.max()) + 1 if batch.numel() else 0
        graph_mean = scatter(born, batch, dim=0, dim_size=number_of_graphs,
                             reduce="mean")
        return born - graph_mean[batch]


class DielectricHead(torch.nn.Module):
    """Permutation-invariant symmetric positive-definite 0e + 2e readout."""

    def __init__(self, irreps_in, baseline, eigenvalue_floor=1.0e-4,
                 floor_beta=10.0):
        super().__init__()
        baseline = torch.as_tensor(baseline, dtype=torch.get_default_dtype())
        if baseline.shape != (3, 3):
            raise ValueError("dielectric baseline must have shape (3, 3)")
        isotropic = torch.eye(3, dtype=baseline.dtype) * \
            torch.diagonal(baseline).mean()
        if not torch.allclose(baseline, isotropic, atol=1.0e-6, rtol=0.0):
            raise ValueError('dielectric baseline must be rotationally isotropic')
        self.register_buffer("baseline", baseline.clone())
        self.readout = EquivariantTensorReadout(irreps_in, "ij=ji")
        self.eigenvalue_floor = float(eigenvalue_floor)
        # Kept in the signature for checkpoint/source compatibility with the
        # earlier eigenvalue-floor prototype; the matrix-exponential SPD map
        # below is stable even for exactly degenerate cubic baselines.
        self.floor_beta = float(floor_beta)

    def forward(self, features, batch):
        number_of_graphs = int(batch.max()) + 1 if batch.numel() else 0
        atom_residual = self.readout(features)
        order = [2, 0, 1]
        atom_residual = atom_residual[..., order, :][..., :, order]
        residual = scatter(atom_residual, batch, dim=0,
                           dim_size=number_of_graphs, reduce="mean")
        residual = 0.5 * (residual + residual.transpose(-1, -2))
        baseline_scalar = torch.diagonal(self.baseline).mean()
        scale = baseline_scalar - self.eigenvalue_floor
        if scale.detach().item() <= 0.0:
            raise ValueError('dielectric baseline must exceed eigenvalue floor')
        identity = torch.eye(3, dtype=residual.dtype,
                             device=residual.device).unsqueeze(0)
        # floor*I + scale*exp(A/scale) is differentiable, rotationally
        # consistent, strictly positive definite, and equals the fixed cubic
        # baseline when the residual head is zero.
        epsilon = (self.eigenvalue_floor * identity
                   + scale * torch.matrix_exp(residual / scale))
        return 0.5 * (epsilon + epsilon.transpose(-1, -2))


class SRBornEpsilonModel(torch.nn.Module):
    """One shared encoder with SR-H, Born-charge, and dielectric heads.

    This wrapper is deliberately separate from :class:`Net`, keeping all
    Hamiltonian-only checkpoints and call sites backward compatible.
    """

    model_type = "sr_born_epsilon"
    direct_full_h_head = False
    analytic_lr_reconstruction = True

    def __init__(self, sr_model, born_baseline, epsilon_baseline,
                 eigenvalue_floor=1.0e-4):
        super().__init__()
        self.sr_model = sr_model
        irreps_node = sr_model.expansion.node_order_expansion_blocks[-1].irreps_out
        self.born_head = BornChargeHead(irreps_node, born_baseline)
        self.epsilon_head = DielectricHead(
            irreps_node, epsilon_baseline, eigenvalue_floor=eigenvalue_floor)

    @property
    def atomic_numbers(self):
        return self.sr_model.atomic_numbers

    def forward(self, data: Batch):
        features, hamiltonian = self.sr_model.forward_features(data)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(len(data.x), dtype=torch.long,
                                device=data.x.device)
        return {
            "node_features": features,
            "hamiltonian": hamiltonian,
            "born": self.born_head(features, data.x, batch),
            "epsilon": self.epsilon_head(features, batch),
        }


def load_sr_born_epsilon_state(model, state_dict):
    """Load either a multitask state or an old SR-only state, fail closed."""
    if (getattr(model, "model_type", None) != "sr_born_epsilon"
            or not hasattr(model, "sr_model")):
        return model.load_state_dict(state_dict)
    if any(key.startswith("sr_model.") for key in state_dict):
        return model.load_state_dict(state_dict)
    remapped = {f"sr_model.{key}": value for key, value in state_dict.items()}
    incompatible = model.load_state_dict(remapped, strict=False)
    allowed_missing = tuple(("born_head.", "epsilon_head."))
    illegal_missing = [key for key in incompatible.missing_keys
                       if not key.startswith(allowed_missing)]
    if illegal_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "SR checkpoint is incompatible; only the two new tensor heads may "
            f"be absent. missing={illegal_missing}, "
            f"unexpected={incompatible.unexpected_keys}")
    return incompatible
