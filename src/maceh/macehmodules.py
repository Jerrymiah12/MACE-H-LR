import torch 
from torch.nn import functional as F
from e3nn import o3, nn
from abc import abstractmethod
from e3nn.util.jit import compile_mode
from typing import Callable, List, Optional, Tuple, Union
from ._vendor.mfn.blocks import (RealAgnosticResidualInteractionBlockUp, RealAgnosticResidualInteractionBlock,
                              RealAgnosticResidualInteractionBlockUpNorm, RealAgnosticResidualInteractionBlockNorm,
                              LinearNodeEmbeddingBlock, RadialEmbeddingBlock, EquivariantProductBasisBlock)
from ._vendor.mfn.scatter import scatter_sum
from ._vendor.mfn.irreps_tools import tp_out_irreps_with_instructions
from ._vendor.nequip.tp_utils import tp_path_exists
import warnings
from .e3modules import SphericalBasis, sort_irreps, e3LayerNorm, e3ElementWise, SkipConnection, SeparateWeightTensorProduct, SelfTp
from .model import get_gate_nonlin, EquiConv
from ._vendor.schnetpack.acsf import GaussianBasis
from e3nn.o3 import Irreps, Linear, FullyConnectedTensorProduct
from typing import Any, Callable, Dict, List, Optional, Type, Tuple
from torch_geometric.data import Batch
from .e3modules import SeparateWeightTensorProduct, e3LayerNorm


# # target_irreps: the o3.Irreps cooresponding to the Hamiltonian block in the direct 
# # sum space



# @compile_mode("script")
# class EdgeUpdateBlock(torch.nn.Module):
#     def __init__(
#         self,
#         node_attrs_irreps: o3.Irreps,
#         node_feats_irreps: o3.Irreps,
#         edge_feats_mid_irreps: o3.Irreps,
#         edge_feats_post_irreps: o3.Irreps,
#         sh_irreps: o3.irreps = None,
#         edge_attrs_irreps: o3.Irreps = None,
#         radial_MLP: Optional[List[int]] = None,
#     ) -> None:
#         """
#         node_attrs_irreps: one-hot encoding of atom species, e.g., o3.Irreps([(num_elements, (0, 1))])
#         node_feats_irreps: atom-wise feature, e.g., (sh_irreps * num_features).sort()[0].simplify()
#         edge_feats_mid_irreps: edge-wise middle irrep in the model
#         edge_feats_post_irreps: edge-wise post irrep in the model
#         sh_irreps: spherical harmonics, o3.Irreps.spherical_harmonics(max_ell), None for diag pairs
#         edge_attrs_irreps: radial embedding of edge length, e.g., o3.Irreps(f"{self.radial_embedding.out_dim}x0e"), None for diag Pairs
#         radial_MLP: defining mlp for the edge length, None for diag pairs
#         """
#         super().__init__()
#         self.node_attrs_irreps = node_attrs_irreps
#         self.node_feats_irreps = node_feats_irreps
#         self.edge_feats_mid_irreps = edge_feats_mid_irreps
#         self.edge_feats_post_irreps = edge_feats_post_irreps
#         self.sh_irreps = sh_irreps        
#         self.edge_attrs_irreps = edge_attrs_irreps
#         self.radial_MLP = radial_MLP

#         self._setup()

#     @abstractmethod
#     def _setup(self) -> None:
#         raise NotImplementedError

#     @abstractmethod
#     def forward(
#         self,
#         node_attrs: torch.Tensor,
#         node_feats: torch.Tensor,
#         edge_attrs: torch.Tensor,
#         edge_feats: torch.Tensor,
#         edge_index: torch.Tensor,
#     ) -> torch.Tensor:
#         raise NotImplementedError


# @compile_mode("script")
# class SelfEdgeUpdateBlock(EdgeUpdateBlock):
#     def _setup(self) -> None:
#         # First linear
#         self.linear_pre = o3.Linear(
#             self.node_feats_irreps,
#             self.edge_feats_mid_irreps,
#             internal_weights=True,
#             shared_weights=True,
#         )

#         #gated activation
#         self.nonlin = get_gate_nonlin(self.edge_feats_mid_irreps, self.edge_feats_mid_irreps, self.edge_feats_post_irreps)

#         irreps_conv_out = self.nonlin.irreps_in

#         #Depth-wise tensor product
#         #Instructions
#         irreps_mid, instructions = tp_out_irreps_with_instructions(
#             self.edge_feats_mid_irreps,
#             self.edge_feats_mid_irreps,
#             irreps_conv_out,
#         )

#         # Tensor Product
#         self.conv_tp_diag = o3.TensorProduct(
#             self.node_feats_irreps,
#             self.node_feats_irreps,
#             irreps_mid,
#             instructions=instructions,
#             shared_weights=True,
#             internal_weights=True,
#         )

#         self.lin_post = o3.Linear(
#             irreps_mid, 
#             irreps_conv_out,
#             internal_weights=True,
#             shared_weights=True,
#             )
        




# def get_gate_nonlin(irreps_in1, irreps_in2, irreps_out, 
#                     act={1: torch.nn.functional.silu, -1: torch.tanh}, 
#                     act_gates={1: torch.sigmoid, -1: torch.tanh}
#                     ):
#     # get gate nonlinearity after tensor product
#     # irreps_in1 and irreps_in2 are irreps to be multiplied in tensor product
#     # irreps_out is desired irreps after gate nonlin
#     # notice that nonlin.irreps_out might not be exactly equal to irreps_out
            
#     irreps_scalars = o3.Irreps([
#         (mul, ir)
#         for mul, ir in irreps_out
#         if ir.l == 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
#     ]).simplify()
#     irreps_gated = o3.Irreps([
#         (mul, ir)
#         for mul, ir in irreps_out
#         if ir.l > 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
#     ]).simplify()
#     if irreps_gated.dim > 0:
#         if tp_path_exists(irreps_in1, irreps_in2, "0e"):
#             ir = "0e"
#         elif tp_path_exists(irreps_in1, irreps_in2, "0o"):
#             ir = "0o"
#             warnings.warn('Using odd representations as gates')
#         else:
#             raise ValueError(
#                 f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} is unable to produce gates needed for irreps_gated={irreps_gated}")
#     else:
#         ir = None
#     irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

#     gate_nonlin = nn.Gate(
#         irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
#         irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
#         irreps_gated  # gated tensors
#     )
    
#     return gate_nonlin


# self.target_irrps

#         # Irreps
#         num_keep_irreps = o3.Irreps(
#             (num_keep * o3.Irreps.spherical_harmonics(2, p=-1))
#             .sort()
#             .irreps.simplify()
#         )
#         irreps_matrix = o3.Irreps(
#             (num_features * o3.Irreps.spherical_harmonics(2, p=-1))
#             .sort()
#             .irreps.simplify()
#             )

#         #Instructions
#         irreps_mid_mid, instructions = tp_out_irreps_with_instructions(
#             self.node_feats_irreps,
#             self.node_feats_irreps,
#             self.node_feats_irreps,
#         )
#         irreps_mid, _ = tp_out_irreps_with_instructions(
#             self.node_feats_irreps,
#             sh_irreps,
#             irreps_matrix,
#         )

#         # Tensor Product
#         self.conv_tp_diag = o3.TensorProduct(
#             self.node_feats_irreps,
#             self.node_feats_irreps,
#             irreps_mid,
#             instructions=instructions,
#             shared_weights=True,
#             internal_weights=True,
#         )


class SelfEquiConv(torch.nn.Module):
    def __init__(self, fc_len_in, irreps_in1, irreps_in2, irreps_out, norm='', nonlin=True, 
                 act = {1: torch.nn.functional.silu, -1: torch.tanh},
                 act_gates = {1: torch.sigmoid, -1: torch.tanh}
                 ):
        super(SelfEquiConv, self).__init__()
        
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)
        
        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out, act, act_gates)
            irreps_tp_out = self.nonlin.irreps_in
        else:
            irreps_tp_out = Irreps([(mul, ir) for mul, ir in irreps_out if tp_path_exists(irreps_in1, irreps_in2, ir)])
        
        self.tp = SeparateWeightTensorProduct(irreps_in1, irreps_in2, irreps_tp_out)

        if nonlin:
            self.cfconv = o3.Linear(self.nonlin.irreps_out, self.nonlin.irreps_out)
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.cfconv = o3.Linear(irreps_tp_out, irreps_tp_out)
            self.irreps_out = irreps_tp_out
        
        # if nonlin:
        #     self.cfconv = e3ElementWise(self.nonlin.irreps_out)
        #     self.irreps_out = self.nonlin.irreps_out
        # else:
        #     self.cfconv = e3ElementWise(irreps_tp_out)
        #     self.irreps_out = irreps_tp_out
        
        # # fully connected net to create tensor product weights
        # linear_act = nn.SiLU()
        # self.fc = nn.Sequential(nn.Linear(fc_len_in, 64),
        #                         linear_act,
        #                         nn.Linear(64, 64),
        #                         linear_act,
        #                         nn.Linear(64, self.cfconv.len_weight)
        #                         )

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.cfconv.irreps_in)
            else:
                raise ValueError(f'unknown norm: {norm}')

    def forward(self, fea_in1, fea_in2, fea_weight, batch_edge):
        z = self.tp(fea_in1, fea_in2)

        if self.nonlin is not None:
            z = self.nonlin(z)

        z = self.cfconv(z)
        # weight = self.fc(fea_weight)
        # z = self.cfconv(z, weight)

        if self.norm is not None:
            z = self.norm(z, batch_edge)

        # TODO self-connection here
        return z



class SelfEdgeUpdateBlock(torch.nn.Module):
    def __init__(self, num_species, fc_len_in, irreps_sh, irreps_in_node, irreps_in_edge, irreps_out_edge,
                 act, act_gates, use_selftp=False, use_sc=True, init_edge=False, nonlin=False, norm='e3LayerNorm', if_sort_irreps=False):
        super(SelfEdgeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_in_edge = Irreps(irreps_in_edge)
        irreps_out_edge = Irreps(irreps_out_edge)

        # irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
        irreps_in1 = irreps_in_node
        if if_sort_irreps:
            self.sort = sort_irreps(irreps_in1)
            irreps_in1 = self.sort.irreps_out
        # irreps_in2 = irreps_sh
        irreps_in2 = irreps_in_node

        # self.lin_pre = o3.Linear(irreps_in=irreps_in_edge, irreps_out=irreps_in_edge, biases=True)
        
        self.nonlin = None
        self.lin_post = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_edge, act, act_gates)
            irreps_conv_out = self.nonlin.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_edge
            conv_nonlin = True
            
        self.conv = SelfEquiConv(fc_len_in, irreps_in1, irreps_in2, irreps_conv_out, nonlin=conv_nonlin, act=act, act_gates=act_gates)
        self.lin_post = Linear(irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out, biases=True)
        
        if use_sc:
            self.sc = FullyConnectedTensorProduct(self.conv.irreps_out, f'{num_species**2}x0e', self.conv.irreps_out)

        # if use_sc:
        #     self.sc = FullyConnectedTensorProduct(irreps_in_edge, f'{num_species**2}x0e', self.conv.irreps_out)

        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')
        
        self.skip_connect = SkipConnection(irreps_in_edge, self.irreps_out) # ! consider init_edge
        
        self.self_tp = None
        if use_selftp:
            self.self_tp = SelfTp(self.irreps_out, self.irreps_out)
            
        self.use_sc = use_sc
        self.init_edge = init_edge
        self.if_sort_irreps = if_sort_irreps
        self.irreps_in_edge = irreps_in_edge

    def forward(self, node_fea, edge_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch):
        
        if not self.init_edge:
            edge_fea_old = edge_fea
            # if self.use_sc:
            #     edge_self_connection = self.sc(edge_fea, edge_one_hot)
            # edge_fea = self.lin_pre(edge_fea)

        # if not self.init_edge:
        #     edge_fea_old = edge_fea
        #     if self.use_sc:
        #         edge_self_connection = self.sc(edge_fea, edge_one_hot)
        #     edge_fea = self.lin_pre(edge_fea)
            
        index_i = edge_index[0]
        index_j = edge_index[1]

        assert index_i == index_j, "the nodes for the self-edges should be the same"

        # fea_in = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
        fea_in = node_fea[index_i]
        if self.if_sort_irreps:
            fea_in = self.sort(fea_in)
        # edge_fea = self.conv(fea_in, edge_sh, edge_length_embedded, batch[edge_index[0]])
        edge_fea = self.conv(fea_in, fea_in, edge_length_embedded, batch[edge_index[0]])
        
        edge_fea = self.lin_post(edge_fea)

        if self.use_sc:
            edge_self_connection = self.sc(edge_fea, edge_one_hot)
            edge_fea = edge_fea + edge_self_connection
            
        if self.nonlin is not None:
            edge_fea = self.nonlin(edge_fea)

        if self.norm is not None:
            edge_fea = self.norm(edge_fea, batch[edge_index[0]])
        
        if not self.init_edge:
            edge_fea = self.skip_connect(edge_fea_old, edge_fea)
        
        if self.self_tp is not None:
            edge_fea = self.self_tp(edge_fea)

        return edge_fea
    


class EdgeUpdateBlock(torch.nn.Module):
    def __init__(self, num_species, fc_len_in, irreps_sh, irreps_in_node, irreps_in_edge, irreps_out_edge,
                 act, act_gates, use_selftp=False, use_sc=True, init_edge=False, nonlin=False, norm='e3LayerNorm', if_sort_irreps=False):
        super(EdgeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_in_edge = Irreps(irreps_in_edge)
        irreps_out_edge = Irreps(irreps_out_edge)

        irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
        if if_sort_irreps:
            self.sort = sort_irreps(irreps_in1)
            irreps_in1 = self.sort.irreps_out
        irreps_in2 = irreps_sh

        self.lin_pre = o3.Linear(irreps_in=irreps_in_edge, irreps_out=irreps_in_edge, biases=True)
        
        self.nonlin = None
        self.lin_post = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_edge, act, act_gates)
            irreps_conv_out = self.nonlin.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_edge
            conv_nonlin = True
            
        self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, irreps_conv_out, nonlin=conv_nonlin, act=act, act_gates=act_gates)
        self.lin_post = Linear(irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out, biases=True)
        
        if use_sc:
            self.sc = FullyConnectedTensorProduct(irreps_in_edge, f'{num_species**2}x0e', self.conv.irreps_out)

        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')
        
        self.skip_connect = SkipConnection(irreps_in_edge, self.irreps_out) # ! consider init_edge
        
        self.self_tp = None
        if use_selftp:
            self.self_tp = SelfTp(self.irreps_out, self.irreps_out)
            
        self.use_sc = use_sc
        self.init_edge = init_edge
        self.if_sort_irreps = if_sort_irreps
        self.irreps_in_edge = irreps_in_edge

    def forward(self, node_fea, edge_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch):
        
        if not self.init_edge:
            edge_fea_old = edge_fea
            if self.use_sc:
                edge_self_connection = self.sc(edge_fea, edge_one_hot)
            edge_fea = self.lin_pre(edge_fea)
            
        index_i = edge_index[0]
        index_j = edge_index[1]
        fea_in = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
        if self.if_sort_irreps:
            fea_in = self.sort(fea_in)
        edge_fea = self.conv(fea_in, edge_sh, edge_length_embedded, batch[edge_index[0]])
        
        edge_fea = self.lin_post(edge_fea)

        if self.use_sc:
            edge_fea = edge_fea + edge_self_connection
            
        if self.nonlin is not None:
            edge_fea = self.nonlin(edge_fea)

        if self.norm is not None:
            edge_fea = self.norm(edge_fea, batch[edge_index[0]])
        
        if not self.init_edge:
            edge_fea = self.skip_connect(edge_fea_old, edge_fea)
        
        if self.self_tp is not None:
            edge_fea = self.self_tp(edge_fea)

        return edge_fea
    

# mace = MACE(6, 8, 5, 4, 2, 2, o3.Irreps("64x0e+32x1o+16x2e+8x3o+8x4e"), 8, 3, [64,64,64])

@compile_mode("script")
class MACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        num_interactions: int,
        num_species: int,
        hidden_irreps: o3.Irreps,
        # MLP_irreps: o3.Irreps,
        # atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        # atomic_numbers: List[int],
        correlation: int,
        # gate: Optional[Callable],
        radial_MLP: Optional[List[int]] = None,
        mace_norm: str = 'e3LayerNorm',
        basis_func: str = 'Bessel',
        num_gaussian: int = 128, 
     ):
        super().__init__()
        # self.register_buffer(
        #     "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int64)
        # )
        # self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        # self.register_buffer(
        #     "num_interactions", torch.tensor(num_interactions, dtype=torch.int64)
        # )
        # self.register_buffer("num_elements", torch.tensor(num_elements, dtype=torch.int64))

        # Embedding
        node_attr_irreps = o3.Irreps([(num_species, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )

        self.basis_func = basis_func
        if basis_func == 'Bessel':
            self.radial_embedding = RadialEmbeddingBlock(
                r_max=r_max,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
            )
        elif basis_func == 'Gaussian':
            self.radial_embedding = GaussianBasis(start=0.0, stop=r_max, n_gaussians=num_gaussian, trainable=True)
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        # Interactions and readout
        # self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        self.norm_bool = False
        if mace_norm:
            if mace_norm == 'e3LayerNorm':
                self.norm_bool = True
            else:
                raise ValueError(f'unknown norm: {mace_norm}')

        if self.norm_bool:
            inter = RealAgnosticResidualInteractionBlockUpNorm(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
        else:
            inter = RealAgnosticResidualInteractionBlockUp(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=node_feats_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )

        self.interactions = torch.nn.ModuleList([inter])

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_species,
            use_sc=True,
        )
        self.products = torch.nn.ModuleList([prod])

        # self.readouts = torch.nn.ModuleList()
        # self.readouts.append(LinearReadoutBlock(hidden_irreps))

        hidden_irreps_out = hidden_irreps
        for i in range(num_interactions - 1):
            # if i == num_interactions - 2:
            #     hidden_irreps_out = str(
            #         hidden_irreps[0]
            #     )  # Select only scalars for last layer
            # else:
            #     hidden_irreps_out = hidden_irreps
            if self.norm_bool:
                inter = RealAgnosticResidualInteractionBlockNorm(
                    node_attrs_irreps=node_attr_irreps,
                    node_feats_irreps=hidden_irreps,
                    edge_attrs_irreps=sh_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    target_irreps=interaction_irreps,
                    hidden_irreps=hidden_irreps_out,
                    avg_num_neighbors=avg_num_neighbors,
                    radial_MLP=radial_MLP,
                )
            else:
                inter = RealAgnosticResidualInteractionBlock(
                    node_attrs_irreps=node_attr_irreps,
                    node_feats_irreps=hidden_irreps,
                    edge_attrs_irreps=sh_irreps,
                    edge_feats_irreps=edge_feats_irreps,
                    target_irreps=interaction_irreps,
                    hidden_irreps=hidden_irreps_out,
                    avg_num_neighbors=avg_num_neighbors,
                    radial_MLP=radial_MLP,
                )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_species,
                use_sc=True,
            )
            self.products.append(prod)
            # if i == num_interactions - 2:
            #     self.readouts.append(
            #         NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate)
            #     )
            # else:
            #     self.readouts.append(LinearReadoutBlock(hidden_irreps))

    def forward(
        self,
        edge_attr: torch.Tensor,
        edge_index: torch.LongTensor,
        node_one_hot: torch.LongTensor,
        batch: torch.LockingLogger = None,
        # data: Dict[str, torch.Tensor],
        # training: bool = False,
        # compute_force: bool = True,
        # compute_virials: bool = False,
        # compute_stress: bool = False,
        # compute_displacement: bool = False,
    ) -> List[torch.Tensor]: # -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        # data["node_attrs"].requires_grad_(True)
        # data["positions"].requires_grad_(True)
        # num_graphs = data["ptr"].numel() - 1
        # displacement = torch.zeros(
        #     (num_graphs, 3, 3),
        #     dtype=data["positions"].dtype,
        #     device=data["positions"].device,
        # )
        # if compute_virials or compute_stress or compute_displacement:
        #     (
        #         data["positions"],
        #         data["shifts"],
        #         displacement,
        #     ) = get_symmetric_displacement(
        #         positions=data["positions"],
        #         unit_shifts=data["unit_shifts"],
        #         cell=data["cell"],
        #         edge_index=data["edge_index"],
        #         num_graphs=num_graphs,
        #         batch=data["batch"],
        #     )

        if self.norm_bool and batch is None:
            raise ValueError(f'batch cannot be None with e3LayerNorm in MACE')

        # Atomic energies
        # node_e0 = self.atomic_energies_fn(data["node_attrs"])
        # e0 = scatter_sum(
        #     src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        # )  # [n_graphs,]

        # Embeddings
        # node_one_hot = F.one_hot(data.x, num_classes=self.num_elements).type(torch.get_default_dtype())
        node_feats = self.node_embedding(node_one_hot)
        # node_feats = self.node_embedding(data["node_attrs"])
        # vectors, lengths = get_edge_vectors_and_lengths(
        #     positions=data["positions"],
        #     edge_index=data["edge_index"],
        #     shifts=data["shifts"],
        # )


        selfloop_edge = torch.logical_and(torch.abs(edge_attr[:, 0]) < 1e-7, edge_index[0] == edge_index[1])
        keep_edge = torch.logical_not(selfloop_edge)

        edge_index = edge_index[:, keep_edge]
        vectors = edge_attr[keep_edge][:, [1,2,3]]    #[:, [2, 3, 1]] # (y, z, x) order

        # selfloop_edge = torch.logical_and(torch.abs(data["edge_attr"][:, 0]) < 1e-7,
        #                                   data["edge_index"][0] == data["edge_index"][1])
        # keep_edge = torch.logical_not(selfloop_edge)

        # edge_index = data["edge_index"][:, keep_edge]
        # vectors = data["edge_attr"][keep_edge][:, [2, 3, 1]] # (y, z, x) order
        # lengths = data['edge_attr'][keep_edge][:, [0]]
        # the MACE workflow doesn't include the self-loop

        edge_attrs = self.spherical_harmonics(vectors)

        if self.basis_func == 'Bessel':
            lengths = edge_attr[keep_edge][:, [0]]        #[:, [0]]
            edge_feats = self.radial_embedding(lengths)
        elif self.basis_func == 'Gaussian':
            lengths = edge_attr[keep_edge][:, 0]        #[:, 0]
            edge_feats = self.radial_embedding(lengths)

        # Interactions
        # energies = [e0]
        # node_energies_list = [node_e0]
        # for interaction, product, readout in zip(
        #     self.interactions, self.products, self.readouts
        # ):
        #     node_feats, sc = interaction(
        #         node_attrs=data["node_attrs"],
        #         node_feats=node_feats,
        #         edge_attrs=edge_attrs,
        #         edge_feats=edge_feats,
        #         edge_index=data["edge_index"],
        #     )
        #     node_feats = product(
        #         node_feats=node_feats,
        #         sc=sc,
        #         node_attrs=data["node_attrs"],
        #     )
        
        node_feats_up = None
        node_feats_hidden_list = []
        for i, (interaction, product) in enumerate(zip(
            self.interactions, self.products
        )):
            if i == 0:
                if self.norm_bool:
                    node_feats, sc, up = interaction(
                        node_attrs=node_one_hot,
                        node_feats=node_feats,
                        edge_attrs=edge_attrs,
                        edge_feats=edge_feats,
                        edge_index=edge_index,
                        batch=batch
                    )
                else:
                    node_feats, sc, up = interaction(
                        node_attrs=node_one_hot,
                        node_feats=node_feats,
                        edge_attrs=edge_attrs,
                        edge_feats=edge_feats,
                        edge_index=edge_index,
                    )
                # the up irreps are same with node_feats irreps in the InteractionBlock as interaction.irreps_out
                node_feats_up = up
            else:
                if self.norm_bool:
                    node_feats, sc = interaction(
                        node_attrs=node_one_hot,
                        node_feats=node_feats,
                        edge_attrs=edge_attrs,
                        edge_feats=edge_feats,
                        edge_index=edge_index,
                        batch=batch
                    )
                else:
                    node_feats, sc = interaction(
                        node_attrs=node_one_hot,
                        node_feats=node_feats,
                        edge_attrs=edge_attrs,
                        edge_feats=edge_feats,
                        edge_index=edge_index,
                    )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_one_hot,
            )
            node_feats_hidden_list.append(node_feats)
        

        #     node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
        #     energy = scatter_sum(
        #         src=node_energies, index=data["batch"], dim=-1, dim_size=num_graphs
        #     )  # [n_graphs,]
        #     energies.append(energy)
        #     node_energies_list.append(node_energies)

        # # Sum over energy contributions
        # contributions = torch.stack(energies, dim=-1)
        # total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        # node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        # node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        # # Outputs
        # forces, virials, stress = get_outputs(
        #     energy=total_energy,
        #     positions=data["positions"],
        #     displacement=displacement,
        #     cell=data["cell"],
        #     training=training,
        #     compute_force=compute_force,
        #     compute_virials=compute_virials,
        #     compute_stress=compute_stress,
        # )
        # print("total_energy", total_energy)
        # return {
        #     "energy": total_energy,
        #     "node_energy": node_energy,
        #     "contributions": contributions,
        #     "forces": forces,
        #     "virials": virials,
        #     "stress": stress,
        #     "displacement": displacement,
        # }

        return node_feats_hidden_list, node_feats_up



class NodeDegreeExpansionBlock(torch.nn.Module):
    def __init__(self, hidden_irreps: o3.Irreps, hidden_post_irreps: o3.Irreps, up_irreps: o3.Irreps, up_post_irreps: o3.Irreps,
                 irreps_post_node: o3.Irreps, num_species: int, use_sc: bool=True, expand_nonlin: bool=True, norm: str = 'e3LayerNorm'):
        super(NodeDegreeExpansionBlock, self).__init__()
        
        self.hidden_linear = o3.Linear(hidden_irreps, hidden_post_irreps, internal_weights=True, shared_weights=True)
        
        self.up_linear = o3.Linear(up_irreps, up_post_irreps, internal_weights=True, shared_weights=True)

        self.nonlin = None
        if expand_nonlin:
            self.nonlin = get_gate_nonlin(self.hidden_linear.irreps_out, self.up_linear.irreps_out, irreps_post_node)
            irreps_tp_out = self.nonlin.irreps_in
            self.irreps_out = self.nonlin.irreps_out
        else:
            irreps_tp_out = Irreps([(mul, ir) for mul, ir in irreps_post_node if tp_path_exists(self.hidden_linear.irreps_out, 
                                                                                                self.up_linear.irreps_out, ir)])
            self.irreps_out = irreps_tp_out

        self.tp = SeparateWeightTensorProduct(self.hidden_linear.irreps_out, self.up_linear.irreps_out, irreps_tp_out)

        self.lin_post = o3.Linear(irreps_in=self.irreps_out, irreps_out=self.irreps_out, biases=True) # after gate nonlinearity

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')

        self.sc = None
        if use_sc:
            self.sc = FullyConnectedTensorProduct(hidden_irreps, f'{num_species}x0e', self.irreps_out)

        self.skip_connect = o3.Linear(irreps_in=hidden_irreps, irreps_out=self.irreps_out, biases=True)

    def forward(self, node_feats_hidden: torch.Tensor, node_feats_up: torch.Tensor, node_one_hot: torch.LongTensor, batch: torch.LongTensor
                ) -> torch.Tensor:

        node_feats_hidden_post = self.hidden_linear(node_feats_hidden)
        node_feats_up_post = self.up_linear(node_feats_up)

        z = self.tp(node_feats_hidden_post, node_feats_up_post)

        if self.nonlin is not None:
            z = self.nonlin(z)

        z = self.lin_post(z)

        if self.sc is not None:
            node_self_connection = self.sc(node_feats_hidden, node_one_hot)
            z = z + node_self_connection

        if self.norm is not None:
            z = self.norm(z, batch)

        z = z + self.skip_connect(node_feats_hidden)

        return z



class NodeDegreeExpansion(torch.nn.Module):
    def __init__(self, hidden_irreps: o3.Irreps, max_ell: int, irreps_post_node: o3.Irreps, 
                 num_interactions: int, num_species: int, use_sc: bool=True, expand_nonlin: bool=True,
                 expand_norm: str = 'e3LayerNorm'):
        super(NodeDegreeExpansion, self).__init__()

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        up_irreps = (sh_irreps * num_features).sort()[0].simplify()

        assert self.check_path(hidden_irreps, up_irreps, num_interactions, irreps_post_node), "the irreps_post_node cannot be fully produced\
             by the curret Expansion block, consider incease the maximum order of hidden_irreps"

        hidden_post_irreps = self.set_irreps(hidden_irreps, irreps_post_node)
        up_post_irreps = self.set_irreps(up_irreps, irreps_post_node)

        self.node_order_expansion_blocks = torch.nn.ModuleList([])
        for _ in range(num_interactions):
            node_order_expansion_block = NodeDegreeExpansionBlock(hidden_irreps, hidden_post_irreps, up_irreps, up_post_irreps, irreps_post_node,
                                                                 num_species, use_sc, expand_nonlin, expand_norm)
            self.node_order_expansion_blocks.append(node_order_expansion_block)
            up_irreps = node_order_expansion_block.irreps_out
            up_post_irreps = self.set_irreps(up_irreps, irreps_post_node)

    def forward(self, node_feats_hidden_list: List[torch.Tensor], node_feats_up: torch.Tensor, node_one_hot: torch.LongTensor, batch: torch.LongTensor
                ) -> List[torch.Tensor]:
        
        post_node_feats_list = []
        for node_order_expansion_block, node_feats_hidden in zip(self.node_order_expansion_blocks, node_feats_hidden_list):
            post_node_feats = node_order_expansion_block(node_feats_hidden, node_feats_up, node_one_hot, batch)
            node_feats_up = post_node_feats
            post_node_feats_list.append(post_node_feats)

        return post_node_feats_list
        

    def check_path(self, hidden_irreps: o3.Irreps, up_irreps: o3.Irreps, num_blocks: int, irreps_post_node: o3.Irreps):
        hidden_irreps_simp = o3.Irreps([(1, irrep) for _, irrep in hidden_irreps])
        up_irreps_simp = o3.Irreps([(1, irrep) for _, irrep in up_irreps])

        for _ in range(num_blocks):
            up_irreps_simp = o3.FullTensorProduct(hidden_irreps_simp, up_irreps_simp).irreps_out.sort()[0].simplify()
            up_irreps_simp = o3.Irreps([(1, irrep) for _, irrep in up_irreps_simp if irrep in irreps_post_node])

        contain_bool = [irrep in up_irreps_simp for _, irrep in irreps_post_node]

        return len(contain_bool) == sum(contain_bool)
        
    
    def set_irreps(self, pre_irreps: o3.Irreps, irreps_post_node:o3.Irreps) -> Tuple[o3.Irreps]:
        
        post_irreps = o3.Irreps([(irreps_post_node.count(irrep), irrep) for _, irrep in pre_irreps]).sort()[0].simplify()

        if pre_irreps.lmax > irreps_post_node.lmax:
            mul = irreps_post_node.count(o3.Irrep(str(irreps_post_node.lmax)+"y"))
            for l in range(irreps_post_node.lmax+1, pre_irreps.lmax+1):
                irrep = o3.Irrep(str(l)+"y")
                post_irreps += o3.Irreps([(mul, irrep)])

        assert len(pre_irreps.sort()[0].simplify()) == len(post_irreps), "the set post-irreps shouldn't delete certain Irrep types"

        return post_irreps



#unify irreps_sh? o3.spherical_harmonics(o3.Irreps.spherical_harmonics(4), torch.tensor([[0.,2.,0.]]).to(torch.float), normalize=True, normalization='component')
#delete use_sbf=True, selftp=False, only_ij=False because they are all false in practice, and no_parity is also set as false

class EdgeUpdate(torch.nn.Module):
    def __init__(self, num_species: int, irreps_edge_init: o3.Irreps, irreps_sh: o3.Irreps, node_irreps_out_list: List[o3.Irreps],
                 irreps_mid_edge, irreps_post_edge, irreps_out_edge, r_max, use_sc=True, num_basis=128, edge_upd=True,
                 act={1: torch.nn.functional.silu, -1: torch.tanh}, act_gates={1: torch.sigmoid, -1: torch.tanh}):
        super(EdgeUpdate, self).__init__()

        self.num_species = num_species
        num_blocks = len(node_irreps_out_list)

        self.basis = GaussianBasis(start=0.0, stop=r_max, n_gaussians=num_basis, trainable=False)

        # distance expansion to initialize edge feature
        irreps_edge_init = Irreps(irreps_edge_init)
        assert irreps_edge_init == Irreps(f'{irreps_edge_init.dim}x0e')
        self.distance_expansion = GaussianBasis(
            start=0.0, stop=6.0, n_gaussians=irreps_edge_init.dim, trainable=False
        )

        self.sh = o3.SphericalHarmonics(irreps_out=irreps_sh, normalize=True, normalization='component',)
        self.irreps_sh = irreps_sh

        irreps_edge_prev = irreps_edge_init
        self.edge_update_blocks = torch.nn.ModuleList([])
        for index_block in range(num_blocks):
            if index_block == num_blocks - 1:
                edge_update_block = EdgeUpdateBlock(num_species, num_basis, irreps_sh, node_irreps_out_list[index_block], irreps_edge_prev, irreps_post_edge, act, act_gates, use_selftp=False, use_sc=use_sc, if_sort_irreps=False)
            else:
                if edge_upd:
                    edge_update_block = EdgeUpdateBlock(num_species, num_basis, irreps_sh, node_irreps_out_list[index_block], irreps_edge_prev, irreps_mid_edge, act, act_gates, use_selftp=False, use_sc=use_sc, if_sort_irreps=False)
                else:
                    edge_update_block = None
            self.edge_update_blocks.append(edge_update_block)
            if edge_update_block is not None:
                irreps_edge_prev = edge_update_block.irreps_out

        irreps_out_edge = Irreps(irreps_out_edge)
        for _, ir in irreps_out_edge:
            assert ir in irreps_edge_prev, f'required ir {ir} in irreps_out_edge cannot be produced by convolution in the last edge update block ({edge_update_block.irreps_in_edge} -> {edge_update_block.irreps_out})'

        self.irreps_out_edge = irreps_out_edge
        self.lin_edge = Linear(irreps_in=irreps_edge_prev, irreps_out=irreps_out_edge, biases=True)

    def forward(self, post_node_feats_list: List[torch.Tensor], edge_attr: torch.Tensor, edge_index: torch.LongTensor,
                x: torch.LongTensor, batch: torch.LongTensor) -> torch.Tensor:

        edge_one_hot = F.one_hot(self.num_species * x[edge_index[0]] + x[edge_index[1]],
                                 num_classes=self.num_species**2).type(torch.get_default_dtype()) # ! might not be good if dataset has many elements

        # edge_length = edge_attr[:, 0] # watch out the shape 
        # edge_vec = edge_attr[:, [2, 3, 1]] # (y, z, x) order
        edge_length = edge_attr[:, 0]
        edge_vec = edge_attr[:, [1, 2, 3]]

        # edge_one_hot = F.one_hot(self.num_species * data.x[data.edge_index[0]] + data.x[data.edge_index[1]],
        #                     num_classes=self.num_species**2).type(torch.get_default_dtype()) # ! might not be good if dataset has many elements

        # edge_length = data['edge_attr'][:, 0] # watch out the shape 
        # edge_vec = data["edge_attr"][:, [2, 3, 1]] # (y, z, x) order
        
        edge_sh = self.sh(edge_vec).type(torch.get_default_dtype())
        edge_length_embedded = self.basis(edge_length)

        edge_fea = self.distance_expansion(edge_length).type(torch.get_default_dtype())
        for node_fea, edge_update_block in zip(post_node_feats_list, self.edge_update_blocks):
            if edge_update_block is not None:
                edge_fea = edge_update_block(node_fea, edge_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch)

        edge_fea = self.lin_edge(edge_fea)
        return None, edge_fea
    
