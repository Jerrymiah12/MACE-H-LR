import torch 
from e3nn import o3
from .data import AijData
from .e3modules import e3TensorDecomp

# def get_mean_std_tensor(dataset: AijData, net_out_irreps: o3.Irreps) -> tuple:
#     # config.net_out_irreps a.k.a irreps_out_edge

#     assert 'label' in dataset._data.keys(), "label is not found in the dataset"

#     atomic_number_edge_i = dataset._data.x[dataset._data.edge_index[0]]
#     atomic_number_edge_j = dataset._data.x[dataset._data.edge_index[1]]
#     # we have to leave the 'source to target' order in pyg alone, since deeph use this order

#     num_species = dataset.info['index_to_Z'].numel()
#     mean_tensor = torch.zeros(num_species, num_species, net_out_irreps.dim).to(dtype=dataset._data.label.dtype, device=dataset._data.label.device)
#     std_tensor = torch.ones(num_species, num_species, net_out_irreps.dim).to(dtype=dataset._data.label.dtype, device=dataset._data.label.device)
#     for x_i in range(num_species):
#          for x_j in range(num_species):
#               label_ij = dataset._data.label[torch.logical_and(atomic_number_edge_i==x_i, atomic_number_edge_j==x_j)]
#               for l, sli in zip(net_out_irreps.ls, net_out_irreps.slices()):
#                    if l == 0:
#                        mean_tensor[x_i, x_j, sli] = label_ij[:, sli].mean()
#                    std_tensor[x_i, x_j, sli] = label_ij[:, sli].std()
   
#     return mean_tensor, std_tensor



# def get_mean_std_tensor(dataset: AijData, net_out_irreps: o3.Irreps) -> tuple:
#     # config.net_out_irreps a.k.a irreps_out_edge

#     assert 'label' in dataset._data.keys(), "label is not found in the dataset"

#     atomic_number_edge_i = dataset._data.x[dataset._data.edge_index[0]]
#     atomic_number_edge_j = dataset._data.x[dataset._data.edge_index[1]]
#     # we have to leave the 'source to target' order in pyg alone, since deeph use this order

#     num_species = dataset.info['index_to_Z'].numel()
#     mean_tensor = torch.zeros(num_species, num_species, net_out_irreps.dim).to(dtype=dataset._data.label.dtype, device=dataset._data.label.device)
#     std_tensor = torch.ones(num_species, num_species, net_out_irreps.dim).to(dtype=dataset._data.label.dtype, device=dataset._data.label.device)
#     for x_i in range(num_species):
#          for x_j in range(num_species):
#               label_ij = dataset._data.label[torch.logical_and(atomic_number_edge_i==x_i, atomic_number_edge_j==x_j)]
#               for l, sli in zip(net_out_irreps.ls, net_out_irreps.slices()):
#                    if l == 0:
#                        mean_tensor[x_i, x_j, sli] = label_ij[:, sli].mean()
#                    std_tensor[x_i, x_j, sli] = label_ij[:, sli].std()
   
#     return mean_tensor, std_tensor



def get_mean_std_tensor(dataset: AijData, net_out_irreps: o3.Irreps, kernel: e3TensorDecomp) -> tuple:
    """Compute shift/scale statistics without materializing all net outputs.

    ``kernel.get_net_out`` returns a tensor as large as ``dataset.label`` and
    internally concatenates another equally large list of orbital blocks.  On
    the MgO production set each copy is about 7.6 GiB, which is enough to
    trigger WSL's OOM killer on top of the resident dataset.  Transform one
    structure at a time and retain only float64 sufficient statistics.

    This preserves the original definition: for every ordered species pair,
    the mean (for scalar irreps) and unbiased standard deviation are taken
    over all edges *and* all components in each irrep slice.
    """
    assert 'label' in dataset._data.keys(), "label is not found in the dataset"

    kernel_args = dict(kernel.args)
    kernel_args['device_torch'] = 'cpu'
    kernel_cpu = type(kernel)(**kernel_args)
    num_species = dataset.info['index_to_Z'].numel()
    num_pairs = num_species * num_species
    out_dim = net_out_irreps.dim

    edge_counts = torch.zeros(num_pairs, dtype=torch.int64)
    sums = torch.zeros(num_pairs, out_dim, dtype=torch.float64)
    square_sums = torch.zeros_like(sums)
    output_dtype = None

    # Training always supplies the complete AijData, not a PyG subset.  Being
    # explicit here prevents silently computing the wrong membership if that
    # calling contract changes later.
    assert dataset._indices is None, \
        "shift/scale statistics require the complete AijData dataset"

    with torch.no_grad():
        for graph_index in range(len(dataset)):
            data = dataset.get(graph_index)
            net_out = kernel_cpu.get_net_out(data.label.cpu())
            output_dtype = net_out.dtype
            edge_i = data.x[data.edge_index[0]]
            edge_j = data.x[data.edge_index[1]]
            pair_index = edge_i * num_species + edge_j

            for pair in range(num_pairs):
                selected = net_out[pair_index == pair]
                if selected.numel() == 0:
                    continue
                selected64 = selected.to(torch.float64)
                edge_counts[pair] += selected.shape[0]
                sums[pair] += selected64.sum(dim=0)
                square_sums[pair] += selected64.square().sum(dim=0)

    # ``get()`` caches per-graph views.  They are cheap, but clearing them here
    # keeps the dataset in the same state as the old all-at-once calculation.
    dataset._data_list = None

    assert output_dtype is not None, "cannot compute statistics of an empty dataset"
    dtype = output_dtype
    mean_tensor = torch.zeros(
        num_species, num_species, out_dim, dtype=dtype)
    std_tensor = torch.ones_like(mean_tensor)
    for x_i in range(num_species):
        for x_j in range(num_species):
            pair = x_i * num_species + x_j
            for l, sli in zip(net_out_irreps.ls,
                              net_out_irreps.slices()):
                width = sli.stop - sli.start
                count = int(edge_counts[pair]) * width
                assert count > 1, \
                    f"not enough samples for species pair ({x_i}, {x_j})"
                total = sums[pair, sli].sum()
                total_sq = square_sums[pair, sli].sum()
                mean = total / count
                # Match torch.std's default correction=1.  Clamp only guards
                # against a tiny negative caused by floating-point round-off.
                variance = (total_sq - total.square() / count) / (count - 1)
                std = variance.clamp_min(0).sqrt()
                if l == 0:
                    mean_tensor[x_i, x_j, sli] = mean.to(dtype)
                std_tensor[x_i, x_j, sli] = std.to(dtype)
 
    return mean_tensor, std_tensor



# def shift_scale_out(edge_feature: torch.Tensor, mean_tensor: torch.Tensor, std_tensor: torch.Tensor, x: torch.Tensor, edge_index: torch.LongTensor,
#                     net_out_irreps: o3.Irreps = None) -> torch.Tensor:
     
#     atomic_number_edge_i = x[edge_index[0]]
#     atomic_number_edge_j = x[edge_index[1]]
#     # we have to leave the 'source to target' order in pyg alone, since deeph use this order

#     num_species = mean_tensor.shape[0]

#     for x_i in range(num_species):
#          for x_j in range(num_species):
#             mask = torch.logical_and(atomic_number_edge_i==x_i, atomic_number_edge_j==x_j)
#             #   for l, sli in zip(net_out_irreps.ls, net_out_irreps.slices()):
#             #     if l == 0:
#             #         edge_feature[mask][:, sli] = edge_feature[mask][:, sli] * std_tensor[x_i, x_j, sli] + mean_tensor[x_i, x_j, sli]
#             #     else:
#             #         edge_feature[mask][:, sli] = edge_feature[mask][:, sli] * std_tensor[x_i, x_j, sli]
#             edge_feature[mask] = edge_feature[mask] * std_tensor[x_i, x_j] + mean_tensor[x_i, x_j]
         
#     return edge_feature



def shift_scale_out(edge_feature: torch.Tensor, mean_tensor: torch.Tensor, std_tensor: torch.Tensor, x: torch.Tensor, edge_index: torch.LongTensor,
                    ) -> torch.Tensor:
     
    atomic_number_edge_i = x[edge_index[0]]
    atomic_number_edge_j = x[edge_index[1]]
    # we have to leave the 'source to target' order in pyg alone, since deeph use this order

    num_species = mean_tensor.shape[0]

    for x_i in range(num_species):
         for x_j in range(num_species):
            mask = torch.logical_and(atomic_number_edge_i==x_i, atomic_number_edge_j==x_j)
            #   for l, sli in zip(net_out_irreps.ls, net_out_irreps.slices()):
            #     if l == 0:
            #         edge_feature[mask][:, sli] = edge_feature[mask][:, sli] * std_tensor[x_i, x_j, sli] + mean_tensor[x_i, x_j, sli]
            #     else:
            #         edge_feature[mask][:, sli] = edge_feature[mask][:, sli] * std_tensor[x_i, x_j, sli]

            # edge_feature[mask] = edge_feature[mask] * std_tensor[x_i, x_j] + mean_tensor[x_i, x_j]

            edge_feature[mask] = edge_feature[mask] + (edge_feature[mask] * (std_tensor[x_i, x_j] - 1.)).detach() + mean_tensor[x_i, x_j] 
         
    return edge_feature



# def inverse_shift_scale_out(edge_feature: torch.Tensor, mean_tensor: torch.Tensor, std_tensor: torch.Tensor, x: torch.Tensor, edge_index: torch.LongTensor,
#                             net_out_irreps: o3.Irreps = None) -> torch.Tensor:
     
#     atomic_number_edge_i = x[edge_index[0]]
#     atomic_number_edge_j = x[edge_index[1]]
#     # we have to leave the 'source to target' order in pyg alone, since deeph use this order

#     num_species = mean_tensor.shape[0]

#     for x_i in range(num_species):
#          for x_j in range(num_species):
#             mask = torch.logical_and(atomic_number_edge_i==x_i, atomic_number_edge_j==x_j)
#             #   for l, sli in zip(net_out_irreps.ls, net_out_irreps.slices()):
#             #     if l == 0:
#             #         edge_feature[mask][:, sli] = (edge_feature[mask][:, sli] - mean_tensor[x_i, x_j, sli]) / std_tensor[x_i, x_j, sli]
#             #     else:
#             #         edge_feature[mask][:, sli] = edge_feature[mask][:, sli] / std_tensor[x_i, x_j, sli]
#             edge_feature[mask] = (edge_feature[mask] - mean_tensor[x_i, x_j]) / std_tensor[x_i, x_j]
         
#     return edge_feature



def inverse_shift_scale_out(edge_feature: torch.Tensor, mean_tensor: torch.Tensor, std_tensor: torch.Tensor, x: torch.Tensor, edge_index: torch.LongTensor,
                            ) -> torch.Tensor:
     
    atomic_number_edge_i = x[edge_index[0]]
    atomic_number_edge_j = x[edge_index[1]]
    # we have to leave the 'source to target' order in pyg alone, since deeph use this order

    num_species = mean_tensor.shape[0]

    for x_i in range(num_species):
         for x_j in range(num_species):
            mask = torch.logical_and(atomic_number_edge_i==x_i, atomic_number_edge_j==x_j)
            #   for l, sli in zip(net_out_irreps.ls, net_out_irreps.slices()):
            #     if l == 0:
            #         edge_feature[mask][:, sli] = (edge_feature[mask][:, sli] - mean_tensor[x_i, x_j, sli]) / std_tensor[x_i, x_j, sli]
            #     else:
            #         edge_feature[mask][:, sli] = edge_feature[mask][:, sli] / std_tensor[x_i, x_j, sli]
            edge_feature[mask] = (edge_feature[mask] - mean_tensor[x_i, x_j]) / std_tensor[x_i, x_j]
         
    return edge_feature
