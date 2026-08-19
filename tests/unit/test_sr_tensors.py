from types import SimpleNamespace

import torch
from e3nn import o3

from maceh.maceh import (BornChargeHead, DielectricHead,
                         SRBornEpsilonModel, load_sr_born_epsilon_state)
from maceh.kernel import assert_sr_tensor_checkpoint


def _randomize(module, seed=7):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(
                parameter.shape, generator=generator,
                dtype=parameter.dtype, device=parameter.device) * 0.05)


def test_tensor_heads_are_equivariant_and_enforce_physical_constraints():
    irreps = o3.Irreps('2x0e+2x1o+2x2e')
    born_head = BornChargeHead(
        irreps, torch.stack((-2.0 * torch.eye(3), 2.0 * torch.eye(3))))
    epsilon_head = DielectricHead(irreps, 3.3 * torch.eye(3))
    _randomize(born_head)
    _randomize(epsilon_head, seed=8)
    features = torch.randn(4, irreps.dim)
    species = torch.tensor([0, 1, 0, 1])
    batch = torch.tensor([0, 0, 1, 1])

    rotation = o3.rand_matrix()
    # MACE-H's encoder coordinates are ordered (y,z,x).
    permutation = torch.tensor([[0., 1., 0.],
                                [0., 0., 1.],
                                [1., 0., 0.]])
    internal_rotation = permutation @ rotation @ permutation.T
    rotated_features = features @ irreps.D_from_matrix(internal_rotation).T

    born = born_head(features, species, batch)
    epsilon = epsilon_head(features, batch)
    born_rotated = born_head(rotated_features, species, batch)
    epsilon_rotated = epsilon_head(rotated_features, batch)
    expected_born = rotation @ born @ rotation.T
    expected_epsilon = rotation @ epsilon @ rotation.T

    assert torch.allclose(born_rotated, expected_born, atol=2e-5, rtol=2e-5)
    assert torch.allclose(epsilon_rotated, expected_epsilon,
                          atol=2e-5, rtol=2e-5)
    for graph_index in range(2):
        assert torch.max(torch.abs(
            born[batch == graph_index].sum(dim=0))) < 2e-6
    assert torch.equal(epsilon, epsilon.transpose(-1, -2))
    assert torch.all(torch.linalg.eigvalsh(epsilon) > 0)


class _FakeSR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 3, bias=False)
        self.register_buffer('atomic_numbers', torch.tensor([8, 12]))
        block = SimpleNamespace(irreps_out=o3.Irreps('1x0e+1x1o+1x2e'))
        self.expansion = SimpleNamespace(
            node_order_expansion_blocks=[block])

    def forward_features(self, data):
        features = self.projection(data.features)
        # Pad the fake scalar/vector features to 0e+1o+2e (dimension 9).
        return torch.nn.functional.pad(features, (0, 6)), data.edge_output


def test_legacy_checkpoint_initializes_only_sr_submodel():
    sr = _FakeSR()
    old_state = {key: value.clone() for key, value in sr.state_dict().items()}
    model = SRBornEpsilonModel(
        sr,
        torch.stack((-2.0 * torch.eye(3), 2.0 * torch.eye(3))),
        3.3 * torch.eye(3))
    incompatible = load_sr_born_epsilon_state(model, old_state)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(key.startswith(('born_head.', 'epsilon_head.'))
               for key in incompatible.missing_keys)
    keys = model.state_dict()
    assert any(key.startswith('born_head.') for key in keys)
    assert any(key.startswith('epsilon_head.') for key in keys)
    assert not any('full_h' in key.lower() for key in keys)
    assert model.model_type == 'sr_born_epsilon'
    assert model.direct_full_h_head is False
    assert model.analytic_lr_reconstruction is True

    manifest = {'path': '/labels/manifest.json', 'sha256': 'ab' * 32}
    checkpoint = {
        'state_dict': keys,
        'model_type': model.model_type,
        'direct_full_h_head': model.direct_full_h_head,
        'analytic_lr_reconstruction': model.analytic_lr_reconstruction,
        'tensor_label_manifest': manifest,
    }
    assert assert_sr_tensor_checkpoint(checkpoint)
