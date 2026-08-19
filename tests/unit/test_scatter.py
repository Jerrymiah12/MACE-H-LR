"""`maceh` aggregates with `torch_geometric.utils.scatter`, not `torch_scatter`.

`torch_scatter` ships compiled kernels that must be built against the exact
torch build in use, so a CPU wheel cannot be moved to a CUDA box and a source
build needs `nvcc`.  `torch_geometric.utils.scatter` is pure PyTorch and has
no such constraint.  The swap is only safe because the two agree on the four
call sites `maceh` actually uses, which is what these tests pin.

`torch_scatter` is not a dependency any more, so the parity tests skip when it
is absent -- what must always hold is the *behaviour*, checked directly.
"""
import pytest
import torch

from torch_geometric.utils import scatter

try:
    from torch_scatter import scatter as reference_scatter
except ImportError:                                          # pragma: no cover
    reference_scatter = None

REAL_SRC = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [-1.0, 0.5]])
COMPLEX_SRC = torch.tensor([[1 + 2j, 3 - 1j], [2 + 0j, 1 + 1j],
                            [5 + 5j, 6 - 6j], [-1 - 1j, 0 + 2j]],
                           dtype=torch.complex64)
INDEX = torch.tensor([0, 0, 1, 2])
DIM_SIZE = 3


def test_scatter_add_matches_by_hand():
    # model.py aggregates edge features onto nodes with reduce='add'; this is
    # the hot path of message passing, so pin it against an explicit sum.
    got = scatter(REAL_SRC, INDEX, dim=0, dim_size=DIM_SIZE, reduce='add')
    want = torch.tensor([[4.0, 6.0], [5.0, 6.0], [-1.0, 0.5]])
    assert torch.equal(got, want)


def test_scatter_mean_matches_by_hand():
    got = scatter(REAL_SRC, INDEX, dim=0, dim_size=DIM_SIZE, reduce='mean')
    want = torch.tensor([[2.0, 3.0], [5.0, 6.0], [-1.0, 0.5]])
    assert torch.allclose(got, want)


def test_scatter_add_supports_complex():
    # e3modules.py subtracts a mean from possibly-complex fields by scattering
    # with reduce='add' and dividing afterwards, precisely because scatter with
    # reduce='mean' does not support complex dtypes.  The 'add' arm must.
    got = scatter(COMPLEX_SRC, INDEX, dim=0, dim_size=DIM_SIZE, reduce='add')
    want = torch.tensor([[3 + 2j, 4 + 0j], [5 + 5j, 6 - 6j], [-1 - 1j, 0 + 2j]],
                        dtype=torch.complex64)
    assert torch.equal(got, want)


def test_scatter_mean_still_rejects_complex():
    # If this ever starts working, the workaround in e3modules.EquivariantLayerNorm
    # (scatter 'add' then divide) could be simplified -- but until then the
    # comment there must stay true.
    with pytest.raises((NotImplementedError, RuntimeError)):
        scatter(COMPLEX_SRC, INDEX, dim=0, dim_size=DIM_SIZE, reduce='mean')


def test_dim_size_pads_trailing_empty_groups():
    # dim_size is always passed explicitly (node count), and indices need not
    # cover it; the result must be zero-padded, not truncated.
    got = scatter(REAL_SRC, INDEX, dim=0, dim_size=5, reduce='add')
    assert got.shape == (5, 2)
    assert torch.equal(got[3:], torch.zeros(2, 2))


@pytest.mark.skipif(reference_scatter is None,
                    reason="torch_scatter is no longer a dependency")
@pytest.mark.parametrize("src", [REAL_SRC, COMPLEX_SRC], ids=["real", "complex"])
@pytest.mark.parametrize("reduce", ["add", "mean"])
def test_parity_with_torch_scatter(src, reduce):
    # the swap this test guards: identical results, or identical refusal.
    kwargs = dict(dim=0, dim_size=DIM_SIZE, reduce=reduce)
    try:
        want = reference_scatter(src, INDEX, **kwargs)
    except Exception as exc:
        with pytest.raises(type(exc)):
            scatter(src, INDEX, **kwargs)
        return
    assert torch.allclose(scatter(src, INDEX, **kwargs), want)
