"""Guard the spherical Bessel basis we borrow from PyTorch Geometric.

``maceh.e3modules`` imports ``bessel_basis`` from
``torch_geometric.nn.models.dimenet_utils``. That module is real and MIT
licensed, but it is not part of PyG's documented public API, so a PyG upgrade
could move or change it. Without these assertions such a change would surface
as a silently different radial basis -- a model that trains but is not the model
that produced the recorded results.

The properties checked here follow from the definition of the basis

    R_{l,i}(x) = sqrt(2) * j_l(z_{l,i} x) / |j_{l+1}(z_{l,i})|,  x in [0, 1]

with ``z_{l,i}`` the (i+1)-th positive root of ``j_l``, so they hold for any
correct implementation and do not encode PyG's particular one.
"""

import numpy as np
import pytest
import sympy as sym

N_ORDERS = 4
N_FREQ = 4


def test_pyg_still_exposes_bessel_basis():
    # The import maceh.e3modules depends on. If PyG relocates it, fail here
    # with an obvious message rather than deep inside model construction.
    from torch_geometric.nn.models.dimenet_utils import bessel_basis

    assert callable(bessel_basis)


@pytest.fixture(scope="module")
def basis():
    from torch_geometric.nn.models.dimenet_utils import bessel_basis

    x = sym.Symbol("x")
    expressions = bessel_basis(N_ORDERS, N_FREQ)
    functions = [[sym.lambdify(x, expr, "numpy") for expr in row]
                 for row in expressions]
    return functions, expressions


def test_shape(basis):
    _, expressions = basis
    assert len(expressions) == N_ORDERS
    assert all(len(row) == N_FREQ for row in expressions)


def test_single_free_symbol(basis):
    _, expressions = basis
    for row in expressions:
        for expr in row:
            assert {s.name for s in expr.free_symbols} == {"x"}


def test_basis_vanishes_at_the_cutoff(basis):
    functions, _ = basis
    # Not tighter than 1e-5 on purpose. The implementation stores its Bessel
    # roots in a float32 array, so z is known to ~7 significant digits and
    # R(1) lands around 1e-6 rather than at machine zero. That is harmless
    # here -- SphericalBasis multiplies by a PolynomialCutoff that is exactly
    # zero at the cutoff -- but a tolerance of 1e-9 would fail on correct code.
    # A basis that had genuinely changed would miss by O(1), not O(1e-6).
    for order in range(N_ORDERS):
        for index in range(N_FREQ):
            assert abs(functions[order][index](1.0)) < 1e-5


def test_basis_is_orthonormal_under_the_radial_weight(basis):
    functions, _ = basis
    # The integrand R_l R_l x^2 is finite at the origin (R_l ~ x^l), but the
    # closed form carries 1/x and 1/x^2 factors that overflow there, so start
    # just above zero; the omitted sliver contributes O(1e-18).
    grid = np.linspace(1e-6, 1.0, 200001)
    weight = grid ** 2
    for order in range(N_ORDERS):
        values = np.array([f(grid) for f in functions[order]])
        gram = np.array([
            [np.trapezoid(values[i] * values[j] * weight, grid)
             for j in range(N_FREQ)]
            for i in range(N_FREQ)])
        assert np.allclose(gram, np.eye(N_FREQ), atol=2e-5), (
            f"l={order} Gram matrix is not the identity:\n{gram}")


def test_spherical_basis_layer_builds_and_respects_the_cutoff():
    torch = pytest.importorskip("torch")
    from maceh.e3modules import SphericalBasis

    layer = SphericalBasis("4x0e+4x1o+2x2e", rcutoff=5.0)
    direction = torch.nn.functional.normalize(torch.randn(8, 3), dim=-1)
    output = layer(torch.rand(8) * 5.0, direction)
    assert output.shape[0] == 8
    assert bool(torch.isfinite(output).all())

    at_cutoff = layer(torch.tensor([5.0]), torch.tensor([[0.0, 0.0, 1.0]]))
    assert float(at_cutoff.abs().max()) == 0.0
