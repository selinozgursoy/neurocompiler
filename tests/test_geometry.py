import numpy as np
import pytest

from neurograph.geometry import (equivariance_error, intertwiner, random_rotation, rep_matrix, schur_linear,
                                 sph_harm_features, wigner)

IRR = "8x0e+4x1o+2x2e"


def test_wigner_is_a_representation():
    rng = np.random.default_rng(0)
    A, B = random_rotation(rng), random_rotation(rng)
    for l in (0, 1, 2):
        assert np.allclose(wigner(l, A @ B), wigner(l, A) @ wigner(l, B), atol=1e-10)
        assert np.allclose(wigner(l, A) @ wigner(l, A).T, np.eye(2 * l + 1), atol=1e-10)


@pytest.mark.parametrize("key", [(1, 1, 0), (1, 1, 1), (1, 1, 2), (2, 1, 1), (2, 2, 2), (1, 2, 2)])
def test_intertwiners_are_equivariant(key):
    rng = np.random.default_rng(1)
    C, R = intertwiner(*key), random_rotation(rng)
    x, y = rng.normal(size=2 * key[0] + 1), rng.normal(size=2 * key[1] + 1)
    lhs = np.einsum("i,j,ijk->k", wigner(key[0], R) @ x, wigner(key[1], R) @ y, C)
    assert np.allclose(lhs, wigner(key[2], R) @ np.einsum("i,j,ijk->k", x, y, C), atol=1e-9)


def test_triangle_rule_forbids_paths():
    assert intertwiner(1, 1, 3) is None and intertwiner(0, 1, 2) is None


def test_spherical_harmonics_are_equivariant():
    rng = np.random.default_rng(2)
    v, R = rng.normal(size=3), random_rotation(rng)
    for l in (1, 2):
        assert np.allclose(sph_harm_features(R @ v)[l], wigner(l, R) @ sph_harm_features(v)[l], atol=1e-9)


def test_shared_cells_keep_equivariance_under_programming_error():
    """The central claim: (W + E) (x) I still commutes with the group action."""
    rng = np.random.default_rng(3)
    shared = schur_linear(IRR, rng, noise=0.05, shared=True)
    dense = schur_linear(IRR, np.random.default_rng(3), noise=0.05, shared=False)
    assert equivariance_error(shared, IRR, np.random.default_rng(4)) < 1e-12
    assert equivariance_error(dense, IRR, np.random.default_rng(4)) > 0.01


def test_rep_matrix_is_orthogonal():
    D = rep_matrix(IRR, random_rotation(np.random.default_rng(5)))
    assert D.shape == (8 + 12 + 10,) * 2 and np.allclose(D @ D.T, np.eye(30), atol=1e-10)
