"""Group-theory helpers: irreps strings, SO(3) representations for l <= 2,
Schur-structured equivariant linear maps and an equivariance-error metric."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

_IRREP = re.compile(r"^(\d+)x(\d+)([eo])$")


@dataclass(frozen=True)
class Irrep:
    mul: int
    l: int
    parity: int  # +1 even, -1 odd

    @property
    def dim(self) -> int:
        return self.mul * (2 * self.l + 1)

    def __str__(self) -> str:
        return f"{self.mul}x{self.l}{'e' if self.parity > 0 else 'o'}"


def parse_irreps(text: str) -> list[Irrep]:
    """Parse '16x0e+8x1o' into Irreps. Raises ValueError on malformed input."""
    out = []
    for part in text.replace(" ", "").split("+"):
        m = _IRREP.match(part)
        if not m:
            raise ValueError(f"malformed irrep {part!r} in {text!r}")
        mul, l = int(m.group(1)), int(m.group(2))
        if mul <= 0:
            raise ValueError(f"multiplicity must be positive in {part!r}")
        out.append(Irrep(mul, l, 1 if m.group(3) == "e" else -1))
    return out


def irreps_dim(text: str) -> int:
    return sum(ir.dim for ir in parse_irreps(text))


def tensor_product_allowed(a: str, b: str, out: str, check_parity: bool = True) -> list[str]:
    """Return the output irreps that violate the selection rules (empty = valid)."""
    A, B = parse_irreps(a), parse_irreps(b)
    bad = []
    for o in parse_irreps(out):
        ok = any(
            abs(x.l - y.l) <= o.l <= x.l + y.l and (not check_parity or o.parity == x.parity * y.parity)
            for x in A for y in B
        )
        if not ok:
            bad.append(f"{o.l}{'e' if o.parity > 0 else 'o'}")
    return bad


def schur_cells(irreps_in: str, irreps_out: str) -> tuple[int, int]:
    """(dense cells, Schur-shared cells) for an equivariant linear map.

    By Schur's lemma the map is block diagonal by irrep type and acts as
    W_l (x) I_(2l+1), so only mul_out x mul_in entries per shared type are free.
    """
    I, O = parse_irreps(irreps_in), parse_irreps(irreps_out)
    dense = sum(i.dim for i in I) * sum(o.dim for o in O)
    shared = sum(i.mul * o.mul for i in I for o in O if (i.l, i.parity) == (o.l, o.parity))
    return dense, shared


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


_S2, _S6 = 1 / math.sqrt(2), 1 / math.sqrt(6)
_B2 = [np.array(m, float) for m in (
    [[0, _S2, 0], [_S2, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, _S2], [0, _S2, 0]],
    [[0, 0, _S2], [0, 0, 0], [_S2, 0, 0]], [[_S2, 0, 0], [0, -_S2, 0], [0, 0, 0]],
    [[-_S6, 0, 0], [0, -_S6, 0], [0, 0, 2 * _S6]])]


def wigner(l: int, R: np.ndarray) -> np.ndarray:
    """Real representation matrix of rotation R for l in {0, 1, 2}."""
    if l == 0:
        return np.eye(1)
    if l == 1:
        return R
    if l == 2:  # action on symmetric traceless matrices
        return np.array([[np.trace(_B2[k] @ R @ _B2[j] @ R.T) for j in range(5)] for k in range(5)])
    raise NotImplementedError("numeric representations are provided for l <= 2")


def rep_matrix(irreps: str, R: np.ndarray) -> np.ndarray:
    """Block-diagonal representation of R on a field with the given irreps."""
    blocks = [np.kron(np.eye(ir.mul), wigner(ir.l, R)) for ir in parse_irreps(irreps)]
    n = sum(b.shape[0] for b in blocks)
    M, o = np.zeros((n, n)), 0
    for b in blocks:
        M[o:o + b.shape[0], o:o + b.shape[0]] = b
        o += b.shape[0]
    return M


def schur_linear(irreps: str, rng: np.random.Generator, noise: float = 0.0, shared: bool = True) -> np.ndarray:
    """A random equivariant endomorphism of `irreps`, optionally with programming noise.

    shared=True perturbs each W_l once (one set of physical cells reused for all
    2l+1 components); shared=False perturbs every cell of each block independently.
    """
    blocks = []
    for ir in parse_irreps(irreps):
        d = 2 * ir.l + 1
        W = rng.normal(size=(ir.mul, ir.mul)) / math.sqrt(ir.mul)
        if shared:
            blocks.append(np.kron(W + noise * rng.normal(size=W.shape), np.eye(d)))
        else:
            B = np.kron(W, np.eye(d))
            blocks.append(B + noise * rng.normal(size=B.shape))
    n = sum(b.shape[0] for b in blocks)
    M, o = np.zeros((n, n)), 0
    for b in blocks:
        M[o:o + b.shape[0], o:o + b.shape[0]] = b
        o += b.shape[0]
    return M


def equivariance_error(M: np.ndarray, irreps: str, rng: np.random.Generator, trials: int = 16) -> float:
    """Mean relative commutator norm ||M D - D M|| / ||M|| over random rotations."""
    errs = []
    for _ in range(trials):
        D = rep_matrix(irreps, random_rotation(rng))
        errs.append(np.linalg.norm(M @ D - D @ M))
    return float(np.mean(errs) / max(np.linalg.norm(M), 1e-12))


_INTERTWINERS: dict[tuple[int, int, int], np.ndarray | None] = {}


def intertwiner(l1: int, l2: int, l3: int) -> np.ndarray | None:
    """The SO(3)-equivariant bilinear map V_l1 x V_l2 -> V_l3 in this module's
    bases, as a tensor C[i, j, k], or None if the triangle rule forbids it.

    Solved numerically: C must satisfy (D1 (x) D2) C = C D3 for every rotation,
    so it spans the null space of that constraint stacked over a few rotations.
    """
    key = (l1, l2, l3)
    if key in _INTERTWINERS:
        return _INTERTWINERS[key]
    if not (abs(l1 - l2) <= l3 <= l1 + l2) or max(key) > 2:
        _INTERTWINERS[key] = None
        return None
    d1, d2, d3 = 2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1
    rng = np.random.default_rng(1234)
    rows = []
    eye = np.eye(d1 * d2 * d3)
    for _ in range(4):
        R = random_rotation(rng)
        D1, D2, D3 = wigner(l1, R), wigner(l2, R), wigner(l3, R)
        # C'[i,j,k] = sum D1[i,a] D2[j,b] C[a,b,c] D3[k,c] must equal C
        T = np.einsum("ia,jb,kc->ijkabc", D1, D2, D3).reshape(d1 * d2 * d3, d1 * d2 * d3)
        rows.append(T - eye)
    _, s, vt = np.linalg.svd(np.vstack(rows))
    if s[-1] > 1e-8:
        _INTERTWINERS[key] = None
        return None
    C = vt[-1].reshape(d1, d2, d3)
    C = C / np.linalg.norm(C) * math.sqrt(d3)
    _INTERTWINERS[key] = C
    return C


def sph_harm_features(x: np.ndarray, lmax: int = 2) -> dict[int, np.ndarray]:
    """Equivariant features of a 3-vector for l = 0, 1, 2 in this module's bases."""
    x = np.asarray(x, float)
    out = {0: np.ones(1), 1: x.copy()}
    if lmax >= 2:
        S = np.outer(x, x) - np.eye(3) * (x @ x) / 3.0
        out[2] = np.array([np.sum(B * S) for B in _B2])
    return out
