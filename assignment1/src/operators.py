"""Discrete geometry operators for Assignment 1.

Fill the TODO sections. Keep these conventions:
- L is a positive-semidefinite stiffness/Laplacian matrix satisfying L @ 1 = 0.
- Cotangent weights build a stiffness matrix for integrated gradient energy.
- The pointwise Laplace-Beltrami operator would be M^{-1} L, where M is the mass matrix.
"""
from __future__ import annotations

from collections import Counter
import numpy as np
import scipy.sparse as sp


def unique_edges(F: np.ndarray) -> np.ndarray:
    """Return sorted unique undirected edges from triangular faces."""
    edges = set()
    for i, j, k in F:
        for a, b in [(i, j), (j, k), (k, i)]:
            if a > b:
                a, b = b, a
            edges.add((int(a), int(b)))
    return np.asarray(sorted(edges), dtype=np.int64)


def boundary_vertices(F: np.ndarray) -> np.ndarray:
    """Return vertices incident on boundary edges."""
    counts: Counter[tuple[int, int]] = Counter()
    for i, j, k in F:
        for a, b in [(i, j), (j, k), (k, i)]:
            if a > b:
                a, b = b, a
            counts[(int(a), int(b))] += 1
    verts = sorted({v for edge, c in counts.items() if c == 1 for v in edge})
    return np.asarray(verts, dtype=np.int64)


def _assemble_laplacian(n_vertices: int, edge_weights: dict[tuple[int, int], float]) -> sp.csr_matrix:
    """Assemble L from undirected edge weights.

    For each edge (i,j) with weight w:
        L_ii += w, L_jj += w, L_ij -= w, L_ji -= w.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    diag = np.zeros(n_vertices, dtype=float)
    for (i, j), w in edge_weights.items():
        diag[i] += w
        diag[j] += w
        rows += [i, j]
        cols += [j, i]
        vals += [-w, -w]
    rows.extend(range(n_vertices))
    cols.extend(range(n_vertices))
    vals.extend(diag.tolist())
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()


def uniform_laplacian(n_vertices: int, F: np.ndarray) -> sp.csr_matrix:
    """Uniform graph Laplacian: one unit weight per mesh edge."""
    # TODO: build a dictionary mapping each unique edge (i,j) to weight 1.0.
    # Hint: use unique_edges(F) and _assemble_laplacian(...).
    edges = unique_edges(F)
    edge_weights = dict()
    edge_weights[(edges)] = 1.0
    return _assemble_laplacian(n_vertices, edge_weights)

def _cotangent(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    """cot(angle between u and v)."""
    cross_norm = np.linalg.norm(np.cross(u, v))
    if cross_norm < eps:
        return 0.0
    return float(np.dot(u, v) / cross_norm)


def cotan_laplacian(V: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
    """Cotangent stiffness matrix.

    For an interior edge (i,j), the edge weight is
        0.5 * (cot(alpha_ij) + cot(beta_ij)),
    where alpha and beta are the angles opposite the edge.

    Do not include a 1/area factor here. Area normalization belongs in M^{-1} L
    when you need a pointwise Laplace-Beltrami operator.
    """
    # TODO: accumulate one 0.5*cot contribution for each triangle edge.
    # Hint for face (i,j,k):
    #   angle at i is opposite edge (j,k), etc.
    #   use _cotangent(vj-vi, vk-vi).
    raise NotImplementedError("Implement cotan_laplacian.")


def lumped_mass(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Return a diagonal/lumped vertex mass vector.

    Each triangle should contribute one third of its area to each of its vertices.
    """
    # TODO: compute triangle areas and scatter area/3 to the three vertices.
    raise NotImplementedError("Implement lumped_mass.")


def operator_diagnostics(L: sp.spmatrix) -> dict[str, float]:
    """Small numerical checks for a Laplacian/stiffness matrix."""
    n = L.shape[0]
    ones = np.ones(n)
    return {
        "symmetry_error": float(sp.linalg.norm((L - L.T).tocsr())),
        "constant_residual": float(np.linalg.norm(L @ ones)),
    }
