"""Optimization solvers for Assignment 1.

Fill the TODO sections. All solvers act on X in R^{n x 3}; solve each coordinate
with the same sparse matrix.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def constraint_matrix(indices: np.ndarray, n_vertices: int) -> sp.csr_matrix:
    """Return C such that C @ X extracts rows X[indices]."""
    indices = np.asarray(indices, dtype=np.int64)
    rows = np.arange(len(indices), dtype=np.int64)
    vals = np.ones(len(indices), dtype=float)
    return sp.coo_matrix((vals, (rows, indices)), shape=(len(indices), n_vertices)).tocsr()


def fairing_solve(X0: np.ndarray, L: sp.spmatrix, mass: np.ndarray, lambda_value: float) -> np.ndarray:
    """Solve the unconstrained fairing system.

    Target equation:
        (M + lambda L) X = M X0
    where M is diagonal with entries `mass`.
    """
    # TODO: assemble Q = M + lambda_value * L and rhs = M X0.
    # Hint: use sp.diags(mass) and spla.spsolve.
    raise NotImplementedError("Implement fairing_solve.")


def hard_constrained_solve(
    X0: np.ndarray,
    L: sp.spmatrix,
    mass: np.ndarray,
    lambda_value: float,
    indices: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Solve the equality-constrained fairing problem.

    Target KKT system:
        [ Q  C^T ] [ X      ] = [ M X0 ]
        [ C   0  ] [ Lambda ]   [ D    ]
    where Q = M + lambda L and C X = D extracts constrained vertices.
    """
    # TODO: build C with constraint_matrix, assemble the KKT block matrix,
    # and solve for X. You may solve all 3 coordinates at once.
    raise NotImplementedError("Implement hard_constrained_solve.")


def soft_constrained_solve(
    X0: np.ndarray,
    L: sp.spmatrix,
    mass: np.ndarray,
    lambda_value: float,
    indices: np.ndarray,
    targets: np.ndarray,
    soft_mu: float,
) -> np.ndarray:
    """Solve fairing plus quadratic penalties for the constraints.

    Target equation:
        (Q + mu C^T C) X = M X0 + mu C^T D.
    """
    # TODO: implement the soft-constrained linear system.
    raise NotImplementedError("Implement soft_constrained_solve.")


def constraint_residual(X: np.ndarray, indices: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Return max and RMS residual at constrained vertices."""
    r = X[indices] - targets
    row_norm = np.linalg.norm(r, axis=1)
    return {"max": float(np.max(row_norm)), "rms": float(np.sqrt(np.mean(row_norm**2)))}
