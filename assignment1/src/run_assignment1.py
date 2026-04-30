"""Run the completed Assignment 1 example solution.

Example:
    python src/run_assignment1.py --operator cotan --lambda-value 0.03 --soft-mu 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from mesh_io import load_obj, write_obj
from operators import (
    boundary_vertices,
    cotan_laplacian,
    lumped_mass,
    operator_diagnostics,
    uniform_laplacian,
)
from solver import fairing_solve, hard_constrained_solve, soft_constrained_solve, constraint_residual
from visualize import plot_height_mesh, plot_lambda_sweep, plot_hard_soft_comparison


def load_constraints(X0: np.ndarray, F: np.ndarray, anchors_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with anchors_path.open("r") as f:
        anchors = json.load(f)

    indices: list[int] = []
    targets: list[np.ndarray] = []

    boundary = boundary_vertices(F)
    if anchors.get("boundary_lock", True):
        for idx in boundary:
            indices.append(int(idx))
            targets.append(X0[idx].copy())

    handle_indices: list[int] = []
    for h in anchors.get("handles", []):
        idx = int(h["index"])
        handle_indices.append(idx)
        indices.append(idx)
        targets.append(np.asarray(h["target"], dtype=float))

    return np.asarray(indices, dtype=np.int64), np.asarray(targets, dtype=float), np.asarray(handle_indices, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", default="data/noisy_heightfield.obj")
    parser.add_argument("--anchors", default="data/anchors.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--operator", choices=["uniform", "cotan"], default="cotan")
    parser.add_argument("--lambda-value", type=float, default=0.03)
    parser.add_argument("--soft-mu", type=float, default=20.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X0, F = load_obj(args.mesh)
    mass = lumped_mass(X0, F)
    if args.operator == "uniform":
        L = uniform_laplacian(X0.shape[0], F)
    else:
        L = cotan_laplacian(X0, F)

    diagnostics = operator_diagnostics(L)
    print("operator diagnostics:", diagnostics)

    lambdas = [0.0, 0.005, 0.03, 0.15]
    Xs = [fairing_solve(X0, L, mass, lam) for lam in lambdas]
    plot_lambda_sweep(Xs, F, lambdas, out / "fig1_lambda_sweep.png")
    for lam, X in zip(lambdas, Xs):
        write_obj(out / f"fairing_lambda_{lam:g}.obj", X, F)

    indices, targets, handle_indices = load_constraints(X0, F, Path(args.anchors))
    X_hard = hard_constrained_solve(X0, L, mass, args.lambda_value, indices, targets)
    X_soft = soft_constrained_solve(X0, L, mass, args.lambda_value, indices, targets, args.soft_mu)

    plot_height_mesh(X0, F, out / "fig0_input_mesh.png", "Input noisy mesh", handles=handle_indices)
    plot_height_mesh(X_hard, F, out / "fig2_hard_constraints.png", "Hard-constrained editing", handles=handle_indices)
    plot_height_mesh(X_soft, F, out / "fig3_soft_constraints.png", "Soft-constrained editing", handles=handle_indices)
    plot_hard_soft_comparison(X0, F, X_hard, X_soft, indices, out / "fig4_hard_vs_soft.png")

    write_obj(out / "hard_constrained.obj", X_hard, F)
    write_obj(out / "soft_constrained.obj", X_soft, F)

    hard_res = constraint_residual(X_hard, indices, targets)
    soft_res = constraint_residual(X_soft, indices, targets)
    metrics = {
        "operator": args.operator,
        "lambda": args.lambda_value,
        "soft_mu": args.soft_mu,
        "operator_diagnostics": diagnostics,
        "hard_constraint_residual": hard_res,
        "soft_constraint_residual": soft_res,
    }
    with (out / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
