"""Plotting helpers for Assignment 1."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def _triangulation(V: np.ndarray, F: np.ndarray) -> Triangulation:
    return Triangulation(V[:, 0], V[:, 1], F)


def plot_height_mesh(
    V: np.ndarray,
    F: np.ndarray,
    path: str | Path,
    title: str,
    handles: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save a 2D height-colored mesh plot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tri = _triangulation(V, F)
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    im = ax.tripcolor(tri, V[:, 2], shading="gouraud", vmin=vmin, vmax=vmax)
    ax.triplot(tri, linewidth=0.25, alpha=0.22)
    if handles is not None and len(handles):
        ax.scatter(V[handles, 0], V[handles, 1], s=50, marker="o", edgecolors="black", linewidths=1.0)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="height z")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lambda_sweep(Vs: list[np.ndarray], F: np.ndarray, lambdas: list[float], path: str | Path) -> None:
    """Save a row of height-colored meshes for different lambda values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_z = np.concatenate([V[:, 2] for V in Vs])
    vmin, vmax = float(all_z.min()), float(all_z.max())
    fig, axes = plt.subplots(1, len(Vs), figsize=(4.0 * len(Vs), 4.0), constrained_layout=True)
    if len(Vs) == 1:
        axes = [axes]
    im = None
    for ax, V, lam in zip(axes, Vs, lambdas):
        tri = _triangulation(V, F)
        im = ax.tripcolor(tri, V[:, 2], shading="gouraud", vmin=vmin, vmax=vmax)
        ax.triplot(tri, linewidth=0.18, alpha=0.18)
        ax.set_aspect("equal")
        ax.set_title(f"lambda={lam:g}")
        ax.set_xticks([])
        ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.75, label="height z")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_hard_soft_comparison(
    X0: np.ndarray,
    F: np.ndarray,
    X_hard: np.ndarray,
    X_soft: np.ndarray,
    indices: np.ndarray,
    path: str | Path,
) -> None:
    """Save a side-by-side comparison of hard and soft constraints."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    all_z = np.concatenate([X0[:, 2], X_hard[:, 2], X_soft[:, 2]])
    vmin, vmax = float(all_z.min()), float(all_z.max())
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), constrained_layout=True)
    for ax, V, title in zip(axes, [X0, X_hard, X_soft], ["Input", "Hard constraints", "Soft constraints"]):
        tri = _triangulation(V, F)
        im = ax.tripcolor(tri, V[:, 2], shading="gouraud", vmin=vmin, vmax=vmax)
        ax.triplot(tri, linewidth=0.18, alpha=0.18)
        ax.scatter(V[indices, 0], V[indices, 1], s=16, edgecolors="black", linewidths=0.6)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="height z")
    fig.savefig(path, dpi=180)
    plt.close(fig)
