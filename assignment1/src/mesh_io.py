"""Minimal OBJ utilities for Assignment 1."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and triangular faces from a simple OBJ file.

    Supports lines of the form `v x y z` and `f i j k`. Texture/normal suffixes
    in face records are ignored if present.
    """
    path = Path(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                tri = []
                for token in parts[1:4]:
                    tri.append(int(token.split("/")[0]) - 1)
                faces.append(tri)
    V = np.asarray(vertices, dtype=float)
    F = np.asarray(faces, dtype=np.int64)
    if F.shape[1] != 3:
        raise ValueError("This assignment expects a triangle mesh.")
    return V, F


def write_obj(path: str | Path, V: np.ndarray, F: np.ndarray) -> None:
    """Write a triangle mesh to OBJ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Assignment 1 output mesh\n")
        for v in V:
            f.write("v {:.8f} {:.8f} {:.8f}\n".format(*v))
        for tri in F:
            f.write("f {} {} {}\n".format(*(tri + 1)))
