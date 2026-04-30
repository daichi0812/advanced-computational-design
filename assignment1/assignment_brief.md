# Assignment 1: Mesh Fairing and Constrained Editing

## Goal

In Weeks 1--4, we described computational design problems using variables, objectives, constraints, sparse matrices, and discrete geometry. In this assignment, you will implement a small version of that pipeline on a triangle mesh.

You will turn the design request

> Smooth a noisy mesh, but keep boundary and handle constraints under control.

into an optimization problem.

## Data

The starter mesh is `data/noisy_heightfield.obj`. The handle constraints are stored in `data/anchors.json`.

The mesh is a synthetic noisy heightfield. It is small enough that sparse direct solvers should run quickly.

## Mathematical model

Let

- `X0 in R^{n x 3}` be the input vertex positions,
- `X in R^{n x 3}` be the optimized vertex positions,
- `M` be a lumped diagonal mass matrix,
- `L` be either a uniform graph Laplacian or cotangent stiffness matrix.

The basic fairing objective is

```text
min_X  1/2 ||X - X0||_M^2 + lambda/2 tr(X^T L X)
```

which gives the sparse linear system

```text
(M + lambda L) X = M X0.
```

For hard constraints, let `C X = D` encode pinned boundary vertices and handle targets. Solve

```text
min_X  1/2 ||X - X0||_M^2 + lambda/2 tr(X^T L X)
subject to C X = D.
```

You may use either elimination or the KKT system:

```text
[ Q  C^T ] [ X     ] = [ M X0 ]
[ C   0  ] [ Lambda]   [ D    ]
```

where `Q = M + lambda L`.

For soft constraints, solve

```text
min_X  1/2 ||X - X0||_M^2 + lambda/2 tr(X^T L X)
       + mu/2 ||C X - D||^2.
```

This gives

```text
(Q + mu C^T C) X = M X0 + mu C^T D.
```

## Required tasks

### Part A: Discrete operators

Implement:

1. uniform graph Laplacian,
2. cotangent stiffness matrix,
3. lumped vertex mass matrix.

Check that `L @ ones` is approximately zero.

### Part B: Unconstrained mesh fairing

Solve the fairing problem for several `lambda` values. Show how the result changes as `lambda` increases.

### Part C: Hard-constrained editing

Use boundary vertices and the provided handles as hard constraints. Solve using KKT or elimination. Verify that the constrained vertices match their targets to numerical precision.

### Part D: Soft-constrained editing

Solve the same editing problem using a soft penalty. Compare with the hard-constrained result. Report the maximum constraint error for both methods.

## Report prompts

Your short report should answer:

1. What are the design variables?
2. What objective did you optimize?
3. Which requirements were hard constraints and which were soft penalties?
4. How did changing `lambda` affect the output?
5. How different were hard and soft constraints numerically and visually?
6. What role did the mass matrix or cotangent weights play in making the operator resolution-aware?

## Suggested grading breakdown

- Discrete operators: 25%
- Fairing solver: 20%
- Hard and soft constraints: 30%
- Figures and numerical checks: 15%
- Short report clarity: 10%
