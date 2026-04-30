# Assignment 1: Mesh Fairing and Constrained Editing

This starter package contains a small noisy triangle mesh, handle constraints, and Python scaffolding for implementing mesh fairing with hard and soft constraints.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## What to implement

Fill the `TODO` sections in:

- `src/operators.py`
- `src/solver.py`

Then run:

```bash
python src/run_assignment1.py --operator cotan --lambda-value 0.03 --soft-mu 20
```

This should write figures and `.obj` files to `outputs/`.

## Submission checklist

Submit a zip file containing:

1. your completed source code,
2. 4--6 figures showing your results,
3. a 1--2 page report using `report_template.md` as a guide,
4. generated meshes (`.obj`) for at least your unconstrained fairing, hard-constrained editing, and soft-constrained editing results.

Do not submit your virtual environment or large cache folders.
