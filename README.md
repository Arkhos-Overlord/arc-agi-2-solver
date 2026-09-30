# ARC-AGI-2 — Solver

Working notes and measured results for competing in
[ARC Prize 2026 – ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2).

> **Never submit without explicit permission.** This project builds and scores
> locally. Any outward action — `kaggle competitions submit`, opening a PR,
> publishing a writeup — requires the owner to ask for it first, every time.
> See `notes/PROJECT.md`.

## Data

1,000 training + 120 evaluation tasks, public and free — no Kaggle access
required:

```bash
bash fetch_data.sh
```

## Current state

| Approach | Result |
|---|---|
| DSL brute-force search, 1000 training tasks, depth 1 | 1.90% |
| DSL brute-force search, 1000 training tasks, depth 2 | 2.60% |
| Search machinery self-test | 5/5 pass |
| TTT transformer, 4 eval tasks | 0.00% (in progress) |

The DSL path is **abandoned on evidence**: depth 1 → 2 cost 50× the compute for
+0.7pp, and both numbers sit below the ARC authors' 5% "not meaningful" floor.
The live work is test-time training.

## Layout

```
solver/prims.py      30 grid primitives (DSL approach, kept as baseline)
solver/search.py     BFS program search with all-pairs verification
solver/selftest.py   5 checks proving the search machinery is correct
solver/augment.py    dihedral + colour-permutation views, invertible
ttt.py               test-time training solver (current focus)
solve_aug.py         augmentation-stability selection over DSL candidates
coverage.py          DSL coverage sweep + primitive effectiveness report
score.py             local scorer using the exact Kaggle metric
diag_seq.py          sequence-length diagnostic (why tasks were rejected)
fetch_data.sh        reproducible dataset fetch
```

## Running

```bash
# environment note: torch lives in the hermes-agent venv on python 3.11,
# while bare `python` is 3.14.7. Use the venv interpreter for anything
# needing torch.
VP="/c/Users/sydne/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe"

python solver/selftest.py                        # prove the machinery works
python coverage.py --split training --depth 2    # DSL coverage report
"$VP" ttt.py --limit 4 --steps 150 --aug 6      # TTT solver
```

## Scoring

`score.py` and `ttt.py` both report the **exact Kaggle metric**: per test
output, `1` if either of the 2 attempts matches exactly, else `0`, summed and
divided by total test outputs. A local number means what a leaderboard number
means.

Reference points (official): human average on public eval **66%**,
Kaggle leaderboard **83.06%**, ARC-AGI-2 2025 winner (NVARC) **24.03%**,
and the authors' floor for "meaningful" is **5%**.

## Key constraints

- No internet during Kaggle evaluation — no GPT/Claude/API calls
- Pre-trained models ARE allowed, if they run locally
- Open sourcing is mandatory for prize eligibility (CC0 / MIT-0 in practice)
- In 2025 the top scorer (55.5%) was **ineligible** for not open-sourcing;
  the winner at 53.5% open-sourced and took the prize

## Environment gotchas (cost real time — recorded so they cost none again)

- **Interpreter split.** `python` is Hermes' 3.14.7; `pip` is the
  hermes-agent venv on **3.11**. `pip install X` puts packages where `python`
  cannot import them. Use `python -m pip`, or call the venv interpreter.
- **A stray `inspect.py` in the home directory** once shadowed the stdlib for
  every Python process started from `~`, breaking both torch and numpy. It has
  been moved to `~/\_stray-scripts-review\`. Never name a scratch script after
  a stdlib module.
- **torch 2.5.1+cu121 with CUDA** (GTX 1650, 4.29 GB) is already installed in
  the venv. It was never missing.
- **Sequence length is the real memory constraint.** A 4-pair leave-one-out
  context on 30×30 grids needs up to **3,693 cells**; a single pair can need
  ~1,800. This is why naive max-cells settings silently reject most tasks.
