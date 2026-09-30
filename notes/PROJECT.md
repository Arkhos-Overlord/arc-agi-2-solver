# ARC-AGI-2 — Project Notes

## Hard rule: NEVER SUBMIT WITHOUT PERMISSION

Owner instruction, 2026-09-30: **never submit without explicit permission.
Always.**

This is binding on every agent and every future session, not just this one.

What this means concretely:

- Building, scoring, benchmarking, and drafting locally = **allowed, do it freely**
- Committing to git locally = allowed
- **Any outward-facing action = STOP and ask first.** That includes:
  - `kaggle competitions submit`
  - opening or updating a GitHub PR / issue / comment
  - claiming a bounty
  - any form submission, upload, or publish
- Permission is **per submission**, not standing. Having approved one submission
  does not authorise the next one.
- A near deadline is **not** an exception. "It closes in 20 minutes" is a reason
  to ask urgently, not a reason to act alone.
- If an unattended cron would ever submit, it must be disabled or gated behind
  an explicit approval flag. Nothing in this project may self-submit.

If you are unsure whether something counts as submitting: it does. Ask.

---

## Competition facts (verified 2026-09-30, primary sources)

Source: `https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2`
and `https://arcprize.org/competitions/2026`

| | |
|---|---|
| Prize pool | $700,000 |
| Progress Prizes | $275,000 (1st $75k → 8th $15k) — **leaderboard** |
| Grand Prize | $275,000 — best **Solution Writeup**, 6 judged criteria, *not* the top score |
| Bonus Prize | $150,000 — only unlocks at **85% accuracy**, split among top 5 |
| Teams | 2,232 (9,794 entrants, 20,996 submissions) |
| Entry deadline | **2026-10-26** — rules must be accepted by here |
| Team merger | 2026-10-26 |
| Final submission | **2026-11-02** |
| Winners announced | 2026-12-04 |
| Submission format | **Kaggle Notebook** (code competition) |
| GPU | Kaggle L4x4, 96GB VRAM available |

### Constraints that shape the design

1. **No internet during evaluation.** No GPT/Claude/API calls at inference.
   The solution must be **self-contained and offline**.
2. **Open source is mandatory** to be eligible for any prize, across all
   tracks, and must happen *before* private evaluation scores are released.
3. **2 attempts per test output.** Submit the top 2 candidate programs — free
   accuracy, and the solver already ranks them.
4. Scoring is **per test output**, not per task: `1` if either of the 2
   attempts matches exactly, else `0`. Sum divided by total test outputs.

## Data (already on disk, no Kaggle access required)

| Split | Count | Notes |
|---|---|---|
| `data/training/` | **1,000** | public, for development |
| `data/evaluation/` | **120** | public, **all 120 ship answers** → offline scoring |
| Kaggle test set | — | blocked (403) until rules accepted |

Grids: max 30×30, colours 0–9, `0` = background.
2–10 train pairs (avg 3.2); 1–4 test inputs (avg 1.1).

**Reference points (official):** human average on public eval **66%**,
Kaggle leader **83.06%**.

## Layout

```
arc-agi-2/
  data/training, data/evaluation   1120 public tasks
  solver/prims.py                  30 grid primitives
  solver/search.py                 BFS program search, all-pairs verification
  score.py                         local scorer using the exact Kaggle metric
  notes/PROJECT.md                 this file
```

## Method

An ARC solution is program synthesis, not a model call. Search compositions of
primitives, keep only those reproducing **every** training pair exactly
(dimensions included) — that constraint eliminates almost everything fast.
Rank survivors, submit the top 2.

Current baseline is depth-2 search over 30 primitives. That is a starting
point, not a ceiling — depth 3+ and a wider DSL are where the score comes from.


## Environment findings (2026-09-30)

- **Interpreter split:** `python` = Hermes' 3.14.7; `pip` = the hermes-agent
  venv on **3.11**. Installing with bare `pip` puts packages where `python`
  cannot see them. Use `python -m pip` for the 3.14 interpreter, or invoke the
  venv python directly.
- **torch 2.5.1+cu121 was already installed** in the hermes-agent venv with
  CUDA working (GTX 1650, 4.29 GB). No install was needed - the earlier
  `ModuleNotFoundError` was a path problem, not a missing package.
- **Stdlib shadowing:** a stray `C:/Users/sydne/inspect.py` (a Reddit debug
  script) shadowed the stdlib `inspect` module for EVERY python process started
  from the home directory, which broke both torch and numpy imports. Moved to
  `C:/Users/sydne/_stray-scripts-review/reddit_debug_inspect.py`.
  Lesson: never name a scratch script after a stdlib module.
- **VRAM is 4.29 GB.** That constrains model size. TRM (7M) fits comfortably;
  75M+ needs gradient checkpointing or CPU offload.

## Measured results

| Experiment | Result |
|---|---|
| DSL search, 10 eval tasks, depth 2 | **0.00%** (0 programs found) |
| Search machinery self-test | **5/5 pass** - machinery proven correct |
| DSL coverage, 1000 training tasks, depth 1 | **1.90%** (19 solved), 0.6s |
| DSL coverage, 1000 training tasks, depth 2 | **2.60%** (26 solved), 31s |

Depth 1 -> 2 cost 50x the compute for +0.7pp. The ceiling is the DSL's
vocabulary, not the search depth. Both figures sit below the ARC authors' 5%
"not meaningful" floor. DSL-only search is abandoned on this evidence.
