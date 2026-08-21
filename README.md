# ml-technical-debt-code

Runnable evidence for three ML technical debt failure modes: entanglement, hidden feedback loops, and undeclared consumers. Plus the static auditor, scoring framework, and characterization tests that catch them.

![Python Version](https://img.shields.io/badge/python-3.12-blue) ![License](https://img.shields.io/badge/license-MIT-green)

Most articles on ML technical debt stop at defining the categories from Google's 2015 paper. This repo runs them: two simulations with planted ground truth, a dependency-free static auditor, a scoring framework, and a characterization test that catches a real, silent regression.

Read the full write-up on EmiTechLogic: [ML Technical Debt: How to Identify, Measure, and Pay It Down](https://emitechlogic.com/ml-technical-debt-how-to-identify-measure-and-pay-it-down/)

## What It Does

```
Simulations                Static analysis              Safety net
------------                ---------------              ----------
01_entanglement.py    ->    03_audit_tool.py       ->    05_characterization_test.py
02_feedback_loop.py         04_scoring_framework.py      06_characterization_test_catches_bug.py
     |                            |                             |
     v                            v                             v
 coefficient/prediction      ranked debt backlog          passing refactor +
 shift under CACE            (severity x blast /          a broken one caught
                              effort)                       mid-flight
```

Six scripts, one `run_all.py` entry point:

| Script | What It Demonstrates |
| --- | --- |
| `01_entanglement.py` | Drops a correlated feature from a trained logistic regression and retrains, measuring how far the *other* coefficients and predictions move (CACE) |
| `02_feedback_loop.py` | A greedy vs. epsilon-greedy recommender over 60 rounds, tracking exposure concentration (Gini) and overlap with the true best items |
| `03_audit_tool.py` | A ~100-line, dependency-free `ast`-based scanner for undeclared consumers (shared paths with no declared interface) and configuration debt (magic numbers outside config) |
| `04_scoring_framework.py` | Turns the auditor's findings into a ranked backlog using `(Severity x Blast Radius) / Effort` |
| `05_characterization_test.py` | Confirms a refactor's output matches the original exactly, byte for byte, on a fixed input |
| `06_characterization_test_catches_bug.py` | Runs the identical test against a deliberately broken refactor and shows it catching a silent regression |

## Installation

```bash
git clone https://github.com/Emmimal/ml-technical-debt-code.git
cd ml-technical-debt-code
pip install -r requirements.txt
```

No exotic dependencies. NumPy, pandas, scikit-learn, pytest, and Flask (only imported by the sample repo's serving script, never executed directly).

## Quick Start

Run everything in article order with one command:

```bash
python run_all.py
```

That executes all six scripts and prints every number quoted in the article, in sequence: entanglement, feedback loop, audit, scoring, then both characterization tests.

To run a single script instead:

```bash
python 01_entanglement.py
```

## Sample Results

Real output from a seed=42 run, Python 3.12.3, NumPy 2.4.4, scikit-learn 1.8.0:

**Entanglement.** Dropping one feature from a correlated set and retraining:

| Condition | Mean coefficient shift | Mean prediction shift |
| --- | --- | --- |
| Correlated features | 110.1% | 0.0323 |
| Independent features | 10.5% | 0.0849 |

**Feedback loop.** 200 items, 60 rounds, 30,000 simulated users:

| Metric | Greedy | Exploring (15%) |
| --- | --- | --- |
| Items ever shown (of 200) | 12 | 66 |
| Overlap with true top-10 | 0 / 10 | 2 / 10 |

**Static audit.** Same 5-file sample pipeline, before and after a refactor:

| Check | Before | After |
| --- | --- | --- |
| Shared artifacts referenced by 2+ files | 4 | 0 |
| Configuration debt (magic numbers) | 13 | 3 |

**Characterization test.** A one-line refactor slip (`rolling(7)` without `min_periods=1`) caught mid-flight:

```
DataFrame.iloc[:, 6] (column name="rolling_7d") values are different (58.68 %)
rolling_7d NaN count - before: 0, broken after: 2934
```

## Project Structure

```
ml-technical-debt-code/
├── 01_entanglement.py                       # CACE simulation
├── 02_feedback_loop.py                      # hidden feedback loop simulation
├── 03_audit_tool.py                         # static auditor (ast-based)
├── 04_scoring_framework.py                  # severity x blast / effort backlog
├── 05_characterization_test.py              # passing refactor check
├── 06_characterization_test_catches_bug.py  # broken refactor caught
├── run_all.py                               # runs all six in order
├── requirements.txt
├── sample_repo_before/                      # the pipeline as audited (unrefactored)
├── sample_repo_after/                       # after centralizing config + declared interface
└── sample_repo_after_broken/                # after_after, with the rolling-window slip
```

`sample_repo_before/`, `sample_repo_after/`, and `sample_repo_after_broken/` are small, self-contained fraud-scoring pipelines (feature generation, training, batch scoring, a Flask serving stub, and a legacy orchestration script). They exist purely as fixtures for the auditor and the characterization tests, not as a real pipeline meant to be deployed.

## When to Use This

Useful as:
- A working reference for what CACE and hidden feedback loops actually look like in code, not just in the abstract
- A starting point for a static auditor tailored to your own repo's artifact-path and config conventions
- A template for adding characterization tests before a refactor on a pipeline with no existing test coverage

Not intended as:
- A production dependency-scanning tool. The auditor is a deliberately narrow heuristic, not a complete dependency analyzer, and it can miss dynamically constructed paths
- A general-purpose recommender system or scoring rubric. Both simulations are built to isolate one failure mode clearly, not to model a real catalog or a real severity policy

## Known Limitations

- The auditor matches string literals exactly. A path built with `Path(DATA_DIR) / filename` or an f-string won't be caught.
- The scoring framework's severity and blast-radius values are illustrative judgment calls made for this article's findings, not a general-purpose rubric. Swap in your own before using it on a real backlog.
- The entanglement simulation uses a 10-feature logistic regression as a minimal, fully controllable example. Real production models with hundreds of features and nonlinear interactions will show the same underlying effect but not necessarily the same magnitude.
- The feedback-loop simulation is a single-slate top-K recommender. It doesn't model session-level ranking, diversity constraints, or multi-objective scoring that many production systems add specifically to counter this failure mode.

## License

MIT, see [LICENSE](LICENSE).
