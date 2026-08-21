"""
Debt Score = (Severity x Blast Radius) / Effort

  Severity     1-5, how bad is the failure mode if this debt bites you
               (silent wrong predictions > slow iteration > code ugliness)
  Blast Radius 1-5, how much of the system this issue touches. Capped on
               the same 1-5 scale as severity and effort on purpose: a raw
               count (4 files vs. an entire 200-item catalog) isn't
               comparable across category types, and letting raw counts
               drive the score lets whichever category happens to use the
               biggest denominator dominate the backlog for the wrong
               reason. The count from the audit tool still decides WHERE on
               the 1-5 scale a finding lands, it just doesn't get to set
               the scale itself.
  Effort       1-5, estimated cost to fix (1 = a config move, 5 = a
               re-architecture)

Higher score = fix first. This is a rubric, not a law of physics - severity,
blast radius, and effort are judgment calls a team makes together, informed
by (not dictated by) what the audit tool and simulations measured.
"""

findings = [
    # name, category, severity, blast_radius(1-5), effort, source
    ("data/features.csv read directly by 4 files", "Undeclared consumers", 5, 4, 2, "audit tool"),
    ("model_v3_final_FINAL.pkl read directly by 3 files", "Undeclared consumers", 4, 3, 2, "audit tool"),
    ("scaler_v3.pkl read directly by 3 files", "Undeclared consumers", 4, 3, 2, "audit tool"),
    ("scores_output.csv read directly by 2 files", "Undeclared consumers", 2, 2, 1, "audit tool"),
    ("13 magic numbers outside config", "Configuration debt", 3, 4, 1, "audit tool"),
    ("Correlated features: 110% coef shift on feature drop", "Entanglement (CACE)", 5, 4, 3, "simulation"),
    ("Greedy recommender locked to 12/200 items, 0/10 true-best overlap", "Hidden feedback loop", 5, 5, 3, "simulation"),
    ("run_pipeline_legacy.py chains stages via subprocess + file drops", "Glue code / pipeline jungle", 3, 3, 2, "manual review"),
]

if __name__ == "__main__":
    scored = []
    for name, category, sev, blast, effort, source in findings:
        score = (sev * blast) / effort
        scored.append((score, name, category, sev, blast, effort, source))

    scored.sort(key=lambda r: -r[0])

    print(f"{'Score':>7} | {'Category':<28} | {'Sev':>3} | {'Blast':>5} | {'Effort':>6} | Finding")
    print("-" * 110)
    for score, name, category, sev, blast, effort, source in scored:
        print(f"{score:7.1f} | {category:<28} | {sev:>3} | {blast:>5} | {effort:>6} | {name}")
