"""
A minimal, dependency-free ML technical debt auditor.

Scans a directory of .py files for two of the Google paper's debt categories
that are mechanically detectable without running the code:

  1. UNDECLARED CONSUMERS: string literals that look like shared data
     artifacts (paths ending in .csv/.json/.pkl/.parquet, or containing
     "data/"), counted by how many distinct files reference the SAME literal
     directly. A literal referenced directly by more than one file is an
     undeclared coupling - those files are consumers of an artifact with no
     declared interface, versioning, or schema contract between them.

  2. CONFIGURATION DEBT: numeric literals embedded directly in function
     calls or assignments outside of a designated config module. Excludes
     trivial values (0, 1, -1, True/False-adjacent) that are rarely "config"
     in intent.

Usage: python 03_audit_tool.py <path-to-repo>
"""
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

ARTIFACT_PATTERN = re.compile(r"^[\w./\\-]+\.(csv|json|pkl|parquet|joblib)$")
DATA_LIKE_PATTERN = re.compile(r"(^data/|^models/|^archive/|^logs/)")
CONFIG_FILE_NAMES = {"config.py", "settings.py"}
TRIVIAL_NUMBERS = {0, 1, -1, 0.0, 1.0, -1.0, 2}


def find_py_files(root: Path):
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def scan_undeclared_consumers(files):
    """Map literal artifact string -> set of files that reference it directly."""
    literal_to_files = defaultdict(set)
    for f in files:
        try:
            tree = ast.parse(f.read_text(), filename=str(f))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val = node.value
                if ARTIFACT_PATTERN.match(val) or DATA_LIKE_PATTERN.match(val):
                    literal_to_files[val].add(f.name)
    return literal_to_files


def scan_configuration_debt(files):
    """Count non-trivial numeric literals appearing outside a config module,
    per file, and return the total plus a per-file breakdown."""
    per_file_counts = {}
    total = 0
    for f in files:
        if f.name in CONFIG_FILE_NAMES:
            continue
        try:
            tree = ast.parse(f.read_text(), filename=str(f))
        except SyntaxError:
            continue
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if isinstance(node.value, bool):
                    continue
                if node.value in TRIVIAL_NUMBERS:
                    continue
                count += 1
        if count:
            per_file_counts[f.name] = count
            total += count
    return total, per_file_counts


def run_audit(repo_path: str, label: str):
    root = Path(repo_path)
    files = find_py_files(root)

    literal_map = scan_undeclared_consumers(files)
    undeclared = {lit: names for lit, names in literal_map.items() if len(names) > 1}

    config_total, config_by_file = scan_configuration_debt(files)

    print(f"\n=== Audit: {label} ({repo_path}) ===")
    print(f"Files scanned: {len(files)}")
    print(f"\nShared artifacts referenced by more than one file directly (undeclared consumers):")
    if not undeclared:
        print("  none found")
    for lit, names in sorted(undeclared.items()):
        print(f"  '{lit}' referenced directly by {len(names)} files: {sorted(names)}")

    print(f"\nConfiguration debt: {config_total} non-trivial numeric literals outside a config module")
    for fname, count in sorted(config_by_file.items(), key=lambda kv: -kv[1]):
        print(f"  {fname}: {count}")

    return {
        "files_scanned": len(files),
        "undeclared_consumer_artifacts": len(undeclared),
        "undeclared_consumer_files_total": sum(len(v) for v in undeclared.values()),
        "config_debt_count": config_total,
    }


if __name__ == "__main__":
    before = run_audit("sample_repo_before", "BEFORE refactor")
    after = run_audit("sample_repo_after", "AFTER refactor")

    print("\n=== Summary ===")
    print(f"Undeclared-consumer artifacts: before={before['undeclared_consumer_artifacts']}, "
          f"after={after['undeclared_consumer_artifacts']}")
    print(f"Total undeclared-consumer file references: before={before['undeclared_consumer_files_total']}, "
          f"after={after['undeclared_consumer_files_total']}")
    print(f"Configuration debt count: before={before['config_debt_count']}, after={after['config_debt_count']}")
