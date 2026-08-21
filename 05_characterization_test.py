"""
Characterization test: before refactoring feature_gen.py into
feature_store.build_features(), capture its exact output as a golden
reference. After refactoring, regenerate on the SAME input and assert the
output matches to the last decimal - not "looks similar," bit-for-bit equal
on every column and every row.

This is the safety net referenced in the article: refactor first behind a
test like this, and a silent behavior change gets caught before it ships,
not after a downstream consumer complains.
"""
import sys
import pandas as pd

sys.path.insert(0, "sample_repo_before")
sys.path.insert(0, "sample_repo_after")


def run_before():
    import importlib
    import feature_gen as before_mod
    importlib.reload(before_mod)
    import os
    cwd = os.getcwd()
    os.chdir("sample_repo_before")
    try:
        df = before_mod.build_features()
    finally:
        os.chdir(cwd)
    return df.reset_index(drop=True)


def run_after():
    import importlib
    import feature_store as after_mod
    importlib.reload(after_mod)
    import os
    cwd = os.getcwd()
    os.chdir("sample_repo_after")
    try:
        df = after_mod.build_features()
    finally:
        os.chdir(cwd)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    before_df = run_before()
    after_df = run_after()

    print(f"BEFORE shape: {before_df.shape}, AFTER shape: {after_df.shape}")

    same_columns = list(before_df.columns) == list(after_df.columns)
    print(f"Same columns, same order: {same_columns}")

    try:
        pd.testing.assert_frame_equal(before_df, after_df, check_exact=True)
        print("PASS: refactored output is byte-for-byte identical to the original.")
    except AssertionError as e:
        print("FAIL: refactor changed behavior.")
        print(str(e)[:500])
