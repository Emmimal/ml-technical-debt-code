import sys
import os
import importlib
import pandas as pd

sys.path.insert(0, "sample_repo_before")


def run_before():
    import feature_gen as before_mod
    importlib.reload(before_mod)
    cwd = os.getcwd()
    os.chdir("sample_repo_before")
    try:
        df = before_mod.build_features()
    finally:
        os.chdir(cwd)
    return df.reset_index(drop=True)


def run_broken_after():
    sys.path.insert(0, "sample_repo_after_broken")
    import feature_store as broken_mod
    importlib.reload(broken_mod)
    cwd = os.getcwd()
    os.chdir("sample_repo_after_broken")
    try:
        df = broken_mod.build_features()
    finally:
        os.chdir(cwd)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    before_df = run_before()
    broken_df = run_broken_after()

    print(f"BEFORE shape: {before_df.shape}, BROKEN AFTER shape: {broken_df.shape}")

    try:
        pd.testing.assert_frame_equal(before_df, broken_df, check_exact=True)
        print("PASS: identical (unexpected)")
    except AssertionError as e:
        print("FAIL: refactor changed behavior, as expected. Details:")
        print(str(e)[:800])

    nan_before = before_df["rolling_7d"].isna().sum()
    nan_after = broken_df["rolling_7d"].isna().sum()
    print(f"\nrolling_7d NaN count - before: {nan_before}, broken after: {nan_after}")
