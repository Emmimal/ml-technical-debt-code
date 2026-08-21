"""
Hidden feedback loop demonstration.

A simple recommender: N items with a true latent quality score. Each round,
the system recommends the top-K items by ESTIMATED quality (based on past
observed engagement), users engage probabilistically as a function of true
quality, and those engagement counts feed back into next round's estimate.

Two policies:
  - GREEDY: always show the current top-K by estimated quality.
  - EXPLORING: epsilon-greedy, occasionally shows a random item outside top-K.

Tracks the Gini coefficient of cumulative exposure across items each round to
quantify how concentrated recommendations become over time - the hidden
feedback loop the Google paper describes, where the model's own outputs
shape the training data it will see next.
"""
import numpy as np

SEED = 42
N_ITEMS = 200
N_USERS_PER_ROUND = 500
N_ROUNDS = 60
TOP_K = 10
EPSILON = 0.15


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def run(policy: str, rng):
    true_quality = rng.beta(2, 5, size=N_ITEMS)  # most items mediocre, few great
    shown_count = np.zeros(N_ITEMS)
    engaged_count = np.zeros(N_ITEMS)
    gini_over_time = []

    for _ in range(N_ROUNDS):
        # Estimated quality = observed engagement rate, with a small prior
        # to avoid divide-by-zero for never-shown items.
        est_quality = (engaged_count + 1) / (shown_count + 5)

        if policy == "greedy":
            ranked = np.argsort(-est_quality)
            recommend_set = ranked[:TOP_K]
        else:  # exploring
            ranked = np.argsort(-est_quality)
            n_explore = max(1, int(TOP_K * EPSILON))
            exploit_set = ranked[: TOP_K - n_explore]
            remaining = ranked[TOP_K - n_explore :]
            explore_set = rng.choice(remaining, size=n_explore, replace=False)
            recommend_set = np.concatenate([exploit_set, explore_set])

        for item in recommend_set:
            n_users = N_USERS_PER_ROUND // TOP_K
            shown_count[item] += n_users
            engagements = rng.binomial(n_users, true_quality[item])
            engaged_count[item] += engagements

        gini_over_time.append(gini(shown_count))

    return gini_over_time, shown_count, true_quality


if __name__ == "__main__":
    rng_greedy = np.random.default_rng(SEED)
    gini_greedy, shown_greedy, quality = run("greedy", rng_greedy)

    rng_explore = np.random.default_rng(SEED)
    gini_explore, shown_explore, _ = run("exploring", rng_explore)

    print("=== Hidden feedback loop: exposure concentration over rounds ===")
    print(f"Round 1  Gini  - greedy: {gini_greedy[0]:.3f}, exploring: {gini_explore[0]:.3f}")
    print(f"Round 10 Gini  - greedy: {gini_greedy[9]:.3f}, exploring: {gini_explore[9]:.3f}")
    print(f"Round 30 Gini  - greedy: {gini_greedy[29]:.3f}, exploring: {gini_explore[29]:.3f}")
    print(f"Round 60 Gini  - greedy: {gini_greedy[-1]:.3f}, exploring: {gini_explore[-1]:.3f}")

    n_shown_greedy = np.sum(shown_greedy > 0)
    n_shown_explore = np.sum(shown_explore > 0)
    print(f"\nItems that ever got shown out of {N_ITEMS}: greedy = {n_shown_greedy}, exploring = {n_shown_explore}")

    # Did greedy converge to the actually-best items, or lock in early noise?
    true_top_k = set(np.argsort(-quality)[:TOP_K])
    greedy_top_k = set(np.argsort(-shown_greedy)[:TOP_K])
    overlap = len(true_top_k & greedy_top_k)
    print(f"Overlap between greedy's top-{TOP_K} most-shown items and the TRUE top-{TOP_K} by quality: {overlap}/{TOP_K}")

    explore_top_k = set(np.argsort(-shown_explore)[:TOP_K])
    overlap_explore = len(true_top_k & explore_top_k)
    print(f"Overlap between exploring's top-{TOP_K} most-shown items and the TRUE top-{TOP_K} by quality: {overlap_explore}/{TOP_K}")
