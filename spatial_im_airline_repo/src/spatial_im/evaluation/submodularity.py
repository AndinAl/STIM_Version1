from __future__ import annotations

import itertools
import numpy as np


def marginal_gain(eval_fn, S, v):
    S = set(S)
    if v in S:
        return 0.0
    return float(eval_fn(S | {v}) - eval_fn(S))


def empirical_submodularity_violation_rate(nodes, eval_fn, samples: int = 100, seed: int = 7, max_set_size: int | None = None):
    rng = np.random.default_rng(seed)
    nodes = list(nodes)
    violations = 0
    checked = 0
    details = []
    for _ in range(samples):
        perm = rng.permutation(nodes)
        if max_set_size is None:
            max_a = max(1, len(nodes) // 3 + 1)
            max_b = max(2, 2 * len(nodes) // 3 + 1)
        else:
            max_a = max(1, min(max_set_size, len(nodes) - 1))
            max_b = max(2, min(max_set_size, len(nodes) - 1))
        a = int(rng.integers(0, max_a))
        b = int(rng.integers(a, max(max_b, a + 1)))
        S = set(perm[:a].tolist())
        T = set(perm[:b].tolist())
        candidates = [x for x in nodes if x not in T]
        if not candidates:
            continue
        v = int(rng.choice(candidates))
        dS = marginal_gain(eval_fn, S, v)
        dT = marginal_gain(eval_fn, T, v)
        checked += 1
        violated = dT > dS + 1e-8
        violations += int(violated)
        details.append({'S': S, 'T': T, 'v': v, 'delta_S': dS, 'delta_T': dT, 'violation': violated})
    rate = float(violations / max(checked, 1))
    return rate, details
