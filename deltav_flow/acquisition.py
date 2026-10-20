"""Next-run recommendations. Label-free: no outcome, field or verification data enters.

Policies:
  mapped-lhs-v1   Task 8 fno_lhs order (static_order8): one Latin hypercube of
                  `sequence_length` points, seed from study.json, each mapped to the
                  nearest unused candidate. Reproduces the recorded example order.
                  Prefixes are not themselves Latin designs.
  maximin-v1      Product policy for general uploads: greedy maximin distance in the
                  unit-scaled declared box to supplied and just-selected settings,
                  deterministic tie-break. New; it has no historical savings evidence.
  uncertainty-diversity-experimental
                  Task 8 fno_active heuristic (raw spread / tolerance x diversity).
                  Experimental: in Task 8 it was weaker than fixed sampling at sparse budgets.
"""
from __future__ import annotations

import numpy as np

from .bundle import setting_key

POLICIES = {
    "mapped-lhs-v1": {"label": "Even coverage, following the study's recorded space-filling plan", "experimental": False},
    "maximin-v1": {"label": "Even coverage, filling the largest gaps first", "experimental": False},
    "uncertainty-diversity-experimental": {"label": "Uncertainty-driven, kept spread out (experimental)", "experimental": True},
}
DEFAULT_BATCH = 8
GENERATED_POOL = 1024
POOL_SEED = 20261020


def unit(x, ranges):
    lo = np.array([r[0] for r in ranges])
    hi = np.array([r[1] for r in ranges])
    return (np.asarray(x, float) - lo) / (hi - lo)


def mapped_lhs_order(candidate_x, ranges, seed, sequence_length):
    """Exact static_order8('fno_lhs'): LHS(3, seed).random(n) -> .01+.98p -> nearest unused candidate."""
    from scipy.stats import qmc

    x = unit(candidate_x, ranges)
    points = 0.01 + 0.98 * qmc.LatinHypercube(3, seed=seed).random(sequence_length)
    remaining = set(range(len(x)))
    order = []
    for point in points:
        if not remaining:
            break
        choice = min(remaining, key=lambda i: (float(np.sum((x[i] - point) ** 2)), i))
        order.append(choice)
        remaining.remove(choice)
    return order


def generated_pool(ranges, n=GENERATED_POOL, seed=POOL_SEED):
    lo = np.array([r[0] for r in ranges])
    hi = np.array([r[1] for r in ranges])
    unit_points = np.random.default_rng(seed).uniform(0.01, 0.99, (n, 3))
    return [f"gen-{i:04d}" for i in range(n)], lo + unit_points * (hi - lo)


def _nearest(xu, known_u):
    if not len(known_u):
        return np.full(len(xu), np.sqrt(3.0))
    return np.sqrt(((xu[:, None, :] - known_u[None, :, :]) ** 2).sum(2)).min(1)


def recommend(policy, candidate_ids, candidate_x, known_x, ranges, batch=DEFAULT_BATCH, acquisition=None,
              sd=None, tolerance=None):
    """Return (chosen indices into candidates, per-row reasons, notes). Never sees labels."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown recommendation policy {policy!r}")
    if not 1 <= int(batch) <= 32:
        raise ValueError("Batch size must be between 1 and 32")
    candidate_x = np.asarray(candidate_x, float).reshape(-1, 3)
    known_x = np.asarray(known_x, float).reshape(-1, 3)
    supplied = {setting_key(v) for v in known_x}
    usable = [i for i, v in enumerate(candidate_x) if setting_key(v) not in supplied]
    notes = []
    xu = unit(candidate_x, ranges)
    if policy == "mapped-lhs-v1":
        if not acquisition:
            raise ValueError("mapped-lhs-v1 needs the study's recorded acquisition seed and sequence length")
        order = mapped_lhs_order(candidate_x, ranges, acquisition["seed"], acquisition["sequence_length"])
        remaining = [i for i in order if setting_key(candidate_x[i]) not in supplied]
        chosen = remaining[:batch]
        position = {i: k for k, i in enumerate(order)}
        reasons = [{"reason": f"Next in the study's recorded space-filling plan (position {position[i] + 1} of {len(order)}); "
                              "fills the space-filling design fixed before any results were seen."} for i in chosen]
        if len(chosen) < batch:
            notes.append(f"The study's {len(order)}-point space-filling plan runs out after {len(chosen)} more run(s). "
                         "For more, choose \"Even coverage, filling the largest gaps first\".")
        if batch != DEFAULT_BATCH:
            notes.append(f"A batch of {batch} takes the first {batch} points of the plan; our research used batches of {DEFAULT_BATCH}.")
    elif policy == "maximin-v1":
        known_u = unit(known_x, ranges)
        chosen, reasons = [], []
        pool = list(usable)
        for _ in range(min(batch, len(pool))):
            distance = _nearest(xu[pool], known_u)
            k = int(np.argmax(distance))  # first maximum: deterministic tie-break by candidate order
            i = pool.pop(k)
            reasons.append({"reason": "Fills the largest gap in your supplied parameter coverage "
                                      f"(nearest supplied or already-selected setting {distance[k] / np.sqrt(3):.3f} of the box diagonal away).",
                            "gap_fraction_of_diagonal": float(distance[k] / np.sqrt(3))})
            chosen.append(i)
            known_u = np.vstack([known_u, xu[i]])
        if batch != DEFAULT_BATCH:
            notes.append(f"Our research used batches of {DEFAULT_BATCH}; a batch of {batch} has not been studied.")
    else:
        if sd is None or tolerance is None:
            raise ValueError("The experimental uncertainty policy needs model spreads")
        sd = np.asarray(sd, float)
        score = (sd / np.array(tolerance)).max(1)
        known_u = unit(known_x, ranges)
        chosen, reasons, used = [], [], set(range(len(candidate_x))) - set(usable)
        for _ in range(min(batch, len(usable))):
            distance = _nearest(xu, known_u) / np.sqrt(3.0)
            utility = score * (0.25 + 0.75 * distance)
            utility[list(used)] = -np.inf
            i = int(np.argmax(utility))
            if not np.isfinite(utility[i]):
                break
            chosen.append(i)
            used.add(i)
            known_u = np.vstack([known_u, xu[i]])
            reasons.append({"reason": "High raw ensemble spread relative to tolerance, spaced from known runs (experimental heuristic).",
                            "raw_spread_over_tolerance": float(score[i]), "gap_fraction_of_diagonal": float(distance[i])})
        notes.append("Experimental: in our research this approach did worse than even coverage when few runs were available.")
    if not usable:
        notes.append("No feasible unused candidates remain. This is exhaustion of the candidate list, not a sign of confidence.")
    elif len(chosen) < batch and policy != "mapped-lhs-v1":
        notes.append(f"Only {len(chosen)} feasible unused candidate(s) remain.")
    gaps = _nearest(xu[chosen], unit(known_x, ranges)) / np.sqrt(3) if chosen else []
    for r, gap, i in zip(reasons, gaps, chosen):
        r.setdefault("gap_fraction_of_diagonal", float(gap))
        r["candidate_id"] = candidate_ids[i]
    return chosen, reasons, notes
