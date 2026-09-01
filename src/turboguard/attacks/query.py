"""Score-based and hard-label query attack primitives."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def random_query_attack(
    X: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray],
    label_fn: Callable[[np.ndarray], np.ndarray],
    eps: float,
    query_budget: int,
    step_size: float,
    seed: int = 42,
    projection: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    hard_label: bool = False,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    evaluate_fn: Optional[
        Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    ] = None,
    proposals_per_round: int = 1,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Derivative-free score or hard-label attack with query accounting.

    Successful samples stop consuming queries. Score mode performs greedy
    random-direction descent. Hard-label mode samples the full bounded region
    because labels provide no ranking signal before the decision flips.
    """
    checkpoints = random_query_attack_checkpoints(
        X=X,
        score_fn=score_fn,
        label_fn=label_fn,
        eps=eps,
        query_budgets=[query_budget],
        step_size=step_size,
        seed=seed,
        projection=projection,
        hard_label=hard_label,
        progress_callback=progress_callback,
        evaluate_fn=evaluate_fn,
        proposals_per_round=proposals_per_round,
    )
    return checkpoints[query_budget]


def random_query_attack_checkpoints(
    X: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray],
    label_fn: Callable[[np.ndarray], np.ndarray],
    eps: float,
    query_budgets: list[int],
    step_size: float,
    seed: int = 42,
    projection: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    hard_label: bool = False,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
    evaluate_fn: Optional[
        Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    ] = None,
    proposals_per_round: int = 32,
) -> dict[int, tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Run one query trajectory and snapshot it at multiple query budgets."""
    budgets = sorted(set(int(value) for value in query_budgets))
    if not budgets or budgets[0] < 1:
        raise ValueError("query_budgets must contain positive integers")
    if proposals_per_round < 1:
        raise ValueError("proposals_per_round must be positive")
    max_budget = budgets[-1]
    rng = np.random.default_rng(seed)
    X0 = np.asarray(X, dtype=np.float32)
    current = X0.copy()
    if evaluate_fn is None:
        current_score = np.asarray(score_fn(current), dtype=float)
        current_label = np.asarray(label_fn(current), dtype=int)
    else:
        score, label = evaluate_fn(current)
        current_score = np.asarray(score, dtype=float)
        current_label = np.asarray(label, dtype=int)
    queries = np.ones(len(X0), dtype=np.int64)
    success = current_label == 0
    snapshots: dict[int, tuple[np.ndarray, dict[str, np.ndarray]]] = {}

    def snapshot(budget: int) -> None:
        snapshots[budget] = (
            current.copy(),
            {
                "success": success.copy(),
                "queries": queries.copy(),
                "score": current_score.copy(),
                "label": current_label.copy(),
            },
        )

    if 1 in budgets:
        snapshot(1)
    used = 1
    while used < max_budget:
        active = np.flatnonzero(~success)
        if not len(active):
            break
        next_checkpoint = min(
            budget for budget in budgets if budget > used
        )
        proposal_count = min(
            proposals_per_round,
            next_checkpoint - used,
            max_budget - used,
        )
        if hard_label:
            candidates = X0[active, None, :] + rng.uniform(
                -eps,
                eps,
                size=(len(active), proposal_count, X0.shape[1]),
            ).astype(np.float32)
        else:
            direction = rng.normal(
                size=(len(active), proposal_count, current.shape[1])
            ).astype(np.float32)
            direction /= np.maximum(
                np.linalg.norm(direction, axis=2, keepdims=True), 1e-8
            )
            candidates = current[active, None, :] + step_size * direction
        candidates = np.minimum(
            np.maximum(
                candidates,
                X0[active, None, :] - eps,
            ),
            X0[active, None, :] + eps,
        )
        candidate_shape = candidates.shape
        candidate = candidates.reshape(-1, X0.shape[1])
        if projection is not None:
            repeated_source = np.repeat(
                X0[active], proposal_count, axis=0
            )
            candidate = projection(repeated_source, candidate)
        if evaluate_fn is None:
            candidate_score = np.asarray(score_fn(candidate), dtype=float)
            candidate_label = np.asarray(label_fn(candidate), dtype=int)
        else:
            score, label = evaluate_fn(candidate)
            candidate_score = np.asarray(score, dtype=float)
            candidate_label = np.asarray(label, dtype=int)
        candidate_score = candidate_score.reshape(
            len(active), proposal_count
        )
        candidate_label = candidate_label.reshape(
            len(active), proposal_count
        )
        candidates = candidate.reshape(candidate_shape)
        best_index = candidate_score.argmin(axis=1)
        row_index = np.arange(len(active))
        best_score = candidate_score[row_index, best_index]
        best_label = candidate_label[row_index, best_index]
        best_candidate = candidates[row_index, best_index]
        queries[active] += proposal_count
        improved_local = best_score < current_score[active]
        improved = active[improved_local]
        current[improved] = best_candidate[improved_local]
        current_score[improved] = best_score[improved_local]
        current_label[improved] = best_label[improved_local]
        success[active] |= (candidate_label == 0).any(axis=1)
        used += proposal_count
        if progress_callback is not None:
            progress_callback(used, max_budget, int(success.sum()))
        if used in budgets:
            snapshot(used)

    for budget in budgets:
        if budget not in snapshots:
            snapshot(budget)
    return snapshots
