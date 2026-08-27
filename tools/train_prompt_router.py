# SPDX-FileCopyrightText: Copyright 2026 황영환
# SPDX-License-Identifier: Apache-2.0

"""Train the prompt-aware pairwise router on public outcomes.

NumPy is a build-time dependency only.  The emitted artifact is consumed by
``ossp_router.prompt_router`` using the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - CLI error path
    np = None

from ossp_router import prompt_router
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    InputBatch,
    Outcome,
    OutcomeBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_outcomes,
    load_policy,
    policy_sha256,
)
from ossp_router.scoring import score_submissions


TARGET_NAMES = prompt_router.PRIMARY_HEAD_NAMES
COMPANION_NAMES = prompt_router.COMPANION_HEAD_NAMES
_NUMBER = re.compile(r"\d+")
_SPACE = re.compile(r"\s+")


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("학습에는 NumPy가 필요합니다.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _outcome_cost(outcome: Outcome, policy: RoutingPolicy) -> float:
    rates = policy.models[outcome.model_id]
    unit = Decimal(policy.token_unit)
    cost = (
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )
    result = float(cost)
    if not math.isfinite(result) or result <= 0:
        raise ProtocolError("학습 outcome의 비용은 0보다 커야 합니다.")
    return result


def _matrix_and_targets(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    word_bins: int,
    char_bins: int,
) -> Tuple[Any, Any]:
    _require_numpy()
    if (
        inputs.schema_version != outcomes.schema_version
        or inputs.challenge_id != outcomes.challenge_id
        or inputs.split != outcomes.split
    ):
        raise ProtocolError("입력과 outcome 메타데이터가 다릅니다.")
    index = {(row.episode_id, row.model_id): row for row in outcomes.outcomes}
    expected = {
        (episode.episode_id, model)
        for episode in inputs.episodes
        for model in MODEL_IDS
    }
    if set(index) != expected:
        raise ProtocolError("outcome이 모든 episode·model 조합을 포함하지 않습니다.")
    matrix = np.asarray(
        [
            prompt_router.raw_feature_vector(episode, word_bins, char_bins)
            for episode in inputs.episodes
        ],
        dtype=np.float64,
    )
    targets = []
    for episode in inputs.episodes:
        light = index[(episode.episode_id, prompt_router.LIGHT)]
        ax31 = index[(episode.episode_id, prompt_router.AX31)]
        think = index[(episode.episode_id, prompt_router.THINK)]
        delta_ax31 = float(ax31.score - light.score)
        delta_think = float(think.score - ax31.score)
        costs = [_outcome_cost(index[(episode.episode_id, model)], policy) for model in MODEL_IDS]
        targets.append(
            [
                delta_ax31,
                delta_think,
                float(delta_ax31 > 0),
                float(delta_think > 0),
                *(math.log(cost) for cost in costs),
            ]
        )
    return matrix, np.asarray(targets, dtype=np.float64)


def _default_groups(inputs: InputBatch) -> Tuple[str, ...]:
    """Group exact numeric-template duplicates without reading IDs."""

    result = []
    for episode in inputs.episodes:
        text = prompt_router.episode_text(episode).casefold()
        normalized = _SPACE.sub(" ", _NUMBER.sub("#", text)).strip()
        result.append(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
    return tuple(result)


def _load_groups(
    inputs: InputBatch,
    path: Optional[Path],
) -> Tuple[Tuple[str, ...], Optional[Tuple[str, ...]], str]:
    if path is None:
        return _default_groups(inputs), None, "numeric-template-sha256"
    _require_numpy()
    raw = np.load(path, allow_pickle=True)
    suffix = inputs.split
    eid_key = f"eids_{suffix}"
    group_key = f"g2n_{suffix}"
    family_key = f"family_{suffix}"
    expected = tuple(episode.episode_id for episode in inputs.episodes)
    if {eid_key, group_key, family_key}.issubset(raw.files):
        actual = tuple(str(value) for value in raw[eid_key].tolist())
        if actual != expected:
            raise ProtocolError("group npz의 episode 순서가 학습 입력과 다릅니다.")
        groups = tuple(str(value) for value in raw[group_key].tolist())
        families = tuple(str(value) for value in raw[family_key].tolist())
        return groups, families, "exact-source-with-g2n"
    # The v2 audit emits one compact Train+Dev source map.  It intentionally
    # contains only episode-to-family alignment, not a runtime feature or a
    # prompt lookup.  For the v1 nested comparator, derive inner duplicate
    # groups from prompt content and use IDs only to align offline family
    # labels with the already-validated input rows.
    if {"eids", "family"}.issubset(raw.files):
        episode_ids = tuple(str(value) for value in raw["eids"].tolist())
        family_values = tuple(str(value) for value in raw["family"].tolist())
        if len(episode_ids) != len(family_values) or len(set(episode_ids)) != len(
            episode_ids
        ):
            raise ProtocolError("compact source map의 episode/family 범위가 잘못되었습니다.")
        family_by_id = dict(zip(episode_ids, family_values))
        missing = tuple(episode_id for episode_id in expected if episode_id not in family_by_id)
        if missing:
            raise ProtocolError(
                f"compact source map에 학습 문항이 누락되었습니다: {missing[:5]}"
            )
        return (
            _default_groups(inputs),
            tuple(family_by_id[episode_id] for episode_id in expected),
            "numeric-template-with-compact-family-map",
        )
    raise ProtocolError("group npz에 지원하는 episode/group/family 배열이 없습니다.")


def _fold_ids(groups: Sequence[str], folds: int) -> Any:
    _require_numpy()
    members: Dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    if folds < 2 or len(members) < folds:
        raise ProtocolError("그룹 수보다 적은 fold를 사용할 수 없습니다.")
    loads = [0] * folds
    assignments: Dict[str, int] = {}
    ordered = sorted(
        members,
        key=lambda group: (
            -len(members[group]),
            hashlib.sha256(group.encode("utf-8")).hexdigest(),
        ),
    )
    for group in ordered:
        fold = min(range(folds), key=lambda item: (loads[item], item))
        assignments[group] = fold
        loads[fold] += len(members[group])
    return np.asarray([assignments[group] for group in groups], dtype=np.int64)


def _mean_scale(matrix: Any) -> Tuple[Any, Any]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    return mean, np.where(scale > 1e-12, scale, 1.0)


def _fit_standardized(matrix: Any, targets: Any, alpha: float) -> Tuple[Any, Any]:
    intercept = targets.mean(axis=0)
    centered = targets - intercept
    rows, columns = matrix.shape
    if rows <= columns:
        system = matrix @ matrix.T + alpha * np.eye(rows)
        coefficients = matrix.T @ np.linalg.solve(system, centered)
    else:
        system = matrix.T @ matrix + alpha * np.eye(columns)
        coefficients = np.linalg.solve(system, matrix.T @ centered)
    return intercept, coefficients


def _fit_raw(
    matrix: Any,
    targets: Any,
    alpha: float,
    columns: Optional[Any] = None,
) -> Tuple[Any, Any, Any, Any]:
    mean, scale = _mean_scale(matrix)
    standardized = (matrix - mean) / scale
    if columns is None:
        columns = np.arange(matrix.shape[1])
    intercept, subset = _fit_standardized(standardized[:, columns], targets, alpha)
    coefficients = np.zeros((matrix.shape[1], targets.shape[1]), dtype=np.float64)
    coefficients[columns] = subset
    return mean, scale, intercept, coefficients


def _predict(
    matrix: Any,
    mean: Any,
    scale: Any,
    intercept: Any,
    coefficients: Any,
) -> Any:
    return (matrix - mean) / scale @ coefficients + intercept


def _oof(
    matrix: Any,
    targets: Any,
    fold_ids: Any,
    alpha: float,
    columns: Optional[Any] = None,
) -> Any:
    result = np.empty_like(targets)
    for fold in sorted(set(int(value) for value in fold_ids.tolist())):
        validation = fold_ids == fold
        training = ~validation
        mean, scale, intercept, coefficients = _fit_raw(
            matrix[training], targets[training], alpha, columns
        )
        result[validation] = _predict(
            matrix[validation], mean, scale, intercept, coefficients
        )
    return result


def _objective(predicted: Any, actual: Any, *, companion: bool) -> float:
    delta = float(np.mean((predicted[:, :2] - actual[:, :2]) ** 2))
    probability = float(np.mean((predicted[:, 2:4] - actual[:, 2:4]) ** 2))
    if companion:
        return delta + 0.10 * probability
    log_cost = float(np.mean((predicted[:, 4:] - actual[:, 4:]) ** 2))
    return delta + 0.10 * probability + 0.02 * log_cost


def _select_alpha(
    matrix: Any,
    targets: Any,
    fold_ids: Any,
    candidates: Sequence[float],
    columns: Optional[Any] = None,
    *,
    companion: bool,
) -> Tuple[float, Any, Mapping[str, float]]:
    best = None
    diagnostics = {}
    for alpha in candidates:
        predictions = _oof(matrix, targets, fold_ids, alpha, columns)
        objective = _objective(predictions, targets, companion=companion)
        diagnostics[format(alpha, ".12g")] = objective
        rank = (objective, alpha)
        if best is None or rank < best[0]:
            best = (rank, alpha, predictions)
    assert best is not None
    return best[1], best[2], diagnostics


def _quantile_conservative(values: Sequence[float], upper: bool) -> float:
    ordered = sorted(float(value) for value in values)
    if upper:
        index = int(math.ceil(0.90 * (len(ordered) - 1)))
        return ordered[index]
    index = int(math.floor(0.10 * (len(ordered) - 1)))
    return ordered[index]


def _cost_bounds(targets: Any, predictions: Any, fold_ids: Any) -> prompt_router.CostBounds:
    by_model: Dict[str, list[float]] = {model: [] for model in MODEL_IDS}
    for fold in sorted(set(int(value) for value in fold_ids.tolist())):
        mask = fold_ids == fold
        for offset, model in enumerate(MODEL_IDS):
            actual = float(np.exp(targets[mask, 4 + offset]).sum())
            predicted = float(np.exp(np.clip(predictions[mask, 4 + offset], -50, 50)).sum())
            by_model[model].append(actual / predicted)
    lower = min(1.0, _quantile_conservative(by_model[prompt_router.LIGHT], upper=False) * 0.98)
    upper = {
        model: max(1.0, _quantile_conservative(by_model[model], upper=True) * 1.02)
        for model in MODEL_IDS
    }
    return prompt_router.CostBounds(lower, upper)


def _prediction_objects(
    inputs: InputBatch,
    primary: Any,
    companion: Any,
) -> Tuple[prompt_router.EpisodePrediction, ...]:
    result = []
    for index, episode in enumerate(inputs.episodes):
        costs = {
            model: math.exp(min(50.0, max(-50.0, float(primary[index, 4 + offset]))))
            for offset, model in enumerate(MODEL_IDS)
        }
        costs[prompt_router.AX31] = max(
            costs[prompt_router.AX31], costs[prompt_router.LIGHT] * (1.0 + 1e-12)
        )
        costs[prompt_router.THINK] = max(
            costs[prompt_router.THINK], costs[prompt_router.AX31] * (1.0 + 1e-12)
        )
        result.append(
            prompt_router.EpisodePrediction(
                text_fingerprint=prompt_router._stable_hash(prompt_router.episode_text(episode)),
                primary_delta_ax31=float(primary[index, 0]),
                primary_delta_think=float(primary[index, 1]),
                companion_delta_ax31=float(companion[index, 0]),
                companion_delta_think=float(companion[index, 1]),
                primary_prob_ax31=min(1.0, max(0.0, float(primary[index, 2]))),
                primary_prob_think=min(1.0, max(0.0, float(primary[index, 3]))),
                companion_prob_ax31=min(1.0, max(0.0, float(companion[index, 2]))),
                companion_prob_think=min(1.0, max(0.0, float(companion[index, 3]))),
                mean_costs=costs,
                is_ood=False,
            )
        )
    return tuple(result)


def _submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    tier: str,
    selected: Sequence[str],
) -> Submission:
    return Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, model)
            for episode, model in zip(inputs.episodes, selected)
        ),
    )


def _score_tier(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    tier: str,
    selected: Sequence[str],
) -> Mapping[str, Any]:
    submissions = []
    for candidate in TIERS:
        models = selected if candidate == tier else [prompt_router.TIER_MODEL[candidate]] * len(inputs.episodes)
        submissions.append(_submission(inputs, policy, candidate, models))
    return score_submissions(inputs, outcomes, submissions, policy)["tiers"][tier]


def _skeleton_artifact(
    policy: RoutingPolicy,
    word_bins: int,
    char_bins: int,
    cost_bounds: prompt_router.CostBounds,
    tier_policies: Mapping[str, prompt_router.TierPolicy],
) -> prompt_router.PromptRouterArtifact:
    length = len(prompt_router.DENSE_FEATURE_NAMES) + word_bins + char_bins
    zero_head = prompt_router.LinearHead(0.0, (0.0,) * length)
    return prompt_router.PromptRouterArtifact(
        word_bins=word_bins,
        char_bins=char_bins,
        feature_mean=(0.0,) * length,
        feature_scale=(1.0,) * length,
        primary_heads={name: zero_head for name in TARGET_NAMES},
        companion_heads={name: zero_head for name in COMPANION_NAMES},
        cost_bounds=cost_bounds,
        ood_thresholds=prompt_router.OodThresholds(1e9, 1e9),
        tier_policies=tier_policies,
        policy_id=policy.policy_id,
        policy_digest=policy_sha256(policy),
        training_summary={},
    )


def _select_thresholds(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    predictions: Sequence[prompt_router.EpisodePrediction],
    word_bins: int,
    char_bins: int,
    cost_bounds: prompt_router.CostBounds,
) -> Tuple[Mapping[str, prompt_router.TierPolicy], Mapping[str, Any]]:
    target_ratios = {
        "fast": 1.15,
        "balanced": 1.60,
        "premium": 3.10,
    }
    candidates = (0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60)
    selected: Dict[str, prompt_router.TierPolicy] = {}
    diagnostics: Dict[str, Any] = {}
    for tier in TIERS:
        rows = []
        best = None
        for threshold in candidates:
            tier_policies = {
                name: prompt_router.TierPolicy(target_ratios[name], threshold)
                for name in TIERS
            }
            artifact = _skeleton_artifact(
                policy, word_bins, char_bins, cost_bounds, tier_policies
            )
            models, predicted_ratio, _fallback, _ood = prompt_router.allocate_predictions(
                inputs, policy, artifact, tier, predictions
            )
            report = _score_tier(inputs, outcomes, policy, tier, models)
            row = {
                "min_win_probability": threshold,
                "quality_score": report["quality_score"],
                "actual_budget_ratio": report["budget_ratio"],
                "budget_passed": report["budget_passed"],
                "conservative_budget_ratio": predicted_ratio,
                "model_counts": report["model_counts"],
            }
            rows.append(row)
            score = Decimal(report["tier_score"])
            rank = (score, -abs(predicted_ratio - target_ratios[tier]), threshold)
            if best is None or rank > best[0]:
                best = (rank, threshold, row)
        assert best is not None
        selected[tier] = prompt_router.TierPolicy(target_ratios[tier], best[1])
        diagnostics[tier] = {"selected": best[2], "candidates": rows}
    return selected, diagnostics


def _heads(intercept: Any, coefficients: Any, names: Sequence[str]) -> Mapping[str, Any]:
    return {
        name: {
            "intercept": float(intercept[index]),
            "coefficients": [float(value) for value in coefficients[:, index]],
        }
        for index, name in enumerate(names)
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _artifact_value(
    *,
    policy: RoutingPolicy,
    word_bins: int,
    char_bins: int,
    mean: Any,
    scale: Any,
    primary_intercept: Any,
    primary_coefficients: Any,
    companion_intercept: Any,
    companion_coefficients: Any,
    cost_bounds: prompt_router.CostBounds,
    ood_thresholds: prompt_router.OodThresholds,
    tier_policies: Mapping[str, prompt_router.TierPolicy],
    training_summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "artifact_type": prompt_router.ARTIFACT_TYPE,
        "schema_version": 1,
        "feature_version": prompt_router.FEATURE_VERSION,
        "hash_algorithm": "fnv1a64-unicode-signed-word12-char3",
        "word_bins": word_bins,
        "char_bins": char_bins,
        "dense_feature_names": list(prompt_router.DENSE_FEATURE_NAMES),
        "model_ids": list(MODEL_IDS),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "feature_mean": [float(value) for value in mean],
        "feature_scale": [float(value) for value in scale],
        "primary_heads": _heads(primary_intercept, primary_coefficients, TARGET_NAMES),
        "companion_heads": _heads(companion_intercept, companion_coefficients, COMPANION_NAMES),
        "cost_bounds": {
            "light_lower_multiplier": cost_bounds.light_lower_multiplier,
            "upper_multipliers": dict(cost_bounds.upper_multipliers),
        },
        "ood_thresholds": {
            "dense_max_abs_z": ood_thresholds.dense_max_abs_z,
            "dense_rms_z": ood_thresholds.dense_rms_z,
        },
        "tier_policies": {
            tier: {
                "target_budget_ratio": tier_policies[tier].target_budget_ratio,
                "min_win_probability": tier_policies[tier].min_win_probability,
            }
            for tier in TIERS
        },
        "training_summary": dict(training_summary),
    }


def train(
    *,
    input_path: Path,
    outcomes_path: Path,
    artifact_path: Path,
    report_path: Path,
    policy: RoutingPolicy,
    word_bins: int,
    char_bins: int,
    folds: int,
    alpha_candidates: Sequence[float],
    groups_path: Optional[Path] = None,
    validation_input_path: Optional[Path] = None,
    validation_outcomes_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    _require_numpy()
    inputs = load_input(input_path)
    outcomes = load_outcomes(outcomes_path)
    matrix, targets = _matrix_and_targets(
        inputs, outcomes, policy, word_bins, char_bins
    )
    groups, families, group_strategy = _load_groups(inputs, groups_path)
    fold_ids = _fold_ids(groups, folds)
    dense = len(prompt_router.DENSE_FEATURE_NAMES)
    companion_columns = np.concatenate(
        (
            np.arange(dense, dtype=np.int64),
            np.arange(dense + word_bins, matrix.shape[1], dtype=np.int64),
        )
    )
    primary_alpha, primary_oof, primary_diagnostics = _select_alpha(
        matrix,
        targets,
        fold_ids,
        alpha_candidates,
        companion=False,
    )
    companion_targets = targets[:, : len(COMPANION_NAMES)]
    companion_alpha, companion_oof, companion_diagnostics = _select_alpha(
        matrix,
        companion_targets,
        fold_ids,
        alpha_candidates,
        companion_columns,
        companion=True,
    )
    bounds = _cost_bounds(targets, primary_oof, fold_ids)
    prediction_objects = _prediction_objects(inputs, primary_oof, companion_oof)
    tier_policies, threshold_diagnostics = _select_thresholds(
        inputs,
        outcomes,
        policy,
        prediction_objects,
        word_bins,
        char_bins,
        bounds,
    )
    oof_artifact = _skeleton_artifact(
        policy, word_bins, char_bins, bounds, tier_policies
    )
    oof_submissions = []
    for tier in TIERS:
        selected_models, _ratio, _fallback, _ood = prompt_router.allocate_predictions(
            inputs, policy, oof_artifact, tier, prediction_objects
        )
        oof_submissions.append(_submission(inputs, policy, tier, selected_models))
    group_oof_score = score_submissions(
        inputs, outcomes, oof_submissions, policy
    )
    group_oof_c2 = score_submissions(
        inputs,
        outcomes,
        [
            _submission(
                inputs,
                policy,
                tier,
                [prompt_router.TIER_MODEL[tier]] * len(inputs.episodes),
            )
            for tier in TIERS
        ],
        policy,
    )

    mean, scale, primary_intercept, primary_coefficients = _fit_raw(
        matrix, targets, primary_alpha
    )
    _, _, companion_intercept, companion_coefficients = _fit_raw(
        matrix, companion_targets, companion_alpha, companion_columns
    )
    standardized_dense = ((matrix - mean) / scale)[:, :dense]
    dense_max = np.max(np.abs(standardized_dense), axis=1)
    dense_rms = np.sqrt(np.mean(standardized_dense**2, axis=1))
    ood = prompt_router.OodThresholds(
        dense_max_abs_z=float(np.quantile(dense_max, 0.995) * 1.05),
        dense_rms_z=float(np.quantile(dense_rms, 0.995) * 1.05),
    )
    training_summary = {
        "optimizer": "numpy-ridge-group-oof-pairwise-v1",
        "num_episodes": len(inputs.episodes),
        "folds": folds,
        "group_strategy": group_strategy,
        "primary_alpha": primary_alpha,
        "companion_alpha": companion_alpha,
        "input_sha256": _file_sha256(input_path),
        "outcomes_sha256": _file_sha256(outcomes_path),
        "dev_used_for_selection": False,
        "family_count": len(set(families)) if families is not None else None,
    }
    value = _artifact_value(
        policy=policy,
        word_bins=word_bins,
        char_bins=char_bins,
        mean=mean,
        scale=scale,
        primary_intercept=primary_intercept,
        primary_coefficients=primary_coefficients,
        companion_intercept=companion_intercept,
        companion_coefficients=companion_coefficients,
        cost_bounds=bounds,
        ood_thresholds=ood,
        tier_policies=tier_policies,
        training_summary=training_summary,
    )
    artifact = prompt_router.parse_artifact(value)
    train_plans = [
        prompt_router.make_prompt_submission(inputs, policy, artifact, tier)
        for tier in TIERS
    ]
    train_report = score_submissions(
        inputs, outcomes, [plan.submission for plan in train_plans], policy
    )
    report: Dict[str, Any] = {
        "report_type": "ossp-prompt-router-training-v1",
        "training_summary": training_summary,
        "feature_dimension": int(matrix.shape[1]),
        "primary_alpha_objectives": primary_diagnostics,
        "companion_alpha_objectives": companion_diagnostics,
        "cost_bounds": value["cost_bounds"],
        "ood_thresholds": value["ood_thresholds"],
        "group_oof_threshold_selection": threshold_diagnostics,
        "group_oof_score": group_oof_score,
        "group_oof_tierfixed_c2": group_oof_c2,
        "group_oof_delta_final_score": float(group_oof_score["final_score"])
        - float(group_oof_c2["final_score"]),
        "fitted_train_self_check": train_report,
    }
    if validation_input_path is not None or validation_outcomes_path is not None:
        if validation_input_path is None or validation_outcomes_path is None:
            raise ProtocolError("validation input과 outcomes는 함께 지정해야 합니다.")
        validation_inputs = load_input(validation_input_path)
        validation_outcomes = load_outcomes(validation_outcomes_path)
        plans = [
            prompt_router.make_prompt_submission(validation_inputs, policy, artifact, tier)
            for tier in TIERS
        ]
        report["untouched_validation"] = {
            "input_sha256": _file_sha256(validation_input_path),
            "outcomes_sha256": _file_sha256(validation_outcomes_path),
            "selection_used_this_split": False,
            "score": score_submissions(
                validation_inputs,
                validation_outcomes,
                [plan.submission for plan in plans],
                policy,
            ),
            "runtime_diagnostics": {
                tier: {
                    "conservative_budget_ratio": plan.conservative_budget_ratio,
                    "fallback_count": plan.fallback_count,
                    "ood_count": plan.ood_count,
                }
                for tier, plan in zip(TIERS, plans)
            },
        }
    _write_json_atomic(artifact_path, value)
    _write_json_atomic(report_path, report)
    return report


def _positive_float_list(value: str) -> Tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alpha 목록을 해석할 수 없습니다.") from exc
    if not result or any(not math.isfinite(item) or item <= 0 for item in result):
        raise argparse.ArgumentTypeError("alpha는 0보다 큰 유한한 수여야 합니다.")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="공개 Train으로 pairwise prompt router를 학습합니다.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--groups-npz", type=Path)
    parser.add_argument("--validation-input", type=Path)
    parser.add_argument("--validation-outcomes", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--word-bins", type=int, default=prompt_router.DEFAULT_WORD_BINS)
    parser.add_argument("--char-bins", type=int, default=prompt_router.DEFAULT_CHAR_BINS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--alphas",
        type=_positive_float_list,
        default=_positive_float_list("0.1,1,10,100,1000"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = load_policy(args.policy) if args.policy is not None else load_bundled_policy()
        report = train(
            input_path=args.input,
            outcomes_path=args.outcomes,
            artifact_path=args.artifact,
            report_path=args.report,
            policy=policy,
            word_bins=args.word_bins,
            char_bins=args.char_bins,
            folds=args.folds,
            alpha_candidates=args.alphas,
            groups_path=args.groups_npz,
            validation_input_path=args.validation_input,
            validation_outcomes_path=args.validation_outcomes,
        )
    except (OSError, ProtocolError, RuntimeError, ValueError, np.linalg.LinAlgError if np is not None else ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(
        "OK: prompt-router artifact를 생성했습니다 "
        f"(Train self-check {report['fitted_train_self_check']['final_score']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
