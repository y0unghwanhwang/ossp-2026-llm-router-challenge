# SPDX-FileCopyrightText: Copyright 2026 황영환
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the prompt router and controls with the official scorer."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - CLI error path
    np = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines"))
sys.path.insert(0, str(ROOT / "tools"))

import feature_budget  # noqa: E402
import hash_regex  # noqa: E402
import train_prompt_router as trainer  # noqa: E402
from ossp_router import prompt_router, tierfixed  # noqa: E402
from ossp_router.heuristic import make_submission as make_heuristic_submission  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    Decision,
    InputBatch,
    OutcomeBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_outcomes,
    load_policy,
)
from ossp_router.scoring import score_submissions  # noqa: E402


def _submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    tier: str,
    models: Sequence[str],
) -> Submission:
    return Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, model)
            for episode, model in zip(inputs.episodes, models)
        ),
    )


def _selected(submission: Submission) -> Tuple[str, ...]:
    return tuple(item.model_id for item in submission.decisions)


def _score(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    submissions: Sequence[Submission],
) -> Mapping[str, Any]:
    return score_submissions(inputs, outcomes, submissions, policy)


def _official_and_prompt_baselines(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    artifact: prompt_router.PromptRouterArtifact,
    hash_artifact_path: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Tuple[Submission, ...]]]:
    all_light = tuple(
        _submission(
            inputs,
            policy,
            tier,
            [prompt_router.LIGHT] * len(inputs.episodes),
        )
        for tier in TIERS
    )
    c2 = tuple(tierfixed.make_submission(inputs, policy, tier) for tier in TIERS)
    heuristic = tuple(
        make_heuristic_submission(inputs, policy, tier) for tier in TIERS
    )
    feature = tuple(
        feature_budget.make_feature_budget_submission(inputs, policy, tier).submission
        for tier in TIERS
    )
    hash_artifact = hash_regex.load_artifact(hash_artifact_path)
    hash_submissions = tuple(
        hash_regex.make_hash_regex_submission(
            inputs, policy, hash_artifact, tier
        ).submission
        for tier in TIERS
    )
    prompt = tuple(
        prompt_router.make_prompt_submission(inputs, policy, artifact, tier).submission
        for tier in TIERS
    )
    submissions = {
        "all_light": all_light,
        "tierfixed_c2": c2,
        "official_prompt_heuristic": heuristic,
        "official_feature_budget": feature,
        "official_hash_regex": hash_submissions,
        "prompt_aware": prompt,
    }
    reports = {
        name: _score(inputs, outcomes, policy, value)
        for name, value in submissions.items()
    }
    return reports, submissions


def _runtime_predictions(
    inputs: InputBatch,
    artifact: prompt_router.PromptRouterArtifact,
) -> Tuple[prompt_router.EpisodePrediction, ...]:
    return tuple(prompt_router.predict_episode(item, artifact) for item in inputs.episodes)


def _cost_only_control(
    inputs: InputBatch,
    policy: RoutingPolicy,
    predictions: Sequence[prompt_router.EpisodePrediction],
    prompt_submissions: Sequence[Submission],
) -> Tuple[Submission, ...]:
    result = []
    for tier, prompt_submission in zip(TIERS, prompt_submissions):
        counts = {model: 0 for model in MODEL_IDS}
        for model in _selected(prompt_submission):
            counts[model] += 1
        if tier in ("fast", "balanced"):
            models = [prompt_router.LIGHT] * len(inputs.episodes)
            ranked = sorted(
                range(len(models)),
                key=lambda index: (
                    predictions[index].mean_costs[prompt_router.AX31]
                    - predictions[index].mean_costs[prompt_router.LIGHT],
                    predictions[index].text_fingerprint,
                ),
            )
            for index in ranked[: counts[prompt_router.AX31]]:
                models[index] = prompt_router.AX31
        else:
            models = [prompt_router.AX31] * len(inputs.episodes)
            light_ranked = sorted(
                range(len(models)),
                key=lambda index: (
                    -(
                        predictions[index].mean_costs[prompt_router.AX31]
                        - predictions[index].mean_costs[prompt_router.LIGHT]
                    ),
                    predictions[index].text_fingerprint,
                ),
            )
            light_indexes = set(light_ranked[: counts[prompt_router.LIGHT]])
            for index in light_indexes:
                models[index] = prompt_router.LIGHT
            think_ranked = sorted(
                (index for index in range(len(models)) if index not in light_indexes),
                key=lambda index: (
                    predictions[index].mean_costs[prompt_router.THINK]
                    - predictions[index].mean_costs[prompt_router.AX31],
                    predictions[index].text_fingerprint,
                ),
            )
            for index in think_ranked[: counts[prompt_router.THINK]]:
                models[index] = prompt_router.THINK
        result.append(_submission(inputs, policy, tier, models))
    return tuple(result)


def _random_same_counts(
    inputs: InputBatch,
    policy: RoutingPolicy,
    prompt_submissions: Sequence[Submission],
    seed: int,
) -> Tuple[Submission, ...]:
    result = []
    for tier, prompt_submission in zip(TIERS, prompt_submissions):
        models = list(_selected(prompt_submission))
        random.Random(f"prompt-router-random:{seed}:{tier}").shuffle(models)
        result.append(_submission(inputs, policy, tier, models))
    return tuple(result)


def _shuffled_prediction_control(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: prompt_router.PromptRouterArtifact,
    predictions: Sequence[prompt_router.EpisodePrediction],
    seed: int,
) -> Tuple[Submission, ...]:
    order = list(range(len(predictions)))
    random.Random(f"prompt-router-shuffle:{seed}").shuffle(order)
    shuffled = tuple(
        replace(
            predictions[source],
            text_fingerprint=prompt_router._stable_hash(
                prompt_router.episode_text(inputs.episodes[target])
            ),
        )
        for target, source in enumerate(order)
    )
    submissions = []
    for tier in TIERS:
        models, _ratio, _fallback, _ood = prompt_router.allocate_predictions(
            inputs, policy, artifact, tier, shuffled
        )
        submissions.append(_submission(inputs, policy, tier, models))
    return tuple(submissions)


def _distribution(values: Sequence[float]) -> Mapping[str, float]:
    ordered = sorted(values)

    def quantile(q: float) -> float:
        return ordered[int(round(q * (len(ordered) - 1)))]

    return {
        "mean": statistics.fmean(values),
        "sd": statistics.pstdev(values),
        "p025": quantile(0.025),
        "p50": quantile(0.50),
        "p975": quantile(0.975),
    }


def _outcome_cost(row: Any, policy: RoutingPolicy) -> float:
    rates = policy.models[row.model_id]
    unit = Decimal(policy.token_unit)
    return float(
        rates.fixed_cost
        + Decimal(row.input_tokens) * rates.input_token_rate / unit
        + Decimal(row.output_tokens) * rates.output_token_rate / unit
    )


def _fixed_policy_bootstrap(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    submissions: Sequence[Submission],
    comparator_submissions: Sequence[Submission],
    repeats: int,
) -> Mapping[str, Any]:
    index = {(row.episode_id, row.model_id): row for row in outcomes.outcomes}
    selected = {
        submission.tier: {
            item.episode_id: item.model_id for item in submission.decisions
        }
        for submission in submissions
    }
    comparator = {
        submission.tier: {
            item.episode_id: item.model_id for item in submission.decisions
        }
        for submission in comparator_submissions
    }
    tier_scores: Dict[str, list[float]] = {tier: [] for tier in TIERS}
    ratios: Dict[str, list[float]] = {tier: [] for tier in TIERS}
    overflow = dict.fromkeys(TIERS, 0)
    finals = []
    paired_deltas = []
    n = len(inputs.episodes)
    for repeat in range(repeats):
        rng = random.Random(f"prompt-router-bootstrap:{repeat}")
        sample = [rng.randrange(n) for _ in range(n)]
        weighted = 0.0
        comparator_weighted = 0.0
        for tier in TIERS:
            quality = 0.0
            cost = 0.0
            light_cost = 0.0
            for position in sample:
                episode = inputs.episodes[position]
                model = selected[tier][episode.episode_id]
                row = index[(episode.episode_id, model)]
                light = index[(episode.episode_id, prompt_router.LIGHT)]
                quality += float(row.score)
                cost += _outcome_cost(row, policy)
                light_cost += _outcome_cost(light, policy)
            ratio = cost / light_cost
            score = quality / n
            if ratio > float(policy.tiers[tier].budget_multiplier):
                score = 0.0
                overflow[tier] += 1
            ratios[tier].append(ratio)
            tier_scores[tier].append(score)
            weighted += score * float(policy.tiers[tier].weight)
            comparator_quality = 0.0
            comparator_cost = 0.0
            for position in sample:
                episode = inputs.episodes[position]
                model = comparator[tier][episode.episode_id]
                row = index[(episode.episode_id, model)]
                comparator_quality += float(row.score)
                comparator_cost += _outcome_cost(row, policy)
            comparator_ratio = comparator_cost / light_cost
            comparator_score = comparator_quality / n
            if comparator_ratio > float(policy.tiers[tier].budget_multiplier):
                comparator_score = 0.0
            comparator_weighted += comparator_score * float(
                policy.tiers[tier].weight
            )
        finals.append(weighted)
        paired_deltas.append(weighted - comparator_weighted)
    return {
        "repeats": repeats,
        "final_score": _distribution(finals),
        "paired_delta_vs_tierfixed_c2": {
            **_distribution(paired_deltas),
            "probability_le_zero": sum(value <= 0 for value in paired_deltas)
            / repeats,
        },
        "tiers": {
            tier: {
                "quality_or_zero": _distribution(tier_scores[tier]),
                "budget_ratio": _distribution(ratios[tier]),
                "overflow_probability": overflow[tier] / repeats,
            }
            for tier in TIERS
        },
    }


def _subset_batches(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    indexes: Sequence[int],
    split: str,
) -> Tuple[InputBatch, OutcomeBatch]:
    episodes = tuple(inputs.episodes[index] for index in indexes)
    ids = {episode.episode_id for episode in episodes}
    return (
        InputBatch(inputs.schema_version, inputs.challenge_id, split, episodes),
        OutcomeBatch(
            outcomes.schema_version,
            outcomes.challenge_id,
            split,
            tuple(row for row in outcomes.outcomes if row.episode_id in ids),
        ),
    )


def _lodo_evaluation(
    *,
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    groups_path: Path,
    word_bins: int,
    char_bins: int,
    folds: int,
    primary_alpha: float,
    companion_alpha: float,
    alpha_candidates: Optional[Sequence[float]] = None,
) -> Mapping[str, Any]:
    if np is None:
        raise RuntimeError("LODO 평가에는 NumPy가 필요합니다.")
    matrix, targets = trainer._matrix_and_targets(
        inputs, outcomes, policy, word_bins, char_bins
    )
    groups, families, _strategy = trainer._load_groups(inputs, groups_path)
    if families is None:
        raise ProtocolError("LODO에는 출처 family 라벨이 필요합니다.")
    family_array = np.asarray(families, dtype=object)
    dense = len(prompt_router.DENSE_FEATURE_NAMES)
    companion_columns = np.concatenate(
        (
            np.arange(dense, dtype=np.int64),
            np.arange(dense + word_bins, matrix.shape[1], dtype=np.int64),
        )
    )
    combined = {tier: [None] * len(inputs.episodes) for tier in TIERS}
    family_reports: Dict[str, Any] = {}
    for family in sorted(set(families)):
        train_mask = family_array != family
        test_indexes = np.where(~train_mask)[0].tolist()
        train_indexes = np.where(train_mask)[0].tolist()
        train_inputs, train_outcomes = _subset_batches(
            inputs, outcomes, train_indexes, f"lodo-fit-{family}"
        )
        test_inputs, test_outcomes = _subset_batches(
            inputs, outcomes, test_indexes, f"lodo-test-{family}"
        )
        inner_groups = tuple(groups[index] for index in train_indexes)
        inner_folds = trainer._fold_ids(inner_groups, min(folds, len(set(inner_groups))))
        train_matrix = matrix[train_mask]
        train_targets = targets[train_mask]
        companion_targets = train_targets[:, : len(trainer.COMPANION_NAMES)]
        if alpha_candidates is None:
            selected_primary_alpha = primary_alpha
            selected_companion_alpha = companion_alpha
            primary_oof = trainer._oof(
                train_matrix, train_targets, inner_folds, selected_primary_alpha
            )
            companion_oof = trainer._oof(
                train_matrix,
                companion_targets,
                inner_folds,
                selected_companion_alpha,
                companion_columns,
            )
            primary_diagnostics = None
            companion_diagnostics = None
        else:
            selected_primary_alpha, primary_oof, primary_diagnostics = (
                trainer._select_alpha(
                    train_matrix,
                    train_targets,
                    inner_folds,
                    alpha_candidates,
                    companion=False,
                )
            )
            selected_companion_alpha, companion_oof, companion_diagnostics = (
                trainer._select_alpha(
                    train_matrix,
                    companion_targets,
                    inner_folds,
                    alpha_candidates,
                    companion_columns,
                    companion=True,
                )
            )
        bounds = trainer._cost_bounds(train_targets, primary_oof, inner_folds)
        oof_predictions = trainer._prediction_objects(
            train_inputs, primary_oof, companion_oof
        )
        tier_policies, _selection = trainer._select_thresholds(
            train_inputs,
            train_outcomes,
            policy,
            oof_predictions,
            word_bins,
            char_bins,
            bounds,
        )
        mean, scale, primary_intercept, primary_coefficients = trainer._fit_raw(
            train_matrix, train_targets, selected_primary_alpha
        )
        _, _, companion_intercept, companion_coefficients = trainer._fit_raw(
            train_matrix,
            companion_targets,
            selected_companion_alpha,
            companion_columns,
        )
        standardized_dense = ((train_matrix - mean) / scale)[:, :dense]
        max_z = np.max(np.abs(standardized_dense), axis=1)
        rms_z = np.sqrt(np.mean(standardized_dense**2, axis=1))
        ood = prompt_router.OodThresholds(
            float(np.quantile(max_z, 0.995) * 1.05),
            float(np.quantile(rms_z, 0.995) * 1.05),
        )
        value = trainer._artifact_value(
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
            training_summary={
                "optimizer": "lodo-diagnostic",
                "held_out_family": family,
                "num_episodes": len(train_inputs.episodes),
                "dev_used_for_selection": False,
            },
        )
        artifact = prompt_router.parse_artifact(value)
        plans = tuple(
            prompt_router.make_prompt_submission(test_inputs, policy, artifact, tier)
            for tier in TIERS
        )
        c2 = tuple(
            tierfixed.make_submission(test_inputs, policy, tier) for tier in TIERS
        )
        score = _score(
            test_inputs,
            test_outcomes,
            policy,
            [plan.submission for plan in plans],
        )
        c2_score = _score(test_inputs, test_outcomes, policy, c2)
        family_reports[family] = {
            "num_episodes": len(test_inputs.episodes),
            "inner_selected_alpha": {
                "primary": selected_primary_alpha,
                "companion": selected_companion_alpha,
                "primary_objectives": primary_diagnostics,
                "companion_objectives": companion_diagnostics,
            },
            "prompt_aware": score,
            "tierfixed_c2": c2_score,
            "delta_final_score": float(score["final_score"]) - float(c2_score["final_score"]),
            "runtime_diagnostics": {
                tier: {
                    "conservative_budget_ratio": plan.conservative_budget_ratio,
                    "ood_count": plan.ood_count,
                }
                for tier, plan in zip(TIERS, plans)
            },
        }
        for tier, plan in zip(TIERS, plans):
            for global_index, decision in zip(test_indexes, plan.submission.decisions):
                combined[tier][global_index] = decision.model_id
    if any(model is None for models in combined.values() for model in models):
        raise AssertionError("LODO 결정 결합에 누락이 있습니다.")
    combined_submissions = tuple(
        _submission(inputs, policy, tier, combined[tier])  # type: ignore[arg-type]
        for tier in TIERS
    )
    c2_full = tuple(tierfixed.make_submission(inputs, policy, tier) for tier in TIERS)
    combined_score = _score(inputs, outcomes, policy, combined_submissions)
    c2_score = _score(inputs, outcomes, policy, c2_full)
    return {
        "protocol": (
            "each family excluded from coefficients, alpha selection, inner "
            "thresholds, and cost bounds"
            if alpha_candidates is not None
            else "each family excluded from coefficients, inner thresholds, and cost bounds"
        ),
        "combined_prompt_aware": combined_score,
        "combined_tierfixed_c2": c2_score,
        "combined_delta_final_score": float(combined_score["final_score"])
        - float(c2_score["final_score"]),
        # Analysis-only outer predictions let the v2 evaluator compare the
        # current v1 and v2 on exactly the same held rows.  They are never
        # serialized into a runtime artifact.
        "combined_models": {
            tier: list(combined[tier]) for tier in TIERS
        },
        "families": family_reports,
    }


def evaluate(
    *,
    input_path: Path,
    outcomes_path: Path,
    artifact_path: Path,
    hash_artifact_path: Path,
    report_path: Path,
    policy: RoutingPolicy,
    random_repeats: int,
    bootstrap_repeats: int,
    train_input_path: Optional[Path] = None,
    train_outcomes_path: Optional[Path] = None,
    groups_path: Optional[Path] = None,
    training_report_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    inputs = load_input(input_path)
    outcomes = load_outcomes(outcomes_path)
    artifact = prompt_router.load_artifact(artifact_path)
    reports, submissions = _official_and_prompt_baselines(
        inputs, outcomes, policy, artifact, hash_artifact_path
    )
    predictions = _runtime_predictions(inputs, artifact)
    cost_only = _cost_only_control(
        inputs, policy, predictions, submissions["prompt_aware"]
    )
    shuffled = _shuffled_prediction_control(
        inputs, policy, artifact, predictions, seed=20260818
    )
    reports["cost_only_same_counts"] = _score(
        inputs, outcomes, policy, cost_only
    )
    reports["prediction_shuffle"] = _score(
        inputs, outcomes, policy, shuffled
    )
    random_scores = []
    random_seed_zero = None
    for seed in range(random_repeats):
        random_submissions = _random_same_counts(
            inputs, policy, submissions["prompt_aware"], seed
        )
        score = _score(inputs, outcomes, policy, random_submissions)
        random_scores.append(float(score["final_score"]))
        if seed == 0:
            random_seed_zero = score
    result: Dict[str, Any] = {
        "report_type": "ossp-prompt-router-evaluation-v1",
        "evaluation_split_used_for_selection": False,
        "official_scorer_reports": reports,
        "random_same_counts": {
            "repeats": random_repeats,
            "score_distribution": _distribution(random_scores),
            "seed_zero_official_report": random_seed_zero,
        },
        "fixed_policy_evaluation_bootstrap": _fixed_policy_bootstrap(
            inputs,
            outcomes,
            policy,
            submissions["prompt_aware"],
            submissions["tierfixed_c2"],
            bootstrap_repeats,
        ),
    }
    if training_report_path is not None:
        training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
        result["same_family_group_oof"] = {
            "prompt_aware": training_report.get("group_oof_score"),
            "tierfixed_c2": training_report.get("group_oof_tierfixed_c2"),
            "delta_final_score": training_report.get(
                "group_oof_delta_final_score"
            ),
        }
    if train_input_path is not None or train_outcomes_path is not None or groups_path is not None:
        if train_input_path is None or train_outcomes_path is None or groups_path is None:
            raise ProtocolError("LODO의 train input, outcomes, groups는 모두 필요합니다.")
        train_inputs = load_input(train_input_path)
        train_outcomes = load_outcomes(train_outcomes_path)
        result["leave_one_domain_out"] = _lodo_evaluation(
            inputs=train_inputs,
            outcomes=train_outcomes,
            policy=policy,
            groups_path=groups_path,
            word_bins=artifact.word_bins,
            char_bins=artifact.char_bins,
            folds=int(artifact.training_summary["folds"]),
            primary_alpha=float(artifact.training_summary["primary_alpha"]),
            companion_alpha=float(artifact.training_summary["companion_alpha"]),
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(report_path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="prompt-aware 라우터와 대조군을 평가합니다.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--hash-artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--random-repeats", type=int, default=200)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--train-input", type=Path)
    parser.add_argument("--train-outcomes", type=Path)
    parser.add_argument("--groups-npz", type=Path)
    parser.add_argument("--training-report", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = load_policy(args.policy) if args.policy is not None else load_bundled_policy()
        result = evaluate(
            input_path=args.input,
            outcomes_path=args.outcomes,
            artifact_path=args.artifact,
            hash_artifact_path=args.hash_artifact,
            report_path=args.report,
            policy=policy,
            random_repeats=args.random_repeats,
            bootstrap_repeats=args.bootstrap_repeats,
            train_input_path=args.train_input,
            train_outcomes_path=args.train_outcomes,
            groups_path=args.groups_npz,
            training_report_path=args.training_report,
        )
    except (OSError, ProtocolError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    prompt_score = result["official_scorer_reports"]["prompt_aware"]["final_score"]
    print(f"OK: 평가 보고서를 생성했습니다 (prompt-aware {prompt_score}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
