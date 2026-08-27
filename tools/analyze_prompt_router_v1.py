#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 황영환
# SPDX-License-Identifier: Apache-2.0

"""Decompose the fixed v1 Premium decisions against tier-fixed C2.

This is an offline diagnostic.  Episode IDs, outcomes, and source labels are
used only to join public evaluation rows after routing; none of them are
available to or imported by the runtime router.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - command-line dependency error
    np = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ossp_router.heuristic import extract_features  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
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
    load_submission,
    write_json,
)
from ossp_router.scoring import score_submissions  # noqa: E402


LIGHT = "ax31-light"
AX31 = "ax31"
THINK = "axk1-think"
POLICY_NAMES = ("c2", "downgrade_only", "upgrade_only", "v1")
ACTION_NAMES = ("downgrade", "upgrade")
DEFAULT_INPUT = ROOT / "data/materialized/dev/inputs.json"
DEFAULT_OUTCOMES = ROOT / "data/dev/outcomes.json"
DEFAULT_C2 = ROOT / "build/first-router/c2-dev"
DEFAULT_V1 = ROOT / "build/first-router/final-dev"
DEFAULT_SOURCE_MAP = ROOT / "build/first-router/diagnostics/source-map.npz"
DEFAULT_REPORT = ROOT / "build/first-router/diagnostics/premium-ablation-report.json"
DOCUMENT_VALUES_BEGIN = "<!-- PROMPT_ROUTER_V1_P0_VALUES"
DOCUMENT_VALUES_END = "PROMPT_ROUTER_V1_P0_VALUES -->"


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _submission_models(submission: Submission) -> Dict[str, str]:
    return {decision.episode_id: decision.model_id for decision in submission.decisions}


def _submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    tier: str,
    models: Mapping[str, str],
) -> Submission:
    return Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, models[episode.episode_id])
            for episode in inputs.episodes
        ),
    )


def build_ablation_models(
    inputs: InputBatch,
    c2_premium: Submission,
    v1_premium: Submission,
) -> Mapping[str, Mapping[str, str]]:
    """Return C2, each disjoint action, and full-v1 Premium model maps."""

    if c2_premium.tier != "premium" or v1_premium.tier != "premium":
        raise ProtocolError("Premium ablation에는 premium submission 두 개가 필요합니다.")
    episode_ids = {episode.episode_id for episode in inputs.episodes}
    c2 = _submission_models(c2_premium)
    v1 = _submission_models(v1_premium)
    if set(c2) != episode_ids or set(v1) != episode_ids:
        raise ProtocolError("C2/v1 Premium 결정이 입력 문항과 일치하지 않습니다.")
    if set(c2.values()) != {AX31}:
        raise ProtocolError("v1 Premium 분해의 C2는 전량 AX31이어야 합니다.")
    invalid = sorted(set(v1.values()) - {LIGHT, AX31, THINK})
    if invalid:
        raise ProtocolError(f"v1 Premium에 알 수 없는 모델이 있습니다: {invalid}")

    downgrade = dict(c2)
    upgrade = dict(c2)
    for episode_id, model_id in v1.items():
        if model_id == LIGHT:
            downgrade[episode_id] = LIGHT
        elif model_id == THINK:
            upgrade[episode_id] = THINK
    return {
        "c2": c2,
        "downgrade_only": downgrade,
        "upgrade_only": upgrade,
        "v1": v1,
    }


def build_ablation_submissions(
    inputs: InputBatch,
    policy: RoutingPolicy,
    c2_submissions: Mapping[str, Submission],
    v1_submissions: Mapping[str, Submission],
) -> Mapping[str, Tuple[Submission, ...]]:
    for tier in TIERS:
        if c2_submissions[tier].tier != tier or v1_submissions[tier].tier != tier:
            raise ProtocolError(f"{tier} submission 등급이 일치하지 않습니다.")
    for tier in ("fast", "balanced"):
        if _submission_models(c2_submissions[tier]) != _submission_models(
            v1_submissions[tier]
        ):
            raise ProtocolError(f"v1 {tier}는 C2와 같아야 Premium만 분해할 수 있습니다.")
    model_maps = build_ablation_models(
        inputs, c2_submissions["premium"], v1_submissions["premium"]
    )
    return {
        name: (
            c2_submissions["fast"],
            c2_submissions["balanced"],
            _submission(inputs, policy, "premium", model_maps[name]),
        )
        for name in POLICY_NAMES
    }


def _outcome_cost(outcome: Outcome, policy: RoutingPolicy) -> Decimal:
    rates = policy.models[outcome.model_id]
    unit = Decimal(policy.token_unit)
    return (
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )


def _load_source_map(path: Path, inputs: InputBatch) -> Tuple[Mapping[str, str], Mapping[str, Any]]:
    if np is None:
        raise RuntimeError("출처별 진단에는 NumPy가 필요합니다.")
    raw = np.load(path, allow_pickle=True)
    if not {"eids", "family"}.issubset(raw.files):
        raise ProtocolError("source-map npz에 eids/family 배열이 필요합니다.")
    episode_ids = [str(value) for value in raw["eids"].tolist()]
    families = [str(value) for value in raw["family"].tolist()]
    if len(episode_ids) != len(families) or len(episode_ids) != len(set(episode_ids)):
        raise ProtocolError("source-map의 ID/family 범위가 잘못되었습니다.")
    mapping = dict(zip(episode_ids, families))
    missing = sorted(
        episode.episode_id for episode in inputs.episodes if episode.episode_id not in mapping
    )
    if missing:
        raise ProtocolError(f"source-map에 Dev ID가 누락되었습니다: {missing[:5]}")
    selected = {episode.episode_id: mapping[episode.episode_id] for episode in inputs.episodes}
    ambiguous = len(raw["ambiguous"]) if "ambiguous" in raw.files else None
    unmatched = len(raw["unmatched"]) if "unmatched" in raw.files else None
    return selected, {
        "path_basis": "generated exact/public-source match; local build input",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows_total": len(mapping),
        "rows_for_evaluation": len(selected),
        "family_counts_for_evaluation": dict(sorted(Counter(selected.values()).items())),
        "ambiguous_total": ambiguous,
        "unmatched_total": unmatched,
    }


def _distribution(values: Sequence[float]) -> Mapping[str, float]:
    ordered = sorted(values)

    def quantile(probability: float) -> float:
        return ordered[int(round(probability * (len(ordered) - 1)))]

    return {
        "mean": statistics.fmean(values),
        "sd": statistics.pstdev(values),
        "p025": quantile(0.025),
        "p50": quantile(0.5),
        "p975": quantile(0.975),
    }


def _random_same_counts(
    *,
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    c2_fast_balanced: Sequence[Submission],
    premium: Submission,
    observed_score: float,
    name: str,
    repeats: int,
    seed: int,
) -> Mapping[str, Any]:
    if repeats < 1:
        raise ValueError("random repeats는 1 이상이어야 합니다.")
    base_models = [decision.model_id for decision in premium.decisions]
    scores = []
    overflows = 0
    for repeat in range(repeats):
        models = list(base_models)
        random.Random(f"v1-diagnostic:{seed}:{name}:{repeat}").shuffle(models)
        shuffled = Submission(
            premium.schema_version,
            premium.challenge_id,
            premium.policy_id,
            premium.split,
            "premium",
            tuple(
                Decision(episode.episode_id, model)
                for episode, model in zip(inputs.episodes, models)
            ),
        )
        report = score_submissions(
            inputs, outcomes, (*c2_fast_balanced, shuffled), policy
        )
        scores.append(float(report["final_score"]))
        if not report["tiers"]["premium"]["budget_passed"]:
            overflows += 1
    return {
        "repeats": repeats,
        "score_distribution": _distribution(scores),
        "observed_score": observed_score,
        "observed_percentile_strict": sum(value < observed_score for value in scores)
        / repeats,
        "add_one_probability_random_ge_observed": (
            1 + sum(value >= observed_score for value in scores)
        )
        / (repeats + 1),
        "premium_overflow_probability": overflows / repeats,
    }


def _feature_summary(rows: Sequence[Tuple[Any, str]]) -> Mapping[str, Any]:
    if not rows:
        return {"count": 0}
    extracted = [extract_features(episode) for episode, _family in rows]
    return {
        "count": len(rows),
        "family_counts": dict(sorted(Counter(family for _episode, family in rows).items())),
        "mean_character_count": statistics.fmean(value.character_count for value in extracted),
        "mean_word_count": statistics.fmean(value.word_count for value in extracted),
        "mean_hangul_ratio": statistics.fmean(value.hangul_ratio for value in extracted),
        "mean_numeric_density": statistics.fmean(value.numeric_density for value in extracted),
        "code_marker_rate": sum(value.code_marker_count > 0 for value in extracted)
        / len(extracted),
        "math_marker_rate": sum(value.math_marker_count > 0 for value in extracted)
        / len(extracted),
        "reasoning_marker_rate": sum(
            value.reasoning_marker_count > 0 for value in extracted
        )
        / len(extracted),
        "long_context_rate": sum(value.long_context for value in extracted)
        / len(extracted),
    }


def _aggregate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    num_episodes: int,
    premium_weight: Decimal,
) -> Mapping[str, Any]:
    items = list(rows)
    score_delta = sum((item["score_delta"] for item in items), Decimal("0"))
    cost_delta = sum((item["cost_delta"] for item in items), Decimal("0"))
    return {
        "changed_count": len(items),
        "beneficial_count": sum(item["score_delta"] > 0 for item in items),
        "neutral_count": sum(item["score_delta"] == 0 for item in items),
        "harmful_count": sum(item["score_delta"] < 0 for item in items),
        "quality_points_delta": _decimal_text(score_delta),
        "premium_quality_score_delta": _decimal_text(
            score_delta / Decimal(num_episodes)
        ),
        "final_score_delta_if_budget_passes": _decimal_text(
            score_delta * premium_weight / Decimal(num_episodes)
        ),
        "cost_delta": _decimal_text(cost_delta),
    }


def _contribution_report(
    *,
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    source_map: Mapping[str, str],
    c2_premium: Submission,
    v1_premium: Submission,
) -> Mapping[str, Any]:
    outcome_index = {
        (outcome.episode_id, outcome.model_id): outcome for outcome in outcomes.outcomes
    }
    episodes = {episode.episode_id: episode for episode in inputs.episodes}
    c2 = _submission_models(c2_premium)
    v1 = _submission_models(v1_premium)
    rows = []
    feature_groups: Dict[str, Dict[str, list[Tuple[Any, str]]]] = {
        action: {label: [] for label in ("beneficial", "neutral", "harmful")}
        for action in ACTION_NAMES
    }
    for episode in inputs.episodes:
        episode_id = episode.episode_id
        before = c2[episode_id]
        after = v1[episode_id]
        if after == before:
            continue
        action = "downgrade" if after == LIGHT else "upgrade"
        base_outcome = outcome_index[(episode_id, before)]
        changed_outcome = outcome_index[(episode_id, after)]
        score_delta = changed_outcome.score - base_outcome.score
        cost_delta = _outcome_cost(changed_outcome, policy) - _outcome_cost(
            base_outcome, policy
        )
        label = (
            "beneficial" if score_delta > 0 else "harmful" if score_delta < 0 else "neutral"
        )
        family = source_map[episode_id]
        rows.append(
            {
                "action": action,
                "family": family,
                "score_delta": score_delta,
                "cost_delta": cost_delta,
                "label": label,
            }
        )
        feature_groups[action][label].append((episodes[episode_id], family))

    premium_weight = policy.tiers["premium"].weight
    by_action = {
        action: _aggregate_rows(
            (row for row in rows if row["action"] == action),
            num_episodes=len(inputs.episodes),
            premium_weight=premium_weight,
        )
        for action in ACTION_NAMES
    }
    families = sorted(set(source_map.values()))
    by_family = {
        family: {
            "all_changes": _aggregate_rows(
                (row for row in rows if row["family"] == family),
                num_episodes=len(inputs.episodes),
                premium_weight=premium_weight,
            ),
            **{
                action: _aggregate_rows(
                    (
                        row
                        for row in rows
                        if row["family"] == family and row["action"] == action
                    ),
                    num_episodes=len(inputs.episodes),
                    premium_weight=premium_weight,
                )
                for action in ACTION_NAMES
            },
        }
        for family in families
    }
    positive_by_family: Dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for row in rows:
        if row["score_delta"] > 0:
            positive_by_family[row["family"]] += row["score_delta"]
    positive_total = sum(positive_by_family.values(), Decimal("0"))
    positive_shares = (
        {family: float(value / positive_total) for family, value in positive_by_family.items()}
        if positive_total
        else {}
    )
    return {
        "all_changes": _aggregate_rows(
            rows,
            num_episodes=len(inputs.episodes),
            premium_weight=premium_weight,
        ),
        "by_action": by_action,
        "by_family": by_family,
        "positive_quality_concentration": {
            "gross_positive_quality_points": _decimal_text(positive_total),
            "family_shares": dict(sorted(positive_shares.items())),
            "largest_family_share": max(positive_shares.values(), default=0.0),
            "herfindahl_index": sum(value * value for value in positive_shares.values()),
        },
        "prompt_feature_profiles": {
            action: {
                label: _feature_summary(feature_groups[action][label])
                for label in ("beneficial", "neutral", "harmful")
            }
            for action in ACTION_NAMES
        },
    }


def analyze(
    *,
    input_path: Path,
    outcomes_path: Path,
    c2_directory: Path,
    v1_directory: Path,
    source_map_path: Path,
    policy: RoutingPolicy,
    random_repeats: int,
    random_seed: int,
) -> Mapping[str, Any]:
    inputs = load_input(input_path)
    outcomes = load_outcomes(outcomes_path)
    c2 = {tier: load_submission(c2_directory / f"{tier}.json") for tier in TIERS}
    v1 = {tier: load_submission(v1_directory / f"{tier}.json") for tier in TIERS}
    source_map, source_metadata = _load_source_map(source_map_path, inputs)
    policies = build_ablation_submissions(inputs, policy, c2, v1)
    official_reports = {
        name: score_submissions(inputs, outcomes, submissions, policy)
        for name, submissions in policies.items()
    }
    summaries = {
        name: {
            "final_score": report["final_score"],
            "premium": report["tiers"]["premium"],
        }
        for name, report in official_reports.items()
    }
    c2_score = Decimal(summaries["c2"]["final_score"])
    for summary in summaries.values():
        summary["final_score_delta_vs_c2"] = _decimal_text(
            Decimal(summary["final_score"]) - c2_score
        )

    contribution = _contribution_report(
        inputs=inputs,
        outcomes=outcomes,
        policy=policy,
        source_map=source_map,
        c2_premium=c2["premium"],
        v1_premium=v1["premium"],
    )
    random_controls = {
        name: _random_same_counts(
            inputs=inputs,
            outcomes=outcomes,
            policy=policy,
            c2_fast_balanced=policies["c2"][:2],
            premium=policies[name][2],
            observed_score=float(official_reports[name]["final_score"]),
            name=name,
            repeats=random_repeats,
            seed=random_seed,
        )
        for name in ("downgrade_only", "upgrade_only", "v1")
    }
    upgrade_passed = bool(
        official_reports["upgrade_only"]["tiers"]["premium"]["budget_passed"]
    )
    premium_limit = policy.tiers["premium"].budget_multiplier
    upgrade_ratio = Decimal(
        official_reports["upgrade_only"]["tiers"]["premium"]["budget_ratio"]
    )
    interaction = (
        Decimal(official_reports["v1"]["final_score"])
        - Decimal(official_reports["downgrade_only"]["final_score"])
        - Decimal(official_reports["upgrade_only"]["final_score"])
        + Decimal(official_reports["c2"]["final_score"])
    )
    return {
        "schema_version": 1,
        "report_type": "ossp-prompt-router-v1-premium-ablation",
        "analysis_boundary": (
            "offline public-outcome diagnostic; IDs, outcomes, and source labels are not runtime features"
        ),
        "inputs": {
            "split": inputs.split,
            "num_episodes": len(inputs.episodes),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "outcomes_sha256": hashlib.sha256(outcomes_path.read_bytes()).hexdigest(),
            "source_map": source_metadata,
        },
        "policy_summaries": summaries,
        "official_scorer_reports": official_reports,
        "contributions_vs_c2_premium": contribution,
        "same_model_count_random_controls": random_controls,
        "budget_dependency": {
            "downgrades_are_required_for_the_ten_upgrades": not upgrade_passed,
            "upgrade_only_budget_passed": upgrade_passed,
            "upgrade_only_budget_ratio": _decimal_text(upgrade_ratio),
            "premium_budget_multiplier": _decimal_text(premium_limit),
            "upgrade_only_budget_ratio_headroom": _decimal_text(
                premium_limit - upgrade_ratio
            ),
            "cost_savings_receive_direct_score_credit": False,
        },
        "additivity_check": {
            "full_minus_downgrade_minus_upgrade_plus_c2": _decimal_text(interaction),
            "expected_when_all_policies_pass_budget": "0",
        },
    }


def document_values(report: Mapping[str, Any]) -> Mapping[str, str]:
    """Return the stable P0 values that the permanent report must reproduce."""

    summaries = report["policy_summaries"]
    actions = report["contributions_vs_c2_premium"]["by_action"]
    concentration = report["contributions_vs_c2_premium"][
        "positive_quality_concentration"
    ]
    controls = report["same_model_count_random_controls"]
    budget = report["budget_dependency"]
    values = {
        "c2.final_score": summaries["c2"]["final_score"],
        "c2.premium_quality": summaries["c2"]["premium"]["quality_score"],
        "c2.premium_cost": summaries["c2"]["premium"]["total_cost"],
        "c2.premium_ratio": summaries["c2"]["premium"]["budget_ratio"],
        "downgrade.final_score": summaries["downgrade_only"]["final_score"],
        "downgrade.final_delta": summaries["downgrade_only"][
            "final_score_delta_vs_c2"
        ],
        "downgrade.premium_quality": summaries["downgrade_only"]["premium"][
            "quality_score"
        ],
        "downgrade.premium_cost": summaries["downgrade_only"]["premium"][
            "total_cost"
        ],
        "downgrade.premium_ratio": summaries["downgrade_only"]["premium"][
            "budget_ratio"
        ],
        "upgrade.final_score": summaries["upgrade_only"]["final_score"],
        "upgrade.final_delta": summaries["upgrade_only"]["final_score_delta_vs_c2"],
        "upgrade.premium_quality": summaries["upgrade_only"]["premium"][
            "quality_score"
        ],
        "upgrade.premium_cost": summaries["upgrade_only"]["premium"]["total_cost"],
        "upgrade.premium_ratio": summaries["upgrade_only"]["premium"]["budget_ratio"],
        "v1.final_score": summaries["v1"]["final_score"],
        "v1.final_delta": summaries["v1"]["final_score_delta_vs_c2"],
        "v1.premium_quality": summaries["v1"]["premium"]["quality_score"],
        "v1.premium_cost": summaries["v1"]["premium"]["total_cost"],
        "v1.premium_ratio": summaries["v1"]["premium"]["budget_ratio"],
        "downgrade.quality_points": actions["downgrade"]["quality_points_delta"],
        "upgrade.quality_points": actions["upgrade"]["quality_points_delta"],
        "upgrade_only.headroom": budget["upgrade_only_budget_ratio_headroom"],
        "positive.largest_family_share": str(concentration["largest_family_share"]),
        "random.repeats": str(controls["v1"]["repeats"]),
        "random.upgrade_p": str(
            controls["upgrade_only"]["add_one_probability_random_ge_observed"]
        ),
    }
    return dict(sorted(values.items()))


def verify_document_values(report: Mapping[str, Any], document_path: Path) -> None:
    """Fail if the report's machine-readable P0 block is stale or incomplete."""

    text = document_path.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(DOCUMENT_VALUES_BEGIN)
        + r"\s*\n(?P<body>.*?)\n"
        + re.escape(DOCUMENT_VALUES_END),
        re.DOTALL,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"문서에 P0 자동 대조 블록이 없습니다: {document_path}")
    actual: Dict[str, str] = {}
    for line in match.group("body").splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in actual:
            raise ValueError(f"잘못된 P0 자동 대조 행입니다: {line!r}")
        actual[key] = value
    expected = document_values(report)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            key for key in set(expected) & set(actual) if expected[key] != actual[key]
        )
        raise ValueError(
            "문서 P0 수치가 원시 JSON과 다릅니다: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="v1 Premium의 52개 하향과 10개 상향을 C2 대비로 분해합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--c2-submissions", type=Path, default=DEFAULT_C2)
    parser.add_argument("--v1-submissions", type=Path, default=DEFAULT_V1)
    parser.add_argument("--source-map-npz", type=Path, default=DEFAULT_SOURCE_MAP)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--random-repeats", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260818)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--verify-document",
        type=Path,
        help="생성한 JSON과 문서의 P0 자동 대조 블록이 같은지 검사합니다.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = load_policy(args.policy) if args.policy else load_bundled_policy()
        report = analyze(
            input_path=args.input,
            outcomes_path=args.outcomes,
            c2_directory=args.c2_submissions,
            v1_directory=args.v1_submissions,
            source_map_path=args.source_map_npz,
            policy=policy,
            random_repeats=args.random_repeats,
            random_seed=args.random_seed,
        )
        write_json(args.report, report)
        if args.verify_document:
            verify_document_values(report, args.verify_document)
    except (OSError, ProtocolError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    summaries = report["policy_summaries"]
    for name in POLICY_NAMES:
        premium = summaries[name]["premium"]
        print(
            f"{name}: final={summaries[name]['final_score']} "
            f"premium_quality={premium['quality_score']} "
            f"premium_cost={premium['budget_ratio']}"
        )
    print(f"OK: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
