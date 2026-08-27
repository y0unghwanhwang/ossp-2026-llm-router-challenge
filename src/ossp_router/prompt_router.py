# SPDX-FileCopyrightText: Copyright 2026 황영환
# SPDX-License-Identifier: Apache-2.0

"""Prompt-aware, batch-budgeted router with a tier-fixed fallback.

The runtime deliberately uses only the Python standard library.  Training is
implemented separately in ``tools/train_prompt_router.py`` and produces one
small JSON artifact containing global linear coefficients.  No episode ID,
split name, input position, outcome, or prompt lookup table is used here.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .heuristic import episode_text, extract_features
from .protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    Episode,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_json,
    load_policy,
    parse_submission,
    policy_sha256,
    submission_to_dict,
)
from .tierfixed import TIER_MODEL, make_submission as make_tierfixed_submission
from .tierfixed import write_submission_atomic


ARTIFACT_TYPE = "ossp-prompt-pairwise-linear-v1"
FEATURE_VERSION = 1
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parent / "resources" / "prompt-router-public.v1.json"
)
DEFAULT_WORD_BINS = 128
DEFAULT_CHAR_BINS = 128
MIN_HASH_BINS = 16
MAX_HASH_BINS = 4_096

LIGHT = "ax31-light"
AX31 = "ax31"
THINK = "axk1-think"
DELTA_AX31 = "delta_ax31"
DELTA_THINK = "delta_think"
PROB_AX31 = "prob_ax31"
PROB_THINK = "prob_think"

_FNV_OFFSET = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
_UINT64_MASK = (1 << 64) - 1
_TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", re.UNICODE)
_FORMAL = re.compile(
    r"\b(?:prove|derive|theorem|lemma|counterexample|induction|"
    r"증명|유도|정리|보조정리|반례|귀납)\b",
    re.IGNORECASE,
)
_PROGRAM = re.compile(
    r"```|\b(?:traceback|exception|complexity|big[- ]?o|"
    r"시간\s*복잡도|공간\s*복잡도|예외|스택\s*추적)\b",
    re.IGNORECASE,
)
_CONSTRAINT = re.compile(
    r"\b(?:exactly|at least|at most|must|only|without|except|"
    r"정확히|이상|이하|반드시|오직|제외하고)\b",
    re.IGNORECASE,
)
_TRANSFORM = re.compile(
    r"\b(?:summari[sz]e|rewrite|translate|list|extract|classify|"
    r"요약|다시\s*쓰|번역|나열|추출|분류)\b",
    re.IGNORECASE,
)
_COMPARE = re.compile(
    r"\b(?:compare|contrast|difference|better|worse|versus|vs\.?|"
    r"비교|차이|장단점|더\s*나은)\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(
    r"\b(?:who|what|when|where|which|why|how|explain|solve|answer|"
    r"누구|무엇|언제|어디|어떤|왜|어떻게|설명|풀어|답)\b|\?",
    re.IGNORECASE,
)

DENSE_FEATURE_NAMES = (
    "log_character_count",
    "log_word_count",
    "log_sentence_count",
    "log_message_count",
    "hangul_ratio",
    "ascii_letter_ratio",
    "log_code_marker_count",
    "log_math_marker_count",
    "numeric_density",
    "newline_density",
    "long_context",
    "log_reasoning_marker_count",
    "formal_reasoning",
    "program_analysis",
    "log_constraint_count",
    "simple_transform",
    "comparison",
    "log_question_marker_count",
    "log_system_message_count",
    "log_assistant_message_count",
    "log_role_transition_count",
)
PRIMARY_HEAD_NAMES = (
    DELTA_AX31,
    DELTA_THINK,
    PROB_AX31,
    PROB_THINK,
    f"log_cost_{LIGHT}",
    f"log_cost_{AX31}",
    f"log_cost_{THINK}",
)
COMPANION_HEAD_NAMES = (DELTA_AX31, DELTA_THINK, PROB_AX31, PROB_THINK)


@dataclass(frozen=True)
class LinearHead:
    intercept: float
    coefficients: Tuple[float, ...]


@dataclass(frozen=True)
class CostBounds:
    light_lower_multiplier: float
    upper_multipliers: Mapping[str, float]


@dataclass(frozen=True)
class OodThresholds:
    dense_max_abs_z: float
    dense_rms_z: float


@dataclass(frozen=True)
class TierPolicy:
    target_budget_ratio: float
    min_win_probability: float


@dataclass(frozen=True)
class PromptRouterArtifact:
    word_bins: int
    char_bins: int
    feature_mean: Tuple[float, ...]
    feature_scale: Tuple[float, ...]
    primary_heads: Mapping[str, LinearHead]
    companion_heads: Mapping[str, LinearHead]
    cost_bounds: CostBounds
    ood_thresholds: OodThresholds
    tier_policies: Mapping[str, TierPolicy]
    policy_id: str
    policy_digest: str
    training_summary: Mapping[str, Any]


@dataclass(frozen=True)
class EpisodePrediction:
    text_fingerprint: int
    primary_delta_ax31: float
    primary_delta_think: float
    companion_delta_ax31: float
    companion_delta_think: float
    primary_prob_ax31: float
    primary_prob_think: float
    companion_prob_ax31: float
    companion_prob_think: float
    mean_costs: Mapping[str, float]
    is_ood: bool


@dataclass(frozen=True)
class PromptRouterPlan:
    submission: Submission
    conservative_budget_ratio: float
    fallback_count: int
    ood_count: int


def _stable_hash(value: str) -> int:
    digest = _FNV_OFFSET
    for character in value:
        # Hash Unicode code points directly.  This is deterministic across
        # Python processes and avoids the randomized built-in hash().
        digest ^= ord(character)
        digest = (digest * _FNV_PRIME) & _UINT64_MASK
    return digest


def _normalized_tokens(text: str) -> Tuple[str, ...]:
    result = []
    for token in _TOKEN.findall(text):
        normalized = token.casefold()
        if normalized.isdecimal():
            normalized = "<number>"
        result.append(normalized)
    return tuple(result)


def _signed_bins(values: Sequence[str], bins: int) -> Tuple[float, ...]:
    vector = [0.0] * bins
    for value in values:
        digest = _stable_hash(value)
        index = digest & (bins - 1)
        vector[index] += -1.0 if digest & (1 << 63) else 1.0
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return tuple(vector)


def _validate_bins(value: int, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not MIN_HASH_BINS <= value <= MAX_HASH_BINS
        or value & (value - 1)
    ):
        raise ValueError(f"{label}는 허용 범위의 2의 거듭제곱이어야 합니다.")


def raw_feature_vector(
    episode: Episode,
    word_bins: int = DEFAULT_WORD_BINS,
    char_bins: int = DEFAULT_CHAR_BINS,
) -> Tuple[float, ...]:
    """Extract dense, word 1/2-gram, and Unicode character 3-gram features."""

    _validate_bins(word_bins, "word_bins")
    _validate_bins(char_bins, "char_bins")
    features = extract_features(episode)
    text = episode_text(episode)
    nonspace = max(1, sum(not character.isspace() for character in text))
    ascii_letters = sum(character.isascii() and character.isalpha() for character in text)
    messages = episode.messages or ()
    roles = tuple(message.role for message in messages)
    system_count = sum(role == "system" for role in roles)
    assistant_count = sum(role == "assistant" for role in roles)
    role_transitions = sum(left != right for left, right in zip(roles, roles[1:]))
    dense = (
        math.log1p(features.character_count),
        math.log1p(features.word_count),
        math.log1p(features.sentence_count),
        math.log1p(features.message_count),
        features.hangul_ratio,
        ascii_letters / nonspace,
        math.log1p(features.code_marker_count),
        math.log1p(features.math_marker_count),
        features.numeric_density,
        text.count("\n") / max(1, len(text)),
        float(features.long_context),
        math.log1p(features.reasoning_marker_count),
        float(bool(_FORMAL.search(text))),
        float(bool(_PROGRAM.search(text))),
        math.log1p(len(_CONSTRAINT.findall(text))),
        float(bool(_TRANSFORM.search(text))),
        float(bool(_COMPARE.search(text))),
        math.log1p(len(_QUESTION.findall(text))),
        math.log1p(system_count),
        math.log1p(assistant_count),
        math.log1p(role_transitions),
    )
    tokens = _normalized_tokens(text)
    word_grams = [f"w1:{token}" for token in tokens]
    word_grams.extend(
        f"w2:{left}\x1f{right}" for left, right in zip(tokens, tokens[1:])
    )
    lowered = text.casefold()
    char_grams = [f"c3:{lowered[index:index + 3]}" for index in range(max(0, len(lowered) - 2))]
    vector = dense + _signed_bins(word_grams, word_bins) + _signed_bins(char_grams, char_bins)
    if any(not math.isfinite(value) for value in vector):
        raise ValueError("프롬프트 특징에 유한하지 않은 값이 있습니다.")
    return vector


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label}은(는) JSON 객체여야 합니다.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    missing = sorted(set(expected) - set(value))
    extra = sorted(set(value) - set(expected))
    if missing or extra:
        raise ProtocolError(f"{label} 필드 오류: 누락={missing}, 초과={extra}")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ProtocolError(f"{label}은(는) 유한한 숫자여야 합니다.")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{label}은(는) 유한한 숫자여야 합니다.")
    return result


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ProtocolError(f"{label} 값이 허용 범위를 벗어났습니다.")
    return value


def _vector(value: Any, length: int, label: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ProtocolError(f"{label}은(는) 길이 {length}의 배열이어야 합니다.")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def _head(value: Any, length: int, label: str) -> LinearHead:
    raw = _object(value, label)
    _exact_keys(raw, ("intercept", "coefficients"), label)
    return LinearHead(
        intercept=_number(raw["intercept"], f"{label}.intercept"),
        coefficients=_vector(raw["coefficients"], length, f"{label}.coefficients"),
    )


def parse_artifact(value: Any) -> PromptRouterArtifact:
    root = _object(value, "artifact")
    expected = (
        "artifact_type",
        "schema_version",
        "feature_version",
        "hash_algorithm",
        "word_bins",
        "char_bins",
        "dense_feature_names",
        "model_ids",
        "policy_id",
        "policy_sha256",
        "feature_mean",
        "feature_scale",
        "primary_heads",
        "companion_heads",
        "cost_bounds",
        "ood_thresholds",
        "tier_policies",
        "training_summary",
    )
    _exact_keys(root, expected, "artifact")
    if root["artifact_type"] != ARTIFACT_TYPE:
        raise ProtocolError("지원하지 않는 prompt-router artifact_type입니다.")
    if _integer(root["schema_version"], "schema_version", 1, 1) != 1:
        raise ProtocolError("지원하지 않는 artifact schema_version입니다.")
    if _integer(root["feature_version"], "feature_version", 1, 1) != FEATURE_VERSION:
        raise ProtocolError("지원하지 않는 feature_version입니다.")
    if root["hash_algorithm"] != "fnv1a64-unicode-signed-word12-char3":
        raise ProtocolError("지원하지 않는 feature hash 방식입니다.")
    word_bins = _integer(root["word_bins"], "word_bins", MIN_HASH_BINS, MAX_HASH_BINS)
    char_bins = _integer(root["char_bins"], "char_bins", MIN_HASH_BINS, MAX_HASH_BINS)
    try:
        _validate_bins(word_bins, "word_bins")
        _validate_bins(char_bins, "char_bins")
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc
    if root["dense_feature_names"] != list(DENSE_FEATURE_NAMES):
        raise ProtocolError("artifact의 dense feature 정의가 런타임과 다릅니다.")
    if root["model_ids"] != list(MODEL_IDS):
        raise ProtocolError("artifact의 모델 집합이 공개 정책과 다릅니다.")
    length = len(DENSE_FEATURE_NAMES) + word_bins + char_bins
    mean = _vector(root["feature_mean"], length, "feature_mean")
    scale = _vector(root["feature_scale"], length, "feature_scale")
    if any(value <= 0 for value in scale):
        raise ProtocolError("feature_scale은 모두 0보다 커야 합니다.")
    primary_raw = _object(root["primary_heads"], "primary_heads")
    companion_raw = _object(root["companion_heads"], "companion_heads")
    _exact_keys(primary_raw, PRIMARY_HEAD_NAMES, "primary_heads")
    _exact_keys(companion_raw, COMPANION_HEAD_NAMES, "companion_heads")
    primary = {name: _head(primary_raw[name], length, f"primary_heads.{name}") for name in PRIMARY_HEAD_NAMES}
    companion = {name: _head(companion_raw[name], length, f"companion_heads.{name}") for name in COMPANION_HEAD_NAMES}

    cost_raw = _object(root["cost_bounds"], "cost_bounds")
    _exact_keys(cost_raw, ("light_lower_multiplier", "upper_multipliers"), "cost_bounds")
    lower = _number(cost_raw["light_lower_multiplier"], "cost_bounds.light_lower_multiplier")
    upper_raw = _object(cost_raw["upper_multipliers"], "cost_bounds.upper_multipliers")
    _exact_keys(upper_raw, MODEL_IDS, "cost_bounds.upper_multipliers")
    upper = {model: _number(upper_raw[model], f"upper_multipliers.{model}") for model in MODEL_IDS}
    if lower <= 0 or any(value <= 0 for value in upper.values()):
        raise ProtocolError("비용 상·하한 배수는 0보다 커야 합니다.")

    ood_raw = _object(root["ood_thresholds"], "ood_thresholds")
    _exact_keys(ood_raw, ("dense_max_abs_z", "dense_rms_z"), "ood_thresholds")
    ood = OodThresholds(
        dense_max_abs_z=_number(ood_raw["dense_max_abs_z"], "ood_thresholds.dense_max_abs_z"),
        dense_rms_z=_number(ood_raw["dense_rms_z"], "ood_thresholds.dense_rms_z"),
    )
    if ood.dense_max_abs_z <= 0 or ood.dense_rms_z <= 0:
        raise ProtocolError("OOD 임계값은 0보다 커야 합니다.")

    tier_raw = _object(root["tier_policies"], "tier_policies")
    _exact_keys(tier_raw, TIERS, "tier_policies")
    tiers: Dict[str, TierPolicy] = {}
    for tier in TIERS:
        item = _object(tier_raw[tier], f"tier_policies.{tier}")
        _exact_keys(item, ("target_budget_ratio", "min_win_probability"), f"tier_policies.{tier}")
        tiers[tier] = TierPolicy(
            target_budget_ratio=_number(item["target_budget_ratio"], f"tier_policies.{tier}.target_budget_ratio"),
            min_win_probability=_number(item["min_win_probability"], f"tier_policies.{tier}.min_win_probability"),
        )
        if not 0 <= tiers[tier].min_win_probability <= 1:
            raise ProtocolError("최소 승률은 0과 1 사이여야 합니다.")
    policy_id = root["policy_id"]
    digest = root["policy_sha256"]
    if not isinstance(policy_id, str) or not policy_id:
        raise ProtocolError("artifact.policy_id가 올바르지 않습니다.")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ProtocolError("artifact.policy_sha256가 올바르지 않습니다.")
    summary = _object(root["training_summary"], "training_summary")
    return PromptRouterArtifact(
        word_bins=word_bins,
        char_bins=char_bins,
        feature_mean=mean,
        feature_scale=scale,
        primary_heads=primary,
        companion_heads=companion,
        cost_bounds=CostBounds(lower, upper),
        ood_thresholds=ood,
        tier_policies=tiers,
        policy_id=policy_id,
        policy_digest=digest,
        training_summary=dict(summary),
    )


def load_artifact(path: Path = DEFAULT_ARTIFACT_PATH) -> PromptRouterArtifact:
    return parse_artifact(load_json(path))


def _linear(head: LinearHead, values: Sequence[float]) -> float:
    return head.intercept + math.fsum(
        coefficient * value for coefficient, value in zip(head.coefficients, values)
    )


def predict_episode(episode: Episode, artifact: PromptRouterArtifact) -> EpisodePrediction:
    raw = raw_feature_vector(episode, artifact.word_bins, artifact.char_bins)
    standardized = tuple(
        (value - mean) / scale
        for value, mean, scale in zip(raw, artifact.feature_mean, artifact.feature_scale)
    )
    primary = {name: _linear(artifact.primary_heads[name], standardized) for name in PRIMARY_HEAD_NAMES}
    companion = {name: _linear(artifact.companion_heads[name], standardized) for name in COMPANION_HEAD_NAMES}
    costs = {
        model: math.exp(min(50.0, max(-50.0, primary[f"log_cost_{model}"])))
        for model in MODEL_IDS
    }
    costs[AX31] = max(costs[AX31], costs[LIGHT] * (1.0 + 1e-12))
    costs[THINK] = max(costs[THINK], costs[AX31] * (1.0 + 1e-12))
    dense_z = standardized[: len(DENSE_FEATURE_NAMES)]
    max_abs_z = max((abs(value) for value in dense_z), default=0.0)
    rms_z = math.sqrt(math.fsum(value * value for value in dense_z) / max(1, len(dense_z)))
    is_ood = (
        max_abs_z > artifact.ood_thresholds.dense_max_abs_z
        or rms_z > artifact.ood_thresholds.dense_rms_z
    )
    text = episode_text(episode)
    return EpisodePrediction(
        text_fingerprint=_stable_hash(text),
        primary_delta_ax31=primary[DELTA_AX31],
        primary_delta_think=primary[DELTA_THINK],
        companion_delta_ax31=companion[DELTA_AX31],
        companion_delta_think=companion[DELTA_THINK],
        primary_prob_ax31=min(1.0, max(0.0, primary[PROB_AX31])),
        primary_prob_think=min(1.0, max(0.0, primary[PROB_THINK])),
        companion_prob_ax31=min(1.0, max(0.0, companion[PROB_AX31])),
        companion_prob_think=min(1.0, max(0.0, companion[PROB_THINK])),
        mean_costs=costs,
        is_ood=is_ood,
    )


def _conservative_budget_ratio(
    selected: Sequence[str],
    predictions: Sequence[Optional[EpisodePrediction]],
    artifact: PromptRouterArtifact,
) -> float:
    valid = [item for item in predictions if item is not None]
    if len(valid) != len(predictions) or not valid:
        return math.inf if predictions else 1.0
    lower_light_total = artifact.cost_bounds.light_lower_multiplier * math.fsum(
        item.mean_costs[LIGHT] for item in valid
    )
    if lower_light_total <= 0 or not math.isfinite(lower_light_total):
        return math.inf
    # All-light has ratio exactly one because the official denominator is the
    # same measured all-light cost.  Only uncertain incremental cost is bounded.
    upper_increment = 0.0
    for model, item in zip(selected, valid):
        if model == LIGHT:
            continue
        model_upper = artifact.cost_bounds.upper_multipliers[model] * item.mean_costs[model]
        light_lower = artifact.cost_bounds.light_lower_multiplier * item.mean_costs[LIGHT]
        upper_increment += max(0.0, model_upper - light_lower)
    return 1.0 + upper_increment / lower_light_total


def conservative_budget_slack(
    selected: Sequence[str],
    predictions: Sequence[Optional[EpisodePrediction]],
    artifact: PromptRouterArtifact,
    budget_multiplier: float,
) -> float:
    """Return the conservative batch analogue of ``sum(c_m - kappa*c_l)``.

    A non-positive value means the official budget inequality is satisfied
    under the artifact's aggregate cost bounds.  The allocator normally uses
    an even smaller target ratio, leaving an additional distribution-shift
    reserve inside the official multiplier.
    """

    if not math.isfinite(budget_multiplier) or budget_multiplier < 1.0:
        raise ValueError("budget_multiplier는 1 이상의 유한한 수여야 합니다.")
    valid = [item for item in predictions if item is not None]
    if len(valid) != len(predictions) or not valid:
        return math.inf if predictions else 0.0
    lower_light_total = artifact.cost_bounds.light_lower_multiplier * math.fsum(
        item.mean_costs[LIGHT] for item in valid
    )
    ratio = _conservative_budget_ratio(selected, predictions, artifact)
    return (ratio - budget_multiplier) * lower_light_total


def _quality_signal(
    prediction: EpisodePrediction,
    transition: str,
) -> Tuple[float, float, bool]:
    if transition == DELTA_AX31:
        first = prediction.primary_delta_ax31
        second = prediction.companion_delta_ax31
        probability = min(prediction.primary_prob_ax31, prediction.companion_prob_ax31)
    elif transition == DELTA_THINK:
        first = prediction.primary_delta_think
        second = prediction.companion_delta_think
        probability = min(prediction.primary_prob_think, prediction.companion_prob_think)
    else:  # pragma: no cover - internal caller uses constants.
        raise AssertionError(transition)
    agrees_positive = first > 0 and second > 0
    conservative_gain = min(first, second) if agrees_positive else min(first, second, 0.0)
    return conservative_gain, probability, agrees_positive


def _downgrade_signal(prediction: EpisodePrediction) -> Tuple[float, float, bool]:
    first = prediction.primary_delta_ax31
    second = prediction.companion_delta_ax31
    agrees_negative = first < 0 and second < 0
    conservative_gain = -max(first, second) if agrees_negative else 0.0
    probability = 1.0 - max(prediction.primary_prob_ax31, prediction.companion_prob_ax31)
    return conservative_gain, probability, agrees_negative


def _content_groups(
    episodes: Sequence[Episode],
    predictions: Sequence[Optional[EpisodePrediction]],
) -> Tuple[Tuple[int, ...], ...]:
    groups: Dict[str, list[int]] = {}
    for index, episode in enumerate(episodes):
        groups.setdefault(episode_text(episode), []).append(index)
    # Sorting by content-derived fingerprint, then text, makes ties independent
    # of episode IDs and input order.  Exact duplicates are always moved together.
    return tuple(
        tuple(groups[text])
        for text in sorted(groups, key=lambda value: (_stable_hash(value), value))
    )


def _ranked_actions(
    actions: Sequence[Tuple[Tuple[int, ...], str, float]],
    selected: Sequence[str],
    predictions: Sequence[Optional[EpisodePrediction]],
    artifact: PromptRouterArtifact,
) -> Tuple[Tuple[Tuple[int, ...], str, float], ...]:
    ranked = []
    current_ratio = _conservative_budget_ratio(selected, predictions, artifact)
    for indexes, model, gain in actions:
        candidate = list(selected)
        for index in indexes:
            candidate[index] = model
        next_ratio = _conservative_budget_ratio(candidate, predictions, artifact)
        extra = max(0.0, next_ratio - current_ratio)
        utility = gain * len(indexes) / max(extra, 1e-12)
        fingerprint = predictions[indexes[0]].text_fingerprint  # type: ignore[union-attr]
        ranked.append((-utility, -gain, fingerprint, indexes, model, gain))
    ranked.sort(key=lambda item: item[:3])
    return tuple((item[3], item[4], item[5]) for item in ranked)


def allocate_predictions(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: PromptRouterArtifact,
    tier: str,
    predictions: Sequence[Optional[EpisodePrediction]],
) -> Tuple[Tuple[str, ...], float, int, int]:
    """Allocate a whole batch under a conservative cost-ratio cap."""

    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")
    if len(predictions) != len(inputs.episodes):
        raise ProtocolError("예측 개수가 입력 episode 개수와 다릅니다.")
    tier_policy = artifact.tier_policies[tier]
    official_limit = float(policy.tiers[tier].budget_multiplier)
    if not 1.0 <= tier_policy.target_budget_ratio < official_limit:
        raise ProtocolError("artifact의 목표 비용 비율이 공식 한도 안쪽이 아닙니다.")
    selected = [TIER_MODEL[tier]] * len(inputs.episodes)
    groups = _content_groups(inputs.episodes, predictions)
    threshold = tier_policy.min_win_probability
    actions = []

    if tier in ("fast", "balanced"):
        for indexes in groups:
            items = [predictions[index] for index in indexes]
            if any(item is None or item.is_ood for item in items):
                continue
            signals = [_quality_signal(item, DELTA_AX31) for item in items if item is not None]
            gain = min(item[0] for item in signals)
            probability = min(item[1] for item in signals)
            if all(item[2] for item in signals) and gain > 0 and probability >= threshold:
                actions.append((indexes, AX31, gain))
        for indexes, model, _gain in _ranked_actions(actions, selected, predictions, artifact):
            candidate = list(selected)
            for index in indexes:
                candidate[index] = model
            if _conservative_budget_ratio(candidate, predictions, artifact) <= tier_policy.target_budget_ratio:
                selected = candidate
    else:
        # Quality-positive downgrades both improve the estimate and release
        # budget, so apply them before ranking think upgrades.
        for indexes in groups:
            items = [predictions[index] for index in indexes]
            if any(item is None or item.is_ood for item in items):
                continue
            signals = [_downgrade_signal(item) for item in items if item is not None]
            gain = min(item[0] for item in signals)
            probability = min(item[1] for item in signals)
            if all(item[2] for item in signals) and gain > 0 and probability >= threshold:
                for index in indexes:
                    selected[index] = LIGHT
        for indexes in groups:
            if any(selected[index] != AX31 for index in indexes):
                continue
            items = [predictions[index] for index in indexes]
            if any(item is None or item.is_ood for item in items):
                continue
            signals = [_quality_signal(item, DELTA_THINK) for item in items if item is not None]
            gain = min(item[0] for item in signals)
            probability = min(item[1] for item in signals)
            if all(item[2] for item in signals) and gain > 0 and probability >= threshold:
                actions.append((indexes, THINK, gain))
        for indexes, model, _gain in _ranked_actions(actions, selected, predictions, artifact):
            candidate = list(selected)
            for index in indexes:
                candidate[index] = model
            if _conservative_budget_ratio(candidate, predictions, artifact) <= tier_policy.target_budget_ratio:
                selected = candidate

    ratio = _conservative_budget_ratio(selected, predictions, artifact)
    # If even the conservative target is violated, do not invent a partially
    # trusted policy: return the single-source C2 baseline for this tier.
    if ratio > tier_policy.target_budget_ratio:
        selected = [TIER_MODEL[tier]] * len(inputs.episodes)
        ratio = _conservative_budget_ratio(selected, predictions, artifact)
    if conservative_budget_slack(
        selected, predictions, artifact, official_limit
    ) > 0:
        selected = [TIER_MODEL[tier]] * len(inputs.episodes)
        ratio = _conservative_budget_ratio(selected, predictions, artifact)
    fallback_count = sum(
        model == TIER_MODEL[tier]
        for model, prediction in zip(selected, predictions)
        if prediction is None or prediction.is_ood
    )
    ood_count = sum(prediction is not None and prediction.is_ood for prediction in predictions)
    return tuple(selected), ratio, fallback_count, ood_count


def make_prompt_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: PromptRouterArtifact,
    tier: str,
) -> PromptRouterPlan:
    if inputs.schema_version != policy.schema_version:
        raise ProtocolError("입력과 정책의 schema_version이 일치하지 않습니다.")
    if artifact.policy_id != policy.policy_id or artifact.policy_digest != policy_sha256(policy):
        raise ProtocolError("artifact가 현재 라우팅 정책과 일치하지 않습니다.")
    predictions: list[Optional[EpisodePrediction]] = []
    for episode in inputs.episodes:
        try:
            predictions.append(predict_episode(episode, artifact))
        except (ArithmeticError, UnicodeError, ValueError):
            predictions.append(None)
    selected, ratio, fallback_count, ood_count = allocate_predictions(
        inputs, policy, artifact, tier, predictions
    )
    submission = Submission(
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
    return PromptRouterPlan(
        submission=parse_submission(submission_to_dict(submission)),
        conservative_budget_ratio=ratio,
        fallback_count=fallback_count,
        ood_count=ood_count,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router-run",
        description="프롬프트 인지 선형 라우터를 한 등급에 대해 실행합니다.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--policy", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = load_policy(args.policy) if args.policy is not None else load_bundled_policy()
    except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    try:
        artifact = load_artifact(args.artifact)
        plan = make_prompt_submission(inputs, policy, artifact, args.tier)
        submission = plan.submission
        message = (
            f"보수 예측 비용비율 {plan.conservative_budget_ratio:.6f}, "
            f"OOD {plan.ood_count}, 개별 fallback {plan.fallback_count}"
        )
    except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as exc:
        # A corrupt, missing, or policy-mismatched learned artifact must not
        # cause an untrusted learned decision.  C2 remains the single fallback.
        submission = make_tierfixed_submission(inputs, policy, args.tier)
        message = f"artifact 무효로 C2 fallback ({exc})"
    try:
        write_submission_atomic(args.output, submission)
    except OSError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(f"OK: {args.tier} 제출 파일을 생성했습니다 ({message}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
