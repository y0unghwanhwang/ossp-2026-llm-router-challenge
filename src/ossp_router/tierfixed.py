# SPDX-FileCopyrightText: Copyright 2026 황영환
# SPDX-License-Identifier: Apache-2.0

"""등급별 고정 라우터 — 프롬프트를 읽지 않는다.

등급 하나마다 모델 하나를 상수로 정한다. 프롬프트 내용, 토큰 수, 학습한 계수,
배치 구성 어느 것도 보지 않는다.

    fast     → ax31-light
    balanced → ax31-light
    premium  → ax31

이 배정을 고른 근거는 별도 기술 보고에 있고, 요약하면 다음 셋이다.

1. 공개 Dev 880문항 공식 채점 0.641051136364. 전량 light(0.619318181818) 대비 +0.021733.
2. 이 배정은 주최측이 먼저 공개한 두 고정 baseline 의 등급별 조합과 같다 —
   `always_light` 의 fast·balanced 와 `prompt_heuristic` 의 premium(조건 없이 ax31)이다.
   따라서 우리가 탐색해서 고른 것이 아니라 외생적으로 주어진 후보다.
3. 예산 여유. premium 을 전량 ax31 로 올려도 비용비율이 2.102 이고 한도가 4.0 이다.
   fast·balanced 는 전량 light 이므로 비율이 정의상 정확히 1.0 이라 예산으로는 소각할 수 없다.

프롬프트를 읽지 않으므로 토크나이저·출처 판별·출력 길이 추정이 전부 필요 없고,
`episode_id`·입력 순서·`challenge_id`·`split` 에도 의존하지 않는다. 같은 등급이면
어떤 배치에서든 같은 선택을 낸다.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .protocol import (
    TIERS,
    Decision,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    dumps_json,
    load_bundled_policy,
    load_input,
    load_policy,
    parse_submission,
    submission_to_dict,
)

__all__ = ["TIER_MODEL", "select_model", "make_submission", "main"]


#: 등급 → 모델. 이 표가 라우터의 전부다.
TIER_MODEL: Mapping[str, str] = {
    "fast": "ax31-light",
    "balanced": "ax31-light",
    "premium": "ax31",
}


def select_model(tier: str, policy: RoutingPolicy) -> str:
    """등급 하나에 대한 모델을 돌려준다. 문항을 인자로 받지 않는다."""

    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")
    model_id = TIER_MODEL[tier]
    # 정책 파일이 바뀌어 모델 이름이 사라지면 조용히 잘못된 제출을 만들지 말고 여기서 멈춘다.
    if model_id not in policy.models:
        raise ProtocolError(
            f"정책에 없는 모델을 선택했습니다: {model_id}"
        )
    return model_id


def make_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    tier: str,
) -> Submission:
    """한 등급의 완전한 v1 제출 객체를 만든다."""

    if inputs.schema_version != policy.schema_version:
        raise ProtocolError("입력과 정책의 schema_version이 일치하지 않습니다.")
    model_id = select_model(tier, policy)
    decisions = tuple(
        Decision(episode.episode_id, model_id) for episode in inputs.episodes
    )
    submission = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=decisions,
    )
    # 생성기와 공개 v1 파서를 같은 엄격 경로에 둔다.
    return parse_submission(submission_to_dict(submission))


def write_submission_atomic(path: Path, submission: Submission) -> None:
    """부분 기록된 JSON 이 유효해 보이는 상태로 남지 않게 쓴다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            dumps_json(submission_to_dict(submission)),
            encoding="utf-8",
        )
        temporary.chmod(0o644)
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router-run",
        description="등급별 고정 라우터를 한 등급에 대해 실행합니다.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = (
            load_policy(args.policy)
            if args.policy is not None
            else load_bundled_policy()
        )
        submission = make_submission(inputs, policy, args.tier)
        write_submission_atomic(args.output, submission)
    except (OSError, ProtocolError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(f"OK: {args.tier} 제출 파일을 생성했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
