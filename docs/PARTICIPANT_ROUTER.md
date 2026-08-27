<!--
SPDX-FileCopyrightText: Copyright 2026 황영환
SPDX-License-Identifier: Apache-2.0
-->

# 제출 라우터: E01 prompt-aware router

이 참가작은 프롬프트 본문만으로 모델을 선택하는 경량 배치 라우터다. 실행 중
외부 네트워크, 모델 호출, 문항 ID, split 이름, 입력 순서, 정답 또는 문항별
평가 결과를 사용하지 않는다.

## 구성

- 21개 길이·언어·수학·코드·추론 표면 특징
- signed-hash 단어 1/2-gram 128차원
- signed-hash Unicode 문자 3-gram 128차원
- Light 대비 AX31·Think의 품질 차이와 비용을 예측하는 선형 head
- 등급별 배치 비용 한도 안에서만 행동을 적용하는 보수적 allocator
- OOD, 개별 예측 오류, artifact 누락·손상·정책 불일치 시 C2 tier-fixed fallback

런타임은 Python 표준 라이브러리만 사용한다. 학습된 JSON artifact는
`src/ossp_router/resources/prompt-router-public.v1.json`이며 SHA-256은
`ef6d1352a0964d7a3adc14bd47ae69ae572b3e8d9c2b2dd59f58f256f6287c29`이다.

## 실행

컨테이너 entrypoint는 다음 인자를 받는다.

```text
--input /challenge/input/inputs.json
--tier fast|balanced|premium
--output /challenge/output/submission.json
```

로컬 테스트와 공식 자원 한도 검사는 다음과 같이 실행한다.

```console
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -B -m unittest discover \
  -s tests -p 'test_*.py'

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -B tools/check_runtime.py \
  --image <linux/arm64-image> --report build/runtime-check-report.json
```

## 주요 파일

- `src/ossp_router/prompt_router.py`: 특징 추출, 예측, 배치 allocator
- `src/ossp_router/tierfixed.py`: C2 fallback
- `tools/train_prompt_router.py`: 공개 Train 기반 학습
- `tools/evaluate_prompt_router.py`: 평가·일반화 진단
- `tools/analyze_prompt_router_v1.py`: 공개 Dev 행동 기여도 분석
- `tests/test_prompt_router.py`: 결정 불변성, 예산, OOD·artifact fallback 검사

공개 Dev 측정은 실제 비공개 평가 성능을 보증하지 않는다. 따라서 이 참가작은
불확실하거나 유효하지 않은 학습 신호를 강제로 적용하지 않고 등급별 C2 선택으로
복귀하도록 설계했다.
