<!--
SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
SPDX-License-Identifier: Apache-2.0
-->

# Efficient LLM Routing Challenge

**프롬프트 난이도·특성에 따라 최적 모델을 선택하는 compute-efficient routing
오픈소스 라우터 개발 챌린지**

> 이 fork의 최종 참가 라우터는 [`docs/PARTICIPANT_ROUTER.md`](docs/PARTICIPANT_ROUTER.md)에
> 정리되어 있습니다. 컨테이너 entrypoint는 E01 prompt-aware router를 실행하고,
> artifact 또는 예측 신뢰성 문제가 있으면 C2 tier-fixed 정책으로 복귀합니다.

이 과제에서는 입력 프롬프트의 내용만 보고 다음 세 평가용 모델 프로필 중 하나를
선택하는 라우터를 만듭니다.

- `ax31-light`
- `ax31`
- `axk1-think`

라우터는 모델을 직접 호출하지 않습니다. 운영자는 라우터가 선택한 모델과
미리 계산해 둔 모델별 평가 결과를 결합하여 품질과 비용을 계산합니다.
따라서 문항마다 프롬프트 내용으로 모델 하나를 한 번 선택하며, 실시간으로
모델 답변을 호출하거나 여러 답변을 비교하는 단계는 없습니다.

## 참가 순서

1. 이 저장소를 참가 팀의 GitHub 계정이나 조직으로 fork합니다.
2. 공개 Train/Dev 자료와 규칙을 확인하고 baseline에서 구현을 시작합니다.
3. `self-check`와 컨테이너 실행으로 세 등급의 선택 결과를 확인합니다.
4. 제출할 코드 커밋을 공개하고, 그 커밋에서 `linux/arm64` 이미지를 빌드해
   공개 레지스트리에 push합니다.
5. 저장소 루트에 `submission-ossp-skt.json`을 추가해 별도 커밋하고, 이
   커밋의 고정된 GitHub 스냅샷 URL을 결과보고서의 `프로젝트 등록 URL`에
   기재합니다.

로컬 clone 등 개발 방법과 브랜치 이름은 자유입니다. 다만 제출 시점부터 평가가
끝날 때까지 평가할 fork와 커밋을 별도 권한 없이 열 수 있어야 합니다.
수상팀은 수상일로부터 5년 동안 제출 저장소를 공개 상태로 유지해야 합니다.

질문과 문서·하네스 오류 신고는 이 저장소의 GitHub Issues에서 받습니다.

## 공개 Train/Dev 준비

참가자에게 Train 1,760문항과 Dev 880문항을 제공합니다. 각 문항에는 라우팅
입력과 모델별 실행 결과에서 산출한 점수 및 토큰 사용량이 포함됩니다. 일부
원천 자료는 라이선스 조건에 따라 고정된 절차로 내려받거나 재현합니다.
비공개 평가 자료의 구성과 분할 기준은 공개하지 않습니다.

재배포 가능한 프롬프트와 모델 답변 본문을 제외한 평가 결과는 `data/train/`과
`data/dev/`에 있습니다. 재배포가 불가한 AIME 원문은 타 repository로부터
Train/Dev에 필요한 고정 파일만 공개 출처에서 받아 결합합니다.
자료 생성에는 Python 3.10 이상이 필요합니다.

```console
python3 -m venv .venv-data
.venv-data/bin/pip install -r data/sources/requirements-materialize-public-data.txt
.venv-data/bin/python tools/materialize_public_data.py
```

완성된 입력은 Git 비추적 경로인 `data/materialized/train/inputs.json`과
`data/materialized/dev/inputs.json`에 생깁니다. 입력 수와 SHA-256은
[`data/public-data.v1.json`](data/public-data.v1.json), 출처와 고지는
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)에서 확인할 수 있습니다.

## 라우터 실행 입력

입력 JSON에는 정수 `schema_version: 1`, `challenge_id`, 데이터 구분을
나타내는 `split`, `episodes`가 들어 있습니다. 각 문항은 최대 128자인
불투명한 `episode_id`와 다음 둘 중 하나만 포함합니다.

- 비어 있지 않은 `prompt`
- `system`, `user`, `assistant` 역할과 `content`로 구성된 비어 있지 않은
  `messages`

공식 평가에서 벤치마크 이름, 데이터 출처, 정답, 모델 답변과 문항별 모델
평가 결과는 라우터 실행 입력으로 제공하지 않습니다. `challenge_id`, `split`,
`episode_id`는 실행 검증과 선택 결과의 문항 연결에만 사용하며 모델 선택에는
사용할 수 없습니다. 해시, 정규식, n-gram, 임베딩처럼 프롬프트 내용에서 직접
계산한 정보는 모델 선택에 사용할 수 있습니다.

공개 Train/Dev에서는 프롬프트와 별도 평가 결과를 연결하고 공개 비용 정책을
적용해 모델별 비용을 계산할 수 있습니다. 이 정보는 학습·검증과 등급별 정책
최적화에 사용할 수 있습니다. 공식 평가 실행 때는 문항별 실제 비용이
제공되지 않으므로, 필요한 경우 공개 정책과 프롬프트 특징으로 비용을
추정할 수 있습니다.

## 라우터 선택 결과

라우터는 `fast`, `balanced`, `premium` 세 등급에 대해 각각 제출 JSON을
만듭니다. 모든 입력 문항마다 `episode_id`와 `model_id`를 정확히 한 번
기록해야 합니다.

| 등급 | 최대 비용 비율 | 최종 점수 가중치 |
| --- | ---: | ---: |
| Fast | 1.25 | 0.4 |
| Balanced | 2.0 | 0.3 |
| Premium | 4.0 | 0.3 |

비용 비율은 같은 입력 전체를 `ax31-light`로 선택했을 때의 비용을 1로 둔
상대값입니다. 한도를 넘은 등급의 점수는 0입니다.

## 왜 이런 평가 방식인가요?

실제 서비스에서는 앞으로 들어올 요청의 분포를 완벽하게 알 수 없으며, 모델
서빙에도 동시성·대기열·메모리 같은 용량 한계가 있습니다. 이 과제는 공개
Train/Dev로 정책을 개발하되 별도 입력에서도 일반화하고, 정해진 비용 안에서
품질을 높이는 상황을 모사합니다. 예산을 넘긴 정책은 대기열 증가, 응답 시간
목표 위반이나 서빙 실패를 일으킬 수 있는 운영 불가능한 구성으로 보아 해당
등급을 0점 처리합니다.

## Quickstart: baseline에서 시작하기

별도 패키지를 설치하지 않고 toy 자료에서 baseline과 채점 흐름을 확인할 수
있습니다. 먼저 모든 문항에 경량 모델을 선택하는 세 등급 결과를 만듭니다.

```console
PYTHONPATH=src python3 baselines/always_light.py \
  --input data/toy/inputs.json \
  --output-dir build/toy-submission

PYTHONPATH=src python3 -m ossp_router.cli self-check \
  --input data/toy/inputs.json \
  --outcomes data/toy/outcomes.json \
  --submissions build/toy-submission \
  --report build/toy-report.json
```

첫 번째 명령은 세 등급의 선택 결과를 만들고, 두 번째 명령은 파일 형식, 문항
누락 여부, 비용 한도와 점수를 검사합니다. 다음으로 프롬프트 길이, 언어,
코드·수학 기호만 사용하는 baseline을 세 등급에 실행해 볼 수 있습니다.

```console
for tier in fast balanced premium; do
  PYTHONPATH=src python3 baselines/prompt_heuristic.py \
    --input data/toy/inputs.json \
    --tier "$tier" \
    --output "build/prompt-heuristic/$tier.json"
done

PYTHONPATH=src python3 -m ossp_router.cli self-check \
  --input data/toy/inputs.json \
  --outcomes data/toy/outcomes.json \
  --submissions build/prompt-heuristic \
  --report build/prompt-heuristic-report.json
```

[`src/ossp_router/heuristic.py`](src/ossp_router/heuristic.py)의 특징 추출과
`select_model`을 바꾸는 것이 가장 짧은 구현 경로입니다. 등급·문항 ID·입력
순서가 아니라 프롬프트 내용만 모델 선택 함수에 전달하십시오. 더 강한 특징
baseline과 공개 Train/Dev로 학습하는 예제는
[baseline 안내](baselines/README.md)에 있습니다.

정책 파일은 패키지에 포함된 동결 v1을 기본으로 사용하며, 별도 파일을
검사할 때만 `--policy`를 지정합니다.

저장소 루트에 기술 제출 정보 파일을 작성한 뒤에는 다음 명령으로 여섯 필드,
코드 커밋 SHA, 이미지 다이제스트와 라이선스 값을 확인합니다.

```console
python3 tools/validate_technical_submission.py
```

최종 이미지의 실행 시간과 자원 제한은 공개 Train/Dev 전체로 미리 확인할 수
있습니다. 로컬 태그는 검사 시작 시 변경 불가능한 이미지 ID로 고정됩니다.

```console
docker build --pull --platform linux/arm64 \
  --file container/Dockerfile --tag my-router:check .

PYTHONPATH=src python3 tools/check_runtime.py \
  --image my-router:check \
  --report build/runtime-check-report.json
```

이 검사는 위 materialization으로 만든 공개 Train 1,760문항과 Dev 880문항만
사용합니다. 공개 모델별 outcome과 최종 평가 자료는 컨테이너에 전달하지
않으며, 공식 장비와 다른 환경에서 측정한 시간은 참고값입니다.

## 문서

이 챌린지를 이해하는 데 가장 중요한 네 문서는 다음과 같습니다.

- [과제 규칙](docs/CHALLENGE_RULES.md)
- [제출 안내](docs/SUBMISSION.md)
- [컨테이너 실행 규격](docs/RUNTIME.md)
- [데이터 카드](docs/DATA_CARD.md)

점수와 예외 처리가 필요할 때 참고해 주세요.

- [점수 계산](docs/SCORING.md)
- [실행 오류와 규칙 집행](docs/ENFORCEMENT.md)
- [데이터 라이선스](DATA_LICENSES.md)

공개 운영 절차와 자원 측정 근거는 [전체 문서 안내](docs/README.md)에 별도로
모았습니다. 라우터 구현에 필요한 필수 문서는 아닙니다.

출품작 제출 마감은 2026년 8월 27일 18:00(대한민국 표준시)이며,
[공식 대회 접수 사이트](https://osscontest.kr/)의 출품작 제출 절차를 따릅니다.
공식 결과보고서 원본 파일과 PDF를 업로드하며, 결과보고서의 `프로젝트 등록
URL`로 공개 저장소를 제출합니다. 마감 전에는 결과보고서를 복수로 제출하거나
자유롭게 다시 업로드할 수 있으며 마지막으로 접수된 파일을 심사합니다.

`submission-ossp-skt.json`은 사이트에 별도로 업로드하지 않고 제출 저장소
루트에 반드시 커밋합니다. 파일 형식과 최종 커밋 순서는
[제출 안내](docs/SUBMISSION.md)에 기록합니다.

## 제공 내용

이 저장소에는 공개 Train/Dev 자료, 네 가지 baseline, 형식·점수 검증 도구,
참가자용 컨테이너 예제와 공개 평가 하네스가 들어 있습니다. 공식 플랫폼은
`linux/arm64`이며 최종 자원 한도는
[컨테이너 실행 규격](docs/RUNTIME.md)에 동결했습니다.

## 라이선스

프로젝트가 직접 작성한 코드와 문서는 [Apache License 2.0](LICENSE)으로
제공합니다. 이 라이선스는 제3자 벤치마크 자료를 재라이선스하지 않습니다.
자료별 조건은 [DATA_LICENSES.md](DATA_LICENSES.md)에 따로 기록합니다.
