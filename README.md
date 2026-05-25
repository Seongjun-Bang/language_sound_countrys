# Language Sound Countrys (비원어민 한국어 발음 STT 오류 기반 모국어 분류 프로젝트)

본 프로젝트는 OpenAI Whisper 등 고성능 STT(Speech-to-Text) 모델을 활용하여, 한국어가 모국어가 아닌 외국인이 발음하는 한국어 음성을 텍스트로 변환하고, **그 과정에서 발생하는 인식 오류(오발음/전사 오류)의 패턴을 분석하여 해당 화자의 모국어(국적)를 분류 및 예측**하는 머신러닝 파이프라인 구축을 목표로 합니다.

---

## 🎯 프로젝트 개요 (Overview)

- **핵심 가설:** "화자의 모국어 발성 습관(Phonology)의 차이가 STT 모델의 인식 오류 패턴에 일관되게 나타날 것이다."
  - *예: 영어권 화자의 'ㄹ'/'ㄴ' 발음 오류, 중일어권 화자의 성조/장단음 및 모음 왜곡 등이 STT 결과에 독특한 오탈자 및 패턴으로 반영됨.*
- **활용 데이터:** AI Hub의 '교육용 각종 언어 모국어 사용자의 한국어 음성 데이터'
  - **040:** 영어 모국어 사용자의 한국어 음성 데이터 (`040.교육용_영어_모국어_사용자의_한국어_음성_데이터`)
  - **041:** 유럽어 모국어 사용자의 한국어 음성 데이터 (`041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터`)
  - **042:** 중·일어 모국어 사용자의 한국어 음성 데이터 (`042.교육용_중·일어_모국어_사용자의_한국어_음성_데이터`)
  - **043:** 아시아어(중·일어 제외) 사용자의 한국어 음성 데이터 (`043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터`)
- **주요 기능:**
  - AI Hub 대용량 오디오 및 라벨링 ZIP 아카이브 실시간 탐색 및 파싱
  - Whisper GPU 가속 배치 추론 및 화자 인적 사항/평가 메타데이터 추출
  - STT 환각(Hallucination) 필터링 및 텍스트 전처리 데이터 정제
  - 분류 모델 학습용 정합성 보정 및 이어하기/오류 데이터 복구 기능

---

## 📂 폴더 구조 및 역할 (Project Structure)

- **`process_dataset_zip.py`**
  - AI Hub 원본 음성 ZIP 파일(`TS*.zip`, `VS*.zip`) 및 라벨링 ZIP 파일(`TL*.zip`, `VL*.zip`)을 실시간으로 탐색하여 매핑하고 파싱합니다.
  - Whisper 모델(`large`)을 로드하여 GPU/CUDA 가속 배치 추론을 수행합니다.
  - 예측 텍스트(`prediction`)와 정답(`ground_truth`), 화자 메타데이터를 매핑하여 `whisper_outputs_*/` 폴더 아래 결과 CSV 파일로 정제해 나갑니다.
  - 대용량 데이터 처리 중 유실을 방지하고 중단 위치부터 이어할 수 있도록 트래커 파일(`processed_files_tracker_zip.txt`)을 관리합니다.

- **`repair_whisper_outputs.py`**
  - 생성된 STT 데이터의 정합성을 검증하고 노이즈를 제거하는 사후 보정 스크립트입니다.
  - Whisper의 빈번한 환각 패턴(예: *"자막 제공"*, *"구독과 좋아요"*, *"시청해주셔서 감사합니다"*, *"SBS 뉴스"* 등)을 정규표현식 기반으로 정밀 필터링합니다.
  - 메타데이터 및 평가 점수의 누락된 값을 표준화하여 최종 훈련 데이터셋으로 정제합니다.

- **`retry_problematic_whisper.py`**
  - ZIP 손상이나 예외 상황 등으로 누락되거나 오류가 발생했던 4대 대역별(영어, 유럽어, 중일어, 아시아어) 음성 데이터를 타겟팅하여 자동으로 재추론 및 복구를 처리하는 스크립트입니다.

- **`Plan.md`**
  - 데이터셋 수집 및 전처리, Whisper 추론, 오류 피처 추출, 분류 모델 모델링에 이르는 프로젝트의 단계별 일정과 구체적인 계획서가 작성되어 있습니다.

- **`output_sample.csv`**
  - 본 파이프라인을 완료하고 보정한 최종 맵핑 데이터셋의 예시입니다. 화자 메타데이터(프로필/평가지표), 오디오 출처, 정답 문장 및 Whisper 예측 결과가 포함되어 있습니다.

---

## 🚀 파이프라인 워크플로우 (Pipeline Workflow)

### 1. 데이터 준비 및 배치
AI Hub 등에서 확보한 데이터셋 폴더를 프로젝트 루트 경로 아래에 다음과 같이 구조화합니다.
```text
.
├── 040.교육용_영어_모국어_사용자의_한국어_음성_데이터/
├── 041.교육용_유럽어_모국어_사용자의_한국어_음성_데이터/
├── 042.교육용_중·일어_모국어_사용자의_한국어_음성_데이터/
├── 043.교육용_아시아어(중·일어_제외)_사용자의_한국어_음성_데이터/
├── process_dataset_zip.py
├── repair_whisper_outputs.py
├── retry_problematic_whisper.py
└── Plan.md
```

### 2. Whisper 배치 추론 및 메타데이터 매핑
가장 먼저 음성 오디오 ZIP을 풀고 Whisper 대형 모델을 통해 STT 문장을 추론하는 동시에 화자 메타데이터를 매핑합니다.
```bash
python process_dataset_zip.py
```
- **추출 대상 화자 메타데이터:**
  - `birth_year`, `gender`, `nationality` (국적), `language` (모국어)
  - `proficiency` (유창성), `topik_score`, `topik_level` (토픽 레벨)
  - `learning_period` (학습 기간), `korea_residency` (국내 거주 여부/기간)
  - `DeliveryEval`, `LanguageUseEval`, `ContentEval` (전문가 한국어 능력 평가지표)

### 3. 환각 필터링 및 텍스트 클렌징
Whisper 결과물 내에 섞인 대표적인 환각 메시지와 특수문자를 지우고 정답(Ground Truth)의 무의미한 텍스트를 걸러냅니다.
```bash
python repair_whisper_outputs.py
```

### 4. 오류 복구 및 완성
예외 상황으로 정상 기록되지 않았던 영역을 추적하여 최종 보완 및 병합을 완료합니다.
```bash
python retry_problematic_whisper.py
```

---

## 📊 데이터셋 스키마 (Dataset Schema)

정제 과정을 완료한 최종 머신러닝 피드용 데이터셋의 주요 스키마는 다음과 같습니다.

| 컬럼명 | 설명 | 예시 |
| :--- | :--- | :--- |
| `filename` | 음성 파일명 (WAV) | `F-041-DE-02-EIG-01-0023.wav` |
| `zip_source` | 원본 오디오 ZIP 아카이브 | `TS_유럽어_독일_02.zip` |
| `ground_truth` | 원본 발화 문장 (정답지) | `저는 한국 음식을 아주 좋아해요.` |
| `prediction` | Whisper ASR 추론 결과 (예측값) | `저는 한국 엄식을 아추 쪼아해요.` |
| `nationality` | 화자의 국적 / 모국어 | `독일 (Germany)` |
| `proficiency` | 한국어 학습 구사 유창 수준 | `Beginner` |
| `learning_period`| 한국어 공식 학습 기간 | `12 months` |
| `DeliveryEval` | 전문가의 전달력(발음) 평가 등급 (1~5) | `3` |

---

## 🛠️ 요구 사항 (Prerequisites)

- **OS:** Linux (Ubuntu 20.04/22.04 권장) 또는 Windows
- **Python:** 3.8 이상
- **주요 라이브러리:**
  - `openai-whisper`
  - `torch` (CUDA 지원 GPU 환경 탑재 권장)
  - `requests`, `pandas`
- **시스템 도구:** 오디오 리샘플링 및 파싱 처리를 위해 시스템에 `ffmpeg`가 설치되어 있어야 합니다.
  ```bash
  # Ubuntu
  sudo apt update && sudo apt install ffmpeg -y
  ```
