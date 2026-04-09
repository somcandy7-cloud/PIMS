# PIMS 현재 구현체 명세

**날짜:** 2026-04-06  
**목적:** 현재 구현된 서비스의 배경·형태·기능·기술 구조를 통합 기록

---

## 1. 배경

### 문제

코크스 공장 OVEN 설비에는 이동차(Transfer Car)가 운행된다. 이동차 구동부에는 인버터·모터·엔코더 등이 연결되어 있고, ibaPDA 시스템이 수백 개의 PLC 신호를 고주파로 수집해 CSV로 내보낸다.

Trip(갑작스러운 정지 또는 이상 동작)이 발생하면 엔지니어가 CSV를 열어 어느 신호가 먼저 급변했는지 수동으로 확인한다. 이 과정은:
- 컬럼 수가 많아(수백 개) 직관적으로 파악이 어렵다
- 신호명이 `DB420.DBW 2` 같은 PLC 주소 형식이라 의미를 모르면 해석 불가
- Trip마다 반복 수작업이 필요하다

### 목표

ibaPDA CSV를 자동으로 분석해 **비전문가도 이해할 수 있는 한국어로 Trip 원인과 이상 신호를 즉시 제시**하는 로컬 도구를 만든다.

### 사용 환경

- 현장 PC에서 로컬 실행 (인터넷 불필요)
- Python + Streamlit 기반 웹 UI (브라우저에서 접근)
- Ollama 로컬 LLM 연동 (선택적)
- 외부 서버·클라우드 없음

---

## 2. 서비스 형태

### 두 가지 실행 모드

**① 대시보드 모드 (주 사용)**

```
streamlit run dashboard.py
```

브라우저에서 대화형으로 CSV를 분석한다. 엔지니어가 CSV 경로를 입력하고 "분석 실행"을 누르면 Trip 이벤트 목록과 시각화가 즉시 나타난다. 신호를 선택하거나 파라미터를 조정하면 실시간으로 재분석된다.

**② 파이프라인 데몬 모드**

```
python main.py              # 폴더 감시 데몬
python main.py --file <경로> # 단일 파일 즉시 분석
```

ibaPDA가 CSV를 자동으로 내보내는 폴더를 감시하다가 신규 파일이 생기면 자동으로 분석·리포트·알람을 실행한다.

### 데이터 흐름

```
ibaPDA Analyzer
    │ CSV 내보내기
    ▼
[폴더 감시] ──→ [분석 파이프라인] ──→ 리포트 (.txt)
                                  └──→ 알람 (Windows 토스트 + 로그)

또는

엔지니어가 CSV 경로 입력
    │
    ▼
[대시보드] ──→ 시각화 + 인터랙티브 분석
```

---

## 3. 프론트엔드 (대시보드)

Streamlit 기반 단일 페이지 앱. URL: `http://localhost:8501`

### 사이드바

| 요소 | 설명 |
|------|------|
| CSV 파일 경로 입력 | 분석할 iba CSV의 전체 경로 |
| RoC Z-Score 임계값 슬라이더 | 1.0 ~ 6.0, 기본 3.0. 높을수록 더 큰 변화만 탐지 |
| 롤링 윈도우 슬라이더 | 10 ~ 200 샘플, 기본 50. 크면 더 긴 패턴을 기준으로 판단 |
| 급변 신호 표시 개수 슬라이더 | 3 ~ 20개, 기본 5 |
| 분석 실행 버튼 | CSV 로드 + Trip 탐지 실행 |
| 제외 신호 관리 | 제외 중인 신호 목록과 복원 버튼 |
| 임계값 조정 신호 관리 | 오버라이드된 신호별 임계값과 초기화 버튼 |

### 메인 영역

#### 요약 카드 (상단 4개)

```
[ 총 샘플 수 ]  [ 기록 신호 수 ]  [ 탐지된 Trip ]  [ 데이터 구간 ]
```

#### Trip 이벤트 선택

드롭다운으로 탐지된 Trip 중 하나를 선택한다.  
레이블 형식: `Trip 01 | 15:49:23 | 2개 신호 동시 급변 (Trip 추정)`

#### 이벤트 요약 + 급변 신호 Bar (2단 레이아웃)

**왼쪽 — 이벤트 요약:**
- 발생 시각 (KST)
- 장애 유형
- 심각도 (🔴 높음 / 🟡 중간 / 🟢 낮음)
- 급변 신호 수
- **원인 추정** (텍스트 박스)
- **점검 항목** (번호 리스트)

**오른쪽 — 급변 신호 Bar 차트:**
- x축: 변화량(절대값), y축: 신호명
- 수평 막대 그래프 (crimson 색상)
- 차트 아래 신호 선택 라디오 버튼 → 추이 차트 연동

#### 신호 추이 차트

선택한 신호의 Trip 전후 ± N초 구간을 표시한다.

- **파란 실선**: 실제 신호값
- **초록 점선**: 정상 평균 (μ) — 전체 데이터 앞 30% 기준
- **주황 점선**: μ±2σ 정상 범위 상·하한
- **빨간 수직선**: Trip 발생 시각
- 구간 슬라이더: ±10 ~ ±300초 조정 가능

#### 급변 신호 통계 테이블

이벤트의 상위 N개 신호 각각에 대해 한 행씩 표시:

```
신호명 | Trip 시점 값 | 정상 평균 | 정상 σ | 변화량 | [제외] [+0.5]
```

- **[제외]**: 해당 신호를 탐지에서 완전히 제외 → `signals.yaml` 저장 → 자동 재분석
- **[+0.5]**: 해당 신호의 임계값 +0.5 상향 → `signals.yaml` 저장 → 자동 재분석

#### 전체 Trip 타임라인

데이터 전체 구간 위에 모든 Trip 발생 시각을 수직선으로 오버레이한다.
- 현재 선택 중인 Trip: 빨간 굵은 선
- 나머지: 주황 얇은 선
- 레이블: T1, T2, T3 …

---

## 4. 기능 명세

### F-01. CSV 로드 및 파싱

- **입력**: iba PDA Analyzer CSV (비표준 포맷)
- **동작**: 인코딩 자동 탐지 → 컬럼명 추출(Row 1) → 레코드 파싱 → UTC DatetimeIndex DataFrame 생성
- **특이사항**: 모든 레코드가 단일 행에 `;`로 연결된 포맷을 처리. 컬럼명 중복 시 자동 suffix(`_1`, `_2`) 부여

### F-02. 신호 전처리

- **동작**: 비수치 컬럼 제거 → 결측치 ffill → bfill → 0 채우기 → float64 변환
- **목적**: 이후 통계 연산에서 NaN/타입 오류 방지

### F-03. Trip 이벤트 탐지

- **방법**: Rolling Rate-of-Change Z-Score
- **아날로그 신호 자동 선별**: unique 값 > 10인 컬럼만 대상 (디지털 ON/OFF 신호 자동 제외)
- **노이즈 제거**: std 하한 = global_std × 0.1 (flat 구간에서 tiny noise 폭발 방지)
- **단발 노이즈 제거**: min_duration 샘플 이상 연속으로 임계값 초과해야 Trip 인정
- **이벤트 병합**: Rising edge 기준으로 연속 구간을 단일 이벤트로 처리
- **출력**: `TripEvent` 리스트 (시각, trigger 유형, 상위 N개 급변 신호)

### F-04. 장애 유형 분류

- **방법**: trigger 유형 + 신호명 패턴 기반 룰 엔진
- **분류 결과**: 장애 유형, 심각도, 점검 항목 리스트, 원인 추정 텍스트
- **trigger 유형**:
  - `multi_signal_drop`: 2개 이상 신호 동시 급감 → 구동부 전원 차단/인버터 보호 동작 추정
  - `single_signal_drop`: 단일 신호 급감 → 센서 불량/단선 추정
  - `spike`: 단일 신호 급등 → 과부하/노이즈 추정

### F-05. 신호 제외 관리

- 오탐지 신호를 대시보드에서 즉시 제외
- `signals.yaml`의 `excluded_signals`에 영속 저장
- 사이드바에서 복원 가능

### F-06. 신호별 임계값 조정

- 특정 신호가 민감하게 탐지될 때 임계값을 +0.5씩 상향
- `signals.yaml`의 `signal_overrides`에 영속 저장
- 사이드바에서 초기화 가능

### F-07. 텍스트 리포트 생성

- 파이프라인 모드에서 Trip당 `.txt` 리포트 파일 생성
- 내용: 발생 시각, 장애 유형, 심각도, 원인 추정, 점검 항목, 급변 신호 목록

### F-08. 알람 발송

- Windows 토스트 알림 (plyer 설치 시)
- `alarm.log`에 WARNING 레벨 기록
- 대시보드 모드에서는 미사용

### F-09. 폴더 감시 자동 실행

- watchdog으로 지정 폴더를 실시간 감시
- 신규 `.csv` 파일 생성 감지 시 F-01~F-08 자동 실행
- 파이프라인 오류 발생 시 로그 기록 후 감시 계속

---

## 5. 기술 구조

### 파일 구조

```
PIMS/
├── dashboard.py                  # Streamlit 대시보드 (주 사용 진입점)
├── main.py                       # CLI / 폴더 감시 데몬
├── config/
│   ├── signals.yaml              # 신호 그룹, 제외 목록, 임계값 오버라이드
│   └── settings.yaml             # 경로, LLM 설정, 분석 파라미터
├── src/
│   ├── data/
│   │   ├── loader.py             # IbaCSVLoader
│   │   └── schema.py             # AlarmThreshold, SignalSchema
│   ├── analysis/
│   │   ├── preprocessor.py       # Preprocessor
│   │   ├── trip_detector.py      # TripDetector, TripEvent
│   │   ├── fault_classifier.py   # FaultClassifier, FaultReport
│   │   ├── statistical.py        # ZScoreDetector (보조)
│   │   └── signal_discovery.py   # SignalGroup, classify_numeric_columns (보조)
│   ├── alarm/
│   │   └── notifier.py           # notify()
│   ├── report/
│   │   └── template_reporter.py  # generate()
│   └── watcher/
│       └── file_watcher.py       # FolderWatcher, CsvHandler
└── tests/                        # pytest 테스트 모음
```

### 핵심 데이터 흐름

```
IbaCSVLoader.load()
    → pd.DataFrame (UTC index, 수백 개 컬럼)

Preprocessor.process()
    → 수치형 컬럼만, 결측치 제거

TripDetector.detect()
    → list[TripEvent]

FaultClassifier.classify(event)
    → FaultReport

template_reporter.generate() / dashboard 시각화
```

### 캐싱 전략 (대시보드)

```python
@st.cache_data  load_and_process(path)
    # 동일 CSV 경로에 대해 로드/전처리 결과 재사용

@st.cache_data  detect_trips(path, thr, win, topn, excluded, overrides_key)
    # 파라미터가 하나라도 바뀌면 재실행
    # 신호 제외/임계값 변경 시: detect_trips.clear() → st.rerun()
```

### 알고리즘 상세 — Rolling RoC Z-Score

```
RoC[t] = |signal[t] - signal[t-1]|

roll_mean[t] = mean(RoC[t-window : t])
roll_std[t]  = max(std(RoC[t-window : t]), global_std × 0.1)

z[t] = (RoC[t] - roll_mean[t]) / roll_std[t]

exceeded[t] = z[t] > threshold (신호별 임계값 적용)

sustained  = rolling(min_duration, sum) >= min_duration
trip_event = rising_edge(sustained)  ← 연속 구간 단일 이벤트 병합
```

### 의존성

| 패키지 | 용도 |
|--------|------|
| pandas | 시계열 데이터프레임 |
| numpy | 통계 연산 |
| streamlit | 대시보드 UI |
| plotly | 인터랙티브 차트 |
| pyyaml | 설정 파일 파싱 |
| watchdog | 폴더 감시 데몬 |
| plyer | Windows 토스트 알림 (optional) |

---

## 6. 현재 미구현 / 보조 모듈

| 항목 | 상태 | 예정 |
|------|------|------|
| `ZScoreDetector` | 구현됨, 미연동 | 보조 분석용 (TripDetector와 병행 가능) |
| `SignalGroup` / `filter_by_patterns` | 구현됨, 미연동 | 신호 역할 파악 단계용 |
| `SignalSchema` / `AlarmThreshold` | 구현됨, 미연동 | signals.yaml 연동 완성 시 활용 |
| 신호명 → 센서명 매핑 | 미구현 | 고도화 Phase 1 |
| 설비 이미지 오버레이 | 미구현 | 고도화 Phase 1 |
| LLM 알람 추천 | 미구현 | 고도화 Phase 1 |
| 이상치 이력 누적 + 패턴 제안 | 미구현 | 고도화 Phase 1 |
| 통계 기반 자동 베이스라인 갱신 | 미구현 | Phase 2 로드맵 |