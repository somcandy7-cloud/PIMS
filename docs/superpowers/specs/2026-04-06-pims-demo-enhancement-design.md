# PIMS 발표 베타 고도화 설계

**날짜:** 2026-04-06  
**목표:** 교육 수료 발표(2주 후)에서 "실제 데이터 기반 이상 탐지 → 설비 위치 시각화 → AI 대응 가이드 → 자기개선 제안"을 보여주는 데모 베타 완성

---

## 배경 및 범위

현재 PIMS는 CSV 데이터 로드 → 이상치 탐지 → 룰 기반 분류 → 대시보드 표시까지 동작한다.  
이번 고도화는 4가지 축을 추가한다:

1. **신호 정체 파악** — PLC 신호명을 실제 센서명으로 매핑
2. **설비 위치 시각화** — 실제 이상치 센서의 설비 내 위치를 이미지로 표시
3. **LLM 알람 추천** — ibaPDA 매뉴얼 기반 대응 가이드 자동 생성
4. **모델 고도화 제안** — 이상치 이력 분석 후 LLM이 config 개선을 제안, 엔지니어가 승인

범위 외: 취급법/위치 정보 자동 수집, 벡터 DB/RAG, 통계 기반 자동 베이스라인 갱신(Phase 2 로드맵).

---

## 아키텍처

```
3OVEN.xlsx / 4OVEN.xlsx
        │
        ▼
scripts/extract_signal_map.py       (일회성 실행)
        │
        ▼
config/signal_map.yaml              (신호명 → 센서명 + 설비 이미지 좌표)
        │
        ├──▶ src/data/signal_map.py
        │           │
        │           ▼
        │    dashboard.py           (센서명·우선순위·설비 이미지 오버레이)
        │
ibaPDA_Analyzer_manual.pdf
        │
        ▼
scripts/extract_manual_text.py      (일회성 실행)
        │
        ▼
config/iba_manual.txt               (청크 텍스트)
        │
        ▼
src/report/llm_advisor.py           (Ollama 호출 → 알람 추천 + 패턴 제안)

분석 실행마다
        │
        ▼
src/analysis/anomaly_logger.py      → logs/anomaly_history.jsonl 누적
        │
        ▼
대시보드 "모델 고도화 제안" 탭      → LLM 패턴 분석 → 제안 목록 → 적용 버튼
```

---

## 신규 파일 목록

| 파일 | 역할 |
|------|------|
| `scripts/extract_signal_map.py` | 3OVEN/4OVEN.xlsx 파싱 → `config/signal_map.yaml` 생성 |
| `scripts/extract_manual_text.py` | ibaPDA PDF 텍스트 추출 → `config/iba_manual.txt` 저장 |
| `src/data/signal_map.py` | YAML 로드, `get_sensor_name()` / `get_position()` 제공 |
| `src/report/llm_advisor.py` | Ollama 호출 래퍼. 알람 추천 + 이력 기반 패턴 제안 생성 |
| `src/analysis/anomaly_logger.py` | 이상치 이벤트 JSONL 누적 기록 |
| `config/signal_map.yaml` | 신호명 → 센서명 + 이미지 내 위치 좌표 |
| `config/iba_manual.txt` | ibaPDA 매뉴얼 추출 텍스트 |
| `assets/equipment_diagram.png` | 사용자 제작 설비 이미지 |
| `logs/anomaly_history.jsonl` | 이상치 이력 누적 파일 |

기존 변경:
| 파일 | 변경 내용 |
|------|----------|
| `dashboard.py` | 센서명 컬럼, 우선순위 배지, 설비 이미지 패널, LLM 추천 섹션, 고도화 제안 탭 추가 |
| `fault_classifier.py` | `signal_map`에서 센서명 받아 `root_cause_hypothesis`에 반영 |

---

## 외부 입력물 준비 기한

| 산출물 | 담당 | 완료 기한 |
|--------|------|----------|
| `assets/equipment_diagram.png` | 사용자 (설비 이미지 제작) | 1주차 말 (4/11까지) |
| `config/iba_manual.txt` | `extract_manual_text.py` 실행 | 1주차 초 (4/8까지) |
| `config/signal_map.yaml` 초안 | `extract_signal_map.py` 실행 | 1주차 초 (4/8까지) |

이 두 파일이 준비되지 않으면 설비 시각화(2번 축)와 LLM 추천(3번 축)이 전부 지연된다.

---

## 구현 실행 순서 (의존성 기반)

```
Week 1:
  Day 1-2: extract_signal_map.py + extract_manual_text.py 실행 (외부 입력물 확보)
           anomaly_logger.py + signal_map.py 구현
  Day 3-4: llm_advisor.py 구현 + Ollama 연동 테스트
           dashboard.py — 센서명 컬럼 + 우선순위 배지 추가
  Day 5:   사용자: 실제 이상치 확인 → equipment_diagram.png 제작

Week 2:
  Day 1-2: dashboard.py — 설비 이미지 오버레이 + LLM 추천 섹션
  Day 3:   dashboard.py — 모델 고도화 제안 탭
  Day 4:   end-to-end 검증 (실제 CSV 전체)
  Day 5:   발표 스크립트 + 리허설
```

---

## 컴포넌트 상세

### 1. `config/signal_map.yaml`

```yaml
DB420.DBW10:
  sensor_name: 이동차 구동모터 속도 FB
  position: {x: 0.32, y: 0.61}   # assets/equipment_diagram.png 내 비율 좌표

DB421.DBW4:
  sensor_name: 리프팅 실린더 압력
  position: {x: 0.70, y: 0.40}
```

- `position`은 사용자가 이미지 제작 후 수동으로 입력한다.
- 매핑 없는 신호는 원래 신호명을 그대로 표시한다.

### 2. `src/data/signal_map.py`

```python
class SignalMap:
    def get_sensor_name(self, signal_id: str) -> str: ...
    def get_position(self, signal_id: str) -> dict | None: ...
    def get_all_mapped(self) -> dict[str, dict]: ...
```

- YAML 로드 실패 시 graceful degradation (빈 매핑 반환).

### 3. `src/report/llm_advisor.py`

두 가지 기능:

**A. 알람 추천 (`get_alarm_recommendation`)**
- 입력: 이상 신호명, 센서명, 이상치 유형, ibaPDA 매뉴얼 관련 청크
- 출력: 한국어 텍스트 (알람 로직 추천, 참고 매뉴얼 항목)
- Ollama 비활성 시: `"LLM 미연결 — 매뉴얼 직접 참조 권장"` 반환 (예외 발생 금지)

프롬프트 구조:
```
[시스템] 당신은 ibaPDA 시스템 전문가입니다. 아래 매뉴얼 발췌와 신호 정보를 바탕으로
         한국어로 알람 로직 추천을 작성하세요. 500자 이내로 간결하게.

[컨텍스트] {iba_manual 관련 청크 — 키워드 매칭으로 추출, 최대 2000자}

[사용자] 신호: {signal_id} ({sensor_name})
         이상 유형: {fault_type}
         변화량: {magnitude:.1f}σ
         → 이 신호에 적합한 ibaPDA 알람 로직을 추천해주세요.
```

응답 캐싱: `functools.lru_cache` 또는 `st.cache_data`로 동일 (signal_id, fault_type) 조합은 재호출 없이 캐시 반환. 데모 중 라이브 호출 대기 방지.

**B. 패턴 제안 (`get_improvement_suggestions`)**
- 입력: `logs/anomaly_history.jsonl` 전체 내용
- 출력: 제안 목록 (각 항목: 신호명, 패턴 설명, 권장 조치, 적용할 config 변경값)
- **이력 부족 시 처리**: 동일 신호 탐지 횟수가 2회 미만이면 LLM 호출 대신 "데이터 누적 중 (현재 {n}회)" 표시. 데모 시 실제 이력이 부족할 경우 `tests/fixtures/sample_anomaly_history.jsonl` mock 데이터로 대체 가능 (발표 시 "시뮬레이션 이력 기반" 명시).
- 예시 출력:
  ```
  신호 'DB420.DBW10': 최근 5회 중 4회 동일 시간대(15:49~16:00) 반복 탐지
  → 스탠바이 사이클 패턴 가능성
  → 권장: signal_overrides에 임계값 +1.5 추가
  ```

### 4. `src/analysis/anomaly_logger.py`

```python
class AnomalyLogger:
    def log(self, event: TripEvent, csv_path: str) -> None: ...
```

- 분석 실행마다 `detect_trips` 결과를 JSONL에 append.
- JSONL 레코드 스키마:
  ```json
  {
    "logged_at": "2026-04-06T15:49:00+09:00",
    "csv_file": "2603201549_oven.csv",
    "event_timestamp": "2026-03-20T15:49:15+09:00",
    "trigger": "multi_signal_drop",
    "severity": "높음",
    "top_signals": [
      {"name": "DB420.DBW10", "magnitude": 4.2},
      {"name": "DB421.DBW4", "magnitude": 3.1}
    ]
  }
  ```
- `llm_advisor.get_improvement_suggestions`는 이 스키마의 `top_signals[].name`과 `event_timestamp`를 기준으로 반복 패턴을 분석한다.

### 5. 설비 이미지 오버레이 (dashboard.py)

렌더링 방식: **Plotly `Figure` + `add_layout_image` + `add_trace(Scatter)`**
- Streamlit 외부 컴포넌트 의존성 없이 기존 plotly로 구현 가능
- `signal_map.yaml`의 `position.x`, `position.y`는 이미지 픽셀 기준 0.0~1.0 비율
- Plotly 좌표계: `xaxis range=[0,1]`, `yaxis range=[0,1]`, `yaxis.scaleanchor="x"`로 이미지 비율 고정

```python
fig = go.Figure()
fig.add_layout_image(
    source=Image.open("assets/equipment_diagram.png"),
    x=0, y=1, xref="x", yref="y",
    sizex=1, sizey=1, sizing="stretch", layer="below"
)
# 마커 오버레이
fig.add_trace(go.Scatter(
    x=[pos["x"] for pos in positions],
    y=[1 - pos["y"] for pos in positions],   # y축 반전
    mode="markers+text",
    marker=dict(color=colors, size=14),
    text=sensor_names,
))
```

- 마커 색상: 이상치 발생 신호 → `"crimson"`, 나머지 → `"lightgray"`
- 마커 클릭: Plotly `clickData` → `st.session_state`로 연동해 하단 추이 차트 신호 변경
- 이미지 없을 경우: `st.warning("assets/equipment_diagram.png를 추가하세요")` 표시 후 섹션 skip

### 6. 우선순위 스코어링

룰 기반으로 3단계 분류:

| 조건 | 우선순위 |
|------|---------|
| `multi_signal_drop` 또는 변화량 > 5σ | 🔴 긴급 |
| 단일 신호, 변화량 2~5σ | 🟡 주의 |
| 변화량 < 2σ | 🟢 정보 |

별도 모듈 없이 `fault_classifier.py` 내 `severity` 필드를 확장한다.

---

## 대시보드 UX 변경

### 급변 신호 테이블 (기존 → 변경)

```
기존: 신호명 | Trip값 | 정상평균 | 정상σ | 변화량 | [제외] [+0.5]
변경: 신호명 | 센서명 | 우선순위 | Trip값 | 정상평균 | 변화량 | [제외] [+0.5]
```

### 신규 섹션

1. **설비 위치 패널** (이벤트 요약 아래)
   - 이상치 발생 센서를 설비 이미지 위에 빨간 마커로 표시

2. **🤖 알람 로직 추천** (신호 선택 시 표시)
   - ibaPDA 매뉴얼 기반 Ollama 추천 텍스트
   - "LLM 분석 중..." 스피너 표시

3. **📈 모델 고도화 제안** (하단 별도 탭)
   - "패턴 분석 실행" 버튼 → 이력 기반 LLM 분석
   - 제안 목록 표시: 신호명, 패턴 설명, 권장 조치
   - 각 항목에 "적용" 버튼 → `signals.yaml`의 `signal_overrides` 또는 `excluded_signals` 자동 업데이트 → `detect_trips.clear()` 후 `st.rerun()`
   - 승인 전 preview: "적용 시 변경 내용 — `signal_overrides.DB420.DBW10`: 3.0 → 4.5" 텍스트 표시

---

## 발표 포지셔닝

| 기능 | 발표 레이블 | 구현 상태 |
|------|------------|----------|
| 신호 매핑 | 설비 문서 기반 신호 정체 파악 | Phase 1 (데모) |
| 설비 이미지 오버레이 | 실제 이상치 센서 위치 시각화 | Phase 1 (데모) |
| LLM 알람 추천 | ibaPDA 매뉴얼 기반 대응 가이드 자동 생성 | Phase 1 (데모) |
| LLM 패턴 제안 + 사람 승인 | AI 제안 → 엔지니어 승인 구조 | Phase 1 (데모) |
| 통계 기반 자동 베이스라인 갱신 | 현장 배포 목표 | Phase 2 (로드맵) |

**핵심 메시지:** "현재는 LLM이 제안하고 엔지니어가 승인하는 구조. 향후 통계 기반 자동 학습으로 고도화 예정."

---

## 잔존 리스크 (Known/Accepted)

| # | 리스크 | 대응 |
|---|--------|------|
| R1 | ibaPDA 매뉴얼 PDF가 스캔 이미지 기반일 경우 텍스트 추출 실패 | 4/8 전에 `pdfplumber`로 텍스트 레이어 확인. 실패 시 OCR(pytesseract) 또는 수동 복사 |
| R2 | `fault_classifier.py` severity 필드 확장 시 하위 소비자 브레이킹 체인지 | 신규 필드는 `Optional[str] = None` 기본값으로 추가하여 하위 호환성 유지 |

---

## 2주 작업 순서 (권장)

1. `scripts/extract_signal_map.py` 작성 + OVEN 문서 파싱 → `signal_map.yaml` 초안 생성
2. 실제 CSV 분석 실행 → 이상치 신호 파악 → 사용자가 설비 이미지 제작
3. `signal_map.yaml`에 이미지 좌표 추가
4. `src/data/signal_map.py` + `src/analysis/anomaly_logger.py` 구현
5. `scripts/extract_manual_text.py` + `src/report/llm_advisor.py` 구현
6. `dashboard.py` 고도화 (센서명 + 이미지 패널 + LLM 추천 + 고도화 탭)
7. 실제 데이터로 end-to-end 검증
8. 발표 스크립트 준비