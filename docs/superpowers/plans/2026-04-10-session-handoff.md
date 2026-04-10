# 2026-04-10 세션 인계 문서

## 즉시 해야 할 것 (다음 세션 시작 시)

### Step 1: 미완료 커밋 처리
detection reasoning panel 파일은 수정 완료, 커밋만 안 됨:
```bash
cd "C:/Users/somca/내문서/project/PIMS"
git add src/ui/charts.py app.py
git commit -m "feat: add detection reasoning panel showing IF score explanation and LLM verdict"
```

### Step 2: LLM 프롬프트 개선 (방향 B)
`config/llm_hitl_prompt.md` Constraints 섹션에 추가:
- 신호들이 단조증가 또는 단조감소 트렌드면 → 기동/관성 구간으로 판단 → REJECT
- 예: "모든 주요 신호가 동일 방향으로 완만하게 변화하는 경우 기동 또는 정지 과도 구간으로 간주하고 REJECT하시오"

---

## 오늘 구현된 전체 내역

| 기능 | 파일 | 상태 |
|------|------|------|
| SignalTypeFilter | src/utils/signal_type_filter.py | ✅ 커밋됨 |
| flag context panel | src/utils/flag_context.py, src/ui/charts.py | ✅ 커밋됨 |
| OperationFilter warmup/cooldown | src/utils/operation_filter.py | ✅ 커밋됨 |
| settings.yaml 업데이트 | config/settings.yaml | ✅ 커밋됨 |
| 사이드바 과도필터 안내 | src/ui/sidebar.py | ✅ 커밋됨 |
| detection reasoning panel | src/ui/charts.py, app.py | ⚠ 파일수정완료/커밋미완 |

## 현재 브랜치 상태
- 브랜치: railway-fix
- 최근 커밋: ed39089 feat: show transient filter info in sidebar, increase LLM max_candidates to 50

## 핵심 설정값 (config/settings.yaml)
```yaml
llm:
  max_candidates_per_run: 50
hitl:
  transient_filter:
    warmup_sec: 60
    cooldown_sec: 60
  context_minutes: 5
```

## 알려진 이슈
- test_signal_type_filter.py: MD 신호 분류 2개 실패 (기존 버그, 현재 무시)
- 대시보드 실행: `python -m streamlit run app.py` (포트 8501)
