# PIMS Dashboard V2 Roadmap (2026-04-09)

## Goal
- Keep current CSV parsing, signal mapping, and HITL pipeline logic.
- Upgrade dashboard UX so operators can verify anomalies faster and leave feedback consistently.

## Phase 1 (Done in this patch)
- Add downloadable/savable Excel verification report.
- Include these sheets:
  - `Summary` (running rows, group count, IF candidates, LLM verified)
  - `DeviceGroups`
  - `IFCandidates`
  - `LLMVerified`
- Add in-dashboard preview tabs for the same data.

## Phase 2 (Immediate UI foundation)
- Left panel layout hardening:
  - file upload + path fallback
  - top-N slider (`0..20`)
  - LLM backend/model selector
  - unified "설정 저장" state (persist in settings + session)
  - explicit "분석 시작하기" action
- Add "분석 제외 신호" and "임계값 오버라이드" as editable table instead of simple list.

## Phase 3 (Main panel anomaly workflow)
- Top-N anomaly horizontal bar list with quick select.
- Per anomaly O/X feedback buttons (one-click) with optional reason.
- Trend panel:
  - before/after windows
  - baseline bands
  - selected signal overlay
- LLM reasoning panel:
  - card mode first (current reason + structured bullets)
  - chat mode next (stateful conversation attached to selected anomaly)

## Phase 4 (Visual context + alarm linkage)
- Image panel for equipment/layout view:
  - static image upload per group/equipment
  - anomaly signal-to-image zone mapping
  - highlight abnormal zone overlay
- Alarm logic panel:
  - map selected signal to configured alarm rules
  - show threshold, delay, latch/reset conditions

## Phase 5 (Validation and operations)
- Add smoke tests for dashboard state transitions.
- Add exporter tests for sheet schema stability.
- Add profile/cache inspector page for group conditions/clusters.
- Add deployment guardrails:
  - startup checks for label files path
  - warning banner when mapping file is missing.
