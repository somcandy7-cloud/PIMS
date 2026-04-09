from __future__ import annotations

import streamlit as st


def apply_dashboard_theme() -> None:
    """dashboard_UI/DESIGN.md 기반 커스텀 테마를 적용한다."""
    st.markdown(
        """
<style>
:root{
  --ia-primary:#00385b;
  --ia-primary-container:#05507d;
  --ia-secondary:#00658e;
  --ia-surface:#f5fafa;
  --ia-surface-low:#eff5f4;
  --ia-surface-lowest:#ffffff;
  --ia-surface-highest:#dee4e3;
  --ia-on-surface:#171d1d;
  --ia-outline:#717880;
  --ia-outline-variant:#c1c7d0;
  --ia-error:#ba1a1a;
}

.stApp{
  background: linear-gradient(180deg, var(--ia-surface) 0%, #eef5f7 100%);
  color: var(--ia-on-surface);
}

[data-testid="stSidebar"]{
  background: rgba(239,245,244,0.90);
  backdrop-filter: blur(10px);
  border-right: none !important;
}

h1,h2,h3{
  color: var(--ia-on-surface) !important;
  letter-spacing:-0.01em;
}

[data-testid="stMetric"]{
  background: var(--ia-surface-low);
  border: none !important;
  border-radius: 10px;
  padding: 10px 12px;
}

[data-testid="stMetricLabel"], [data-testid="stMetricValue"]{
  color: var(--ia-on-surface) !important;
}

.ia-card{
  background: var(--ia-surface-low);
  border-radius: 12px;
  padding: 12px;
}

.ia-glass{
  background: rgba(222,228,227,0.60);
  backdrop-filter: blur(16px);
  border-radius: 12px;
  padding: 12px;
}

.ia-section-title{
  font-size: 12px;
  font-weight: 700;
  color: var(--ia-primary);
  text-transform: uppercase;
  letter-spacing: .06em;
  margin-bottom: 8px;
}

.stButton>button{
  border-radius: 10px !important;
  border: none !important;
}

.stButton>button[kind="primary"]{
  background: linear-gradient(135deg,var(--ia-primary),var(--ia-primary-container)) !important;
  color: #fff !important;
}

[data-testid="stDownloadButton"] > button{
  border-radius: 10px !important;
  border: none !important;
  background: linear-gradient(135deg,var(--ia-primary),var(--ia-primary-container)) !important;
  color: #fff !important;
}

.ia-row{
  background: var(--ia-surface-lowest);
  border-radius: 10px;
  padding: 8px 10px;
}

.ia-kv{
  font-size:12px;
  color:var(--ia-outline);
}

.ia-kv strong{
  color:var(--ia-on-surface);
}
</style>
        """,
        unsafe_allow_html=True,
    )
