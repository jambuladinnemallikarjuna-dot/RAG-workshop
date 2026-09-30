import streamlit as st


def status_badge(label: str, ok: bool = True) -> None:
    tone = "status-ok" if ok else "status-error"
    st.markdown(f'<span class="status-pill {tone}">{label}</span>', unsafe_allow_html=True)


def metric_box(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="muted" style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em;">{label}</div>
            <div style="font-size: 1.2rem; font-weight: 700; margin-top: 0.4rem;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
