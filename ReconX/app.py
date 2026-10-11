import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from recon import analyze_target

st.set_page_config(page_title="ReconX | Security Assessment", page_icon="🛡️", layout="centered")


def custom_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .result-box {
            background: #f7f9fc;
            border: 1px solid #dfe7f1;
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1rem;
        }
        .score-badge {
            font-size: 1.8rem;
            font-weight: 700;
            color: #0f172a;
        }
        .muted {
            color: #475569;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_palette(value):
    if value is None:
        return "#8ea4c6"
    if value >= 75:
        return "#fc5c7d"
    if value >= 45:
        return "#fbbf24"
    return "#39d98a"


def risk_level_for_score(score):
    if score >= 75:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def severity_badge(severity):
    if severity == "Critical":
        return "critical"
    if severity == "High":
        return "high"
    if severity == "Medium":
        return "medium"
    return "low"


def load_history():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    return st.session_state["history"]


def show_result_card(title, value, accent="#0f172a"):
    st.markdown(
        f"""
        <div class='result-box'>
            <div class='muted' style='font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;'>{title}</div>
            <div class='score-badge' style='color: {accent}; margin-top: 0.35rem;'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history_table(history):
    if not history:
        st.info("No prior assessments recorded yet.")
        return

    rows = []
    for item in history:
        rows.append({
            "Target": item["target"],
            "Risk": item["risk"],
            "Score": item["score"],
            "HTTPS": "Yes" if item["https"] else "No",
            "Status": item["status"] if item["status"] is not None else "N/A",
            "Time": item["timestamp"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_main_dashboard(result=None):
    if result is None:
        st.info("Run a target assessment to populate the dashboard.")
        return

    score = result["score"]
    risk_label = risk_level_for_score(score)
    risk_color = "#16a34a" if risk_label == "Low" else "#d97706" if risk_label == "Medium" else "#dc2626"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        show_result_card("Risk Score", score, accent=risk_color)
    with c2:
        show_result_card("Risk Level", risk_label, accent=risk_color)
    with c3:
        show_result_card("Findings", len(result["findings"]))
    with c4:
        show_result_card("HTTPS", "Enabled" if result["https"] else "Not confirmed", accent="#0ea5e9" if result["https"] else "#d97706")
    with c5:
        show_result_card("Resolved IPs", len(result["ips"]))

    st.markdown("<hr style='margin: 1rem 0;' />", unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Security posture")
        st.progress(score / 100)
        st.write(f"Current score: {score}/100")
        st.write(f"ML risk class: {result['risk']}")
    with right:
        st.subheader("Risk distribution")
        risk_points = [
            ("Low", 1 if result["risk"] == "LOW" else 0),
            ("Medium", 1 if result["risk"] == "MEDIUM" else 0),
            ("High", 1 if result["risk"] == "HIGH" else 0),
        ]
        risk_df = pd.DataFrame(risk_points, columns=["level", "count"])
        st.bar_chart(risk_df.set_index("level"), use_container_width=True)

    st.markdown("<hr style='margin: 1rem 0;' />", unsafe_allow_html=True)

    findings_col, config_col = st.columns(2)
    with findings_col:
        st.subheader("Findings")
        for item in result["findings"]:
            st.write(f"- {item}")
    with config_col:
        st.subheader("Configuration summary")
        st.write(f"Target: {result['target']}")
        st.write(f"HTTPS: {'Yes' if result['https'] else 'No'}")
        st.write(f"HTTP status: {result['status'] if result['status'] is not None else 'N/A'}")
        st.write(f"Resolved IPs: {', '.join(result['ips']) if result['ips'] else 'Not available'}")
        if result["recommendations"]:
            st.write("Recommendations:")
            for recommendation in result["recommendations"][:3]:
                st.write(f"- {recommendation}")

    st.markdown("<hr style='margin: 1rem 0;' />", unsafe_allow_html=True)
    st.subheader("Recent assessment history")
    render_history_table(load_history())


def render_results_page(result):
    st.subheader("Assessment results")
    st.caption(f"Target: {result['target']} • Timestamp: {result['timestamp']}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_result_card("Risk Score", result["score"])
    with c2:
        show_result_card("Risk Level", result["risk"])
    with c3:
        show_result_card("HTTPS", "Enabled" if result["https"] else "Not confirmed")
    with c4:
        show_result_card("HTTP", result["status"] if result["status"] is not None else "N/A")

    details, recs = st.columns([1.2, 1])
    with details:
        st.subheader("Target details")
        st.write(f"Target: {result['target']}")
        st.write(f"Resolved addresses: {', '.join(result['ips']) if result['ips'] else 'Not available'}")
        st.write(f"HTTPS status: {'Enabled' if result['https'] else 'Not confirmed'}")
        st.write(f"HTTP status code: {result['status'] if result['status'] is not None else 'Not available'}")
        st.write(f"ML prediction: {result['risk']}")
        st.caption("Educational note: the ML model uses a small synthetic dataset and is intended for demonstration, not production-grade risk intelligence.")

    with recs:
        st.subheader("Remediation actions")
        for rec in result["recommendations"]:
            st.write(f"- {rec}")

    st.markdown("<hr style='margin: 1rem 0;' />", unsafe_allow_html=True)
    findings_df = pd.DataFrame([{"Finding": item} for item in result["findings"]])
    st.dataframe(findings_df, use_container_width=True, hide_index=True)

    report = pd.DataFrame([{
        "target": result["target"],
        "risk_level": result["risk"],
        "score": result["score"],
        "https": result["https"],
        "http_status": result["status"],
        "resolved_ips": ", ".join(result["ips"]),
        "findings": " | ".join(result["findings"]),
        "recommendations": " | ".join(result["recommendations"]),
    }])
    st.download_button("Download CSV report", report.to_csv(index=False).encode("utf-8"), file_name="reconx_report.csv", mime="text/csv")


def main():
    custom_css()

    st.title("ReconX")
    st.caption("Authorized target security assessment")

    if "result" in st.session_state:
        current_result = st.session_state["result"]
    else:
        current_result = None

    with st.sidebar:
        st.header("ReconX")
        st.write("Dashboard")
        st.write("New Assessment")
        st.write("Scan History")
        st.write("Risk Analytics")
        st.write("Reports")
        st.write("Settings")
        st.write("System online")

    target_value = st.text_input("Target", value=st.session_state.get("target_input", ""), placeholder="https://example.com")
    run_button = st.button("Run Assessment")

    if run_button:
        if not target_value.strip():
            st.session_state["error"] = "Please enter a valid URL or hostname."
            st.session_state.pop("result", None)
        else:
            st.session_state["target_input"] = target_value
            st.session_state["error"] = None
            with st.spinner("Running authorized DNS and HTTP security checks..."):
                try:
                    result = analyze_target(target_value.strip())
                    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["result"] = result
                    history = load_history()
                    history.insert(0, {
                        "target": result["target"],
                        "risk": result["risk"],
                        "score": result["score"],
                        "https": result["https"],
                        "status": result["status"],
                        "timestamp": result["timestamp"],
                    })
                    st.session_state["history"] = history[:6]
                except ValueError as exc:
                    st.session_state.pop("result", None)
                    st.session_state["error"] = str(exc)
                except Exception as exc:
                    st.session_state.pop("result", None)
                    st.session_state["error"] = f"Assessment failed: {exc}"

    if "error" in st.session_state and st.session_state["error"]:
        st.error(st.session_state["error"])

    if current_result is not None:
        render_results_page(current_result)
        st.markdown("<hr style='margin: 1rem 0;'/>", unsafe_allow_html=True)
        st.subheader("Overview")
        render_main_dashboard(current_result)
    else:
        render_main_dashboard(None)


if __name__ == "__main__":
    main()