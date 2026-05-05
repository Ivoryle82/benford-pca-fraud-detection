"""
Customer Fraud Analytics Dashboard
====================================
Interactive Streamlit dashboard for business owners to:
  1. View portfolio-level customer insights
  2. Inspect any individual user's Benford-law risk profile
  3. Score a new customer by pasting in their transaction amounts
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.benford_core import mad, digit_distribution, BENFORD_P

C = {
    "bg":        "#1C2127",   # app background
    "sidebar":   "#111418",   # sidebar / panel backgrounds
    "surface":   "#252A31",   # card / panel surface
    "border":    "#383F47",   # dividers, cell borders
    "txt":       "#F6F7F9",   # primary text
    "muted":     "#ABB3BF",   # secondary / label text
    "dim":       "#5F6B7C",   # disabled / very muted
    "blue":      "#2D72D2",   # Blueprint primary blue
    "blue_lt":   "#4C90F0",  
    "green":     "#238551",   # success / legitimate
    "green_lt":  "#32A467",
    "orange":    "#C87619",   # warning / medium risk
    "orange_lt": "#EC9A3C",
    "red":       "#AC2F33",   # danger / high risk
    "red_lt":    "#E76A6E",
    "gold":      "#D1980B",   # highlight accent
}

# Page config
st.set_page_config(
    page_title="Fraud Analytics using Benford",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent.parent / "data"

# Global CSS (Blueprint dark theme)
GLOBAL_CSS = f"""
<style>
/* ---- base ---- */
html, body, [data-testid="stAppViewContainer"] {{
    background-color: {C['bg']};
    color: {C['txt']};
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
}}

/* ---- hide default streamlit chrome ---- */
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{
    background-color: {C['sidebar']};
    border-right: 1px solid {C['border']};
}}
[data-testid="stSidebar"] * {{ color: {C['muted']}; }}
[data-testid="stSidebar"] .stRadio label {{ color: {C['muted']}; font-size: 12px; }}

/* ---- main content padding ---- */
[data-testid="stMainBlockContainer"] {{ padding: 20px 28px; }}

/* ---- headings ---- */
h1 {{ font-size: 16px !important; font-weight: 600 !important;
      color: {C['txt']} !important; letter-spacing: .02em;
      border-bottom: 1px solid {C['border']}; padding-bottom: 8px;
      margin-bottom: 16px !important; text-transform: uppercase; }}
h2 {{ font-size: 13px !important; font-weight: 600 !important;
      color: {C['txt']} !important; text-transform: uppercase;
      letter-spacing: .05em; }}
h3 {{ font-size: 12px !important; font-weight: 500 !important;
      color: {C['muted']} !important; text-transform: uppercase;
      letter-spacing: .05em; }}

/* ---- KPI cards ---- */
.bp-stat {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-top: 2px solid {C['blue']};
    padding: 14px 16px;
    min-height: 84px;
}}
.bp-stat-label {{
    font-size: 10px; color: {C['muted']}; text-transform: uppercase;
    letter-spacing: .08em; margin-bottom: 6px;
}}
.bp-stat-value {{
    font-size: 22px; font-weight: 700; color: {C['txt']};
    font-family: "Courier New", monospace;
}}
.bp-stat-sub {{
    font-size: 10px; color: {C['dim']}; margin-top: 4px;
}}

/* ---- tags ---- */
.bp-tag {{
    display: inline-block; padding: 2px 8px;
    font-size: 10px; font-weight: 600;
    letter-spacing: .06em; text-transform: uppercase;
    border: 1px solid;
}}
.bp-tag-green  {{ color: {C['green_lt']}; border-color: {C['green']};
                  background: rgba(35,133,81,.12); }}
.bp-tag-red    {{ color: {C['red_lt']};   border-color: {C['red']};
                  background: rgba(172,47,51,.12); }}
.bp-tag-orange {{ color: {C['orange_lt']};border-color: {C['orange']};
                  background: rgba(200,118,25,.12); }}
.bp-tag-blue   {{ color: {C['blue_lt']};  border-color: {C['blue']};
                  background: rgba(45,114,210,.12); }}

/* ---- risk banner ---- */
.bp-risk-banner {{
    padding: 12px 18px;
    border-left: 3px solid;
    margin: 12px 0;
    display: flex; align-items: center; gap: 14px;
}}
.bp-risk-label {{
    font-size: 11px; text-transform: uppercase;
    letter-spacing: .08em; font-weight: 600;
}}
.bp-risk-score {{
    font-size: 28px; font-weight: 700;
    font-family: "Courier New", monospace;
}}

/* ---- section divider ---- */
.bp-divider {{
    border: none; border-top: 1px solid {C['border']};
    margin: 18px 0;
}}

/* ---- inputs ---- */
.stTextInput input, .stTextArea textarea, .stSelectbox {{
    background: {C['surface']} !important;
    border: 1px solid {C['border']} !important;
    border-radius: 2px !important;
    color: {C['txt']} !important;
    font-size: 12px !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
    border-color: {C['blue']} !important;
    box-shadow: 0 0 0 1px {C['blue']} !important;
}}

/* ---- buttons ---- */
.stButton > button {{
    background: {C['blue']} !important;
    border: 1px solid {C['blue']} !important;
    border-radius: 2px !important;
    color: {C['txt']} !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
    padding: 6px 14px !important;
    transition: background .15s;
}}
.stButton > button:hover {{
    background: {C['blue_lt']} !important;
    border-color: {C['blue_lt']} !important;
}}

/* ---- metrics ---- */
[data-testid="stMetricValue"] {{
    font-family: "Courier New", monospace;
    font-size: 18px !important;
    color: {C['txt']} !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 10px !important;
    color: {C['muted']} !important;
    text-transform: uppercase;
    letter-spacing: .07em;
}}
[data-testid="stMetricDelta"] {{
    font-size: 10px !important;
}}

/* ---- dataframe / table ---- */
[data-testid="stDataFrame"] {{ border: 1px solid {C['border']}; border-radius: 0 !important; }}
[data-testid="stDataFrame"] th {{
    background: {C['sidebar']} !important;
    color: {C['muted']} !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: .06em !important;
    border-bottom: 1px solid {C['border']} !important;
}}
[data-testid="stDataFrame"] td {{
    font-size: 11px !important;
    font-family: "Courier New", monospace;
    color: {C['txt']} !important;
}}

/* ---- expander ---- */
[data-testid="stExpander"] {{
    border: 1px solid {C['border']} !important;
    border-radius: 0 !important;
    background: {C['surface']};
}}
summary {{ color: {C['muted']} !important; font-size: 11px !important;
           text-transform: uppercase; letter-spacing: .06em; }}

/* ---- radio (nav) ---- */
.stRadio > div {{ gap: 2px !important; }}
.stRadio label {{
    padding: 6px 10px !important;
    font-size: 11px !important;
    letter-spacing: .06em;
    text-transform: uppercase;
    cursor: pointer;
    border-left: 2px solid transparent;
    transition: border-color .1s;
}}
.stRadio label:hover {{ border-left-color: {C['blue']}; }}

/* ---- caption ---- */
[data-testid="stCaptionContainer"] {{
    color: {C['dim']} !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}}

/* ---- info / warning / error ---- */
[data-testid="stAlert"] {{
    border-radius: 0 !important;
    border-left: 3px solid;
    font-size: 11px !important;
}}
</style>
"""


# Chart theme
LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=C["surface"],
    font=dict(family="-apple-system, 'Segoe UI', Roboto, sans-serif",
              size=11, color=C["muted"]),
    xaxis=dict(gridcolor=C["border"], linecolor=C["border"],
               zeroline=False, tickfont=dict(size=10, color=C["dim"])),
    yaxis=dict(gridcolor=C["border"], linecolor=C["border"],
               zeroline=False, tickfont=dict(size=10, color=C["dim"])),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"],
                borderwidth=1, font=dict(size=10, color=C["muted"])),
    margin=dict(t=24, b=16, l=8, r=8),
    hoverlabel=dict(bgcolor=C["sidebar"], bordercolor=C["border"],
                    font_size=11, font_color=C["txt"]),
)


def apply_layout(fig, **overrides):
    d = {**LAYOUT_BASE, **overrides}
    fig.update_layout(**d)
    return fig


# Data loading
@st.cache_data
def load_user_features():
    return pd.read_csv(DATA_DIR / "user_features.csv")


@st.cache_data
def load_model_results():
    return pd.read_csv(DATA_DIR / "model_results.csv")


@st.cache_data
def load_emad_table():
    return pd.read_csv(DATA_DIR / "expected_mad_table.csv")


# Maths helpers
def expected_mad_for_n(n, emad_table):
    if n < emad_table["N"].min():
        return emad_table["E_MAD"].iloc[0]
    if n > emad_table["N"].max():
        n_max = emad_table["N"].max()
        return emad_table["E_MAD"].iloc[-1] * np.sqrt(n_max / n)
    return float(np.interp(n, emad_table["N"], emad_table["E_MAD"]))


def excess_mad(values, emad_table):
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    n = len(values)
    if n < 9:
        return np.nan
    raw = mad(values)
    return np.nan if np.isnan(raw) else raw - expected_mad_for_n(n, emad_table)


def risk_percentile(value, reference_series):
    clean = reference_series.dropna()
    if len(clean) == 0:
        return 50.0
    return float(np.mean(clean <= value) * 100)


def risk_tier(pct):
    if pct >= 85:
        return "HIGH RISK",   C["red"],    C["red_lt"],    "bp-tag-red"
    if pct >= 60:
        return "MEDIUM RISK", C["orange"], C["orange_lt"], "bp-tag-orange"
    return     "LOW RISK",    C["green"],  C["green_lt"],  "bp-tag-green"


# HTML helpers
def stat_card(label, value, sub="", accent=None):
    top_color = accent or C["blue"]
    return (
        f'<div class="bp-stat" style="border-top-color:{top_color}">'
        f'<div class="bp-stat-label">{label}</div>'
        f'<div class="bp-stat-value">{value}</div>'
        f'<div class="bp-stat-sub">{sub}</div>'
        f'</div>'
    )


def tag(text, cls="bp-tag-blue"):
    return f'<span class="bp-tag {cls}">{text}</span>'


def risk_banner(pct, label_txt, accent, accent_lt):
    return (
        f'<div class="bp-risk-banner" style="background:rgba(0,0,0,.18);border-left-color:{accent}">'
        f'<div class="bp-risk-score" style="color:{accent_lt}">{pct:.0f}<span style="font-size:14px">th</span></div>'
        f'<div>'
        f'<div class="bp-risk-label" style="color:{accent_lt}">{label_txt}</div>'
        f'<div style="font-size:10px;color:{C["dim"]};margin-top:3px">anomaly percentile vs. portfolio</div>'
        f'</div>'
        f'</div>'
    )


def divider():
    return '<hr class="bp-divider">'


# App entry
def main():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    feat_df    = load_user_features()
    model_df   = load_model_results()
    emad_table = load_emad_table()

    n_total    = len(feat_df)
    n_fraud    = int(feat_df["isFraud"].sum())
    n_legit    = n_total - n_fraud
    fraud_rate = n_fraud / n_total * 100

    # Sidebar
    with st.sidebar:
        st.markdown(
            f"<div style='padding:16px 0 8px;'>"
            f"<div style='font-size:11px;font-weight:700;color:{C['txt']};"
            f"letter-spacing:.1em;text-transform:uppercase'>Fraud Analytics</div>"
            f"<div style='font-size:9px;color:{C['dim']};margin-top:2px;"
            f"letter-spacing:.06em'>BENFORD ANOMALY DETECTION</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<hr style='border-color:{C['border']};margin:8px 0'>",
                    unsafe_allow_html=True)

        page = st.radio(
            "nav",
            ["OVERVIEW", "CUSTOMER LOOKUP", "SCORE NEW CUSTOMER"],
            label_visibility="collapsed",
        )

        st.markdown(f"<hr style='border-color:{C['border']};margin:8px 0'>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:9px;color:{C['dim']};line-height:1.6;"
            f"padding-bottom:8px'>"
            f"DATASET<br>"
            f"<span style='color:{C['muted']}'>{n_total:,} customers &nbsp;·&nbsp; "
            f"{n_fraud:,} flagged</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # PAGE 1 — OVERVIEW
    if page == "OVERVIEW":
        st.markdown("# Portfolio Overview")
        st.caption(f"All {n_total:,} customers · Benford's Law anomaly analysis")

        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(stat_card("Total Customers", f"{n_total:,}"), unsafe_allow_html=True)
        with c2:
            st.markdown(stat_card("Flagged", f"{n_fraud:,}",
                                  f"{fraud_rate:.1f}% of portfolio", C["red"]),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(stat_card("Avg Transaction",
                                  f"${feat_df['mean_amt'].mean():,.0f}"), unsafe_allow_html=True)
        with c4:
            st.markdown(stat_card("Avg Txns / Customer",
                                  f"{feat_df['n_txn'].mean():.0f}"), unsafe_allow_html=True)

        st.markdown(divider(), unsafe_allow_html=True)

        # Row 1: distribution panel + scatter
        col_l, col_r = st.columns([1, 2])

        with col_l:
            st.markdown("## Fraud Breakdown")

            # Horizontal stacked bar — cleaner than a donut for enterprise use
            fig_bar = go.Figure(go.Bar(
                y=["Portfolio"],
                x=[n_legit],
                name="Legitimate",
                orientation="h",
                marker=dict(color=C["green"], line_width=0),
            ))
            fig_bar.add_trace(go.Bar(
                y=["Portfolio"],
                x=[n_fraud],
                name="Flagged",
                orientation="h",
                marker=dict(color=C["red"], line_width=0),
            ))
            apply_layout(
                fig_bar,
                barmode="stack",
                height=90,
                showlegend=True,
                margin=dict(t=8, b=8, l=4, r=4),
                xaxis=dict(showgrid=False, showticklabels=False,
                           zeroline=False, linecolor=C["border"]),
                yaxis=dict(showgrid=False, zeroline=False,
                           linecolor=C["border"]),
                legend=dict(orientation="h", y=-0.5, x=0,
                            font=dict(size=10)),
                plot_bgcolor=C["sidebar"],
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Quick stats table
            stats = [
                ("Legitimate",  f"{n_legit:,}",  f"{100-fraud_rate:.1f}%",  "green"),
                ("Flagged",     f"{n_fraud:,}",   f"{fraud_rate:.1f}%",      "red"),
            ]
            rows_html = "".join(
                f"<tr>"
                f"<td style='padding:5px 8px;border-bottom:1px solid {C['border']}'>"
                f"{tag(status, f'bp-tag-{cls}')}</td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid {C['border']};"
                f"font-family:Courier New;font-size:12px;color:{C['txt']}'>{count}</td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid {C['border']};"
                f"font-size:11px;color:{C['dim']}'>{pct}</td>"
                f"</tr>"
                for status, count, pct, cls in stats
            )
            st.markdown(
                f"<table style='width:100%;border-collapse:collapse;"
                f"border:1px solid {C['border']};margin-top:8px'>"
                f"<thead><tr>"
                f"<th style='padding:5px 8px;background:{C['sidebar']};color:{C['muted']};"
                f"font-size:9px;text-transform:uppercase;letter-spacing:.06em;"
                f"text-align:left;border-bottom:1px solid {C['border']}'>Status</th>"
                f"<th style='padding:5px 8px;background:{C['sidebar']};color:{C['muted']};"
                f"font-size:9px;text-transform:uppercase;letter-spacing:.06em;"
                f"text-align:left;border-bottom:1px solid {C['border']}'>Count</th>"
                f"<th style='padding:5px 8px;background:{C['sidebar']};color:{C['muted']};"
                f"font-size:9px;text-transform:uppercase;letter-spacing:.06em;"
                f"text-align:left;border-bottom:1px solid {C['border']}'>Share</th>"
                f"</tr></thead><tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True,
            )

        with col_r:
            st.markdown("## Benford Anomaly Space — Amount vs. Time")

            plot_df = feat_df.dropna(subset=["benford_amt", "benford_time"]).copy()

            if len(plot_df) > 0:
                # Create simple scatter with go.Scatter to avoid KeyError
                fig_sc = go.Figure(data=go.Scatter(
                    x=plot_df["benford_amt"],
                    y=plot_df["benford_time"],
                    mode="markers",
                    marker=dict(color=C["blue"], size=6),
                    text=[str(u) if pd.notna(u) else "" for u in plot_df.get("user_id", [])],
                    hovertemplate="<b>User:</b> %{text}<br><b>Amount MAD:</b> %{x:.5f}<br><b>Time MAD:</b> %{y:.5f}<extra></extra>",
                ))
                fig_sc.add_vline(x=0, line_dash="dot", line_color=C["border"], line_width=1)
                fig_sc.add_hline(y=0, line_dash="dot", line_color=C["border"], line_width=1)
                fig_sc.update_xaxes(title="Excess MAD — Amount")
                fig_sc.update_yaxes(title="Excess MAD — Time")
                apply_layout(fig_sc, height=290)
                st.plotly_chart(fig_sc, width='stretch')
            else:
                st.warning("No Benford data available.")

        st.markdown(divider(), unsafe_allow_html=True)

        # Row 2: box plots + model perf
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("## Benford Feature Distributions by Class")
            fm = feat_df[["benford_amt", "benford_time", "benford_ratio"]].dropna().copy()
            fm = fm.melt(
                value_vars=["benford_amt", "benford_time", "benford_ratio"],
                var_name="Feature", value_name="Excess MAD")
            fm["Feature"] = fm["Feature"].map({
                "benford_amt":   "Amount",
                "benford_time":  "Time Delta",
                "benford_ratio": "Ratio",
            })
            # Simple box plot without color grouping
            fig_bx = px.box(
                fm, x="Feature", y="Excess MAD",
                points=False,
            )
            fig_bx.update_traces(line_width=1.2)
            apply_layout(fig_bx, height=300)
            st.plotly_chart(fig_bx, width='stretch')

        with col_b:
            st.markdown("## Model Performance Comparison")

            metrics = ["auc_roc", "precision", "recall", "f1"]
            metric_labels = ["AUC-ROC", "Precision", "Recall", "F1"]
            metric_colors = [C["blue"], C["green"], C["orange"], C["gold"]]

            fig_perf = go.Figure()
            for metric, label, color in zip(metrics, metric_labels, metric_colors):
                fig_perf.add_trace(go.Bar(
                    name=label,
                    x=model_df["model"],
                    y=model_df[metric],
                    marker=dict(color=color, line_width=0),
                    text=model_df[metric].map("{:.3f}".format),
                    textposition="outside",
                    textfont=dict(size=9, color=C["muted"]),
                ))
            apply_layout(
                fig_perf,
                barmode="group",
                height=300,
                yaxis=dict(range=[0.5, 0.82], gridcolor=C["border"],
                           tickformat=".2f", tickfont=dict(size=10)),
                xaxis=dict(tickfont=dict(size=10, color=C["muted"])),
                bargap=0.22,
                bargroupgap=0.05,
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        st.markdown(divider(), unsafe_allow_html=True)

        # Risk table
        st.markdown("## Top 20 Highest-Risk Customers")

        risk_df = feat_df.dropna(subset=["benford_amt"]).copy()
        risk_df["risk_score"] = (
            risk_df["benford_amt"].rank(pct=True) * 0.5
            + risk_df["benford_ratio"].fillna(0).rank(pct=True) * 0.3
            + risk_df["benford_time"].fillna(0).rank(pct=True) * 0.2
        ) * 100

        top20 = (
            risk_df.sort_values("risk_score", ascending=False)
            .head(20)[["user_id", "n_txn", "mean_amt", "benford_amt",
                        "benford_time", "benford_ratio", "risk_score", "isFraud"]]
            .reset_index(drop=True)
        )
        top20.columns = ["User ID", "Txns", "Avg Amt", "B-Amount",
                          "B-Time", "B-Ratio", "Risk Score", "isFraud"]
        top20["Avg Amt"]    = top20["Avg Amt"].map("${:.2f}".format)
        top20["Risk Score"] = top20["Risk Score"].map("{:.1f}".format)
        for col in ["B-Amount", "B-Time", "B-Ratio"]:
            top20[col] = top20[col].map("{:.5f}".format)
        top20["Status"] = top20.pop("isFraud").map({0: "LEGITIMATE", 1: "FLAGGED"})

        st.dataframe(top20, use_container_width=True, hide_index=True, height=420)

    # PAGE 2 — CUSTOMER LOOKUP
    elif page == "CUSTOMER LOOKUP":
        st.markdown("# Customer Lookup")
        st.caption("Select a customer to view their full Benford risk profile")

        search = st.text_input("Filter by ID", placeholder="e.g. 10023_325")
        options = feat_df["user_id"].tolist()
        if search:
            options = [u for u in options if search.lower() in u.lower()]

        if not options:
            st.warning("No matching customers.")
            return

        selected_id = st.selectbox("Customer ID", options)
        row = feat_df[feat_df["user_id"] == selected_id].iloc[0]

        st.markdown(divider(), unsafe_allow_html=True)

        # Risk score + profile
        b_amt_pct = risk_percentile(row["benford_amt"], feat_df["benford_amt"])
        tier_label, accent, accent_lt, tag_cls = risk_tier(b_amt_pct)

        col_risk, col_profile = st.columns([1, 2])

        with col_risk:
            st.markdown("## Risk Score")
            st.markdown(risk_banner(b_amt_pct, tier_label, accent, accent_lt),
                        unsafe_allow_html=True)

            actual_cls  = "bp-tag-red" if row["isFraud"] == 1 else "bp-tag-green"
            actual_text = "FRAUD" if row["isFraud"] == 1 else "LEGITIMATE"
            st.markdown(
                f"<div style='margin-top:10px;font-size:10px;color:{C['dim']};"
                f"text-transform:uppercase;letter-spacing:.06em'>Ground truth &nbsp;"
                f"{tag(actual_text, actual_cls)}</div>",
                unsafe_allow_html=True,
            )

            # Dimension breakdown table
            dim_rows = []
            for feat, dim in [("benford_amt",   "Amount"),
                               ("benford_time",  "Time Delta"),
                               ("benford_ratio", "Ratio")]:
                val = row[feat]
                if not np.isnan(val):
                    dpct = risk_percentile(val, feat_df[feat])
                    dlabel, _, _, dtag_cls = risk_tier(dpct)
                else:
                    dpct, dlabel, dtag_cls = None, "N/A", "bp-tag-blue"
                dim_rows.append((dim,
                                 f"{val:.5f}" if not np.isnan(val) else "—",
                                 f"{dpct:.0f}th" if dpct is not None else "—",
                                 dlabel, dtag_cls))

            rows_html = "".join(
                f"<tr>"
                f"<td style='padding:5px 8px;border-bottom:1px solid {C['border']};"
                f"font-size:10px;color:{C['muted']};text-transform:uppercase;"
                f"letter-spacing:.05em'>{dim}</td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid {C['border']};"
                f"font-family:Courier New;font-size:11px;color:{C['txt']}'>{val}</td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid {C['border']};"
                f"font-family:Courier New;font-size:11px;color:{C['dim']}'>{pct}</td>"
                f"<td style='padding:5px 8px;border-bottom:1px solid {C['border']}'>"
                f"{tag(lbl, tc)}</td>"
                f"</tr>"
                for dim, val, pct, lbl, tc in dim_rows
            )
            st.markdown(
                f"<table style='width:100%;border-collapse:collapse;"
                f"border:1px solid {C['border']};margin-top:12px'>"
                f"<thead><tr>"
                + "".join(
                    f"<th style='padding:5px 8px;background:{C['sidebar']};"
                    f"color:{C['muted']};font-size:9px;text-transform:uppercase;"
                    f"letter-spacing:.06em;text-align:left;"
                    f"border-bottom:1px solid {C['border']}'>{h}</th>"
                    for h in ["Dimension", "Excess MAD", "Percentile", "Signal"]
                )
                + f"</tr></thead><tbody>{rows_html}</tbody></table>",
                unsafe_allow_html=True,
            )

        with col_profile:
            st.markdown("## Transaction Profile")
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Transactions",   f"{int(row['n_txn']):,}")
            pc2.metric("Avg Amount",     f"${row['mean_amt']:,.2f}")
            pc3.metric("Max Amount",     f"${row['max_amt']:,.2f}")

            pc4, pc5, pc6 = st.columns(3)
            pc4.metric("Std Dev",        f"${row['std_amt']:,.2f}")
            pc5.metric("Median Amount",  f"${row['median_amt']:,.2f}")
            pc6.metric("Median Gap",
                       f"{row['median_time_delta']/3600:.1f} hrs"
                       if not np.isnan(row["median_time_delta"]) else "N/A")

            st.markdown(divider(), unsafe_allow_html=True)
            st.markdown("## Excess MAD vs. Portfolio Median")

            comp_rows = []
            for feat, name in [("benford_amt", "Amount"),
                                ("benford_time", "Time Delta"),
                                ("benford_ratio", "Ratio")]:
                pct_v = risk_percentile(row[feat], feat_df[feat])
                comp_rows.append({
                    "Feature":          name,
                    "Customer":         float(row[feat]) if not np.isnan(row[feat]) else 0.0,
                    "Population Median": float(feat_df[feat].median()),
                    "pct":              pct_v,
                })
            comp = pd.DataFrame(comp_rows)

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                name="This Customer",
                x=comp["Feature"],
                y=comp["Customer"],
                marker=dict(
                    color=[risk_tier(p)[1] for p in comp["pct"]],
                    line_width=0,
                ),
                width=0.35,
            ))
            fig_cmp.add_trace(go.Scatter(
                name="Population Median",
                x=comp["Feature"],
                y=comp["Population Median"],
                mode="markers",
                marker=dict(symbol="line-ew", size=16,
                            color=C["muted"], line_width=2),
            ))
            apply_layout(fig_cmp, height=260)
            st.plotly_chart(fig_cmp, use_container_width=True)

    # PAGE 3 — SCORE NEW CUSTOMER
    elif page == "SCORE NEW CUSTOMER":
        st.markdown("# Score New Customer")
        st.caption(
            "Enter transaction data to compute a Benford anomaly score "
            "relative to the portfolio baseline"
        )

        with st.expander("METHODOLOGY", expanded=False):
            st.markdown(
                f"""
Benford's Law predicts that in naturally occurring financial data the leading
digit follows **P(d) = log₁₀(1 + 1/d)**. Structured fraud — card testing,
synthetic identities, laundering — disrupts this distribution in measurable ways.

**Excess MAD** = observed MAD − E[MAD | n], where E[MAD | n] is the expected
MAD for a random sample of size n (estimated via Monte Carlo). A high positive
value indicates the customer's digit distribution deviates more than chance alone
would predict.

| Excess MAD | Assessment |
|---|---|
| < 0 | Strongly Benford-conformant |
| 0.000 – 0.006 | Within normal range |
| 0.006 – 0.015 | Slightly anomalous |
| > 0.015 | Significant non-conformance |
| > 0.030 | High risk — common in card-testing or structuring |
""")

        col_in, col_out = st.columns([1, 1])

        with col_in:
            st.markdown("## Transaction Input")
            amounts_raw = st.text_area(
                "Transaction amounts (one per line or comma-separated)",
                height=200,
                placeholder="9.99\n14.99\n249.00\n49.95\n...",
                label_visibility="visible",
            )
            time_raw = st.text_area(
                "Inter-transaction gaps in seconds — optional",
                height=90,
                placeholder="86400\n3600\n7200\n...",
                label_visibility="visible",
            )
            run = st.button("Run Analysis", use_container_width=True)

        with col_out:
            st.markdown("## Risk Assessment")

            if not run:
                st.markdown(
                    f"<div style='border:1px solid {C['border']};padding:16px;"
                    f"font-size:11px;color:{C['dim']};text-align:center;"
                    f"margin-top:4px'>Enter amounts and click RUN ANALYSIS</div>",
                    unsafe_allow_html=True,
                )
            else:
                # Parse amounts
                try:
                    amounts = np.array([
                        float(x.strip())
                        for x in amounts_raw.replace(",", "\n").splitlines()
                        if x.strip()
                    ])
                except ValueError:
                    st.error("Parse error — enter numbers only.")
                    st.stop()

                time_deltas = np.array([])
                if time_raw.strip():
                    try:
                        time_deltas = np.array([
                            float(x.strip())
                            for x in time_raw.replace(",", "\n").splitlines()
                            if x.strip()
                        ])
                    except ValueError:
                        st.warning("Could not parse time gaps — skipping.")

                if len(amounts) == 0:
                    st.error("No valid amounts.")
                    st.stop()

                n        = len(amounts)
                mean_amt = amounts.mean()
                ratios   = amounts / mean_amt if mean_amt > 0 else np.array([])

                b_amt   = excess_mad(amounts, emad_table)
                b_time  = excess_mad(time_deltas, emad_table) if len(time_deltas) >= 9 else np.nan
                b_ratio = excess_mad(ratios, emad_table)      if len(ratios)       >= 9 else np.nan

                if np.isnan(b_amt):
                    st.warning(
                        f"Only {n} amounts provided. Need ≥ 9 for a reliable score. "
                        "Showing illustrative result."
                    )
                    b_amt_safe = 0.0
                else:
                    b_amt_safe = b_amt

                pct = risk_percentile(b_amt_safe, feat_df["benford_amt"])
                tier_label, accent, accent_lt, tag_cls = risk_tier(pct)

                st.markdown(risk_banner(pct, tier_label, accent, accent_lt),
                            unsafe_allow_html=True)

                # Dimension table
                dim_data = [
                    (b_amt,   "Amount",     feat_df["benford_amt"]),
                    (b_time,  "Time Delta", feat_df["benford_time"]),
                    (b_ratio, "Ratio",      feat_df["benford_ratio"]),
                ]
                rows_html = ""
                for fval, dim, ref in dim_data:
                    if not np.isnan(fval):
                        fpct  = risk_percentile(fval, ref)
                        flbl  = risk_tier(fpct)[0]
                        ftag  = risk_tier(fpct)[3]
                        fpct_s = f"{fpct:.0f}th"
                    else:
                        fval   = float("nan")
                        fpct_s = "—"
                        flbl   = "INSUFFICIENT DATA"
                        ftag   = "bp-tag-blue"
                    rows_html += (
                        f"<tr>"
                        f"<td style='padding:5px 8px;border-bottom:1px solid {C['border']};"
                        f"font-size:10px;color:{C['muted']};text-transform:uppercase;"
                        f"letter-spacing:.05em'>{dim}</td>"
                        f"<td style='padding:5px 8px;border-bottom:1px solid {C['border']};"
                        f"font-family:Courier New;font-size:11px;color:{C['txt']}'>"
                        f"{'—' if np.isnan(fval) else f'{fval:.5f}'}</td>"
                        f"<td style='padding:5px 8px;border-bottom:1px solid {C['border']};"
                        f"font-family:Courier New;font-size:11px;color:{C['dim']}'>{fpct_s}</td>"
                        f"<td style='padding:5px 8px;border-bottom:1px solid {C['border']}'>"
                        f"{tag(flbl, ftag)}</td>"
                        f"</tr>"
                    )

                st.markdown(
                    f"<table style='width:100%;border-collapse:collapse;"
                    f"border:1px solid {C['border']};margin-top:12px'>"
                    f"<thead><tr>"
                    + "".join(
                        f"<th style='padding:5px 8px;background:{C['sidebar']};"
                        f"color:{C['muted']};font-size:9px;text-transform:uppercase;"
                        f"letter-spacing:.06em;text-align:left;"
                        f"border-bottom:1px solid {C['border']}'>{h}</th>"
                        for h in ["Dimension", "Excess MAD", "Percentile", "Signal"]
                    )
                    + f"</tr></thead><tbody>{rows_html}</tbody></table>",
                    unsafe_allow_html=True,
                )

        # Digit distribution chart
        if run and amounts_raw.strip():
            try:
                amounts_plot = np.array([
                    float(x.strip())
                    for x in amounts_raw.replace(",", "\n").splitlines()
                    if x.strip()
                ])
                if len(amounts_plot) >= 2:
                    st.markdown(divider(), unsafe_allow_html=True)
                    st.markdown("## Leading-Digit Distribution vs. Benford's Law")

                    _, observed = digit_distribution(amounts_plot)
                    digits = list(range(1, 10))

                    fig_bd = go.Figure()
                    fig_bd.add_trace(go.Bar(
                        name="Customer",
                        x=digits,
                        y=observed,
                        marker=dict(color=C["blue"], line_width=0),
                        width=0.5,
                    ))
                    fig_bd.add_trace(go.Scatter(
                        name="Benford Expected",
                        x=digits,
                        y=BENFORD_P,
                        mode="lines+markers",
                        line=dict(color=C["orange_lt"], width=1.5, dash="dash"),
                        marker=dict(size=5),
                    ))
                    apply_layout(
                        fig_bd,
                        height=300,
                        xaxis=dict(title="Leading Digit", tickvals=digits,
                                   tickfont=dict(size=10)),
                        yaxis=dict(title="Proportion", tickformat=".0%"),
                        bargap=0.3,
                    )
                    st.plotly_chart(fig_bd, use_container_width=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
