"""
UI Components and Theme Styling for EduPredict AI
Provides modern, polished, responsive CSS styles, headers, metric cards, 
Plotly themes, and badge components across all Streamlit pages.
"""

from typing import Optional
import plotly.graph_objects as go
import streamlit as st


def apply_custom_css():
    """Apply high-end modern stylesheet with refined typography, cards, badges, and controls."""
    st.markdown(
        """
        <style>
        /* Import Modern Typography */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        html, body, [class*="css"], [class*="st-"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            letter-spacing: -0.01em;
        }

        /* Pulsing Dot Animation */
        @keyframes pulseGlow {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        .pulse-dot {
            width: 10px;
            height: 10px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
            animation: pulseGlow 2s infinite;
            vertical-align: middle;
        }

        /* Hero Header Banner */
        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e293b 100%);
            border-radius: 16px;
            padding: 28px 32px;
            color: #ffffff;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.08);
            position: relative;
            overflow: hidden;
        }

        .hero-banner::before {
            content: "";
            position: absolute;
            top: -50%;
            right: -10%;
            width: 320px;
            height: 320px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.18) 0%, rgba(99, 102, 241, 0) 70%);
            border-radius: 50%;
            pointer-events: none;
        }

        .hero-pill {
            display: inline-flex;
            align-items: center;
            background: rgba(99, 102, 241, 0.25);
            border: 1px solid rgba(129, 140, 248, 0.35);
            color: #c7d2fe;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 12px;
        }

        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 8px;
            letter-spacing: -0.03em;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: #cbd5e1;
            line-height: 1.55;
            max-width: 900px;
        }

        /* Metric Cards */
        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 20px 22px;
            box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.04), 0 4px 10px -2px rgba(0, 0, 0, 0.03);
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            position: relative;
            overflow: hidden;
        }

        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 20px -4px rgba(15, 23, 42, 0.08), 0 4px 6px -2px rgba(15, 23, 42, 0.04);
            border-color: #cbd5e1;
        }

        .metric-card-accent {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6, #6366f1);
        }

        .metric-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }

        .metric-title {
            color: #64748b;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        .metric-icon-badge {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            background: #f1f5f9;
        }

        .metric-value {
            color: #0f172a;
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.03em;
        }

        .metric-subtitle {
            color: #64748b;
            font-size: 0.8rem;
            font-weight: 500;
            margin-top: 6px;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        /* Glass Content Container */
        .glass-container {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 24px 26px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
            margin-bottom: 20px;
        }

        /* Section Titles */
        .section-header-box {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
            margin-bottom: 14px;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a;
            letter-spacing: -0.02em;
            margin: 0;
        }

        .section-subtitle {
            font-size: 0.85rem;
            color: #64748b;
            margin-top: 2px;
        }

        /* Pill badges */
        .badge-at-risk {
            background-color: #fef2f2;
            color: #b91c1c;
            padding: 5px 14px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border: 1px solid #fecaca;
        }

        .badge-satisfactory {
            background-color: #fffbeb;
            color: #b45309;
            padding: 5px 14px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border: 1px solid #fde68a;
        }

        .badge-good {
            background-color: #f0fdf4;
            color: #15803d;
            padding: 5px 14px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border: 1px solid #bbf7d0;
        }

        /* Recommendation cards */
        .rec-card-urgent {
            background-color: #fff1f2;
            border: 1px solid #fecdd3;
            border-left: 5px solid #e11d48;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }

        .rec-card-recommended {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 5px solid #3b82f6;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }

        .rec-card-positive {
            background-color: #f0fdf4;
            border: 1px solid #dcfce7;
            border-left: 5px solid #10b981;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }

        /* Sidebar Brand Box */
        .sidebar-brand-box {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 12px;
            padding: 16px;
            color: #ffffff;
            margin-bottom: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .sidebar-brand-title {
            font-size: 1.15rem;
            font-weight: 800;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .sidebar-brand-sub {
            font-size: 0.75rem;
            color: #94a3b8;
            margin-top: 4px;
            font-weight: 500;
        }

        .sidebar-status-pill {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 6px;
            padding: 6px 10px;
            font-size: 0.78rem;
            color: #34d399;
            font-weight: 600;
            margin-top: 10px;
            display: flex;
            align-items: center;
        }

        /* Streamlit Button & Widget Polish */
        div.stButton > button:first-child {
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.95rem;
            padding: 0.5rem 1.2rem;
            transition: all 0.2s ease;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(79, 70, 229, 0.45);
        }

        /* Custom Tabs Polish */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #f8fafc;
            padding: 6px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            font-size: 0.9rem;
            color: #64748b;
        }

        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #1e293b !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        /* Disclaimer container */
        .disclaimer-box {
            background-color: #f8fafc;
            border: 1px dashed #cbd5e1;
            border-radius: 12px;
            padding: 16px 20px;
            color: #475569;
            font-size: 0.825rem;
            line-height: 1.5;
            margin-top: 28px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(
    title: str,
    subtitle: str,
    icon: str = "🎓",
    badge: str = "EduPredict AI • Applied ML Platform"
):
    """Render consistent modern page header with glassmorphism accent."""
    apply_custom_css()
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-pill">{badge}</div>
            <div class="hero-title">{icon} {title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    icon: Optional[str] = None,
    accent_color: str = "#3b82f6"
):
    """Render a modern SaaS-styled metric card with micro-interactions."""
    icon_html = f'<div class="metric-icon-badge">{icon}</div>' if icon else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-accent" style="background: linear-gradient(90deg, {accent_color}, #6366f1);"></div>
            <div class="metric-header">
                <div class="metric-title">{title}</div>
                {icon_html}
            </div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand(
    model_ready: bool = True,
    winning_model: str = "Logistic Regression",
    accuracy_str: str = "87.3%"
):
    """Render a unified, high-tech sidebar brand widget."""
    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand-box">
            <div class="sidebar-brand-title">🎓 EduPredict AI</div>
            <div class="sidebar-brand-sub">Machine Learning & Applied AI Internship</div>
            <div class="sidebar-status-pill">
                <span class="pulse-dot"></span>
                <span>Active Model: <strong>{winning_model}</strong> ({accuracy_str})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str = "", icon: str = "📌"):
    """Render a clean modern section title."""
    st.markdown(
        f"""
        <div class="section-header-box">
            <span style="font-size: 1.3rem;">{icon}</span>
            <div>
                <h3 class="section-title">{title}</h3>
                {f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_plotly_styling(fig, height: int = 380):
    """Apply consistent clean modern theme to any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#334155", size=12),
        title_font=dict(size=14, color="#0f172a", family="Plus Jakarta Sans, sans-serif"),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.6)",
    )
    fig.update_xaxes(
        gridcolor="#f1f5f9",
        zerolinecolor="#e2e8f0",
        linecolor="#e2e8f0"
    )
    fig.update_yaxes(
        gridcolor="#f1f5f9",
        zerolinecolor="#e2e8f0",
        linecolor="#e2e8f0"
    )
    return fig


def render_disclaimer():
    """Render mandatory educational prediction disclaimer."""
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Educational Advisory Notice & Limitations:</strong><br>
            Predictions generated by EduPredict AI are probabilistic decision-support estimates derived from historical student cohorts. 
            They are intended strictly as early-warning heuristics for educators, academic counselors, and students. 
            Predictions do <strong>not</strong> represent guaranteed final grades nor deterministic boundaries on academic achievement.
        </div>
        """,
        unsafe_allow_html=True,
    )


