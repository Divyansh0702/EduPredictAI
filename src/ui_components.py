"""
UI Components and Theme Styling for EduPredict AI
Provides consistent CSS styles, headers, metric cards, and badge components
across all Streamlit pages.
"""

import streamlit as st


def apply_custom_css():
    """Apply modern, professional stylesheet with responsive cards, typography, and badges."""
    st.markdown(
        """
        <style>
        /* Global typography and layout tweaks */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Metric card container */
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
        }
        .metric-title {
            color: #64748b;
            font-size: 0.875rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .metric-value {
            color: #0f172a;
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.1;
        }
        .metric-subtitle {
            color: #94a3b8;
            font-size: 0.8rem;
            margin-top: 6px;
        }

        /* Hero banner */
        .hero-banner {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 16px;
            padding: 32px 36px;
            color: #ffffff;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
            border-left: 6px solid #3b82f6;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #cbd5e1;
            line-height: 1.5;
            max-width: 850px;
        }

        /* Pill badges */
        .badge-at-risk {
            background-color: #fee2e2;
            color: #991b1b;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
            border: 1px solid #fecaca;
        }
        .badge-satisfactory {
            background-color: #fef3c7;
            color: #92400e;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
            border: 1px solid #fde68a;
        }
        .badge-good {
            background-color: #d1fae5;
            color: #065f46;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.85rem;
            display: inline-block;
            border: 1px solid #a7f3d0;
        }

        /* Recommendation cards */
        .rec-card-urgent {
            background-color: #fff1f2;
            border-left: 5px solid #e11d48;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 14px;
        }
        .rec-card-recommended {
            background-color: #f8fafc;
            border-left: 5px solid #3b82f6;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 14px;
        }
        .rec-card-positive {
            background-color: #f0fdf4;
            border-left: 5px solid #10b981;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 14px;
        }

        /* Disclaimer container */
        .disclaimer-box {
            background-color: #f8fafc;
            border: 1px dashed #94a3b8;
            border-radius: 10px;
            padding: 14px 18px;
            color: #475569;
            font-size: 0.85rem;
            line-height: 1.45;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str, icon: str = "🎓"):
    """Render consistent page header."""
    apply_custom_css()
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">{icon} {title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, subtitle: str = ""):
    """Render styled metric card."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    """Render mandatory educational prediction disclaimer."""
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Educational Advisory Notice & Limitations:</strong><br>
            Predictions generated by EduPredict AI are statistical estimates derived from historical student cohort patterns 
            (UCI Student Performance Dataset). They are intended strictly as decision-support heuristics for educators, 
            academic advisors, and students. Predictions do <strong>not</strong> represent guaranteed individual outcomes 
            nor causal determinants of academic potential. Real-world academic success depends on dynamic personal, 
            instructional, psychological, and environmental factors.
        </div>
        """,
        unsafe_allow_html=True,
    )

