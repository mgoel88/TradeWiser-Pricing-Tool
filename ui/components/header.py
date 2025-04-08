"""
Header component for WIZX agricultural platform.
"""

import streamlit as st

def render_header():
    """Render the main application header."""
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #f0f0f0;">
            <div style="flex: 1;">
                <h1 style="color: #1E88E5; margin-bottom: 0;">WIZX</h1>
                <p style="color: #555; margin-top: 0;">Agricultural Commodity Pricing Platform</p>
            </div>
            <div style="text-align: right; color: #666;">
                WIZX Index Platform v1.0
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_subheader(title="", description="", icon=None):
    """Render a section subheader with optional icon."""
    icon_html = f'<i class="fas fa-{icon}"></i> ' if icon else ''
    
    st.markdown(
        f"""
        <div style="margin: 1.5rem 0 1rem 0;">
            <h2 style="color: #1E88E5; font-size: 1.5rem; margin-bottom: 0.5rem;">
                {icon_html}{title}
            </h2>
            <p style="color: #555; margin-top: 0; font-size: 0.9rem;">
                {description}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )