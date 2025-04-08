"""
Header component for WIZX agricultural platform.
"""

import streamlit as st
from ui.components.notification_center import render_notification_center

def render_header():
    """Render the main application header."""
    # Create columns for layout
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        # Main title and description
        st.markdown(
            """
            <div style="margin-bottom: 1rem;">
                <h1 style="color: #1E88E5; margin-bottom: 0;">WIZX</h1>
                <p style="color: #555; margin-top: 0;">Agricultural Commodity Pricing Platform</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        # Notifications bell
        render_notification_center(user_id="default_user")
    
    with col3:
        # Version info
        st.markdown(
            """
            <div style="text-align: right; color: #666; margin-top: 1rem;">
                <small>v1.0</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Horizontal line
    st.markdown(
        """
        <hr style="margin: 0 0 1.5rem 0; border: none; height: 1px; background-color: #f0f0f0;">
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