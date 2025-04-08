"""
Header component for the WIZX application.
"""

import streamlit as st
from datetime import datetime

def render_header():
    """Render the application header with logo and title."""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        st.image("assets/img/logo.png", width=80)
        
    with col2:
        st.markdown(
            "<h1 style='text-align: center; margin-bottom: 0;'>WIZX Agricultural Platform</h1>"
            "<p style='text-align: center; color: #555; margin-top: 0;'>Advanced Commodity Pricing & Analysis</p>",
            unsafe_allow_html=True
        )
        
    with col3:
        st.markdown(
            f"<div style='text-align: right; font-size: 0.8rem; color: #777;'>"
            f"<p>Today: {datetime.now().strftime('%d %b %Y')}</p>"
            "</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("<hr style='margin: 0; padding: 0;'>", unsafe_allow_html=True)


def render_subheader(title, description=None, icon=None):
    """
    Render a subheader for a page or section.
    
    Args:
        title: The title of the page or section
        description: Optional description text
        icon: Optional icon name (from Font Awesome)
    """
    icon_html = f"<i class='fas fa-{icon}'></i> " if icon else ""
    
    st.markdown(
        f"<h2 style='margin-bottom: 0;'>{icon_html}{title}</h2>",
        unsafe_allow_html=True
    )
    
    if description:
        st.markdown(
            f"<p style='color: #666; margin-bottom: 1.5rem;'>{description}</p>",
            unsafe_allow_html=True
        )
        
    # Add some space after the header
    st.markdown("<br>", unsafe_allow_html=True)