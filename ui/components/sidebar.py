"""
Sidebar component for the WIZX application.
"""

import streamlit as st

def render_sidebar():
    """Render the application sidebar navigation."""
    st.sidebar.markdown(
        "<div style='text-align: center; margin-bottom: 20px;'>"
        "<h3 style='margin-bottom: 0;'>WIZX Platform</h3>"
        "<p style='color: #666; margin-top: 0;'>Agricultural Analytics</p>"
        "</div>",
        unsafe_allow_html=True
    )
    
    # Main navigation
    st.sidebar.markdown("### Main Navigation")
    
    # Define navigation items as (label, icon, page_name) tuples
    nav_items = [
        ("Dashboard", "dashboard", "dashboard"),
        ("Price Calculator", "calculator", "price_calculator"),
        ("Market Analysis", "chart-line", "market_analysis"),
        ("Quality Analysis", "microscope", "quality_analysis"),
        ("Data Viewer", "table", "data_viewer"),
        ("Price Forecasting", "chart-line", "forecasting"),
        ("WIZX Index", "chart-bar", "index"),
        ("Submit Data", "upload", "submit"),
        ("Data Cleaning", "broom", "data_cleaning"),
        ("Transportation", "truck", "transportation"),
        ("Batch Processor", "layer-group", "batch")
    ]
    
    # Create navigation buttons with icons
    selected_page = None
    for label, icon, page_name in nav_items:
        if st.sidebar.button(
            f":{icon}: {label}",
            key=f"nav_{page_name}",
            use_container_width=True
        ):
            selected_page = page_name
            # Store the selected page in session state
            st.session_state["current_page"] = page_name
    
    # Utilities section
    st.sidebar.markdown("### Utilities")
    
    utility_items = [
        ("API Documentation", "api", "api_docs"),
        ("Settings", "gear", "settings"),
        ("Help & Support", "question-circle", "help")
    ]
    
    for label, icon, page_name in utility_items:
        if st.sidebar.button(
            f":{icon}: {label}",
            key=f"util_{page_name}",
            use_container_width=True
        ):
            selected_page = page_name
            st.session_state["current_page"] = page_name
    
    # Return the selected page
    return selected_page or st.session_state.get("current_page", "dashboard")


def render_user_section():
    """Render the user profile section in the sidebar."""
    st.sidebar.markdown("---")
    
    # User profile section in the bottom of sidebar
    st.sidebar.markdown(
        "<div style='text-align: center;'>"
        "<p style='margin-bottom: 5px;'><b>Welcome</b></p>"
        "<p style='color: #666; margin-top: 0; font-size: 0.9rem;'>User</p>"
        "</div>",
        unsafe_allow_html=True
    )
    
    # Logout button
    if st.sidebar.button("Logout", use_container_width=True):
        # In a real application, this would handle user logout
        st.session_state["authenticated"] = False


def render_complete_sidebar():
    """Render the complete sidebar with navigation and user section."""
    selected_page = render_sidebar()
    render_user_section()
    return selected_page