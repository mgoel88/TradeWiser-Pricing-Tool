"""
Sidebar component for WIZX agricultural platform.
"""

import streamlit as st
from datetime import datetime

def render_user_section():
    """Render the user information section."""
    st.sidebar.markdown(
        """
        <div style="padding: 1rem 0; border-bottom: 1px solid #f0f0f0;">
            <div style="font-weight: bold; color: #1E88E5;">Welcome!</div>
            <div style="color: #666; font-size: 0.8rem;">
                Today: {today}
            </div>
        </div>
        """.format(today=datetime.now().strftime("%d %b %Y")),
        unsafe_allow_html=True
    )


def render_sidebar(active_page="Dashboard"):
    """Render the main sidebar with navigation."""
    # Add user section at the top
    render_user_section()
    
    # Navigation
    st.sidebar.markdown("### Navigation")
    
    # Create navigation buttons
    pages = {
        "Dashboard": {"icon": "📊", "desc": "Market overview and trends"},
        "Price Calculator": {"icon": "🧮", "desc": "Calculate commodity prices"},
        "Quality Analysis": {"icon": "🔍", "desc": "Analyze commodity quality"},
        "Advanced Analysis": {"icon": "📈", "desc": "Advanced price analytics"},
        "Notifications": {"icon": "🔔", "desc": "Manage price alerts"}
    }
    
    # Get all commodities
    try:
        from database_sql import get_all_commodities
        commodities = get_all_commodities()
    except Exception:
        commodities = []
    
    selected_page = None
    
    for page, info in pages.items():
        # Create a button with active state
        button_style = "primary" if page == active_page else "secondary"
        if st.sidebar.button(
            f"{info['icon']} {page}", 
            key=f"nav_{page.lower().replace(' ', '_')}",
            use_container_width=True,
            type=button_style
        ):
            selected_page = page
    
    # Filters section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")
    
    # Commodity filter
    selected_commodity = "All"
    if commodities:
        options = ["All"] + commodities
        selected_commodity = st.sidebar.selectbox(
            "Commodity",
            options,
            index=0
        )
    
    # Region filter (enabled only if specific commodity is selected)
    selected_region = None
    if selected_commodity != "All":
        try:
            from database_sql import get_regions_for_commodity
            regions = get_regions_for_commodity(selected_commodity) or []
            
            if regions:
                region_options = ["All"] + regions
                selected_region = st.sidebar.selectbox(
                    "Region",
                    region_options,
                    index=0
                )
        except Exception:
            pass
    
    # Time period filter
    time_periods = {
        "1 Week": 7,
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365
    }
    
    selected_period = st.sidebar.select_slider(
        "Time Period",
        options=list(time_periods.keys()),
        value="1 Month"
    )
    
    time_period_days = time_periods[selected_period]
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        """
        **WIZX Platform** provides comprehensive agricultural commodity pricing with advanced 
        quality assessment capabilities. The platform uses a proprietary index system similar 
        to SENSEX or NYMEX for standardized commodity pricing.
        """
    )
    
    # Return selected values
    return {
        "page": selected_page or active_page,
        "commodity": selected_commodity,
        "region": selected_region,
        "time_period": time_period_days
    }


def render_complete_sidebar(active_page="Dashboard"):
    """Render the complete sidebar and return selection values."""
    # Add sidebar logo at the top
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #1E88E5; margin-bottom: 0;">WIZX</h1>
            <p style="color: #555; margin-top: 0; font-size: 0.8rem;">Agricultural Commodity Platform</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Render main sidebar and get values
    return render_sidebar(active_page)