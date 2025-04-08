"""
Notifications settings page for the WIZX application.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Import components
from ui.components.header import render_subheader
from ui.components.notification_center import render_notification_settings

# Import notification service
from notification_service import (
    subscribe_to_price_alerts, 
    unsubscribe_from_price_alerts,
    get_user_subscriptions,
    get_notifications,
    check_price_changes,
    start_monitoring,
    setup_test_subscriptions
)


def render():
    """Render the notifications settings page."""
    render_subheader(
        title="Notification Settings",
        description="Manage your price alerts and notification preferences",
        icon="bell"
    )
    
    # Display recent notifications and settings in two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_notifications_overview()
    
    with col2:
        render_notification_settings(user_id="default_user")
    
    # Debugging section
    if st.checkbox("Show Test Controls", value=False):
        render_test_controls()
    
    # Automatically initialize the notification service
    if "notification_service_initialized" not in st.session_state:
        start_monitoring()
        st.session_state.notification_service_initialized = True


def render_notifications_overview():
    """Render an overview of recent notifications."""
    st.markdown("### Recent Notifications")
    
    # Get recent notifications
    notifications = get_notifications(user_id="default_user", limit=20)
    
    if notifications:
        # Create a DataFrame for display
        notification_data = []
        
        for notification in notifications:
            notification_data.append({
                "Commodity": notification.commodity,
                "Region": notification.region,
                "Change": f"{notification.change_percentage:+.2f}%",
                "New Price": f"₹{notification.new_price:.2f}",
                "Time": notification.timestamp.strftime("%d %b, %H:%M")
            })
        
        # Create DataFrame and display
        if notification_data:
            df = pd.DataFrame(notification_data)
            
            # Apply color to price change
            def highlight_change(val):
                if val.startswith("+"):
                    return 'color: green'
                elif val.startswith("-"):
                    return 'color: red'
                return ""
            
            # Style the DataFrame
            styled_df = df.style.applymap(highlight_change, subset=["Change"])
            
            # Display the table
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
    else:
        st.info("No recent notifications to display")
        
    # Last checked time
    st.caption(f"Last checked: {datetime.now().strftime('%d %b %Y, %H:%M:%S')}")


def render_test_controls():
    """Render test controls for demonstration purposes."""
    st.markdown("### Test Controls")
    
    # Set up test subscriptions
    if st.button("Set Up Test Subscriptions"):
        setup_test_subscriptions()
        st.success("Test subscriptions created")
    
    # Manual check for price changes
    if st.button("Check for Price Changes"):
        alerts = check_price_changes(threshold_percentage=1.0)  # Lower threshold for testing
        if alerts:
            st.success(f"Generated {len(alerts)} price alerts")
        else:
            st.info("No significant price changes detected")
    
    # Simulate a price alert
    with st.expander("Generate Test Alert", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            from database_sql import get_all_commodities
            commodities = get_all_commodities()
            test_commodity = st.selectbox("Commodity", commodities)
            
            from database_sql import get_regions_for_commodity
            regions = get_regions_for_commodity(test_commodity) or ["Test Region"]
            test_region = st.selectbox("Region", regions)
            
        with col2:
            test_old_price = st.number_input("Old Price", value=1000.0, step=100.0)
            test_change = st.slider("Price Change %", min_value=-20.0, max_value=20.0, value=5.0, step=0.5)
            
        test_new_price = test_old_price * (1 + test_change/100)
        
        if st.button("Generate Test Alert"):
            from notification_service import PriceAlert, notification_queue
            
            # Create alert
            alert = PriceAlert(
                commodity=test_commodity,
                region=test_region,
                old_price=test_old_price,
                new_price=test_new_price,
                change_percentage=test_change
            )
            
            # Add to queue
            notification_queue.put(alert)
            
            st.success("Test alert generated successfully")
            st.rerun()