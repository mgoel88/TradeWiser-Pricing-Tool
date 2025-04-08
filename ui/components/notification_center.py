"""
Notification center component for WIZX agricultural platform.
"""

import streamlit as st
from datetime import datetime

# Import notification service
from notification_service import get_notifications, subscribe_to_price_alerts, unsubscribe_from_price_alerts


def format_time_ago(timestamp):
    """Format a timestamp as a relative time (e.g., '5 minutes ago')."""
    now = datetime.now()
    delta = now - timestamp
    
    seconds = delta.total_seconds()
    
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    else:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"


def render_notification_bell(user_id="default_user"):
    """Render a notification bell icon with a badge for unread notifications."""
    # Get notifications for the user
    notifications = get_notifications(user_id=user_id)
    
    # Count unread notifications
    unread_count = sum(1 for n in notifications if not n.read)
    
    # Create notification bell
    if unread_count > 0:
        st.markdown(
            f"""
            <div style="position: relative; display: inline-block; cursor: pointer;"
                 onClick="document.getElementById('notification-panel').style.display = 
                         document.getElementById('notification-panel').style.display === 'none' ? 'block' : 'none';">
                <span style="font-size: 1.5rem;">🔔</span>
                <span style="position: absolute; top: -5px; right: -10px; background-color: #FF5252; 
                            color: white; border-radius: 50%; width: 20px; height: 20px; 
                            display: flex; align-items: center; justify-content: center; 
                            font-size: 0.7rem; font-weight: bold;">
                    {unread_count}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="display: inline-block; cursor: pointer;"
                 onClick="document.getElementById('notification-panel').style.display = 
                         document.getElementById('notification-panel').style.display === 'none' ? 'block' : 'none';">
                <span style="font-size: 1.5rem;">🔔</span>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_notification_panel(user_id="default_user"):
    """Render a panel displaying recent notifications."""
    # Get notifications for the user
    notifications = get_notifications(user_id=user_id)
    
    # Create notification panel (initially hidden)
    st.markdown(
        """
        <div id="notification-panel" style="display: none; position: absolute; 
                top: 60px; right: 20px; width: 350px; max-height: 400px;
                background-color: white; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                z-index: 1000; overflow-y: auto; padding: 1rem;">
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #f0f0f0; padding-bottom: 0.5rem;">
                <span style="font-weight: bold;">Notifications</span>
                <span style="cursor: pointer;" 
                      onClick="document.getElementById('notification-panel').style.display = 'none'">✕</span>
            </div>
            <div id="notification-list">
        """,
        unsafe_allow_html=True
    )
    
    # Render notification items
    if notifications:
        for notification in notifications:
            # Determine styling based on read status
            bg_color = "#f9f9f9" if notification.read else "#e3f2fd"
            border_left = "none" if notification.read else "3px solid #1E88E5"
            
            # Format time
            time_ago = format_time_ago(notification.timestamp)
            
            # Format price change 
            is_increase = notification.change_percentage > 0
            change_color = "#4CAF50" if is_increase else "#F44336"
            change_arrow = "↑" if is_increase else "↓"
            
            # Render notification item
            st.markdown(
                f"""
                <div style="padding: 0.75rem; margin-bottom: 0.5rem; 
                           background-color: {bg_color}; border-left: {border_left};
                           border-radius: 3px;">
                    <div style="font-weight: bold;">{notification.commodity} Price Alert</div>
                    <div style="font-size: 0.9rem;">
                        {notification.region} price has <span style="color: {change_color}; font-weight: bold;">
                        {change_arrow} {abs(notification.change_percentage):.2f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                        <span style="font-size: 0.8rem; color: #666;">
                            ₹{notification.old_price:.2f} → ₹{notification.new_price:.2f}
                        </span>
                        <span style="font-size: 0.8rem; color: #666;">{time_ago}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        # No notifications message
        st.markdown(
            """
            <div style="padding: 1rem; text-align: center; color: #666;">
                No notifications to display
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Close notification list and panel
    st.markdown(
        """
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_notification_center(user_id="default_user"):
    """Render the complete notification center."""
    with st.container():
        # Add the notification bell and panel to the UI
        render_notification_bell(user_id)
        render_notification_panel(user_id)


def render_notification_settings(user_id="default_user"):
    """Render notification settings panel for the user."""
    st.markdown("### Price Alert Settings")
    
    # Get all commodities from the database
    from database_sql import get_all_commodities
    commodities = get_all_commodities()
    
    # Create an expandable section for notification settings
    with st.expander("Manage Price Alerts", expanded=False):
        st.markdown("Set up price alerts for commodities you're interested in:")
        
        # Create columns for side-by-side inputs
        col1, col2 = st.columns(2)
        
        with col1:
            # Select commodity
            selected_commodity = st.selectbox(
                "Commodity",
                options=commodities,
                key="notification_commodity"
            )
            
            # Threshold setting
            threshold = st.slider(
                "Price Change Threshold (%)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                key="notification_threshold"
            )
        
        with col2:
            # Get regions for selected commodity
            from database_sql import get_regions_for_commodity
            regions = get_regions_for_commodity(selected_commodity) or []
            
            # Select regions (multi-select)
            selected_regions = st.multiselect(
                "Regions (Leave empty for all)",
                options=regions,
                key="notification_regions"
            )
            
            # Contact information
            phone_number = st.text_input(
                "WhatsApp Number (include country code)",
                key="notification_phone",
                placeholder="+1234567890"
            )
        
        # Subscribe button
        if st.button("Subscribe to Price Alerts", use_container_width=True, type="primary"):
            if subscribe_to_price_alerts(
                user_id=user_id,
                commodity=selected_commodity,
                regions=selected_regions if selected_regions else None,
                phone=phone_number if phone_number else None
            ):
                st.success(f"Successfully subscribed to price alerts for {selected_commodity}")
            else:
                st.error("Failed to subscribe to price alerts. Please try again.")
        
        # Show current subscriptions
        st.markdown("---")
        st.markdown("#### Your Price Alert Subscriptions")
        
        # Get user subscriptions
        from notification_service import get_user_subscriptions
        subscriptions = get_user_subscriptions(user_id)
        
        if subscriptions:
            for commodity, regions in subscriptions.items():
                col_sub1, col_sub2, col_sub3 = st.columns([3, 4, 1])
                
                with col_sub1:
                    st.markdown(f"**{commodity}**")
                
                with col_sub2:
                    region_text = ", ".join(regions) if regions else "All regions"
                    st.markdown(f"{region_text}")
                
                with col_sub3:
                    if st.button("❌", key=f"unsub_{commodity}"):
                        if unsubscribe_from_price_alerts(user_id, commodity):
                            st.rerun()  # Refresh the UI
        else:
            st.info("You don't have any price alert subscriptions yet.")