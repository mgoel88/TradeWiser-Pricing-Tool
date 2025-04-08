"""
Stats card component for displaying data metrics in the dashboard.
"""

import streamlit as st

def render_stat_card(title, value, description=None, delta=None, delta_description=None, icon=None):
    """
    Render a statistic card with value, change percentage, and description.
    
    Args:
        title: Title of the statistic
        value: The main value/metric to display
        description: Optional description text
        delta: Optional change value (can be percent or absolute)
        delta_description: Optional description of what the delta represents
        icon: Optional icon name (from Font Awesome)
    """
    
    # Determine delta color
    if delta is not None:
        if isinstance(delta, (int, float)):
            delta_color = "#4CAF50" if delta >= 0 else "#F44336"
            delta_symbol = "+" if delta > 0 else ""
            delta_text = f"{delta_symbol}{delta:.1f}%" if abs(delta) < 100 else f"{delta_symbol}{int(delta)}%"
        else:
            delta_color = "#1E88E5"  # Default blue if not a number
            delta_text = str(delta)
    
    # HTML for the card
    icon_html = f"<i class='fas fa-{icon}'></i>" if icon else ""
    
    st.markdown(
        f"""
        <div style="background-color: white; border-radius: 8px; padding: 15px; 
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1); height: 100%;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h4 style="margin: 0; color: #555; font-size: 1rem;">{title}</h4>
                <div style="color: #1E88E5; font-size: 1.5rem;">{icon_html}</div>
            </div>
            <div style="font-size: 1.8rem; font-weight: 600; margin: 10px 0;">{value}</div>
            
            {f'<div style="font-size: 0.9rem; color: {delta_color}; margin-bottom: 5px;">{delta_text} {delta_description or ""}</div>' if delta is not None else ''}
            
            {f'<div style="font-size: 0.8rem; color: #777;">{description}</div>' if description else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_stats_grid(stats, columns=3):
    """
    Render a grid of statistic cards.
    
    Args:
        stats: List of dicts with keys (title, value, description, delta, delta_description, icon)
        columns: Number of columns in the grid
    """
    # Create a grid with specified number of columns
    cols = st.columns(columns)
    
    # Place each stat in the grid
    for i, stat in enumerate(stats):
        with cols[i % columns]:
            render_stat_card(
                title=stat.get("title", ""),
                value=stat.get("value", ""),
                description=stat.get("description", None),
                delta=stat.get("delta", None),
                delta_description=stat.get("delta_description", None),
                icon=stat.get("icon", None)
            )