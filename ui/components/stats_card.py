"""
Stats card component for WIZX agricultural platform.
"""

import streamlit as st

def render_stat_card(title, value, delta=None, delta_description=None, icon=None, description=None):
    """
    Render a single stat card.
    
    Args:
        title (str): Card title
        value (str): Main value to display
        delta (float, optional): Delta value for trend
        delta_description (str, optional): Description for the delta
        icon (str, optional): Icon name (FontAwesome)
        description (str, optional): Additional description text
    """
    # Create delta text and color
    delta_text = ""
    delta_color = ""
    
    if delta is not None:
        if delta > 0:
            delta_text = f"+{delta}"
            delta_color = "green"
        elif delta < 0:
            delta_text = f"{delta}"
            delta_color = "red"
        else:
            delta_text = "0"
            delta_color = "gray"
    
    # Create icon HTML
    icon_html = f'<i class="fas fa-{icon}"></i>' if icon else ''
    
    # Create the card
    st.markdown(
        f"""
        <div style="background-color: white; border-radius: 5px; padding: 1rem; 
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); height: 100%;">
            <div style="color: #666; font-size: 0.8rem; margin-bottom: 0.5rem; 
                       display: flex; justify-content: space-between; align-items: center;">
                <span>{title}</span>
                <span style="color: #1E88E5;">{icon_html}</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #333; margin-bottom: 0.5rem;">
                {value}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="color: {delta_color}; font-size: 0.9rem;">
                    {delta_text} {delta_description or ''}
                </div>
                <div style="color: #666; font-size: 0.8rem;">
                    {description or ''}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_stats_grid(stats, columns=4):
    """
    Render a grid of stat cards.
    
    Args:
        stats (list): List of dictionaries containing card data
        columns (int): Number of columns in the grid
    """
    # Create columns
    cols = st.columns(columns)
    
    # Render cards
    for i, stat in enumerate(stats):
        with cols[i % columns]:
            render_stat_card(
                title=stat.get("title", ""),
                value=stat.get("value", ""),
                delta=stat.get("delta"),
                delta_description=stat.get("delta_description"),
                icon=stat.get("icon"),
                description=stat.get("description")
            )