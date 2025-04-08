"""
Price card component for WIZX agricultural platform.
"""

import streamlit as st
import plotly.graph_objects as go

def render_price_card(
    commodity_name, 
    region_name, 
    base_price, 
    final_price, 
    price_components=None, 
    currency="₹", 
    unit="per kg"
):
    """
    Render a price card with breakdown.
    
    Args:
        commodity_name (str): Name of the commodity
        region_name (str): Name of the region
        base_price (float): Base price value
        final_price (float): Final calculated price
        price_components (dict): Dictionary of price adjustment components
        currency (str): Currency symbol
        unit (str): Unit of measurement
    """
    # Set default price components if not provided
    if price_components is None:
        price_components = {
            "Quality": 0,
            "Location": 0,
            "Market": 0
        }
    
    # Create a container
    with st.container():
        # Background and styling
        st.markdown(
            """
            <style>
            .price-card {
                background-color: white;
                padding: 1rem;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Header
        st.markdown(
            f"""
            <div class="price-card">
                <h3 style="margin-top: 0; color: #1E88E5;">{commodity_name}</h3>
                <div style="color: #666; margin-bottom: 1rem;">
                    Region: {region_name}
                </div>
            """,
            unsafe_allow_html=True
        )
        
        # Price display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="color: #666; font-size: 0.8rem;">Base Price ({unit})</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #333;">
                        {currency}{base_price:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col2:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="color: #666; font-size: 0.8rem;">Final Price ({unit})</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #1E88E5;">
                        {currency}{final_price:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Create breakdown waterfall chart
        components = []
        values = []
        
        components.append("Base Price")
        values.append(base_price)
        
        for component, value in price_components.items():
            if value != 0:
                components.append(component)
                values.append(value)
        
        components.append("Final Price")
        values.append(final_price)
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Price Components",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(price_components) + ["total"],
            x=components,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#4CAF50"}},
            decreasing={"marker": {"color": "#F44336"}},
            totals={"marker": {"color": "#1E88E5"}}
        ))
        
        # Update layout
        fig.update_layout(
            title="Price Breakdown",
            showlegend=False,
            height=300,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Close the container div
        st.markdown("</div>", unsafe_allow_html=True)