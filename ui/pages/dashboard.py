"""
Dashboard page for the WIZX application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import components
from ui.components.header import render_subheader
from ui.components.stats_card import render_stats_grid

# Import backend functionality
from wizx_index import calculate_composite_index, calculate_all_indices
from database_sql import get_all_commodities, get_price_history


def render():
    """Render the dashboard page with an overview of market data."""
    render_subheader(
        title="Agricultural Market Dashboard",
        description="Real-time overview of agricultural commodity markets",
        icon="dashboard"
    )
    
    # Top stats section
    render_market_stats()
    
    # Render main content in two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_price_trends()
        render_index_performance()
    
    with col2:
        render_top_movers()
        render_market_breakdown()


def render_market_stats():
    """Render the top statistics cards."""
    # Get today's date
    today = datetime.now()
    
    # Calculate the composite index
    composite_index_result = calculate_composite_index(30) or {}
    
    # Extract values from the result dictionary
    current_index = 100
    if isinstance(composite_index_result, dict) and composite_index_result.get("success"):
        current_index = composite_index_result.get("value", 100)
        prev_index = composite_index_result.get("previous_value", 100)
        # Calculate index change or use the provided change_percentage
        index_change = composite_index_result.get("change_percentage", 0)
    else:
        # Use default values if no valid index data
        current_index = 100
        prev_index = 100
        index_change = 0
    
    # Set up market stats
    market_stats = [
        {
            "title": "WIZX Composite Index",
            "value": f"{current_index:.2f}",
            "delta": index_change,
            "delta_description": "past day",
            "icon": "chart-line"
        },
        {
            "title": "Active Commodities",
            "value": f"{len(get_all_commodities())}",
            "description": "Tracked in the platform",
            "icon": "wheat-awn"
        },
        {
            "title": "Average Price Volatility",
            "value": f"{4.8}%",
            "delta": -0.5,
            "delta_description": "past week",
            "icon": "wave-square"
        },
        {
            "title": "Data Submissions",
            "value": f"{126}",
            "delta": 23,
            "delta_description": "past week",
            "icon": "upload"
        }
    ]
    
    # Render stats grid
    render_stats_grid(market_stats, columns=4)


def render_price_trends():
    """Render the price trends chart showing multiple commodities."""
    st.markdown("### Price Trends")
    
    # Get commodity data for the chart
    commodities = get_all_commodities()[:5]  # Take top 5 for simplicity
    days = 60
    
    # Create DataFrame for the chart
    all_data = []
    for commodity in commodities:
        try:
            # Get price history for the commodity (using first region)
            regions = ["Karnataka", "Maharashtra", "Punjab", "Uttar Pradesh"]  # Example regions
            region = regions[commodities.index(commodity) % len(regions)]  # Cycle through regions
            
            price_history = get_price_history(commodity, region, days)
            
            if price_history and len(price_history) > 0:
                # Create data points
                for entry in price_history:
                    all_data.append({
                        "Date": entry.get("date"),
                        "Price": entry.get("price"),
                        "Commodity": commodity,
                        "Region": region
                    })
        except Exception as e:
            st.error(f"Error getting price history for {commodity}: {str(e)}")
    
    # Create DataFrame from collected data
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Create line chart with Plotly
        fig = px.line(
            df, 
            x="Date", 
            y="Price", 
            color="Commodity",
            line_shape="spline",
            title="Commodity Price Trends (60 Days)",
            labels={"Price": "Price (₹)", "Date": ""},
            height=400
        )
        
        # Customize layout
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=10, r=10, t=50, b=10),
            hovermode="x unified",
            paper_bgcolor="white",
            plot_bgcolor="rgba(245, 247, 249, 0.8)",  # Light gray
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No price data available to display trends.")


def render_index_performance():
    """Render the index performance chart."""
    st.markdown("### WIZX Index Performance")
    
    # Get index data
    days = 30
    composite_index_result = calculate_composite_index(days) or {}
    index_history = []
    
    # Check if we have valid history data
    if isinstance(composite_index_result, dict) and composite_index_result.get("success"):
        # For a real application, we would fetch historical index data
        # For now, generate sample data
        current_value = composite_index_result.get("value", 100)
        
        # Generate sample historical data
        today = datetime.now().date()
        for i in range(days):
            date_val = today - timedelta(days=days-i-1)
            # Create some variability for the demo
            value = current_value * (1 + (np.sin(i/5)/10))
            index_history.append({
                "date": date_val,
                "value": value
            })
    
    if index_history:
        # Create DataFrame
        df = pd.DataFrame(index_history)
        
        # Create combined chart
        fig = go.Figure()
        
        # Add area chart for index
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["value"],
            fill='tozeroy',
            fillcolor='rgba(30, 136, 229, 0.2)',
            line=dict(color='#1E88E5', width=2),
            name="WIZX Composite"
        ))
        
        # Calculate 7-day moving average
        df["MA7"] = df["value"].rolling(window=7).mean()
        
        # Add moving average line
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["MA7"],
            line=dict(color='#FF7043', width=2, dash='dash'),
            name="7-Day MA"
        ))
        
        # Customize layout
        fig.update_layout(
            title="WIZX Composite Index (30 Days)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=10, r=10, t=50, b=10),
            hovermode="x unified",
            height=300,
            paper_bgcolor="white",
            plot_bgcolor="rgba(245, 247, 249, 0.8)"  # Light gray
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No index data available to display performance.")


def render_top_movers():
    """Render the top movers in the market."""
    st.markdown("### Market Movers (24h)")
    
    # Normally we would get this data from the database
    # For now, using example data
    gainers = [
        {"commodity": "Rice", "price_change": 5.2, "region": "Karnataka"},
        {"commodity": "Wheat", "price_change": 3.8, "region": "Punjab"},
        {"commodity": "Maize", "price_change": 2.1, "region": "Maharashtra"}
    ]
    
    losers = [
        {"commodity": "Potato", "price_change": -4.1, "region": "Uttar Pradesh"},
        {"commodity": "Onion", "price_change": -3.7, "region": "Maharashtra"},
        {"commodity": "Cotton", "price_change": -2.9, "region": "Gujarat"}
    ]
    
    # Create tabs for gainers and losers
    tabs = st.tabs(["Top Gainers", "Top Losers"])
    
    # Render gainers
    with tabs[0]:
        for item in gainers:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; 
                           padding: 10px; margin-bottom: 8px; background-color: rgba(200, 230, 201, 0.4); border-radius: 5px;">
                    <div>
                        <div style="font-weight: 600;">{item['commodity']}</div>
                        <div style="font-size: 0.8rem; color: #555;">{item['region']}</div>
                    </div>
                    <div style="color: #4CAF50; font-weight: 600; display: flex; align-items: center;">
                        +{item['price_change']}% ↑
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Render losers
    with tabs[1]:
        for item in losers:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; 
                           padding: 10px; margin-bottom: 8px; background-color: rgba(255, 205, 210, 0.4); border-radius: 5px;">
                    <div>
                        <div style="font-weight: 600;">{item['commodity']}</div>
                        <div style="font-size: 0.8rem; color: #555;">{item['region']}</div>
                    </div>
                    <div style="color: #F44336; font-weight: 600; display: flex; align-items: center;">
                        {item['price_change']}% ↓
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_market_breakdown():
    """Render market breakdown by sector."""
    st.markdown("### Market Sector Breakdown")
    
    # Get sector indices or use default values if not available
    sector_indices_result = calculate_all_indices() or {}
    
    # Create data for the pie chart with default values
    sectors = ["Cereals", "Pulses", "Vegetables", "Fruits", "Oilseeds"]
    # Default balanced values for demonstration
    default_values = [25, 20, 18, 15, 22]
    
    # Use indices from result if available, otherwise use defaults
    if isinstance(sector_indices_result, dict) and sector_indices_result.get("indices"):
        indices = sector_indices_result.get("indices", {})
        values = [indices.get(s, {}).get("latest_value", default_values[i]) for i, s in enumerate(sectors)]
    else:
        values = default_values
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=sectors,
        values=values,
        hole=.4,
        marker=dict(
            colors=['#1E88E5', '#42A5F5', '#90CAF9', '#5C6BC0', '#7986CB'],
            line=dict(color='white', width=2)
        )
    )])
    
    # Update layout
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    # Show chart
    st.plotly_chart(fig, use_container_width=True)