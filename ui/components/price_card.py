"""
Price card component for displaying commodity pricing information.
"""

import streamlit as st
import plotly.graph_objects as go

def render_price_card(
    commodity_name,
    region_name,
    base_price,
    final_price,
    quality_delta=0,
    location_delta=0,
    market_delta=0,
    currency="₹",
    unit="kg",
    quality_score=None
):
    """
    Render a price card with price breakdown and gauge visualization.
    
    Args:
        commodity_name: Name of the commodity
        region_name: Name of the region
        base_price: Base price value
        final_price: Final price value
        quality_delta: Price adjustment based on quality
        location_delta: Price adjustment based on location
        market_delta: Price adjustment based on market conditions
        currency: Currency symbol
        unit: Unit of measurement
        quality_score: Optional quality score (0-100)
    """
    
    # Card container with shadow effect
    st.markdown(
        """
        <style>
        .price-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .price-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        .price-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            align-items: center;
        }
        .price-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin: 0;
        }
        .price-region {
            font-size: 0.9rem;
            color: #777;
            margin: 0;
        }
        .price-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1E88E5;
            margin: 10px 0;
        }
        .price-unit {
            font-size: 0.9rem;
            color: #777;
            margin: 0;
        }
        .price-breakdown {
            margin-top: 15px;
            font-size: 0.9rem;
        }
        .price-breakdown-item {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .delta-positive {
            color: #4CAF50;
        }
        .delta-negative {
            color: #F44336;
        }
        </style>
        
        <div class="price-card">
            <div class="price-header">
                <div>
                    <p class="price-title">{commodity_name}</p>
                    <p class="price-region">{region_name}</p>
                </div>
            </div>
            <p class="price-value">{currency}{final_price:.2f}</p>
            <p class="price-unit">per {unit}</p>
            <div class="price-breakdown">
                <div class="price-breakdown-item">
                    <span>Base Price:</span>
                    <span>{currency}{base_price:.2f}</span>
                </div>
                <div class="price-breakdown-item">
                    <span>Quality Adjustment:</span>
                    <span class="{get_delta_class(quality_delta)}">{format_delta(quality_delta, currency)}</span>
                </div>
                <div class="price-breakdown-item">
                    <span>Location Adjustment:</span>
                    <span class="{get_delta_class(location_delta)}">{format_delta(location_delta, currency)}</span>
                </div>
                <div class="price-breakdown-item">
                    <span>Market Adjustment:</span>
                    <span class="{get_delta_class(market_delta)}">{format_delta(market_delta, currency)}</span>
                </div>
                <hr style="margin: 10px 0;">
                <div class="price-breakdown-item" style="font-weight: 600;">
                    <span>Final Price:</span>
                    <span>{currency}{final_price:.2f}</span>
                </div>
            </div>
        </div>
        """.format(
            commodity_name=commodity_name,
            region_name=region_name,
            base_price=base_price,
            final_price=final_price,
            quality_delta=quality_delta,
            location_delta=location_delta,
            market_delta=market_delta,
            currency=currency,
            unit=unit,
            get_delta_class=lambda delta: "delta-positive" if delta >= 0 else "delta-negative",
            format_delta=lambda delta, currency: f"{currency}{abs(delta):.2f}" if delta < 0 else f"+{currency}{delta:.2f}"
        ),
        unsafe_allow_html=True
    )
    
    # If quality score is provided, render a gauge
    if quality_score is not None:
        # Create gauge for quality score
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=quality_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Quality Score", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1E88E5"},
                'bar': {'color': "#1E88E5"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 60], 'color': '#FF9E80'},
                    {'range': [60, 80], 'color': '#FFECB3'},
                    {'range': [80, 100], 'color': '#C8E6C9'}
                ],
            }
        ))
        
        fig.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="white",
        )
        
        st.plotly_chart(fig, use_container_width=True)