"""
Price calculator page for the WIZX application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import components
from ui.components.header import render_subheader
from ui.components.price_card import render_price_card

# Import backend functionality
from database_sql import (
    get_commodity_data, get_regions, get_all_commodities,
    get_price_history, get_price_recommendation
)
from pricing_engine import calculate_price
from models import predict_price_trend


def render():
    """Render the price calculator page."""
    render_subheader(
        title="Agricultural Commodity Price Calculator",
        description="Calculate prices based on commodity quality parameters and market conditions",
        icon="calculator"
    )
    
    # Main layout with two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Commodity and region selection
        st.markdown("### Select Commodity & Region")
        
        # Get list of commodities
        commodities = get_all_commodities()
        
        # Commodity selection
        selected_commodity = st.selectbox(
            "Select Commodity",
            options=commodities,
            index=0 if commodities else None,
            key="price_calc_commodity"
        )
        
        if selected_commodity:
            # Get commodity data
            commodity_data = get_commodity_data(selected_commodity)
            
            # Show commodity description
            if commodity_data and "description" in commodity_data:
                st.markdown(
                    f"""
                    <div style="background-color: #f8f9fa; border-radius: 5px; padding: 10px; margin-bottom: 15px;">
                        {commodity_data["description"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Get regions for selected commodity
            regions = get_regions(selected_commodity)
            
            # Region selection
            selected_region = st.selectbox(
                "Select Region",
                options=regions,
                index=0 if regions else None,
                key="price_calc_region"
            )
            
            # Quality parameters input
            st.markdown("### Quality Parameters")
            
            quality_params = {}
            if commodity_data and "quality_parameters" in commodity_data:
                # Create sliders for each quality parameter
                for param, param_data in commodity_data["quality_parameters"].items():
                    # Skip internal parameters
                    if param.startswith("_"):
                        continue
                    
                    # Get parameter metadata
                    min_val = param_data.get("min", 0)
                    max_val = param_data.get("max", 100)
                    standard_val = param_data.get("standard_value", (min_val + max_val) / 2)
                    step = (max_val - min_val) / 100
                    
                    # Create a slider for the parameter
                    param_value = st.slider(
                        f"{param.replace('_', ' ').title()}",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(standard_val),
                        step=float(step),
                        format="%.2f",
                        key=f"quality_{param}"
                    )
                    
                    # Add to quality parameters
                    quality_params[param] = param_value
            
            # Calculate price button
            if st.button("Calculate Price", type="primary", use_container_width=True):
                if selected_commodity and selected_region and quality_params:
                    with st.spinner("Calculating price..."):
                        # Calculate price using pricing engine
                        price_result = calculate_price(
                            selected_commodity,
                            quality_params,
                            selected_region
                        )
                        
                        if price_result:
                            # Store in session state for display
                            st.session_state["price_result"] = price_result
                        else:
                            st.error("Error calculating price. Please try again.")
    
    with col2:
        st.markdown("### Price Result")
        
        # Check if we have a price result to display
        if "price_result" in st.session_state:
            price_data = st.session_state["price_result"]
            
            # Extract values from price result
            base_price = price_data.get("base_price", 0)
            final_price = price_data.get("final_price", 0)
            quality_delta = price_data.get("quality_delta", 0)
            location_delta = price_data.get("location_delta", 0)
            market_delta = price_data.get("market_delta", 0)
            quality_score = price_data.get("quality_score", None)
            
            # Render the price card
            render_price_card(
                commodity_name=selected_commodity,
                region_name=selected_region,
                base_price=base_price,
                final_price=final_price,
                quality_delta=quality_delta,
                location_delta=location_delta,
                market_delta=market_delta,
                quality_score=quality_score
            )
            
            # Generate and display price forecast
            st.markdown("### Price Forecast")
            
            # Get days for forecast
            days_options = {
                "7 Days": 7,
                "14 Days": 14,
                "30 Days": 30,
                "60 Days": 60
            }
            
            forecast_period = st.selectbox(
                "Forecast Period",
                options=list(days_options.keys()),
                index=1,  # Default to 14 days
                key="forecast_period"
            )
            
            days = days_options[forecast_period]
            
            with st.spinner("Generating forecast..."):
                # Get historical data for context
                price_history = get_price_history(selected_commodity, selected_region, 60)
                
                if price_history and len(price_history) > 0:
                    # Predict future prices
                    forecast = predict_price_trend(selected_commodity, selected_region, days)
                    
                    if forecast and "data" in forecast:
                        future_prices = forecast["data"]
                        
                        # Combine historical and forecast data
                        all_data = []
                        
                        # Add historical data
                        for entry in price_history:
                            all_data.append({
                                "date": entry.get("date"),
                                "price": entry.get("price"),
                                "type": "Historical"
                            })
                        
                        # Add forecast data
                        for entry in future_prices:
                            all_data.append({
                                "date": entry.get("date"),
                                "price": entry.get("price"),
                                "type": "Forecast"
                            })
                        
                        # Create DataFrame for plotting
                        df = pd.DataFrame(all_data)
                        
                        # Create line chart
                        fig = px.line(
                            df, 
                            x="date", 
                            y="price", 
                            color="type",
                            title=f"Price Forecast ({days} Days)",
                            labels={"price": "Price (₹)", "date": ""},
                            color_discrete_map={
                                "Historical": "#1E88E5",
                                "Forecast": "#FF7043"
                            },
                            line_dash_map={
                                "Historical": "solid",
                                "Forecast": "dash"
                            }
                        )
                        
                        # Add confidence interval
                        if "confidence_interval" in forecast:
                            ci = forecast["confidence_interval"]
                            if len(ci) > 0:
                                # Extract forecast dates
                                forecast_dates = [entry.get("date") for entry in future_prices]
                                
                                # Extract upper and lower bounds
                                upper_bound = [entry.get("upper", entry.get("price") * 1.1) for entry in future_prices]
                                lower_bound = [entry.get("lower", entry.get("price") * 0.9) for entry in future_prices]
                                
                                # Add confidence interval
                                fig.add_trace(
                                    go.Scatter(
                                        x=forecast_dates + forecast_dates[::-1],
                                        y=upper_bound + lower_bound[::-1],
                                        fill="toself",
                                        fillcolor="rgba(255, 112, 67, 0.2)",
                                        line=dict(color="rgba(255, 112, 67, 0)"),
                                        hoverinfo="skip",
                                        showlegend=False
                                    )
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
                            height=300,
                            paper_bgcolor="white",
                            plot_bgcolor="rgba(245, 247, 249, 0.8)"  # Light gray
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show confidence level
                        confidence = forecast.get("confidence", 0.7)
                        st.markdown(
                            f"""
                            <div style="text-align: center; margin-top: -20px;">
                                <span style="font-size: 0.9rem; color: #777;">
                                    Forecast Confidence: <b>{confidence:.0%}</b>
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("Unable to generate price forecast. Insufficient data.")
                else:
                    st.info("No historical price data available for forecasting.")
        else:
            # Show empty state
            st.markdown(
                """
                <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; 
                            text-align: center; height: 200px; display: flex; flex-direction: column; 
                            justify-content: center; align-items: center; margin-bottom: 20px;">
                    <i class="fas fa-calculator" style="font-size: 3rem; color: #1E88E5; margin-bottom: 15px;"></i>
                    <h3 style="margin-bottom: 10px;">No Price Calculated Yet</h3>
                    <p style="color: #777;">Select a commodity and region, then enter quality parameters and click "Calculate Price".</p>
                </div>
                """,
                unsafe_allow_html=True
            )