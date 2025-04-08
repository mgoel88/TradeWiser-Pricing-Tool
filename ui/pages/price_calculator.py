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
from database_sql import get_all_commodities, get_regions_for_commodity, get_price_history
from pricing_engine import calculate_price


def render():
    """Render the price calculator page."""
    render_subheader(
        title="Commodity Price Calculator",
        description="Calculate prices based on quality parameters, location, and market conditions",
        icon="calculator"
    )
    
    # Create layout with two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Selection inputs
        st.markdown("### Select Commodity and Region")
        
        commodities = get_all_commodities()
        if not commodities:
            st.error("No commodities available. Please initialize the database first.")
            return
        
        selected_commodity = st.selectbox(
            "Commodity",
            commodities
        )
        
        # Get regions for the selected commodity
        regions = get_regions_for_commodity(selected_commodity)
        if not regions:
            st.error(f"No regions available for {selected_commodity}.")
            return
        
        selected_region = st.selectbox(
            "Region",
            regions
        )
        
        # Quality parameters
        st.markdown("### Quality Parameters")
        
        # Set default quality parameters based on commodity
        if selected_commodity == "Wheat":
            quality_params = {
                "protein_content": st.slider("Protein Content (%)", 8.0, 16.0, 12.5, 0.1),
                "moisture_content": st.slider("Moisture Content (%)", 8.0, 15.0, 11.2, 0.1),
                "test_weight": st.slider("Test Weight (kg/hl)", 70.0, 85.0, 79.5, 0.1),
                "damaged_kernels": st.slider("Damaged Kernels (%)", 0.0, 5.0, 0.8, 0.1)
            }
        elif selected_commodity == "Rice":
            quality_params = {
                "broken_percentage": st.slider("Broken Rice (%)", 0.0, 20.0, 7.2, 0.1),
                "moisture_content": st.slider("Moisture Content (%)", 8.0, 15.0, 12.0, 0.1),
                "foreign_matter": st.slider("Foreign Matter (%)", 0.0, 3.0, 0.5, 0.1),
                "head_rice_recovery": st.slider("Head Rice Recovery (%)", 60.0, 90.0, 78.0, 0.1)
            }
        else:
            # Generic parameters for other commodities
            quality_params = {
                "moisture_content": st.slider("Moisture Content (%)", 8.0, 15.0, 12.0, 0.1),
                "foreign_matter": st.slider("Foreign Matter (%)", 0.0, 3.0, 0.8, 0.1)
            }
        
        # Calculate button
        st.markdown("### Calculate Price")
        
        calculate_button = st.button("Calculate Price", use_container_width=True, type="primary")
    
    # Results column
    with col2:
        st.markdown("### Price Results")
        
        if calculate_button:
            # Call the pricing engine
            try:
                price_result = calculate_price(
                    commodity=selected_commodity,
                    region=selected_region,
                    quality_params=quality_params
                )
                
                if price_result:
                    # Display price card with breakdown
                    render_price_card(
                        commodity_name=selected_commodity,
                        region_name=selected_region,
                        base_price=price_result.get("base_price", 0),
                        final_price=price_result.get("final_price", 0),
                        price_components={
                            "Quality": price_result.get("quality_delta", 0),
                            "Location": price_result.get("location_delta", 0),
                            "Market": price_result.get("market_delta", 0)
                        },
                        unit=price_result.get("unit", "per kg")
                    )
                    
                    # Display price history if available
                    price_history = get_price_history(selected_commodity, selected_region, 30)
                    
                    if price_history and len(price_history) > 0:
                        st.markdown("### Price History (30 days)")
                        
                        # Convert to DataFrame
                        df = pd.DataFrame(price_history)
                        
                        # Create line chart
                        fig = px.line(
                            df,
                            x="date",
                            y="price",
                            title=f"{selected_commodity} Price History - {selected_region}",
                            labels={"price": "Price (₹)", "date": "Date"}
                        )
                        
                        # Add marker for current price
                        fig.add_trace(
                            go.Scatter(
                                x=[datetime.now().date()],
                                y=[price_result.get("final_price", 0)],
                                mode="markers",
                                marker=dict(size=12, color="red"),
                                name="Today's Price"
                            )
                        )
                        
                        # Update layout
                        fig.update_layout(
                            height=300,
                            margin=dict(l=10, r=10, t=50, b=10),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to calculate price. Please try again.")
            except Exception as e:
                st.error(f"Error calculating price: {str(e)}")
        else:
            # Placeholder content when no calculation is performed
            st.info(
                """
                Select a commodity and region, adjust quality parameters, and click 
                'Calculate Price' to see the pricing breakdown.
                """
            )