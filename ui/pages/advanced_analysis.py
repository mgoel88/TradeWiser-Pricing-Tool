"""
Advanced Analysis page for the WIZX application.

This page provides advanced visualizations and analytical tools for
commodity price data, including price trend heatmaps, regional comparisons,
quality price matrices, and price forecasting.
"""

# Note: This function is called 'render' to match the import in app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import components
from ui.components.header import render_subheader

# Import visualization modules
from visualization import (
    create_price_trend_chart,
    create_quality_impact_chart,
    create_price_distribution_chart,
    create_regional_comparison_chart,
    create_quality_price_matrix,
    create_seasonal_price_chart,
    create_price_trend_heatmap
)

from map_visualization import (
    create_regional_price_map,
    create_price_heatmap,
    create_price_spread_analysis
)

# Import price forecasting
from price_forecasting import (
    train_forecast_model,
    analyze_seasonality
)

# Import data access functions
from database_sql import (
    get_all_commodities,
    get_regions_for_commodity,
    get_price_history,
    get_commodity_data
)


def render():
    """Render the advanced analysis page."""
    render_subheader(
        title="Advanced Price Analysis",
        description="Comprehensive visualization and forecasting tools for commodity prices",
        icon="chart-line"
    )
    
    # Get navigation state from sidebar
    # (commodity, region, time_period are passed from the sidebar)
    try:
        commodity = st.session_state.get("selected_commodity", "Wheat")
        region = st.session_state.get("selected_region")
        time_period = st.session_state.get("selected_time_period", 30)
    except:
        # Default values if not in session state
        commodity = "Wheat"
        region = None
        time_period = 30
    
    # Get all commodities
    commodities = get_all_commodities()
    if not commodity in commodities and commodities:
        commodity = commodities[0]
    
    # Page layout with tabs for different analysis types
    tab1, tab2, tab3, tab4 = st.tabs([
        "Price Trends & Forecasting", 
        "Regional Analysis", 
        "Quality Impact Analysis",
        "Seasonal Patterns"
    ])
    
    with tab1:
        render_price_trends_tab(commodity, region, time_period)
    
    with tab2:
        render_regional_analysis_tab(commodity, time_period)
    
    with tab3:
        render_quality_analysis_tab(commodity, region)
    
    with tab4:
        render_seasonal_analysis_tab(commodity, region)


def render_price_trends_tab(commodity, region, time_period):
    """Render the price trends and forecasting tab."""
    st.markdown("### Price Trends & Forecasting")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Analysis settings
        st.markdown("#### Analysis Settings")
        
        # Commodity selector
        selected_commodity = st.selectbox(
            "Commodity",
            options=get_all_commodities(),
            index=get_all_commodities().index(commodity) if commodity in get_all_commodities() else 0,
            key="trend_commodity"
        )
        
        # Region selector
        regions = get_regions_for_commodity(selected_commodity) or []
        selected_region = st.selectbox(
            "Region",
            options=["All"] + regions,
            index=0 if region is None else (regions.index(region) + 1 if region in regions else 0),
            key="trend_region"
        )
        if selected_region == "All":
            selected_region = None
        
        # Time period selector
        time_periods = {
            "1 Week": 7,
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365
        }
        selected_period_name = st.select_slider(
            "Time Period",
            options=list(time_periods.keys()),
            value=[k for k, v in time_periods.items() if v == time_period][0] if time_period in time_periods.values() else "1 Month",
            key="trend_period"
        )
        selected_period = time_periods[selected_period_name]
        
        # Forecast settings
        st.markdown("#### Forecast Settings")
        forecast_days = st.slider("Forecast Days", 
                                min_value=7, 
                                max_value=90, 
                                value=30, 
                                step=7,
                                key="forecast_days")
    
    with col2:
        # Price trend chart
        st.markdown("#### Historical Price Trend")
        
        # Get price history data
        price_history = get_price_history(selected_commodity, selected_region, days=selected_period)
        
        if price_history and len(price_history) > 0:
            # Create DataFrame for visualization
            df = pd.DataFrame(price_history)
            df['date'] = pd.to_datetime(df['date'])
            
            # Plot the price trend
            fig = create_price_trend_chart(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Price forecast
            st.markdown("#### Price Forecast")
            
            # Generate forecast if we have enough data
            if len(df) >= 14:  # Need at least 2 weeks of data for a meaningful forecast
                try:
                    # Train forecast model
                    forecast_results = train_forecast_model(price_history, forecast_days=forecast_days)
                    
                    # Generate DataFrame for plotting
                    forecast_df = pd.DataFrame({
                        'date': forecast_results['dates'],
                        'price': forecast_results['predictions']
                    })
                    
                    # Combine historical and forecast data
                    combined_df = pd.concat([
                        df[['date', 'price']],
                        forecast_df
                    ])
                    
                    # Add forecast flag column
                    combined_df['is_forecast'] = combined_df['date'] > df['date'].max()
                    
                    # Create figure
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(
                        go.Scatter(
                            x=df['date'],
                            y=df['price'],
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='blue')
                        )
                    )
                    
                    # Add forecast data
                    fig.add_trace(
                        go.Scatter(
                            x=forecast_df['date'],
                            y=forecast_df['price'],
                            mode='lines',
                            name='Forecast',
                            line=dict(color='red', dash='dash')
                        )
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Price Forecast - {selected_commodity} ({selected_region or 'All Regions'})",
                        xaxis_title="Date",
                        yaxis_title="Price (₹/Quintal)",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=400
                    )
                    
                    # Add current date vertical line
                    fig.add_vline(
                        x=df['date'].max(),
                        line_width=1,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Today",
                        annotation_position="top right"
                    )
                    
                    # Display the figure
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show forecast statistics
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    
                    with col_stats1:
                        last_price = forecast_results['last_price']
                        last_forecast = forecast_results['predictions'][-1]
                        change = ((last_forecast - last_price) / last_price) * 100
                        
                        st.metric(
                            "Predicted Price Change",
                            f"{change:+.2f}%",
                            delta=f"₹{last_forecast - last_price:+.2f}"
                        )
                    
                    with col_stats2:
                        # Get min and max of the forecast
                        min_forecast = min(forecast_results['predictions'])
                        max_forecast = max(forecast_results['predictions'])
                        
                        st.metric(
                            "Forecast Range",
                            f"₹{min_forecast:.2f} - ₹{max_forecast:.2f}",
                            delta=f"₹{max_forecast - min_forecast:.2f}"
                        )
                    
                    with col_stats3:
                        # Calculate forecast volatility
                        forecast_std = np.std(forecast_results['predictions'])
                        forecast_mean = np.mean(forecast_results['predictions'])
                        volatility = (forecast_std / forecast_mean) * 100
                        
                        st.metric(
                            "Forecast Volatility",
                            f"{volatility:.2f}%",
                            delta=None
                        )
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")
                    
            else:
                st.warning(f"Insufficient data for forecasting. Need at least 2 weeks of data, but only have {len(df)} days.")
        else:
            st.warning(f"No price data available for {selected_commodity} in {selected_region or 'All Regions'}")
    
    # Price trend heatmap (shown full width)
    st.markdown("#### Price Trend Heatmap")
    
    try:
        # Get regions for heatmap
        if selected_region is None:
            heatmap_regions = get_regions_for_commodity(selected_commodity)
        else:
            heatmap_regions = [selected_region]
        
        if heatmap_regions and len(heatmap_regions) > 0:
            # Create the heatmap
            fig = create_price_trend_heatmap(selected_commodity, heatmap_regions, days=selected_period)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No regions available for heatmap visualization of {selected_commodity}")
    except Exception as e:
        st.error(f"Error creating price trend heatmap: {e}")


def render_regional_analysis_tab(commodity, time_period):
    """Render the regional analysis tab."""
    st.markdown("### Regional Price Analysis")
    
    # Commodity selector
    selected_commodity = st.selectbox(
        "Commodity",
        options=get_all_commodities(),
        index=get_all_commodities().index(commodity) if commodity in get_all_commodities() else 0,
        key="regional_commodity"
    )
    
    # Get regional price data
    regions = get_regions_for_commodity(selected_commodity) or []
    
    if regions and len(regions) > 0:
        price_data_by_region = {}
        
        for region in regions:
            price_history = get_price_history(selected_commodity, region, days=time_period)
            if price_history and len(price_history) > 0:
                # Calculate average price for the region
                prices = [p.get('price', 0) for p in price_history]
                price_data_by_region[region] = sum(prices) / len(prices)
        
        if price_data_by_region:
            # Create regional comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Price Comparison by Region")
                fig = create_regional_comparison_chart(selected_commodity, price_data_by_region)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Price Spread Analysis")
                fig = create_price_spread_analysis(price_data_by_region, selected_commodity)
                st.plotly_chart(fig, use_container_width=True)
            
            # Create map visualization
            st.markdown("#### Regional Price Map")
            fig = create_regional_price_map(price_data_by_region, selected_commodity)
            st.plotly_chart(fig, use_container_width=True)
            
            # Price heatmap
            st.markdown("#### Regional Price Heatmap")
            fig = create_price_heatmap(price_data_by_region, selected_commodity)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No price data available for regions of {selected_commodity}")
    else:
        st.warning(f"No regions available for {selected_commodity}")


def render_quality_analysis_tab(commodity, region):
    """Render the quality analysis tab."""
    st.markdown("### Quality Impact Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Commodity selector
        selected_commodity = st.selectbox(
            "Commodity",
            options=get_all_commodities(),
            index=get_all_commodities().index(commodity) if commodity in get_all_commodities() else 0,
            key="quality_commodity"
        )
        
        # Region selector
        regions = get_regions_for_commodity(selected_commodity) or []
        selected_region = st.selectbox(
            "Region",
            options=regions,
            index=regions.index(region) if region in regions else 0,
            key="quality_region"
        ) if regions else None
        
        # Get quality parameters for selected commodity
        commodity_data = get_commodity_data(selected_commodity)
        
        if commodity_data and 'quality_parameters' in commodity_data:
            quality_params = commodity_data['quality_parameters']
            
            st.markdown("#### Quality Parameters")
            
            # Allow user to adjust quality parameters
            user_quality_params = {}
            
            for param, details in quality_params.items():
                min_val = details.get('min', 0)
                max_val = details.get('max', 100)
                std_val = details.get('standard_value', (min_val + max_val) / 2)
                unit = details.get('unit', '')
                
                user_val = st.slider(
                    f"{param} ({unit})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(std_val),
                    step=(max_val - min_val) / 100,
                    key=f"quality_{param}"
                )
                
                user_quality_params[param] = user_val
    
    with col2:
        # Show quality impact chart
        if selected_region and user_quality_params:
            st.markdown("#### Quality Impact on Price")
            
            try:
                fig = create_quality_impact_chart(selected_commodity, user_quality_params, selected_region)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating quality impact chart: {e}")
        
    # Quality-Price Matrix (shown full width)
    if selected_region and 'quality_parameters' in commodity_data and len(quality_params) >= 2:
        st.markdown("#### Quality-Price Matrix")
        
        col_matrix1, col_matrix2 = st.columns([1, 3])
        
        with col_matrix1:
            # Select parameters for matrix
            param_options = list(quality_params.keys())
            
            param1 = st.selectbox(
                "Parameter 1",
                options=param_options,
                index=0,
                key="matrix_param1"
            )
            
            remaining_params = [p for p in param_options if p != param1]
            param2 = st.selectbox(
                "Parameter 2",
                options=remaining_params,
                index=0,
                key="matrix_param2"
            )
        
        with col_matrix2:
            # Show the matrix
            try:
                fig = create_quality_price_matrix(selected_commodity, param1, param2, selected_region)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating quality-price matrix: {e}")


def render_seasonal_analysis_tab(commodity, region):
    """Render the seasonal analysis tab."""
    st.markdown("### Seasonal Price Patterns")
    
    # Commodity selector
    selected_commodity = st.selectbox(
        "Commodity",
        options=get_all_commodities(),
        index=get_all_commodities().index(commodity) if commodity in get_all_commodities() else 0,
        key="seasonal_commodity"
    )
    
    # Region selector
    regions = get_regions_for_commodity(selected_commodity) or []
    selected_region = st.selectbox(
        "Region",
        options=["All"] + regions,
        index=0 if region is None else (regions.index(region) + 1 if region in regions else 0),
        key="seasonal_region"
    )
    if selected_region == "All":
        selected_region = None
    
    # Time period selector for seasonality analysis
    years = st.slider(
        "Years of Data",
        min_value=1,
        max_value=5,
        value=3,
        key="seasonal_years"
    )
    
    # Seasonal price chart
    try:
        st.markdown("#### Annual Price Cycles")
        fig = create_seasonal_price_chart(selected_commodity, selected_region, years=years)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating seasonal price chart: {e}")
    
    # Seasonality analysis
    try:
        # Get price history
        price_history = get_price_history(selected_commodity, selected_region, days=365*years)
        
        if price_history and len(price_history) > 100:  # Need enough data for meaningful analysis
            # Analyze seasonality
            seasonality = analyze_seasonality(price_history)
            
            st.markdown("#### Monthly Price Patterns")
            
            # Create monthly patterns chart
            monthly_avg = seasonality['monthly_avg']
            monthly_std = seasonality['monthly_std']
            
            # Convert to DataFrame
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_data = {
                'month': months,
                'avg_price': [monthly_avg.get(i+1, 0) for i in range(12)],
                'std_dev': [monthly_std.get(i+1, 0) for i in range(12)]
            }
            monthly_df = pd.DataFrame(monthly_data)
            
            # Calculate error bars
            monthly_df['upper'] = monthly_df['avg_price'] + monthly_df['std_dev']
            monthly_df['lower'] = monthly_df['avg_price'] - monthly_df['std_dev']
            
            # Create figure
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Add average line
            fig.add_trace(
                go.Scatter(
                    x=monthly_df['month'],
                    y=monthly_df['avg_price'],
                    mode='lines+markers',
                    name='Average Price',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Add error band
            fig.add_trace(
                go.Scatter(
                    x=monthly_df['month'],
                    y=monthly_df['upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_df['month'],
                    y=monthly_df['lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(0, 0, 255, 0.2)',
                    name='Price Range'
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f"Monthly Price Patterns - {selected_commodity} ({selected_region or 'All Regions'})",
                xaxis_title="Month",
                yaxis_title="Price (₹/Quintal)",
                height=400
            )
            
            # Show the figure
            st.plotly_chart(fig, use_container_width=True)
            
            # Show quarterly analysis
            st.markdown("#### Quarterly Price Patterns")
            
            quarterly_avg = seasonality['quarterly_avg']
            
            # Convert to DataFrame
            quarters = ['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)']
            quarterly_data = {
                'quarter': quarters,
                'avg_price': [quarterly_avg.get(i+1, 0) for i in range(4)]
            }
            quarterly_df = pd.DataFrame(quarterly_data)
            
            # Create bar chart
            import plotly.express as px
            
            fig = px.bar(
                quarterly_df,
                x='quarter',
                y='avg_price',
                title=f"Quarterly Price Patterns - {selected_commodity} ({selected_region or 'All Regions'})",
                labels={'quarter': 'Quarter', 'avg_price': 'Average Price (₹/Quintal)'},
                color='avg_price',
                color_continuous_scale=px.colors.sequential.Blues
            )
            
            # Add average line
            overall_avg = quarterly_df['avg_price'].mean()
            
            fig.add_hline(
                y=overall_avg,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Yearly Avg: ₹{overall_avg:.2f}",
                annotation_position="top right"
            )
            
            # Update layout
            fig.update_layout(
                height=400,
                coloraxis_showscale=False
            )
            
            # Show the figure
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning(f"Not enough historical data for seasonal analysis of {selected_commodity}")
    
    except Exception as e:
        st.error(f"Error performing seasonality analysis: {e}")