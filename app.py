import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import os
import json
from PIL import Image

# Import modules from our application
from data_crawler import fetch_latest_agmarknet_data, fetch_commodity_list
from data_processor import process_data, standardize_commodity
from pricing_engine import calculate_price, get_price_history
from visualization import create_price_trend_chart, create_quality_impact_chart
from quality_analyzer import analyze_quality_from_image, analyze_report
from models import predict_price_trend

# Import our new SQL database module
from database_sql import (
    get_commodity_data, get_regions, get_all_commodities,
    save_user_input, query_similar_qualities, get_price_recommendation,
    get_price_history as get_sql_price_history,
    calculate_wizx_index, get_wizx_indices,
    initialize_database
)

# Import data cleaning module
from data_cleaning import (
    detect_price_anomalies, fix_missing_data,
    validate_price_curve, clean_data_pipeline
)

# Import WIZX index module
from wizx_index import (
    calculate_all_indices, calculate_composite_index,
    get_sector_indices, historical_index_performance,
    compare_indices, export_indices
)

# Import user submissions module
from user_submissions import (
    submit_price_data, get_pending_submissions,
    get_user_submission_status, auto_verify_submission,
    get_leaderboard
)

# Set page configuration
st.set_page_config(
    page_title="AgriPrice Engine",
    page_icon="🌾",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'selected_commodity' not in st.session_state:
    st.session_state.selected_commodity = None
if 'quality_parameters' not in st.session_state:
    st.session_state.quality_parameters = {}
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = None
if 'calculated_price' not in st.session_state:
    st.session_state.calculated_price = None

# App title and description
st.title("AgriPrice Engine")
st.markdown("### Agricultural Commodity Pricing & Analysis Platform")

# Initialize database if this is the first run
try:
    initialize_database()
except Exception as e:
    st.error(f"Error initializing database: {e}")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", 
    ["Price Dashboard", "Quality Analysis", "Market Trends", "WIZX Index", "Data Submission", "Data Explorer", "Data Cleaning"])

# Main content based on selected page
if page == "Price Dashboard":
    st.header("Commodity Price Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Commodity selection
        commodities = fetch_commodity_list()
        selected_commodity = st.selectbox(
            "Select Commodity", 
            commodities,
            index=0 if commodities else None
        )
        
        if selected_commodity:
            st.session_state.selected_commodity = selected_commodity
            
            # Get regions for selected commodity
            regions = get_regions(selected_commodity)
            selected_region = st.selectbox(
                "Select Region", 
                regions,
                index=0 if regions else None
            )
            
            if selected_region:
                st.session_state.selected_region = selected_region
            
            # Get standard grade parameters for selected commodity
            commodity_data = get_commodity_data(selected_commodity)
            
            # Display quality parameters with sliders
            st.subheader("Quality Parameters")
            quality_params = {}
            
            if commodity_data and 'quality_parameters' in commodity_data:
                for param, details in commodity_data['quality_parameters'].items():
                    default_value = details.get('standard_value', (details.get('min', 0) + details.get('max', 100)) / 2)
                    min_value = details.get('min', 0)
                    max_value = details.get('max', 100)
                    
                    param_value = st.slider(
                        f"{param} ({details.get('unit', '')})", 
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=float(default_value),
                        step=details.get('step', 0.1)
                    )
                    
                    quality_params[param] = param_value
                
                st.session_state.quality_parameters = quality_params
            
            # Calculate button
            if st.button("Calculate Price"):
                if quality_params and selected_region:
                    final_price, base_price, quality_delta, location_delta, market_delta = calculate_price(
                        selected_commodity, 
                        quality_params, 
                        selected_region
                    )
                    
                    st.session_state.calculated_price = {
                        'final_price': final_price,
                        'base_price': base_price,
                        'quality_delta': quality_delta,
                        'location_delta': location_delta,
                        'market_delta': market_delta
                    }
    
    with col2:
        # Display calculation results
        if st.session_state.calculated_price:
            price_data = st.session_state.calculated_price
            
            st.subheader("Price Calculation Results")
            
            # Display in a nice format
            st.markdown(f"""
            ### Final Price: ₹{price_data['final_price']:.2f} / Quintal
            
            **Breakdown:**
            - Base Price: ₹{price_data['base_price']:.2f}
            - Quality Adjustment: ₹{price_data['quality_delta']:.2f}
            - Location Factor: ₹{price_data['location_delta']:.2f}
            - Market Conditions: ₹{price_data['market_delta']:.2f}
            """)
            
            # Create a waterfall chart to visualize price components
            fig = go.Figure(go.Waterfall(
                name="Price Components",
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "total"],
                x=["Base Price", "Quality Δ", "Location Δ", "Market Δ", "Final Price"],
                textposition="outside",
                text=[f"₹{price_data['base_price']:.2f}", 
                      f"₹{price_data['quality_delta']:.2f}", 
                      f"₹{price_data['location_delta']:.2f}", 
                      f"₹{price_data['market_delta']:.2f}", 
                      f"₹{price_data['final_price']:.2f}"],
                y=[price_data['base_price'], 
                   price_data['quality_delta'], 
                   price_data['location_delta'], 
                   price_data['market_delta'], 
                   0],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(
                title="Price Component Breakdown",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    # Price history and trends
    if st.session_state.selected_commodity and st.session_state.selected_region:
        st.subheader("Price History & Trends")
        
        time_period = st.radio(
            "Select Time Period",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year"],
            horizontal=True
        )
        
        days_map = {
            "Last 7 Days": 7,
            "Last 30 Days": 30,
            "Last 90 Days": 90,
            "Last Year": 365
        }
        
        days = days_map[time_period]
        
        price_history = get_price_history(
            st.session_state.selected_commodity, 
            st.session_state.selected_region, 
            days
        )
        
        if price_history is not None and len(price_history) > 0:
            # Create trend chart
            fig = create_price_trend_chart(price_history)
            st.plotly_chart(fig, use_container_width=True)
            
            # Future price prediction
            st.subheader("Price Prediction (Next 30 days)")
            future_prices = predict_price_trend(
                st.session_state.selected_commodity, 
                st.session_state.selected_region
            )
            
            if future_prices is not None and len(future_prices) > 0:
                dates = [date.strftime("%Y-%m-%d") for date in future_prices['date']]
                prices = future_prices['price']
                
                fig = px.line(
                    x=dates, 
                    y=prices, 
                    labels={'x': 'Date', 'y': 'Predicted Price (₹/Quintal)'},
                    title="Price Prediction - Next 30 Days"
                )
                
                # Add confidence interval
                if 'lower_bound' in future_prices and 'upper_bound' in future_prices:
                    fig.add_trace(
                        go.Scatter(
                            x=dates + dates[::-1],
                            y=list(future_prices['upper_bound']) + list(future_prices['lower_bound'])[::-1],
                            fill='toself',
                            fillcolor='rgba(0,176,246,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Confidence Interval'
                        )
                    )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for price prediction")
        else:
            st.info("No price history data available for the selected commodity and region")

elif page == "Quality Analysis":
    st.header("Quality Analysis")
    
    analysis_method = st.radio(
        "Choose Analysis Method",
        ["Manual Entry", "Upload Lab Report", "Upload Sample Images"],
        horizontal=True
    )
    
    if analysis_method == "Manual Entry":
        st.subheader("Enter Quality Parameters Manually")
        
        # Commodity selection
        commodities = fetch_commodity_list()
        selected_commodity = st.selectbox(
            "Select Commodity", 
            commodities,
            index=0 if commodities else None,
            key="quality_commodity_select"
        )
        
        if selected_commodity:
            # Get standard grade parameters for selected commodity
            commodity_data = get_commodity_data(selected_commodity)
            
            if commodity_data and 'quality_parameters' in commodity_data:
                quality_params = {}
                
                for param, details in commodity_data['quality_parameters'].items():
                    default_value = details.get('standard_value', (details.get('min', 0) + details.get('max', 100)) / 2)
                    min_value = details.get('min', 0)
                    max_value = details.get('max', 100)
                    
                    param_value = st.slider(
                        f"{param} ({details.get('unit', '')})", 
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=float(default_value),
                        step=details.get('step', 0.1),
                        key=f"quality_{param}"
                    )
                    
                    quality_params[param] = param_value
                
                if st.button("Analyze Quality"):
                    # Display quality impact on price
                    st.subheader("Quality Impact Analysis")
                    
                    standard_quality = {k: v.get('standard_value', 0) for k, v in commodity_data['quality_parameters'].items()}
                    
                    # Show comparison to standard grade
                    comparison_df = pd.DataFrame({
                        'Parameter': list(quality_params.keys()),
                        'Your Sample': list(quality_params.values()),
                        'Standard Grade': [standard_quality[k] for k in quality_params.keys()],
                        'Difference': [quality_params[k] - standard_quality[k] for k in quality_params.keys()]
                    })
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Calculate quality delta
                    regions = get_regions(selected_commodity)
                    
                    if regions:
                        selected_region = st.selectbox(
                            "Select Region for Price Impact", 
                            regions,
                            index=0,
                            key="quality_region_select"
                        )
                        
                        quality_impact = create_quality_impact_chart(
                            selected_commodity, 
                            quality_params, 
                            selected_region
                        )
                        
                        st.plotly_chart(quality_impact, use_container_width=True)
                        
                        # Save analysis to database
                        save_user_input(selected_commodity, quality_params, selected_region)
                        
    elif analysis_method == "Upload Lab Report":
        st.subheader("Upload Lab Report")
        st.write("Upload a lab report PDF or image to automatically extract quality parameters.")
        
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            st.info("Analyzing lab report... Please wait.")
            
            # Analyze the report (using a mock function for now)
            commodity, quality_params = analyze_report(uploaded_file)
            
            if commodity and quality_params:
                st.success(f"Successfully analyzed report for {commodity}")
                
                # Display extracted parameters
                st.subheader("Extracted Quality Parameters")
                
                params_df = pd.DataFrame({
                    'Parameter': list(quality_params.keys()),
                    'Value': list(quality_params.values())
                })
                
                st.dataframe(params_df, use_container_width=True)
                
                # Get regions for this commodity
                regions = get_regions(commodity)
                
                if regions:
                    selected_region = st.selectbox(
                        "Select Region for Price Calculation", 
                        regions,
                        index=0,
                        key="report_region_select"
                    )
                    
                    if st.button("Calculate Price"):
                        final_price, base_price, quality_delta, location_delta, market_delta = calculate_price(
                            commodity, 
                            quality_params, 
                            selected_region
                        )
                        
                        st.subheader("Price Calculation Results")
                        
                        # Display in a nice format
                        st.markdown(f"""
                        ### Final Price: ₹{final_price:.2f} / Quintal
                        
                        **Breakdown:**
                        - Base Price: ₹{base_price:.2f}
                        - Quality Adjustment: ₹{quality_delta:.2f}
                        - Location Factor: ₹{location_delta:.2f}
                        - Market Conditions: ₹{market_delta:.2f}
                        """)
            else:
                st.error("Could not analyze the report. Please ensure it's a valid lab report.")
    
    elif analysis_method == "Upload Sample Images":
        st.subheader("Upload Sample Images")
        st.write("Upload images of your commodity samples for quality analysis.")
        
        commodities = fetch_commodity_list()
        selected_commodity = st.selectbox(
            "Select Commodity Type", 
            commodities,
            index=0 if commodities else None,
            key="image_commodity_select"
        )
        
        uploaded_files = st.file_uploader(
            "Upload one or more sample images", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )
        
        if uploaded_files and selected_commodity:
            st.info("Analyzing images... This may take a moment.")
            
            # Process and analyze each image
            all_results = []
            
            for i, file in enumerate(uploaded_files):
                quality_params = analyze_quality_from_image(file, selected_commodity)
                
                if quality_params:
                    all_results.append(quality_params)
                    
                    st.subheader(f"Sample {i+1} Analysis")
                    
                    # Display image
                    st.image(file, caption=f"Sample {i+1}", width=300)
                    
                    # Display extracted parameters
                    params_df = pd.DataFrame({
                        'Parameter': list(quality_params.keys()),
                        'Value': list(quality_params.values())
                    })
                    
                    st.dataframe(params_df, use_container_width=True)
            
            if all_results:
                # Calculate average quality parameters
                avg_params = {}
                
                for param in all_results[0].keys():
                    avg_params[param] = sum(result[param] for result in all_results) / len(all_results)
                
                st.subheader("Average Quality Parameters")
                
                avg_df = pd.DataFrame({
                    'Parameter': list(avg_params.keys()),
                    'Value': list(avg_params.values())
                })
                
                st.dataframe(avg_df, use_container_width=True)
                
                # Get regions for this commodity
                regions = get_regions(selected_commodity)
                
                if regions:
                    selected_region = st.selectbox(
                        "Select Region for Price Calculation", 
                        regions,
                        index=0,
                        key="image_region_select"
                    )
                    
                    if st.button("Calculate Price"):
                        final_price, base_price, quality_delta, location_delta, market_delta = calculate_price(
                            selected_commodity, 
                            avg_params, 
                            selected_region
                        )
                        
                        st.subheader("Price Calculation Results")
                        
                        # Display in a nice format
                        st.markdown(f"""
                        ### Final Price: ₹{final_price:.2f} / Quintal
                        
                        **Breakdown:**
                        - Base Price: ₹{base_price:.2f}
                        - Quality Adjustment: ₹{quality_delta:.2f}
                        - Location Factor: ₹{location_delta:.2f}
                        - Market Conditions: ₹{market_delta:.2f}
                        """)

elif page == "Market Trends":
    st.header("Market Trends & Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Filters
        st.subheader("Filters")
        
        commodities = fetch_commodity_list()
        selected_commodities = st.multiselect(
            "Select Commodities", 
            commodities,
            default=[commodities[0]] if commodities else None
        )
        
        timeframe = st.radio(
            "Select Timeframe",
            ["1 Month", "3 Months", "6 Months", "1 Year"],
            horizontal=True
        )
        
        days_map = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365
        }
        
        days = days_map[timeframe]
        
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Price Trends", "Regional Comparison", "Quality-Price Correlation", "Seasonal Analysis"]
        )
    
    with col2:
        if not selected_commodities:
            st.info("Please select at least one commodity.")
        else:
            if analysis_type == "Price Trends":
                st.subheader("Price Trends Analysis")
                
                # Get data for all selected commodities
                all_data = []
                
                for commodity in selected_commodities:
                    regions = get_regions(commodity)
                    
                    if regions:
                        all_region_data = []
                        
                        for region in regions:
                            price_history = get_price_history(commodity, region, days)
                            
                            if price_history is not None and len(price_history) > 0:
                                for entry in price_history:
                                    entry['Commodity'] = commodity
                                    entry['Region'] = region
                                
                                all_region_data.extend(price_history)
                        
                        all_data.extend(all_region_data)
                
                if all_data:
                    # Create a DataFrame
                    df = pd.DataFrame(all_data)
                    
                    # Plot the trends
                    fig = px.line(
                        df, 
                        x="date", 
                        y="price", 
                        color="Commodity",
                        line_dash="Region",
                        labels={"date": "Date", "price": "Price (₹/Quintal)"},
                        title="Commodity Price Trends"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Price volatility
                    st.subheader("Price Volatility")
                    
                    volatility_data = []
                    
                    for commodity in selected_commodities:
                        commodity_data = df[df['Commodity'] == commodity]
                        if not commodity_data.empty:
                            # Calculate std dev for each region
                            regions = commodity_data['Region'].unique()
                            
                            for region in regions:
                                region_data = commodity_data[commodity_data['Region'] == region]
                                volatility = region_data['price'].std()
                                mean_price = region_data['price'].mean()
                                cv = (volatility / mean_price) * 100  # Coefficient of variation
                                
                                volatility_data.append({
                                    'Commodity': commodity,
                                    'Region': region,
                                    'Volatility (₹)': volatility,
                                    'Coefficient of Variation (%)': cv
                                })
                    
                    if volatility_data:
                        vol_df = pd.DataFrame(volatility_data)
                        
                        fig = px.bar(
                            vol_df,
                            x="Commodity",
                            y="Coefficient of Variation (%)",
                            color="Region",
                            barmode="group",
                            title="Price Volatility by Commodity and Region"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available for the selected commodities and timeframe.")
            
            elif analysis_type == "Regional Comparison":
                st.subheader("Regional Price Comparison")
                
                if len(selected_commodities) > 0:
                    commodity = selected_commodities[0]
                    regions = get_regions(commodity)
                    
                    if regions:
                        all_region_data = []
                        
                        for region in regions:
                            price_history = get_price_history(commodity, region, days)
                            
                            if price_history is not None and len(price_history) > 0:
                                latest_price = price_history[-1]['price']
                                
                                all_region_data.append({
                                    'Region': region,
                                    'Latest Price': latest_price
                                })
                        
                        if all_region_data:
                            region_df = pd.DataFrame(all_region_data)
                            
                            fig = px.bar(
                                region_df,
                                x="Region",
                                y="Latest Price",
                                title=f"Latest Price Comparison by Region - {commodity}",
                                labels={"Latest Price": "Price (₹/Quintal)"}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Heatmap of prices across regions and time
                            st.subheader("Regional Price Heatmap")
                            
                            # Get all data for heatmap
                            heatmap_data = []
                            
                            for region in regions:
                                price_history = get_price_history(commodity, region, days)
                                
                                if price_history is not None and len(price_history) > 0:
                                    for entry in price_history:
                                        heatmap_data.append({
                                            'Date': entry['date'],
                                            'Region': region,
                                            'Price': entry['price']
                                        })
                            
                            if heatmap_data:
                                heatmap_df = pd.DataFrame(heatmap_data)
                                
                                # Group by date and region
                                pivot_df = heatmap_df.pivot_table(
                                    index='Date', 
                                    columns='Region', 
                                    values='Price',
                                    aggfunc='mean'
                                )
                                
                                fig = px.imshow(
                                    pivot_df, 
                                    labels=dict(x="Region", y="Date", color="Price (₹/Quintal)"),
                                    title=f"Price Heatmap Across Regions - {commodity}",
                                    color_continuous_scale="RdBu_r"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No regional price data available for {commodity}.")
                    else:
                        st.info(f"No regions found for {commodity}.")
                else:
                    st.info("Please select a commodity for regional comparison.")
            
            elif analysis_type == "Quality-Price Correlation":
                st.subheader("Quality-Price Correlation Analysis")
                
                if len(selected_commodities) > 0:
                    commodity = selected_commodities[0]
                    commodity_data = get_commodity_data(commodity)
                    
                    if commodity_data and 'quality_parameters' in commodity_data:
                        # Get quality parameter to analyze
                        quality_params = list(commodity_data['quality_parameters'].keys())
                        
                        selected_param = st.selectbox(
                            "Select Quality Parameter", 
                            quality_params,
                            index=0 if quality_params else None
                        )
                        
                        if selected_param:
                            # Get regions
                            regions = get_regions(commodity)
                            
                            if regions:
                                selected_region = st.selectbox(
                                    "Select Region", 
                                    regions,
                                    index=0 if regions else None
                                )
                                
                                if selected_region:
                                    # Generate correlation data
                                    param_range = commodity_data['quality_parameters'][selected_param]
                                    min_val = param_range.get('min', 0)
                                    max_val = param_range.get('max', 100)
                                    standard_val = param_range.get('standard_value', (min_val + max_val) / 2)
                                    
                                    # Create a range of values
                                    param_values = np.linspace(min_val, max_val, 20)
                                    price_values = []
                                    
                                    # Calculate price for each parameter value
                                    for val in param_values:
                                        # Create a quality parameter dictionary with standard values
                                        quality_params_dict = {
                                            k: v.get('standard_value', (v.get('min', 0) + v.get('max', 100)) / 2) 
                                            for k, v in commodity_data['quality_parameters'].items()
                                        }
                                        
                                        # Replace the selected parameter with the current value
                                        quality_params_dict[selected_param] = val
                                        
                                        # Calculate price
                                        final_price, _, _, _, _ = calculate_price(
                                            commodity, 
                                            quality_params_dict, 
                                            selected_region
                                        )
                                        
                                        price_values.append(final_price)
                                    
                                    # Create correlation chart
                                    corr_df = pd.DataFrame({
                                        'Parameter Value': param_values,
                                        'Price': price_values
                                    })
                                    
                                    fig = px.line(
                                        corr_df,
                                        x="Parameter Value",
                                        y="Price",
                                        labels={"Parameter Value": f"{selected_param} ({param_range.get('unit', '')})", "Price": "Price (₹/Quintal)"},
                                        title=f"Price Correlation with {selected_param} - {commodity} ({selected_region})"
                                    )
                                    
                                    # Add vertical line at standard value
                                    fig.add_vline(
                                        x=standard_val,
                                        line_dash="dash",
                                        line_color="green",
                                        annotation_text=f"Standard Value ({standard_val})",
                                        annotation_position="top right"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Calculate sensitivity
                                    price_range = max(price_values) - min(price_values)
                                    param_range_val = max_val - min_val
                                    sensitivity = (price_range / param_range_val) / (sum(price_values) / len(price_values))
                                    
                                    st.info(f"Price Sensitivity to {selected_param}: {sensitivity:.4f} (relative change in price per unit change in parameter)")
                            else:
                                st.info(f"No regions found for {commodity}.")
                    else:
                        st.info(f"No quality parameters found for {commodity}.")
                else:
                    st.info("Please select a commodity for quality-price correlation analysis.")
            
            elif analysis_type == "Seasonal Analysis":
                st.subheader("Seasonal Price Analysis")
                
                # Get data for all selected commodities
                all_data = []
                
                for commodity in selected_commodities:
                    regions = get_regions(commodity)
                    
                    if regions:
                        for region in regions:
                            # Get a full year of data, regardless of selected timeframe
                            price_history = get_price_history(commodity, region, 365)
                            
                            if price_history is not None and len(price_history) > 0:
                                for entry in price_history:
                                    entry['Commodity'] = commodity
                                    entry['Region'] = region
                                    # Extract month
                                    entry['Month'] = entry['date'].strftime('%b')
                                    entry['MonthNum'] = entry['date'].month
                                
                                all_data.extend(price_history)
                
                if all_data:
                    # Create a DataFrame
                    df = pd.DataFrame(all_data)
                    
                    # Sort by month
                    df = df.sort_values('MonthNum')
                    
                    # Group by commodity, region, and month
                    monthly_avg = df.groupby(['Commodity', 'Region', 'Month', 'MonthNum'])['price'].mean().reset_index()
                    
                    # Plot the seasonal trends
                    fig = px.line(
                        monthly_avg, 
                        x="Month", 
                        y="price", 
                        color="Commodity",
                        line_dash="Region",
                        labels={"Month": "Month", "price": "Average Price (₹/Quintal)"},
                        title="Seasonal Price Trends",
                        category_orders={"Month": ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Month-over-month price changes
                    st.subheader("Month-over-Month Price Changes")
                    
                    # Prepare data for MoM changes
                    mom_changes = []
                    
                    for commodity in selected_commodities:
                        for region in df[df['Commodity'] == commodity]['Region'].unique():
                            commodity_region_data = monthly_avg[
                                (monthly_avg['Commodity'] == commodity) & 
                                (monthly_avg['Region'] == region)
                            ].sort_values('MonthNum')
                            
                            prev_price = None
                            
                            for _, row in commodity_region_data.iterrows():
                                if prev_price is not None:
                                    change_pct = ((row['price'] - prev_price) / prev_price) * 100
                                    
                                    mom_changes.append({
                                        'Commodity': commodity,
                                        'Region': region,
                                        'Month': row['Month'],
                                        'MonthNum': row['MonthNum'],
                                        'Change (%)': change_pct
                                    })
                                
                                prev_price = row['price']
                    
                    if mom_changes:
                        mom_df = pd.DataFrame(mom_changes)
                        
                        fig = px.bar(
                            mom_df,
                            x="Month",
                            y="Change (%)",
                            color="Commodity",
                            facet_row="Region",
                            labels={"Month": "Month", "Change (%)": "Price Change (%)"},
                            title="Month-over-Month Price Changes",
                            category_orders={"Month": ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']}
                        )
                        
                        fig.update_layout(height=500 * len(mom_df['Region'].unique()))
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available for the selected commodities.")

elif page == "WIZX Index":
    st.header("WIZX Agricultural Commodity Index")
    
    st.markdown("""
    Welcome to the WIZX Index - a standardized commodity price index for agricultural products,
    similar to financial indices like SENSEX and NYMEX. The WIZX Index provides a reliable benchmark
    for tracking agricultural commodity prices across different regions and qualities.
    """)
    
    # Tabs for different WIZX index views
    tab1, tab2, tab3, tab4 = st.tabs(["Index Dashboard", "Commodity Indices", "Sector Indices", "Analysis"])
    
    with tab1:
        st.subheader("WIZX Composite Index Dashboard")
        
        # Time period selection
        time_period = st.radio(
            "Select Time Period",
            ["1 Month", "3 Months", "6 Months", "1 Year"],
            horizontal=True,
            key="wizx_time_period"
        )
        
        days_map = {
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365
        }
        
        days = days_map[time_period]
        
        # Calculate composite index
        with st.spinner("Calculating WIZX Composite Index..."):
            try:
                # Calculate all indices first
                calculate_all_indices()
                
                # Calculate composite index
                composite_index = calculate_composite_index(date_val=date.today())
                
                if composite_index.get("success", False):
                    # Get historical performance
                    historical_data = historical_index_performance("WIZX-Composite", days)
                    
                    # Display current value
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "WIZX-Composite",
                            f"{composite_index['value']:.2f}",
                            f"{composite_index['change_percentage']:.2f}%"
                        )
                    
                    with col2:
                        # Calculate monthly change
                        if historical_data.get("success", False) and "periodic_returns" in historical_data:
                            monthly_return = historical_data["periodic_returns"]["last_month"]
                            if monthly_return is not None:
                                st.metric(
                                    "Monthly Change",
                                    f"{monthly_return:.2f}%",
                                    delta=None
                                )
                    
                    with col3:
                        # Display volatility
                        if historical_data.get("success", False) and "volatility" in historical_data:
                            volatility = historical_data["volatility"]
                            if volatility is not None:
                                st.metric(
                                    "Volatility (Std Dev)",
                                    f"{volatility:.2f}",
                                    delta=None
                                )
                    
                    # Display historical chart
                    if historical_data.get("success", False) and "values" in historical_data:
                        values = historical_data["values"]
                        
                        # Create dataframe for chart
                        df = pd.DataFrame(values)
                        
                        # Create line chart
                        fig = px.line(
                            df,
                            x="date",
                            y="value",
                            title="WIZX Composite Index - Historical Trend",
                            labels={"date": "Date", "value": "Index Value"}
                        )
                        
                        # Add reference line for initial value
                        if len(df) > 0:
                            fig.add_hline(
                                y=1000,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Base Value (1000)"
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate and display component contributions
                        if "components" in composite_index:
                            components = composite_index["components"]
                            
                            st.subheader("Component Contributions")
                            
                            # Create dataframe
                            component_df = pd.DataFrame([
                                {
                                    "Commodity": k,
                                    "Index Value": v["index_value"],
                                    "Weight": v["weight"],
                                    "Weighted Value": v["weighted_value"]
                                }
                                for k, v in components.items()
                            ])
                            
                            # Sort by weighted value
                            component_df = component_df.sort_values("Weighted Value", ascending=False)
                            
                            # Display table
                            st.dataframe(component_df, use_container_width=True)
                            
                            # Create bar chart of component contributions
                            fig = px.bar(
                                component_df,
                                x="Commodity",
                                y="Weighted Value",
                                title="Component Contributions to WIZX Composite Index",
                                color="Weighted Value",
                                labels={"Weighted Value": "Contribution to Index"}
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Failed to calculate composite index: {composite_index.get('message', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error calculating WIZX index: {e}")
    
    with tab2:
        st.subheader("Individual Commodity Indices")
        
        # Commodity selection
        commodities = get_all_commodities()
        selected_commodities = st.multiselect(
            "Select Commodities to Compare",
            commodities,
            default=[commodities[0]] if commodities else None,
            key="wizx_commodity_select"
        )
        
        if selected_commodities:
            # Time period selection
            time_period = st.radio(
                "Select Time Period",
                ["1 Month", "3 Months", "6 Months", "1 Year"],
                horizontal=True,
                key="commodity_time_period"
            )
            
            days_map = {
                "1 Month": 30,
                "3 Months": 90,
                "6 Months": 180,
                "1 Year": 365
            }
            
            days = days_map[time_period]
            
            # Display indices
            with st.spinner("Fetching commodity indices..."):
                try:
                    # Get indices for selected time period
                    end_date = date.today()
                    start_date = end_date - timedelta(days=days)
                    
                    indices = get_wizx_indices(start_date=start_date, end_date=end_date)
                    
                    if indices and any(c in indices for c in selected_commodities):
                        # Create metrics for current values
                        columns = st.columns(min(len(selected_commodities), 4))
                        
                        for i, commodity in enumerate(selected_commodities):
                            if commodity in indices:
                                commodity_indices = indices[commodity]
                                
                                if commodity_indices:
                                    latest_index = commodity_indices[-1]
                                    
                                    with columns[i % 4]:
                                        delta = latest_index["change_percentage"] if "change_percentage" in latest_index else None
                                        
                                        st.metric(
                                            commodity,
                                            f"{latest_index['index_value']:.2f}",
                                            f"{delta:.2f}%" if delta is not None else None
                                        )
                        
                        # Create chart with all selected commodities
                        st.subheader("Commodity Index Comparison")
                        
                        # Prepare data for chart
                        chart_data = []
                        
                        for commodity in selected_commodities:
                            if commodity in indices:
                                for idx in indices[commodity]:
                                    chart_data.append({
                                        "Commodity": commodity,
                                        "Date": idx["date"],
                                        "Index Value": idx["index_value"]
                                    })
                        
                        if chart_data:
                            chart_df = pd.DataFrame(chart_data)
                            
                            # Create line chart
                            fig = px.line(
                                chart_df,
                                x="Date",
                                y="Index Value",
                                color="Commodity",
                                title="WIZX Commodity Indices Comparison",
                                labels={"Date": "Date", "Index Value": "Index Value"}
                            )
                            
                            # Add reference line
                            fig.add_hline(
                                y=1000,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text="Base Value (1000)"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display correlation matrix
                            comparison = compare_indices(selected_commodities, days)
                            
                            if comparison and "correlation_matrix" in comparison:
                                st.subheader("Correlation Matrix")
                                
                                # Convert correlation dict to dataframe
                                corr_matrix = comparison["correlation_matrix"]
                                
                                if corr_matrix:
                                    corr_df = pd.DataFrame(corr_matrix)
                                    
                                    # Create heatmap
                                    fig = px.imshow(
                                        corr_df,
                                        text_auto=True,
                                        color_continuous_scale="RdBu_r",
                                        labels=dict(x="Commodity", y="Commodity", color="Correlation"),
                                        title="Price Correlation Between Commodities"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No index data available for the selected commodities and time period.")
                except Exception as e:
                    st.error(f"Error fetching commodity indices: {e}")
        else:
            st.info("Please select at least one commodity to view indices.")
    
    with tab3:
        st.subheader("Sector Indices")
        
        # Get sector indices
        with st.spinner("Calculating sector indices..."):
            try:
                # Define sectors
                sector_definitions = {
                    "WIZX-Cereals": ["Wheat", "Rice", "Maize"],
                    "WIZX-Pulses": ["Tur Dal"],
                    "WIZX-Oilseeds": ["Soyabean"]
                }
                
                # Calculate sector indices
                sector_indices = get_sector_indices(sector_definitions)
                
                if sector_indices and "sector_indices" in sector_indices:
                    sectors = sector_indices["sector_indices"]
                    
                    # Display metrics
                    columns = st.columns(len(sectors))
                    
                    for i, (sector, data) in enumerate(sectors.items()):
                        if data.get("success", False):
                            with columns[i]:
                                st.metric(
                                    sector,
                                    f"{data['value']:.2f}",
                                    f"{data['change_percentage']:.2f}%"
                                )
                    
                    # Time period selection for historical data
                    time_period = st.radio(
                        "Select Time Period",
                        ["1 Month", "3 Months", "6 Months", "1 Year"],
                        horizontal=True,
                        key="sector_time_period"
                    )
                    
                    days_map = {
                        "1 Month": 30,
                        "3 Months": 90,
                        "6 Months": 180,
                        "1 Year": 365
                    }
                    
                    days = days_map[time_period]
                    
                    # Get historical data for each sector
                    all_sector_data = []
                    
                    for sector in sectors.keys():
                        historical = historical_index_performance(sector, days)
                        
                        if historical.get("success", False) and "values" in historical:
                            for entry in historical["values"]:
                                all_sector_data.append({
                                    "Sector": sector,
                                    "Date": entry["date"],
                                    "Index Value": entry["value"]
                                })
                    
                    if all_sector_data:
                        sector_df = pd.DataFrame(all_sector_data)
                        
                        # Create line chart
                        fig = px.line(
                            sector_df,
                            x="Date",
                            y="Index Value",
                            color="Sector",
                            title="WIZX Sector Indices Comparison",
                            labels={"Date": "Date", "Index Value": "Index Value"}
                        )
                        
                        # Add reference line
                        fig.add_hline(
                            y=1000,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Base Value (1000)"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display sector composition
                        st.subheader("Sector Composition")
                        
                        for sector, commodities in sector_definitions.items():
                            st.write(f"**{sector}**: {', '.join(commodities)}")
                else:
                    st.error("Failed to calculate sector indices.")
            except Exception as e:
                st.error(f"Error calculating sector indices: {e}")
    
    with tab4:
        st.subheader("WIZX Index Analysis & Tools")
        
        analysis_option = st.selectbox(
            "Select Analysis",
            [
                "Performance Comparison",
                "Volatility Analysis",
                "Seasonal Patterns",
                "Export Data"
            ]
        )
        
        if analysis_option == "Performance Comparison":
            st.subheader("Performance Comparison")
            
            # Select time periods
            col1, col2 = st.columns(2)
            
            with col1:
                commodities = get_all_commodities() + ["WIZX-Composite", "WIZX-Cereals", "WIZX-Pulses", "WIZX-Oilseeds"]
                selected_items = st.multiselect(
                    "Select Indices to Compare",
                    commodities,
                    default=["WIZX-Composite"] if "WIZX-Composite" in commodities else None
                )
            
            with col2:
                time_period = st.selectbox(
                    "Time Period",
                    ["1 Month", "3 Months", "6 Months", "1 Year"]
                )
                
                days_map = {
                    "1 Month": 30,
                    "3 Months": 90,
                    "6 Months": 180,
                    "1 Year": 365
                }
                
                days = days_map[time_period]
            
            if selected_items:
                with st.spinner("Calculating performance metrics..."):
                    try:
                        # Get performance data
                        performance_data = []
                        
                        for item in selected_items:
                            performance = historical_index_performance(item, days)
                            
                            if performance.get("success", False) and "periodic_returns" in performance:
                                returns = performance["periodic_returns"]
                                
                                performance_data.append({
                                    "Index": item,
                                    "Current Value": performance.get("current_value", "N/A"),
                                    "Daily Change": returns.get("last_day", "N/A"),
                                    "Weekly Return": returns.get("last_week", "N/A"),
                                    "Monthly Return": returns.get("last_month", "N/A"),
                                    "Quarterly Return": returns.get("last_quarter", "N/A"),
                                    "Yearly Return": returns.get("last_year", "N/A"),
                                    "Volatility": performance.get("volatility", "N/A")
                                })
                        
                        if performance_data:
                            # Display as table
                            st.dataframe(pd.DataFrame(performance_data), use_container_width=True)
                            
                            # Create chart for periodic returns
                            periodic_data = []
                            
                            for item in performance_data:
                                # Daily change
                                if item["Daily Change"] != "N/A":
                                    periodic_data.append({
                                        "Index": item["Index"],
                                        "Period": "Daily",
                                        "Return": item["Daily Change"]
                                    })
                                
                                # Weekly return
                                if item["Weekly Return"] != "N/A":
                                    periodic_data.append({
                                        "Index": item["Index"],
                                        "Period": "Weekly",
                                        "Return": item["Weekly Return"]
                                    })
                                
                                # Monthly return
                                if item["Monthly Return"] != "N/A":
                                    periodic_data.append({
                                        "Index": item["Index"],
                                        "Period": "Monthly",
                                        "Return": item["Monthly Return"]
                                    })
                                
                                # Quarterly return
                                if item["Quarterly Return"] != "N/A":
                                    periodic_data.append({
                                        "Index": item["Index"],
                                        "Period": "Quarterly",
                                        "Return": item["Quarterly Return"]
                                    })
                            
                            if periodic_data:
                                periodic_df = pd.DataFrame(periodic_data)
                                
                                # Create grouped bar chart
                                fig = px.bar(
                                    periodic_df,
                                    x="Period",
                                    y="Return",
                                    color="Index",
                                    barmode="group",
                                    title="Periodic Returns Comparison",
                                    labels={"Period": "Time Period", "Return": "Return (%)"}
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No performance data available for selected indices.")
                    except Exception as e:
                        st.error(f"Error calculating performance metrics: {e}")
        
        elif analysis_option == "Volatility Analysis":
            st.subheader("Volatility Analysis")
            
            # Select time periods
            col1, col2 = st.columns(2)
            
            with col1:
                commodities = get_all_commodities() + ["WIZX-Composite", "WIZX-Cereals", "WIZX-Pulses", "WIZX-Oilseeds"]
                selected_items = st.multiselect(
                    "Select Indices to Analyze",
                    commodities,
                    default=["WIZX-Composite"] if "WIZX-Composite" in commodities else None
                )
            
            with col2:
                time_period = st.selectbox(
                    "Time Period",
                    ["1 Month", "3 Months", "6 Months", "1 Year"],
                    key="volatility_period"
                )
                
                days_map = {
                    "1 Month": 30,
                    "3 Months": 90,
                    "6 Months": 180,
                    "1 Year": 365
                }
                
                days = days_map[time_period]
            
            if selected_items:
                with st.spinner("Calculating volatility metrics..."):
                    try:
                        # Get volatility data
                        volatility_data = []
                        
                        for item in selected_items:
                            performance = historical_index_performance(item, days)
                            
                            if performance.get("success", False):
                                volatility_data.append({
                                    "Index": item,
                                    "Volatility (StdDev)": performance.get("volatility", "N/A"),
                                    "Min Value": performance.get("min_value", "N/A"),
                                    "Max Value": performance.get("max_value", "N/A"),
                                    "Range": performance.get("max_value", 0) - performance.get("min_value", 0) if "max_value" in performance and "min_value" in performance else "N/A"
                                })
                        
                        if volatility_data:
                            # Display as table
                            st.dataframe(pd.DataFrame(volatility_data), use_container_width=True)
                            
                            # Create volatility chart
                            valid_items = [item for item in volatility_data if item["Volatility (StdDev)"] != "N/A"]
                            
                            if valid_items:
                                volatility_df = pd.DataFrame(valid_items)
                                
                                # Create bar chart for volatility
                                fig = px.bar(
                                    volatility_df,
                                    x="Index",
                                    y="Volatility (StdDev)",
                                    title="Volatility Comparison",
                                    labels={"Index": "Index", "Volatility (StdDev)": "Volatility (Standard Deviation)"},
                                    color="Volatility (StdDev)"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Create range chart
                                range_data = []
                                
                                for item in valid_items:
                                    if item["Min Value"] != "N/A" and item["Max Value"] != "N/A":
                                        range_data.append({
                                            "Index": item["Index"],
                                            "Min": item["Min Value"],
                                            "Max": item["Max Value"]
                                        })
                                
                                if range_data:
                                    range_df = pd.DataFrame(range_data)
                                    
                                    # Create range chart
                                    fig = go.Figure()
                                    
                                    for i, row in range_df.iterrows():
                                        fig.add_trace(go.Bar(
                                            name=row["Index"],
                                            x=[row["Index"]],
                                            y=[row["Max"] - row["Min"]],
                                            base=row["Min"],
                                            marker_color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                                        ))
                                    
                                    fig.update_layout(
                                        title="Price Range During Period",
                                        xaxis_title="Index",
                                        yaxis_title="Index Value",
                                        barmode="group"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No volatility data available for selected indices.")
                    except Exception as e:
                        st.error(f"Error calculating volatility metrics: {e}")
        
        elif analysis_option == "Seasonal Patterns":
            st.subheader("Seasonal Patterns Analysis")
            
            # Select commodity and time period
            col1, col2 = st.columns(2)
            
            with col1:
                commodities = get_all_commodities()
                selected_commodity = st.selectbox(
                    "Select Commodity",
                    commodities,
                    index=0 if commodities else None
                )
            
            with col2:
                year_count = st.slider("Number of Years to Analyze", 1, 5, 2)
            
            if selected_commodity:
                with st.spinner("Analyzing seasonal patterns..."):
                    try:
                        # Calculate days
                        days = year_count * 365
                        
                        # Get historical data
                        historical = historical_index_performance(selected_commodity, days)
                        
                        if historical.get("success", False) and "values" in historical:
                            values = historical["values"]
                            
                            # Create dataframe
                            df = pd.DataFrame(values)
                            
                            # Convert date to datetime if it's not already
                            if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                                df["date"] = pd.to_datetime(df["date"])
                            
                            # Extract month and year
                            df["month"] = df["date"].dt.month
                            df["year"] = df["date"].dt.year
                            df["month_name"] = df["date"].dt.strftime("%b")
                            
                            # Calculate monthly averages
                            monthly_avg = df.groupby(["year", "month", "month_name"])["value"].mean().reset_index()
                            
                            # Create heatmap
                            pivot_table = monthly_avg.pivot_table(
                                values="value",
                                index="year",
                                columns="month_name",
                                aggfunc="mean"
                            )
                            
                            # Reorder columns by month
                            month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                            pivot_table = pivot_table[pivot_table.columns.intersection(month_order).sort_values(key=lambda m: month_order.index(m))]
                            
                            # Create heatmap
                            fig = px.imshow(
                                pivot_table,
                                labels=dict(x="Month", y="Year", color="Index Value"),
                                title=f"Seasonal Pattern for {selected_commodity}",
                                color_continuous_scale="Viridis"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate overall monthly averages
                            overall_monthly = df.groupby("month_name")["value"].mean().reset_index()
                            
                            # Sort by month order
                            overall_monthly["month_idx"] = overall_monthly["month_name"].apply(lambda x: month_order.index(x) if x in month_order else 999)
                            overall_monthly = overall_monthly.sort_values("month_idx")
                            
                            # Create bar chart
                            fig = px.bar(
                                overall_monthly,
                                x="month_name",
                                y="value",
                                title=f"Average Monthly Values for {selected_commodity}",
                                labels={"month_name": "Month", "value": "Average Index Value"},
                                color="value"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display month-over-month changes
                            if len(df) > 30:
                                # Calculate monthly average changes
                                df["pct_change"] = df.groupby("year")["value"].pct_change() * 100
                                
                                monthly_change = df.groupby("month_name")["pct_change"].mean().reset_index()
                                monthly_change["month_idx"] = monthly_change["month_name"].apply(lambda x: month_order.index(x) if x in month_order else 999)
                                monthly_change = monthly_change.sort_values("month_idx")
                                
                                # Create bar chart for monthly changes
                                fig = px.bar(
                                    monthly_change,
                                    x="month_name",
                                    y="pct_change",
                                    title=f"Average Monthly Price Changes for {selected_commodity}",
                                    labels={"month_name": "Month", "pct_change": "Average Change (%)"},
                                    color="pct_change",
                                    color_continuous_scale="RdBu_r"
                                )
                                
                                # Add zero line
                                fig.add_hline(
                                    y=0,
                                    line_dash="dash",
                                    line_color="gray"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Insufficient historical data available for seasonal analysis.")
                    except Exception as e:
                        st.error(f"Error analyzing seasonal patterns: {e}")
        
        elif analysis_option == "Export Data":
            st.subheader("Export WIZX Index Data")
            
            # Select export options
            col1, col2 = st.columns(2)
            
            with col1:
                # Select commodities
                commodities = get_all_commodities() + ["WIZX-Composite", "WIZX-Cereals", "WIZX-Pulses", "WIZX-Oilseeds"]
                selected_items = st.multiselect(
                    "Select Indices to Export",
                    commodities,
                    default=["WIZX-Composite"] if "WIZX-Composite" in commodities else []
                )
            
            with col2:
                # Select date range
                date_range = st.date_input(
                    "Select Date Range",
                    value=(date.today() - timedelta(days=30), date.today()),
                    max_value=date.today()
                )
            
            if len(date_range) == 2 and selected_items:
                start_date, end_date = date_range
                
                if st.button("Export WIZX Data"):
                    with st.spinner("Preparing export..."):
                        try:
                            export_result = export_indices(
                                start_date=start_date,
                                end_date=end_date,
                                commodities=selected_items
                            )
                            
                            if export_result.get("success", False):
                                st.success(f"Data exported successfully to {export_result['file_path']}")
                                
                                # Display preview
                                try:
                                    preview_df = pd.read_csv(export_result['file_path'])
                                    st.subheader("Data Preview")
                                    st.dataframe(preview_df.head(10), use_container_width=True)
                                    
                                    # Provide download link
                                    with open(export_result['file_path'], 'rb') as f:
                                        st.download_button(
                                            label="Download CSV File",
                                            data=f,
                                            file_name=os.path.basename(export_result['file_path']),
                                            mime="text/csv"
                                        )
                                except Exception as e:
                                    st.error(f"Error displaying preview: {e}")
                            else:
                                st.error(f"Export failed: {export_result.get('message', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"Error exporting data: {e}")

elif page == "Data Submission":
    st.header("Data Submission & Community Contribution")
    
    st.markdown("""
    Help improve the accuracy and coverage of our price data by submitting your own observations.
    Contributors with verified submissions earn reward points that can be redeemed for benefits.
    """)
    
    tab1, tab2, tab3 = st.tabs(["Submit Data", "Check Status", "Leaderboard"])
    
    with tab1:
        st.subheader("Submit Market Price Data")
        
        # User identification
        user_email = st.text_input("Your Email (for tracking contributions)")
        
        # Price data form
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic information
            commodities = get_all_commodities()
            selected_commodity = st.selectbox(
                "Commodity",
                commodities,
                index=0 if commodities else None,
                key="submit_commodity"
            )
            
            if selected_commodity:
                regions = get_regions(selected_commodity)
                selected_region = st.selectbox(
                    "Region/Market",
                    regions,
                    index=0 if regions else None,
                    key="submit_region"
                )
                
                market_name = st.text_input("Specific Market/Mandi (Optional)")
                
                observed_date = st.date_input(
                    "Observation Date",
                    value=date.today(),
                    max_value=date.today()
                )
                
                price = st.number_input(
                    "Price (₹ per Quintal)",
                    min_value=0.0,
                    value=0.0,
                    step=100.0
                )
        
        with col2:
            st.subheader("Quality Parameters (Optional)")
            
            quality_params = {}
            
            if selected_commodity:
                commodity_data = get_commodity_data(selected_commodity)
                
                if commodity_data and 'quality_parameters' in commodity_data:
                    for param, details in commodity_data['quality_parameters'].items():
                        default_value = details.get('standard_value', (details.get('min', 0) + details.get('max', 100)) / 2)
                        min_value = details.get('min', 0)
                        max_value = details.get('max', 100)
                        
                        param_value = st.slider(
                            f"{param} ({details.get('unit', '')})", 
                            min_value=float(min_value),
                            max_value=float(max_value),
                            value=float(default_value),
                            step=details.get('step', 0.1),
                            key=f"submit_quality_{param}"
                        )
                        
                        quality_params[param] = param_value
        
        # Source details
        st.subheader("Source Information")
        
        source_options = ["Direct Observation", "Local Dealer", "Farmer", "Market Committee", "News/Media", "Other"]
        source_type = st.selectbox("Source Type", source_options)
        
        source_details = st.text_area("Additional Details (Optional)", height=100)
        
        if st.button("Submit Price Data"):
            if not user_email:
                st.error("Please provide your email address for tracking contributions.")
            elif not selected_commodity or not selected_region or price <= 0:
                st.error("Please fill in all required fields: Commodity, Region, and Price.")
            else:
                with st.spinner("Submitting data..."):
                    try:
                        # Prepare source details
                        source_info = {
                            "type": source_type,
                            "details": source_details,
                            "market": market_name
                        }
                        
                        # Submit data
                        result = submit_price_data(
                            user_email=user_email,
                            commodity=selected_commodity,
                            region=selected_region,
                            price=price,
                            date_val=observed_date,
                            quality_params=quality_params,
                            market=market_name,
                            source_details=source_info
                        )
                        
                        if result.get("success", False):
                            st.success(result.get("message", "Data submitted successfully!"))
                            
                            # Verify automatically
                            verify_result = auto_verify_submission(result.get("submission_id"))
                            
                            if verify_result.get("verified", False):
                                st.success("Your submission has been automatically verified!")
                            elif verify_result.get("requires_manual_review", False):
                                st.info("Your submission will be reviewed manually by our team.")
                            
                            # Calculate price recommendation for comparison
                            recommendation = get_price_recommendation(
                                selected_commodity, 
                                quality_params,
                                selected_region
                            )
                            
                            if recommendation and 'recommended_price' in recommendation:
                                st.subheader("Your submission compared to recommendations")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "Your Submitted Price",
                                        f"₹{price:.2f}"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Recommended Price",
                                        f"₹{recommendation['recommended_price']:.2f}",
                                        f"{((price - recommendation['recommended_price']) / recommendation['recommended_price'] * 100):.2f}%"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Confidence",
                                        recommendation['confidence'].title()
                                    )
                        else:
                            st.error(result.get("message", "Failed to submit data."))
                    except Exception as e:
                        st.error(f"Error submitting data: {e}")
    
    with tab2:
        st.subheader("Check Submission Status")
        
        user_identifier = st.text_input("Enter Your Email or User ID")
        
        if user_identifier:
            if st.button("Check Status"):
                with st.spinner("Fetching submission status..."):
                    try:
                        # Get user submission status
                        status = get_user_submission_status(user_identifier)
                        
                        if status and "user_id" in status:
                            if status.get("total_submissions", 0) > 0:
                                # Display metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Submissions", status.get("total_submissions", 0))
                                
                                with col2:
                                    st.metric("Verified", status.get("verified_submissions", 0))
                                
                                with col3:
                                    st.metric("Pending Review", status.get("pending_submissions", 0))
                                
                                with col4:
                                    st.metric("Reward Points", status.get("total_reward_points", 0))
                                
                                # Display recent submissions
                                if "recent_submissions" in status:
                                    st.subheader("Recent Submissions")
                                    
                                    # Convert to dataframe
                                    submissions_df = pd.DataFrame(status["recent_submissions"])
                                    
                                    # Format dataframe
                                    if not submissions_df.empty:
                                        # Reorder columns
                                        columns = ["id", "commodity", "region", "date", "price", "is_verified", "verification_score", "reward_points"]
                                        display_columns = [c for c in columns if c in submissions_df.columns]
                                        
                                        # Rename columns
                                        rename_map = {
                                            "id": "ID",
                                            "commodity": "Commodity",
                                            "region": "Region",
                                            "date": "Date",
                                            "price": "Price (₹)",
                                            "is_verified": "Verified",
                                            "verification_score": "Score",
                                            "reward_points": "Points"
                                        }
                                        
                                        # Display
                                        st.dataframe(
                                            submissions_df[display_columns].rename(columns=rename_map),
                                            use_container_width=True
                                        )
                                
                                # Display commodity breakdown
                                if "commodity_breakdown" in status and status["commodity_breakdown"]:
                                    st.subheader("Commodity Breakdown")
                                    
                                    # Convert to dataframe
                                    breakdown_data = []
                                    
                                    for commodity, data in status["commodity_breakdown"].items():
                                        breakdown_data.append({
                                            "Commodity": commodity,
                                            "Submissions": data.get("submissions", 0),
                                            "Points": data.get("points", 0)
                                        })
                                    
                                    if breakdown_data:
                                        breakdown_df = pd.DataFrame(breakdown_data)
                                        
                                        # Create bar chart
                                        fig = px.bar(
                                            breakdown_df,
                                            x="Commodity",
                                            y="Points",
                                            title="Points Earned by Commodity",
                                            color="Submissions",
                                            labels={"Points": "Reward Points"}
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("You have not submitted any data yet.")
                        else:
                            st.warning("No submissions found for this email or user ID.")
                    except Exception as e:
                        st.error(f"Error fetching submission status: {e}")
    
    with tab3:
        st.subheader("Community Leaderboard")
        
        # Time period selection
        period_options = {
            "Last Week": 7,
            "Last Month": 30,
            "Last Quarter": 90,
            "Last Year": 365,
            "All Time": 3650
        }
        
        selected_period = st.selectbox("Time Period", list(period_options.keys()))
        days = period_options[selected_period]
        
        with st.spinner("Fetching leaderboard..."):
            try:
                # Get leaderboard
                leaderboard_data = get_leaderboard(days=days, limit=50)
                
                if leaderboard_data and "leaderboard" in leaderboard_data:
                    leaderboard = leaderboard_data["leaderboard"]
                    
                    if leaderboard:
                        # Display top contributors
                        st.subheader(f"Top Contributors - {selected_period}")
                        
                        # Convert to dataframe
                        leaderboard_df = pd.DataFrame([
                            {
                                "Rank": item["rank"],
                                "User ID": item["user_id"],
                                "Submissions": item["submissions"],
                                "Points": item["points"],
                                "Top Commodities": ", ".join([f"{c['name']} ({c['submissions']})" for c in item["top_commodities"]]) if "top_commodities" in item else ""
                            }
                            for item in leaderboard
                        ])
                        
                        # Display table
                        st.dataframe(leaderboard_df, use_container_width=True)
                        
                        # Create bar chart for top contributors
                        top_10 = leaderboard_df.head(10).copy()
                        
                        if not top_10.empty:
                            # Create horizontal bar chart
                            fig = px.bar(
                                top_10,
                                y="User ID",
                                x="Points",
                                title="Top 10 Contributors",
                                color="Points",
                                orientation="h",
                                labels={"Points": "Reward Points"}
                            )
                            
                            # Sort by points
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No contributors found for the selected period.")
                else:
                    st.info("No leaderboard data available.")
            except Exception as e:
                st.error(f"Error fetching leaderboard: {e}")

elif page == "Data Cleaning":
    st.header("Data Cleaning & Quality Control")
    
    st.markdown("""
    This page provides tools for cleaning and validating commodity price data.
    Use these tools to ensure data quality and consistency for accurate price analysis.
    """)
    
    tab1, tab2, tab3 = st.tabs(["Anomaly Detection", "Data Validation", "Data Cleaning"])
    
    with tab1:
        st.subheader("Price Anomaly Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Commodity selection
            commodities = get_all_commodities()
            selected_commodity = st.selectbox(
                "Select Commodity",
                ["All Commodities"] + commodities,
                index=0,
                key="anomaly_commodity"
            )
            
            commodity_param = None if selected_commodity == "All Commodities" else selected_commodity
            
            # Region selection
            if commodity_param:
                regions = ["All Regions"] + get_regions(commodity_param)
                selected_region = st.selectbox(
                    "Select Region",
                    regions,
                    index=0,
                    key="anomaly_region"
                )
                
                region_param = None if selected_region == "All Regions" else selected_region
            else:
                region_param = None
        
        with col2:
            # Detection parameters
            time_period = st.selectbox(
                "Time Period",
                ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year"],
                index=1,
                key="anomaly_time_period"
            )
            
            days_map = {
                "Last 7 Days": 7,
                "Last 30 Days": 30,
                "Last 90 Days": 90,
                "Last Year": 365
            }
            
            days = days_map[time_period]
            
            method = st.selectbox(
                "Detection Method",
                ["isolation_forest", "dbscan"],
                index=0,
                key="anomaly_method"
            )
        
        if st.button("Detect Anomalies"):
            with st.spinner("Detecting price anomalies..."):
                try:
                    # Detect anomalies
                    anomalies = detect_price_anomalies(commodity_param, region_param, days, method)
                    
                    if anomalies and "results" in anomalies:
                        results = anomalies["results"]
                        
                        if results:
                            st.success(f"Detected {anomalies.get('total_anomalies', 0)} anomalies out of {anomalies.get('total_points', 0)} price points.")
                            
                            # Display results
                            for result in results:
                                st.subheader(f"{result['commodity']} - {result['region']}")
                                
                                st.write(f"Found {len(result['anomalies'])} anomalies out of {result['total_points']} price points ({result['anomaly_percentage']:.1f}%).")
                                
                                # Create chart
                                if result['anomalies']:
                                    # Create dataframe
                                    anomaly_df = pd.DataFrame([
                                        {
                                            "id": a["id"],
                                            "date": a["date"],
                                            "price": a["price"],
                                            "expected_min": a["expected_range"][0],
                                            "expected_max": a["expected_range"][1]
                                        }
                                        for a in result['anomalies']
                                    ])
                                    
                                    # Create chart
                                    fig = px.scatter(
                                        anomaly_df,
                                        x="date",
                                        y="price",
                                        title=f"Price Anomalies for {result['commodity']} in {result['region']}",
                                        labels={"date": "Date", "price": "Price (₹/Quintal)"}
                                    )
                                    
                                    # Add range
                                    if len(anomaly_df) > 0:
                                        min_date = min(anomaly_df["date"])
                                        max_date = max(anomaly_df["date"])
                                        avg_min = anomaly_df["expected_min"].mean()
                                        avg_max = anomaly_df["expected_max"].mean()
                                        
                                        fig.add_shape(
                                            type="rect",
                                            x0=min_date,
                                            x1=max_date,
                                            y0=avg_min,
                                            y1=avg_max,
                                            line=dict(color="Green"),
                                            fillcolor="rgba(0,255,0,0.1)",
                                            name="Expected Range"
                                        )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Display anomalies table
                                    st.write("Anomalous Prices:")
                                    st.dataframe(anomaly_df, use_container_width=True)
                        else:
                            st.info("No anomalies detected for the specified parameters.")
                    else:
                        st.info("No anomalies detected.")
                except Exception as e:
                    st.error(f"Error detecting anomalies: {e}")
    
    with tab2:
        st.subheader("Price Curve Validation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Commodity selection
            commodities = get_all_commodities()
            validate_commodity = st.selectbox(
                "Select Commodity",
                commodities,
                index=0 if commodities else None,
                key="validate_commodity"
            )
            
            if validate_commodity:
                regions = ["All Regions"] + get_regions(validate_commodity)
                validate_region = st.selectbox(
                    "Select Region",
                    regions,
                    index=0,
                    key="validate_region"
                )
                
                region_param = None if validate_region == "All Regions" else validate_region
            else:
                region_param = None
        
        with col2:
            # Validation parameters
            time_period = st.selectbox(
                "Time Period",
                ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year"],
                index=1,
                key="validate_time_period"
            )
            
            days_map = {
                "Last 7 Days": 7,
                "Last 30 Days": 30,
                "Last 90 Days": 90,
                "Last Year": 365
            }
            
            days = days_map[time_period]
        
        if validate_commodity and st.button("Validate Price Curves"):
            with st.spinner("Validating price curves..."):
                try:
                    # Validate price curve
                    validation = validate_price_curve(validate_commodity, region_param, days)
                    
                    if validation and "results" in validation:
                        results = validation["results"]
                        
                        if results:
                            # Display summary
                            valid_count = sum(1 for r in results if r["valid"])
                            st.write(f"Validation completed for {len(results)} regions. {valid_count} valid, {len(results) - valid_count} with issues.")
                            
                            # Create tabs for each region
                            region_tabs = st.tabs([r["region"] for r in results])
                            
                            for i, result in enumerate(results):
                                with region_tabs[i]:
                                    if result["valid"]:
                                        st.success(f"Price curve for {result['region']} is valid.")
                                    else:
                                        st.warning(f"Found {len(result['issues'])} issues in price curve for {result['region']}.")
                                        
                                        # Display issues
                                        for issue in result["issues"]:
                                            issue_type = issue["type"]
                                            
                                            if issue_type == "large_jump":
                                                st.warning(f"Large price jump detected on {issue['date']}: {issue['change']:.1f}% change to ₹{issue['price']:.2f}")
                                            elif issue_type == "constant_price":
                                                st.info(f"Constant price period: ₹{issue['price']:.2f} for {issue['days']} days starting {issue['start_date']}")
                                            elif issue_type == "missing_dates":
                                                st.error(f"Missing data: {issue['count']} missing dates")
                                                if issue['count'] < 10:
                                                    st.write(f"Missing dates: {', '.join(str(d) for d in issue['dates'])}")
                                        
                                        # Offer to fix issues
                                        st.write("Would you like to fix these issues?")
                                        fix_options = []
                                        
                                        for issue in result["issues"]:
                                            if issue["type"] == "missing_dates":
                                                fix_options.append("Fill missing dates with interpolation")
                                            elif issue["type"] == "large_jump":
                                                fix_options.append("Smooth large price jumps")
                                        
                                        if fix_options:
                                            selected_fixes = st.multiselect(
                                                "Select fixes to apply",
                                                fix_options,
                                                key=f"fixes_{result['region']}"
                                            )
                                            
                                            if selected_fixes and st.button(f"Apply Fixes for {result['region']}"):
                                                with st.spinner("Applying fixes..."):
                                                    if "Fill missing dates with interpolation" in selected_fixes:
                                                        missing_fix = fix_missing_data(validate_commodity, days=days)
                                                        if missing_fix.get("fixed", 0) > 0:
                                                            st.success(f"Fixed {missing_fix['fixed']} missing data points.")
                                                        else:
                                                            st.info("No missing data points were fixed.")
                                                    
                                                    # Run validation again
                                                    new_validation = validate_price_curve(validate_commodity, result["region"], days)
                                                    if new_validation and "results" in new_validation:
                                                        new_result = next((r for r in new_validation["results"] if r["region"] == result["region"]), None)
                                                        if new_result:
                                                            if new_result["valid"]:
                                                                st.success(f"Price curve for {new_result['region']} is now valid.")
                                                            else:
                                                                st.warning(f"Price curve for {new_result['region']} still has {len(new_result['issues'])} issues.")
                        else:
                            st.info("No validation results available.")
                    else:
                        st.error("Validation failed.")
                except Exception as e:
                    st.error(f"Error validating price curves: {e}")
    
    with tab3:
        st.subheader("Run Data Cleaning Pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Commodity selection
            commodities = get_all_commodities()
            cleaning_commodity = st.selectbox(
                "Select Commodity",
                ["All Commodities"] + commodities,
                index=0,
                key="cleaning_commodity"
            )
            
            commodity_param = None if cleaning_commodity == "All Commodities" else cleaning_commodity
        
        with col2:
            # Cleaning parameters
            time_period = st.selectbox(
                "Time Period",
                ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year"],
                index=1,
                key="cleaning_time_period"
            )
            
            days_map = {
                "Last 7 Days": 7,
                "Last 30 Days": 30,
                "Last 90 Days": 90,
                "Last Year": 365
            }
            
            days = days_map[time_period]
        
        if st.button("Run Data Cleaning"):
            with st.spinner("Running data cleaning pipeline..."):
                try:
                    # Run data cleaning pipeline
                    cleaning_results = clean_data_pipeline(commodity_param, days)
                    
                    if cleaning_results:
                        if "error" in cleaning_results:
                            st.error(f"Error in data cleaning pipeline: {cleaning_results['error']}")
                        else:
                            st.success(cleaning_results.get("summary", "Data cleaning completed successfully."))
                            
                            # Display cleaning stats
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Anomalies Detected", cleaning_results.get("anomalies_detected", 0))
                            
                            with col2:
                                st.metric("Missing Data Fixed", cleaning_results.get("missing_data_fixed", 0))
                            
                            with col3:
                                st.metric("Validation Issues", cleaning_results.get("validation_issues", 0))
                            
                            with col4:
                                st.metric("Rules Applied", cleaning_results.get("rules_applied", 0))
                    else:
                        st.error("Data cleaning failed.")
                except Exception as e:
                    st.error(f"Error running data cleaning pipeline: {e}")

elif page == "Data Explorer":
    st.header("Data Explorer")
    
    data_type = st.selectbox(
        "Select Data Type",
        ["Commodity Pricing Data", "Market Data", "Quality Parameter Data", "Regional Data"]
    )
    
    if data_type == "Commodity Pricing Data":
        st.subheader("Explore Commodity Pricing Data")
        
        # Filters
        commodities = fetch_commodity_list()
        selected_commodity = st.selectbox(
            "Select Commodity", 
            commodities,
            index=0 if commodities else None
        )
        
        if selected_commodity:
            regions = get_regions(selected_commodity)
            
            if regions:
                selected_region = st.selectbox(
                    "Select Region", 
                    regions,
                    index=0 if regions else None
                )
                
                if selected_region:
                    # Date range
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        start_date = st.date_input(
                            "Start Date",
                            datetime.now() - timedelta(days=30)
                        )
                    
                    with col2:
                        end_date = st.date_input(
                            "End Date",
                            datetime.now()
                        )
                    
                    # Calculate days difference
                    days_diff = (end_date - start_date).days
                    
                    if days_diff > 0:
                        # Get price data
                        price_data = get_price_history(
                            selected_commodity, 
                            selected_region, 
                            days_diff,
                            start_date=start_date
                        )
                        
                        if price_data is not None and len(price_data) > 0:
                            # Convert to DataFrame
                            df = pd.DataFrame(price_data)
                            
                            # Add additional stats
                            df['7-Day MA'] = df['price'].rolling(window=7).mean()
                            df['30-Day MA'] = df['price'].rolling(window=30).mean()
                            
                            # Plot data
                            st.subheader(f"Price Data: {selected_commodity} in {selected_region}")
                            
                            fig = go.Figure()
                            
                            # Add price line
                            fig.add_trace(go.Scatter(
                                x=df['date'],
                                y=df['price'],
                                mode='lines',
                                name='Daily Price'
                            ))
                            
                            # Add moving averages if enough data
                            if len(df) >= 7:
                                fig.add_trace(go.Scatter(
                                    x=df['date'],
                                    y=df['7-Day MA'],
                                    mode='lines',
                                    line=dict(dash='dash'),
                                    name='7-Day MA'
                                ))
                            
                            if len(df) >= 30:
                                fig.add_trace(go.Scatter(
                                    x=df['date'],
                                    y=df['30-Day MA'],
                                    mode='lines',
                                    line=dict(dash='dot'),
                                    name='30-Day MA'
                                ))
                            
                            fig.update_layout(
                                title=f"{selected_commodity} Price Trends in {selected_region}",
                                xaxis_title="Date",
                                yaxis_title="Price (₹/Quintal)",
                                legend=dict(y=0.99, x=0.01),
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Statistics
                            st.subheader("Price Statistics")
                            
                            stats = {
                                "Average Price": df['price'].mean(),
                                "Minimum Price": df['price'].min(),
                                "Maximum Price": df['price'].max(),
                                "Price Range": df['price'].max() - df['price'].min(),
                                "Standard Deviation": df['price'].std(),
                                "Coefficient of Variation (%)": (df['price'].std() / df['price'].mean()) * 100
                            }
                            
                            stats_df = pd.DataFrame({
                                'Statistic': list(stats.keys()),
                                'Value': list(stats.values())
                            })
                            
                            st.dataframe(stats_df, use_container_width=True)
                            
                            # Raw data
                            st.subheader("Raw Data")
                            
                            st.dataframe(
                                df[['date', 'price', '7-Day MA', '30-Day MA']],
                                use_container_width=True
                            )
                            
                            # Download option
                            csv = df.to_csv(index=False).encode('utf-8')
                            
                            st.download_button(
                                "Download Data as CSV",
                                data=csv,
                                file_name=f"{selected_commodity}_{selected_region}_{start_date}_to_{end_date}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No data available for the selected parameters.")
                    else:
                        st.error("End date must be after start date.")
            else:
                st.info(f"No regions found for {selected_commodity}.")
    
    elif data_type == "Market Data":
        st.subheader("Explore Market Data")
        
        market_data_type = st.radio(
            "Select Market Data Type",
            ["Market Arrivals", "Trade Volumes", "Price Spreads"],
            horizontal=True
        )
        
        if market_data_type == "Market Arrivals":
            # Commodity selection
            commodities = fetch_commodity_list()
            selected_commodity = st.selectbox(
                "Select Commodity", 
                commodities,
                index=0 if commodities else None,
                key="arrival_commodity"
            )
            
            if selected_commodity:
                regions = get_regions(selected_commodity)
                
                if regions:
                    selected_regions = st.multiselect(
                        "Select Regions", 
                        regions,
                        default=[regions[0]] if regions else []
                    )
                    
                    if selected_regions:
                        # Date range
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            start_date = st.date_input(
                                "Start Date",
                                datetime.now() - timedelta(days=30),
                                key="arrival_start"
                            )
                        
                        with col2:
                            end_date = st.date_input(
                                "End Date",
                                datetime.now(),
                                key="arrival_end"
                            )
                        
                        # Fetch and display market arrival data
                        st.info("Fetching market arrival data... This may take a moment.")
                        
                        # Placeholder for actual data
                        arrivals_data = []
                        
                        # Generate some example data
                        date_range = pd.date_range(start=start_date, end=end_date)
                        
                        for region in selected_regions:
                            for date in date_range:
                                arrivals_data.append({
                                    'Date': date,
                                    'Region': region,
                                    'Arrivals (MT)': np.random.randint(50, 500)
                                })
                        
                        arrivals_df = pd.DataFrame(arrivals_data)
                        
                        # Plot data
                        fig = px.line(
                            arrivals_df,
                            x="Date",
                            y="Arrivals (MT)",
                            color="Region",
                            title=f"Market Arrivals for {selected_commodity}",
                            labels={"Date": "Date", "Arrivals (MT)": "Arrivals (Metric Tons)"}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show raw data
                        st.subheader("Raw Arrival Data")
                        st.dataframe(arrivals_df, use_container_width=True)
                    else:
                        st.info("Please select at least one region.")
                else:
                    st.info(f"No regions found for {selected_commodity}.")
        
        elif market_data_type == "Trade Volumes":
            st.info("Trade volume data is currently under development.")
        
        elif market_data_type == "Price Spreads":
            # Commodity selection
            commodities = fetch_commodity_list()
            selected_commodity = st.selectbox(
                "Select Commodity", 
                commodities,
                index=0 if commodities else None,
                key="spread_commodity"
            )
            
            if selected_commodity:
                regions = get_regions(selected_commodity)
                
                if len(regions) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        region1 = st.selectbox(
                            "Select Base Region", 
                            regions,
                            index=0,
                            key="spread_region1"
                        )
                    
                    with col2:
                        # Filter out the first selected region
                        regions2 = [r for r in regions if r != region1]
                        region2 = st.selectbox(
                            "Select Comparison Region", 
                            regions2,
                            index=0,
                            key="spread_region2"
                        )
                    
                    # Date range
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        start_date = st.date_input(
                            "Start Date",
                            datetime.now() - timedelta(days=30),
                            key="spread_start"
                        )
                    
                    with col2:
                        end_date = st.date_input(
                            "End Date",
                            datetime.now(),
                            key="spread_end"
                        )
                    
                    # Calculate days difference
                    days_diff = (end_date - start_date).days
                    
                    if days_diff > 0:
                        # Get price data for both regions
                        prices1 = get_price_history(
                            selected_commodity, 
                            region1, 
                            days_diff,
                            start_date=start_date
                        )
                        
                        prices2 = get_price_history(
                            selected_commodity, 
                            region2, 
                            days_diff,
                            start_date=start_date
                        )
                        
                        if prices1 and prices2:
                            # Convert to DataFrames and merge
                            df1 = pd.DataFrame(prices1)
                            df2 = pd.DataFrame(prices2)
                            
                            # Rename columns
                            df1.rename(columns={'price': f'Price {region1}'}, inplace=True)
                            df2.rename(columns={'price': f'Price {region2}'}, inplace=True)
                            
                            # Merge on date
                            merged_df = pd.merge(df1, df2, on='date', how='inner')
                            
                            # Calculate spread
                            merged_df['Price Spread'] = merged_df[f'Price {region1}'] - merged_df[f'Price {region2}']
                            merged_df['Percentage Spread'] = (merged_df['Price Spread'] / merged_df[f'Price {region1}']) * 100
                            
                            # Plot data
                            st.subheader(f"Price Spread Analysis: {region1} vs {region2}")
                            
                            # Create tabs for different visualizations
                            tab1, tab2, tab3 = st.tabs(["Price Comparison", "Absolute Spread", "Percentage Spread"])
                            
                            with tab1:
                                fig = go.Figure()
                                
                                fig.add_trace(go.Scatter(
                                    x=merged_df['date'],
                                    y=merged_df[f'Price {region1}'],
                                    mode='lines',
                                    name=f'{region1} Price'
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=merged_df['date'],
                                    y=merged_df[f'Price {region2}'],
                                    mode='lines',
                                    name=f'{region2} Price'
                                ))
                                
                                fig.update_layout(
                                    title=f"{selected_commodity} Price Comparison",
                                    xaxis_title="Date",
                                    yaxis_title="Price (₹/Quintal)",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab2:
                                fig = px.line(
                                    merged_df,
                                    x='date',
                                    y='Price Spread',
                                    title=f"Absolute Price Spread ({region1} - {region2})",
                                    labels={"date": "Date", "Price Spread": "Price Spread (₹/Quintal)"}
                                )
                                
                                # Add zero line
                                fig.add_hline(
                                    y=0,
                                    line_dash="dash",
                                    line_color="gray"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with tab3:
                                fig = px.line(
                                    merged_df,
                                    x='date',
                                    y='Percentage Spread',
                                    title=f"Percentage Price Spread ({region1} vs {region2})",
                                    labels={"date": "Date", "Percentage Spread": "Percentage Spread (%)"}
                                )
                                
                                # Add zero line
                                fig.add_hline(
                                    y=0,
                                    line_dash="dash",
                                    line_color="gray"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Statistics
                            st.subheader("Spread Statistics")
                            
                            stats = {
                                "Average Spread (₹/Quintal)": merged_df['Price Spread'].mean(),
                                "Minimum Spread (₹/Quintal)": merged_df['Price Spread'].min(),
                                "Maximum Spread (₹/Quintal)": merged_df['Price Spread'].max(),
                                "Spread Range (₹/Quintal)": merged_df['Price Spread'].max() - merged_df['Price Spread'].min(),
                                "Average Percentage Spread (%)": merged_df['Percentage Spread'].mean(),
                                "Standard Deviation of Spread": merged_df['Price Spread'].std()
                            }
                            
                            stats_df = pd.DataFrame({
                                'Statistic': list(stats.keys()),
                                'Value': list(stats.values())
                            })
                            
                            st.dataframe(stats_df, use_container_width=True)
                            
                            # Show raw data
                            st.subheader("Raw Spread Data")
                            st.dataframe(
                                merged_df[['date', f'Price {region1}', f'Price {region2}', 'Price Spread', 'Percentage Spread']],
                                use_container_width=True
                            )
                        else:
                            st.info("Insufficient data available for the selected parameters.")
                    else:
                        st.error("End date must be after start date.")
                else:
                    st.info(f"Insufficient regions found for {selected_commodity} to calculate spreads.")
    
    elif data_type == "Quality Parameter Data":
        st.subheader("Explore Quality Parameter Data")
        
        # Commodity selection
        commodities = fetch_commodity_list()
        selected_commodity = st.selectbox(
            "Select Commodity", 
            commodities,
            index=0 if commodities else None,
            key="quality_data_commodity"
        )
        
        if selected_commodity:
            # Get commodity data
            commodity_data = get_commodity_data(selected_commodity)
            
            if commodity_data and 'quality_parameters' in commodity_data:
                # Display quality parameters
                st.subheader(f"Quality Parameters for {selected_commodity}")
                
                params_data = []
                
                for param, details in commodity_data['quality_parameters'].items():
                    params_data.append({
                        'Parameter': param,
                        'Standard Value': details.get('standard_value', 'N/A'),
                        'Unit': details.get('unit', ''),
                        'Minimum': details.get('min', 'N/A'),
                        'Maximum': details.get('max', 'N/A')
                    })
                
                params_df = pd.DataFrame(params_data)
                
                st.dataframe(params_df, use_container_width=True)
                
                # Quality impact visualization
                st.subheader("Quality Parameter Impact Visualization")
                
                # Select parameter to visualize
                quality_params = list(commodity_data['quality_parameters'].keys())
                
                selected_param = st.selectbox(
                    "Select Quality Parameter", 
                    quality_params,
                    index=0 if quality_params else None,
                    key="quality_data_param"
                )
                
                if selected_param:
                    regions = get_regions(selected_commodity)
                    
                    if regions:
                        selected_region = st.selectbox(
                            "Select Region", 
                            regions,
                            index=0 if regions else None,
                            key="quality_data_region"
                        )
                        
                        if selected_region:
                            # Generate quality impact visualization
                            param_range = commodity_data['quality_parameters'][selected_param]
                            min_val = param_range.get('min', 0)
                            max_val = param_range.get('max', 100)
                            standard_val = param_range.get('standard_value', (min_val + max_val) / 2)
                            
                            # Create a range of values
                            param_values = np.linspace(min_val, max_val, 20)
                            delta_values = []
                            
                            # Calculate price delta for each parameter value
                            for val in param_values:
                                # Create a quality parameter dictionary with standard values
                                quality_params_dict = {
                                    k: v.get('standard_value', (v.get('min', 0) + v.get('max', 100)) / 2) 
                                    for k, v in commodity_data['quality_parameters'].items()
                                }
                                
                                # Replace the selected parameter with the current value
                                quality_params_dict[selected_param] = val
                                
                                # Calculate price
                                _, base_price, quality_delta, _, _ = calculate_price(
                                    selected_commodity, 
                                    quality_params_dict, 
                                    selected_region
                                )
                                
                                delta_values.append(quality_delta)
                            
                            # Create impact chart
                            impact_df = pd.DataFrame({
                                'Parameter Value': param_values,
                                'Price Delta': delta_values
                            })
                            
                            fig = px.line(
                                impact_df,
                                x="Parameter Value",
                                y="Price Delta",
                                labels={"Parameter Value": f"{selected_param} ({param_range.get('unit', '')})", "Price Delta": "Price Delta (₹/Quintal)"},
                                title=f"Price Impact of {selected_param} - {selected_commodity} ({selected_region})"
                            )
                            
                            # Add vertical line at standard value
                            fig.add_vline(
                                x=standard_val,
                                line_dash="dash",
                                line_color="green",
                                annotation_text=f"Standard Value ({standard_val})",
                                annotation_position="top right"
                            )
                            
                            # Add horizontal line at zero
                            fig.add_hline(
                                y=0,
                                line_dash="dash",
                                line_color="red"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Impact ranges
                            st.subheader("Quality Impact Analysis")
                            
                            min_delta = min(delta_values)
                            max_delta = max(delta_values)
                            delta_range = max_delta - min_delta
                            
                            st.markdown(f"""
                            - **Maximum Premium**: ₹{max_delta:.2f} / Quintal
                            - **Maximum Discount**: ₹{min_delta:.2f} / Quintal
                            - **Total Impact Range**: ₹{delta_range:.2f} / Quintal
                            """)
                            
                            # Impact on final price
                            st.subheader("Impact on Final Price")
                            
                            # Get current base price
                            standard_quality = {k: v.get('standard_value', (v.get('min', 0) + v.get('max', 100)) / 2) 
                                               for k, v in commodity_data['quality_parameters'].items()}
                            
                            final_price, base_price, _, _, _ = calculate_price(
                                selected_commodity, 
                                standard_quality, 
                                selected_region
                            )
                            
                            # Calculate impact percentage
                            impact_pct = (delta_range / base_price) * 100
                            
                            st.markdown(f"""
                            - **Current Base Price**: ₹{base_price:.2f} / Quintal
                            - **Maximum Price Range Due to {selected_param}**: ₹{base_price + min_delta:.2f} to ₹{base_price + max_delta:.2f} / Quintal
                            - **Percentage Impact**: {impact_pct:.2f}% of base price
                            """)
                    else:
                        st.info(f"No regions found for {selected_commodity}.")
            else:
                st.info(f"No quality parameter data available for {selected_commodity}.")
    
    elif data_type == "Regional Data":
        st.subheader("Explore Regional Data")
        
        # Filters
        regions = st.multiselect(
            "Select Regions",
            ["North India", "South India", "East India", "West India", "Central India"],
            default=["North India"]
        )
        
        if regions:
            # Commodities by region
            st.subheader("Commodities by Region")
            
            # Generate example data
            region_data = []
            
            for region in regions:
                # Get commodities for this region
                commodities = fetch_commodity_list(region=region)
                
                for commodity in commodities:
                    # Get current price
                    price = np.random.randint(2000, 8000)
                    
                    region_data.append({
                        'Region': region,
                        'Commodity': commodity,
                        'Current Price (₹/Quintal)': price
                    })
            
            if region_data:
                region_df = pd.DataFrame(region_data)
                
                # Create a heat map
                pivot_df = region_df.pivot_table(
                    index='Commodity', 
                    columns='Region', 
                    values='Current Price (₹/Quintal)',
                    aggfunc='mean'
                )
                
                fig = px.imshow(
                    pivot_df, 
                    labels=dict(x="Region", y="Commodity", color="Price (₹/Quintal)"),
                    title="Commodity Prices by Region",
                    color_continuous_scale="RdBu_r"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show raw data
                st.subheader("Regional Price Data")
                st.dataframe(region_df, use_container_width=True)
            else:
                st.info("No data available for the selected regions.")
        else:
            st.info("Please select at least one region.")

# Footer with information about the data sources
st.markdown("---")
st.markdown("""
**Data Sources**: AGMARKNET, National Agriculture Market (eNAM), State Agricultural Marketing Boards, NCDEX

**Disclaimer**: This tool provides pricing indicators based on available market data and is intended for informational purposes only. Actual market prices may vary.
""")
