import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data_crawler import fetch_latest_agmarknet_data, fetch_commodity_list
from data_processor import process_data, standardize_commodity
from pricing_engine import calculate_price, get_price_history
from visualization import create_price_trend_chart, create_quality_impact_chart
from database import get_commodity_data, save_user_input, get_regions
from quality_analyzer import analyze_quality_from_image, analyze_report
from models import predict_price_trend

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

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", 
    ["Price Dashboard", "Quality Analysis", "Market Trends", "Data Explorer"])

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
                        min_value=min_value,
                        max_value=max_value,
                        value=default_value,
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
                        min_value=min_value,
                        max_value=max_value,
                        value=default_value,
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
                    days_diff = (end_date - start_date.date()).days
                    
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
                    days_diff = (end_date - start_date.date()).days
                    
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
