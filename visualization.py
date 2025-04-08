"""
Visualization module for creating interactive charts and visualizations.
"""

import logging
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from database import get_commodity_data
from pricing_engine import calculate_price_curve, calculate_price
from database_sql import get_price_history, get_regions, get_all_commodities

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_price_trend_chart(price_history):
    """
    Create a line chart showing price trends over time.
    
    Args:
        price_history (list): List of price data points
        
    Returns:
        go.Figure: Plotly figure object
    """
    logger.info("Creating price trend chart")
    
    # Convert to DataFrame if not already
    if not isinstance(price_history, pd.DataFrame):
        price_history = pd.DataFrame(price_history)
    
    # Create the figure
    fig = px.line(
        price_history, 
        x="date", 
        y="price", 
        labels={"date": "Date", "price": "Price (₹/Quintal)"},
        title="Price Trend"
    )
    
    # Add moving averages if enough data
    if len(price_history) >= 7:
        # 7-day moving average
        ma7 = price_history['price'].rolling(window=7).mean()
        fig.add_trace(
            go.Scatter(
                x=price_history['date'],
                y=ma7,
                mode='lines',
                name='7-Day MA',
                line=dict(dash='dash', color='rgba(255, 165, 0, 0.7)')
            )
        )
    
    if len(price_history) >= 30:
        # 30-day moving average
        ma30 = price_history['price'].rolling(window=30).mean()
        fig.add_trace(
            go.Scatter(
                x=price_history['date'],
                y=ma30,
                mode='lines',
                name='30-Day MA',
                line=dict(dash='dot', color='rgba(255, 0, 0, 0.7)')
            )
        )
    
    # Update layout
    fig.update_layout(
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
    
    return fig

def create_quality_impact_chart(commodity, quality_params, region):
    """
    Create a chart showing the impact of quality parameters on price.
    
    Args:
        commodity (str): The commodity
        quality_params (dict): Quality parameters
        region (str): The region
        
    Returns:
        go.Figure: Plotly figure object
    """
    logger.info(f"Creating quality impact chart for {commodity}")
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data or 'quality_parameters' not in commodity_data:
        logger.warning(f"No quality parameters found for commodity: {commodity}")
        return go.Figure()
    
    # Create data for chart
    chart_data = []
    
    # Get standard quality parameters
    standard_params = {
        param: details.get('standard_value', (details.get('min', 0) + details.get('max', 100)) / 2)
        for param, details in commodity_data['quality_parameters'].items()
    }
    
    # Calculate price with standard parameters
    standard_price_result = calculate_price(
        commodity, standard_params, region
    )
    standard_price = standard_price_result[0]  # Final price
    
    # Calculate price with actual parameters
    actual_price_result = calculate_price(
        commodity, quality_params, region
    )
    actual_price = actual_price_result[0]  # Final price
    
    # Calculate delta for each parameter
    for param, value in quality_params.items():
        if param in standard_params:
            # Create test parameters with just this parameter changed
            test_params = standard_params.copy()
            test_params[param] = value
            
            # Calculate price with just this parameter changed
            test_price_result = calculate_price(
                commodity, test_params, region
            )
            test_price = test_price_result[0]  # Final price
            
            # Calculate the impact
            impact = test_price - standard_price
            
            # Add to chart data
            chart_data.append({
                "Parameter": param,
                "Impact": impact,
                "Value": value,
                "Standard": standard_params[param],
                "Unit": commodity_data['quality_parameters'][param].get('unit', '')
            })
    
    # Sort by absolute impact
    chart_data = sorted(chart_data, key=lambda x: abs(x["Impact"]), reverse=True)
    
    # Create the chart
    fig = go.Figure()
    
    # Add bars for parameter impacts
    fig.add_trace(
        go.Bar(
            y=[item["Parameter"] for item in chart_data],
            x=[item["Impact"] for item in chart_data],
            orientation='h',
            marker_color=['red' if x < 0 else 'green' for x in [item["Impact"] for item in chart_data]],
            name='Quality Impact'
        )
    )
    
    # Add annotations for values
    for i, item in enumerate(chart_data):
        parameter = item["Parameter"]
        impact = item["Impact"]
        value = item["Value"]
        std = item["Standard"]
        unit = item["Unit"]
        
        # Format the annotation text
        deviation = value - std
        deviation_str = f"+{deviation:.2f}" if deviation > 0 else f"{deviation:.2f}"
        
        annotation_text = f"{value:.2f} {unit} ({deviation_str} from standard)"
        
        fig.add_annotation(
            y=parameter,
            x=impact,
            text=annotation_text,
            showarrow=True,
            arrowhead=7,
            ax=40 if impact >= 0 else -40,
            ay=0
        )
    
    # Update layout
    fig.update_layout(
        title=f"Quality Parameter Impact on Price - {commodity}",
        xaxis_title="Price Impact (₹/Quintal)",
        yaxis_title="Quality Parameter",
        height=400 + (len(chart_data) * 40)  # Dynamic height based on number of parameters
    )
    
    # Add a vertical line at zero
    fig.add_vline(
        x=0,
        line_width=1,
        line_dash="dash",
        line_color="gray"
    )
    
    # Add total impact annotation
    total_impact = actual_price - standard_price
    impact_str = f"+{total_impact:.2f}" if total_impact > 0 else f"{total_impact:.2f}"
    
    fig.add_annotation(
        x=0.5,
        y=1.05,
        xref="paper",
        yref="paper",
        text=f"Total Quality Impact: {impact_str} ₹/Quintal",
        showarrow=False,
        font=dict(
            size=14,
            color="black"
        ),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.3)",
        borderwidth=1,
        borderpad=4
    )
    
    return fig

def create_price_distribution_chart(price_data, commodity):
    """
    Create a histogram showing the distribution of prices.
    
    Args:
        price_data (list or pd.DataFrame): Price data
        commodity (str): The commodity
        
    Returns:
        go.Figure: Plotly figure object
    """
    logger.info(f"Creating price distribution chart for {commodity}")
    
    # Convert to DataFrame if not already
    if not isinstance(price_data, pd.DataFrame):
        price_data = pd.DataFrame(price_data)
    
    # Create the histogram
    fig = px.histogram(
        price_data, 
        x="price",
        nbins=20,
        labels={"price": "Price (₹/Quintal)"},
        title=f"Price Distribution - {commodity}"
    )
    
    # Add a vertical line for the mean
    mean_price = price_data['price'].mean()
    
    fig.add_vline(
        x=mean_price,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: ₹{mean_price:.2f}",
        annotation_position="top right"
    )
    
    # Add a vertical line for the median
    median_price = price_data['price'].median()
    
    fig.add_vline(
        x=median_price,
        line_width=2,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Median: ₹{median_price:.2f}",
        annotation_position="top left"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Price (₹/Quintal)",
        yaxis_title="Frequency",
        height=400
    )
    
    return fig

def create_regional_comparison_chart(commodity, price_data_by_region):
    """
    Create a bar chart comparing prices across different regions.
    
    Args:
        commodity (str): The commodity
        price_data_by_region (dict): Price data by region
        
    Returns:
        go.Figure: Plotly figure object
    """
    logger.info(f"Creating regional comparison chart for {commodity}")
    
    # Prepare data for chart
    regions = []
    prices = []
    
    for region, data in price_data_by_region.items():
        regions.append(region)
        # Calculate average price for the region
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and 'price' in data[0]:
                avg_price = sum(item['price'] for item in data) / len(data)
            else:
                avg_price = sum(data) / len(data)
        elif isinstance(data, (int, float)):
            avg_price = data
        else:
            avg_price = 0
        
        prices.append(avg_price)
    
    # Create the bar chart
    fig = px.bar(
        x=regions,
        y=prices,
        labels={"x": "Region", "y": "Average Price (₹/Quintal)"},
        title=f"Regional Price Comparison - {commodity}"
    )
    
    # Add price labels
    for i, price in enumerate(prices):
        fig.add_annotation(
            x=regions[i],
            y=price,
            text=f"₹{price:.2f}",
            showarrow=False,
            yshift=10
        )
    
    # Calculate and display national average
    if prices:
        national_avg = sum(prices) / len(prices)
        
        fig.add_hline(
            y=national_avg,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"National Avg: ₹{national_avg:.2f}",
            annotation_position="top right"
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Region",
        yaxis_title="Average Price (₹/Quintal)",
        height=400
    )
    
    return fig

def create_quality_price_matrix(commodity, param1, param2, region):
    """
    Create a heatmap showing how two quality parameters together affect price.
    
    Args:
        commodity (str): The commodity
        param1 (str): First quality parameter
        param2 (str): Second quality parameter
        region (str): The region
        
    Returns:
        go.Figure: Plotly figure object
    """
    logger.info(f"Creating quality-price matrix for {commodity}, {param1} vs {param2}")
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data or 'quality_parameters' not in commodity_data:
        logger.warning(f"No quality parameters found for commodity: {commodity}")
        return go.Figure()
    
    # Check if parameters exist
    if param1 not in commodity_data['quality_parameters'] or param2 not in commodity_data['quality_parameters']:
        logger.warning(f"Parameters {param1} or {param2} not found for {commodity}")
        return go.Figure()
    
    # Get parameter ranges
    param1_data = commodity_data['quality_parameters'][param1]
    param2_data = commodity_data['quality_parameters'][param2]
    
    param1_min = param1_data.get('min', 0)
    param1_max = param1_data.get('max', 100)
    param2_min = param2_data.get('min', 0)
    param2_max = param2_data.get('max', 100)
    
    # Generate parameter values
    param1_values = np.linspace(param1_min, param1_max, 10)
    param2_values = np.linspace(param2_min, param2_max, 10)
    
    # Get standard quality parameters
    standard_params = {
        param: details.get('standard_value', (details.get('min', 0) + details.get('max', 100)) / 2)
        for param, details in commodity_data['quality_parameters'].items()
    }
    
    # Calculate prices for each combination
    price_matrix = np.zeros((len(param2_values), len(param1_values)))
    
    for i, val2 in enumerate(param2_values):
        for j, val1 in enumerate(param1_values):
            # Update parameters
            test_params = standard_params.copy()
            test_params[param1] = val1
            test_params[param2] = val2
            
            # Calculate price
            price_result = calculate_price(commodity, test_params, region)
            price_matrix[i, j] = price_result[0]  # Final price
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=price_matrix,
        x=param1_values,
        y=param2_values,
        colorscale='RdBu_r',
        colorbar=dict(title="Price (₹/Quintal)")
    ))
    
    # Add standard value markers
    standard_val1 = standard_params[param1]
    standard_val2 = standard_params[param2]
    
    # Find closest indices
    idx1 = np.abs(param1_values - standard_val1).argmin()
    idx2 = np.abs(param2_values - standard_val2).argmin()
    
    fig.add_shape(
        type="circle",
        x0=param1_values[idx1] - (param1_max - param1_min) / 20,
        y0=param2_values[idx2] - (param2_max - param2_min) / 20,
        x1=param1_values[idx1] + (param1_max - param1_min) / 20,
        y1=param2_values[idx2] + (param2_max - param2_min) / 20,
        line=dict(color="black", width=2),
        fillcolor="rgba(0,0,0,0)"
    )
    
    # Update layout
    fig.update_layout(
        title=f"Price Impact Matrix - {param1} vs {param2}",
        xaxis_title=f"{param1} ({param1_data.get('unit', '')})",
        yaxis_title=f"{param2} ({param2_data.get('unit', '')})",
        height=600,
        width=700
    )
    
    # Add annotation for standard values
    fig.add_annotation(
        x=standard_val1,
        y=standard_val2,
        text="Standard",
        showarrow=True,
        arrowhead=7,
        ax=40,
        ay=40
    )
    
    return fig

def create_seasonal_price_chart(commodity, region, years=3):
    """
    Create a chart showing seasonal price patterns.
    
    Args:
        commodity (str): The commodity
        region (str): The region
        years (int): Number of years to analyze
        
    Returns:
        go.Figure: Plotly figure object
    """
    logger.info(f"Creating seasonal price chart for {commodity} in {region}")
    
    # For a real implementation, this would fetch historical data
    # For now, generate synthetic data with realistic seasonal patterns
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price data with seasonality
    # Get commodity data for baseline price
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data:
        logger.warning(f"No data found for commodity: {commodity}")
        return go.Figure()
    
    # Get price range for this commodity
    price_range = commodity_data.get('price_range', {})
    min_price = price_range.get('min', 2000)
    max_price = price_range.get('max', 5000)
    
    base_price = (min_price + max_price) / 2
    
    # Different commodities have different seasonal patterns
    # This is a simplified model for demonstration
    seasonal_patterns = {
        "Wheat": {"amplitude": 0.15, "peak_month": 3},  # Peak in March
        "Rice": {"amplitude": 0.10, "peak_month": 11},  # Peak in November
        "Maize": {"amplitude": 0.20, "peak_month": 8},  # Peak in August
        "Tur Dal": {"amplitude": 0.12, "peak_month": 1},  # Peak in January
        "Soyabean": {"amplitude": 0.18, "peak_month": 10}  # Peak in October
    }
    
    pattern = seasonal_patterns.get(commodity, {"amplitude": 0.15, "peak_month": 6})
    
    # Generate prices with seasonality
    prices = []
    
    for date in date_range:
        month = date.month
        year = date.year
        
        # Calculate days from peak month
        days_from_peak = ((month - pattern["peak_month"]) % 12) * 30
        
        # Calculate seasonal factor using cosine function (1-year period)
        seasonal_factor = pattern["amplitude"] * np.cos(2 * np.pi * days_from_peak / 365)
        
        # Add long-term trend
        years_from_start = (date - start_date).days / 365
        trend_factor = 1 + (0.05 * years_from_start)  # 5% annual growth
        
        # Add some noise
        noise = np.random.normal(0, 0.02)
        
        # Calculate price
        price = base_price * (1 + seasonal_factor) * trend_factor * (1 + noise)
        
        prices.append({
            "date": date,
            "price": price,
            "month": date.strftime('%b'),
            "month_num": month,
            "year": year
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(prices)
    
    # Calculate monthly averages
    monthly_avg = df.groupby(['month', 'month_num', 'year'])['price'].mean().reset_index()
    
    # Create the chart
    fig = px.line(
        monthly_avg, 
        x="month_num", 
        y="price", 
        color="year",
        labels={"month_num": "Month", "price": "Average Price (₹/Quintal)", "year": "Year"},
        title=f"Seasonal Price Pattern - {commodity} in {region}",
        markers=True
    )
    
    # Set x-axis to show month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig.update_xaxes(
        tickvals=list(range(1, 13)),
        ticktext=month_names
    )
    
    # Calculate average seasonal pattern across years
    seasonal_avg = monthly_avg.groupby('month_num')['price'].mean().reset_index()
    
    fig.add_trace(
        go.Scatter(
            x=seasonal_avg['month_num'],
            y=seasonal_avg['price'],
            mode='lines+markers',
            name='Average Pattern',
            line=dict(width=3, color='black', dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Price (₹/Quintal)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    return fig


def get_multi_region_price_data(commodity, regions=None, days=30):
    """
    Get price history data for multiple regions for a single commodity.
    
    Args:
        commodity (str): The commodity name
        regions (list, optional): List of regions to include (if None, all regions are included)
        days (int): Number of days of history to fetch
        
    Returns:
        dict: Data suitable for creating a heatmap visualization
    """
    logger.info(f"Getting multi-region price data for {commodity} across {days} days")
    
    # If regions not specified, get all regions for this commodity
    if regions is None:
        regions = get_regions(commodity)
    
    if not regions:
        logger.warning(f"No regions found for commodity {commodity}")
        return None
    
    # Get price history for each region
    price_data = {}
    all_dates = set()
    
    for region in regions:
        history = get_price_history(commodity, region, days)
        if history:
            price_data[region] = history
            # Collect all dates
            all_dates.update(item["date"] for item in history)
    
    if not price_data:
        logger.warning(f"No price data found for {commodity} in any region")
        return None
    
    # Create a sorted list of all dates
    all_dates = sorted(list(all_dates))
    
    # Create a matrix of prices
    date_indices = {date: i for i, date in enumerate(all_dates)}
    price_matrix = np.zeros((len(regions), len(all_dates)))
    price_matrix.fill(np.nan)  # Fill with NaN to represent missing data
    
    for i, region in enumerate(regions):
        if region in price_data:
            for item in price_data[region]:
                j = date_indices[item["date"]]
                price_matrix[i, j] = item["price"]
    
    # Return data suitable for heatmap
    return {
        "commodity": commodity,
        "regions": regions,
        "dates": all_dates,
        "price_matrix": price_matrix
    }


def create_price_trend_heatmap(commodity, regions=None, days=30):
    """
    Create a heatmap visualization of price trends across multiple regions.
    
    Args:
        commodity (str): The commodity name
        regions (list, optional): List of regions to include (if None, all regions are included)
        days (int): Number of days of history to display
        
    Returns:
        go.Figure: Plotly figure object with interactive price trend heatmap
    """
    logger.info(f"Creating price trend heatmap for {commodity} across {days} days")
    
    # Get the data
    heatmap_data = get_multi_region_price_data(commodity, regions, days)
    
    if not heatmap_data:
        logger.warning(f"No data available for price trend heatmap for {commodity}")
        return go.Figure()
    
    # Extract data components
    regions = heatmap_data["regions"]
    dates = heatmap_data["dates"]
    price_matrix = heatmap_data["price_matrix"]
    
    # Format dates for display
    date_labels = [d.strftime('%Y-%m-%d') for d in dates]
    
    # Calculate min, max, and mean prices (excluding NaNs)
    valid_prices = price_matrix[~np.isnan(price_matrix)]
    min_price = np.min(valid_prices) if valid_prices.size > 0 else 0
    max_price = np.max(valid_prices) if valid_prices.size > 0 else 0
    mean_price = np.mean(valid_prices) if valid_prices.size > 0 else 0
    
    # Create a custom color scale centered around the mean
    mid_point = (mean_price - min_price) / (max_price - min_price) if max_price > min_price else 0.5
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=price_matrix,
        x=date_labels,
        y=regions,
        colorscale='RdBu_r',  # Red-White-Blue color scale (reversed)
        zmid=mean_price,      # Center the color scale at the mean
        zmin=min_price,
        zmax=max_price,
        colorbar=dict(
            title="Price (₹/Quintal)",
            titleside="right"
        ),
        hoverongaps=False,
        hovertemplate='Region: %{y}<br>Date: %{x}<br>Price: ₹%{z:.2f}/Quintal<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Price Trend Heatmap - {commodity} (Last {days} Days)",
        xaxis_title="Date",
        yaxis_title="Region",
        height=max(400, 100 + (len(regions) * 30)),  # Dynamic height based on number of regions
        width=max(700, 300 + (len(dates) * 15)),     # Dynamic width based on date range
        xaxis=dict(
            tickangle=-45,
            type='category',
            tickmode='array',
            tickvals=date_labels[::max(1, len(dates)//10)],  # Show every nth date label to avoid crowding
            ticktext=date_labels[::max(1, len(dates)//10)]
        ),
        yaxis=dict(
            type='category'
        ),
        margin=dict(l=100, r=20, t=80, b=80)
    )
    
    # Add annotation for mean price
    fig.add_annotation(
        x=0.5,
        y=1.05,
        xref="paper",
        yref="paper",
        text=f"Average Price: ₹{mean_price:.2f}/Quintal",
        showarrow=False,
        font=dict(
            size=14,
            color="black"
        ),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.3)",
        borderwidth=1,
        borderpad=4
    )
    
    return fig


if __name__ == "__main__":
    # Test functions
    print("Testing visualization module...")
