
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

def create_regional_price_map(prices_by_region, commodity):
    """
    Create an interactive choropleth map of India showing prices by region.
    """
    # Simplified GeoJSON with major regions of India
    india_regions = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "North India"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[76, 28], [80, 28], [80, 32], [76, 32], [76, 28]]]
                }
            },
            {
                "type": "Feature", 
                "properties": {"name": "South India"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[74, 8], [80, 8], [80, 14], [74, 14], [74, 8]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "East India"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[84, 20], [88, 20], [88, 26], [84, 26], [84, 20]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "West India"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[70, 18], [74, 18], [74, 24], [70, 24], [70, 18]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Central India"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[76, 20], [82, 20], [82, 24], [76, 24], [76, 20]]]
                }
            }
        ]
    }
    
    # Convert prices to DataFrame
    df = pd.DataFrame([
        {"region": region, "price": price}
        for region, price in prices_by_region.items()
    ])
    
    # Create choropleth map
    fig = px.choropleth(
        df,
        geojson=india_regions,
        locations="region",
        featureidkey="properties.name",
        color="price",
        color_continuous_scale="Viridis",
        range_color=[df["price"].min(), df["price"].max()],
        title=f"{commodity} Prices by Region",
        labels={"price": "Price (₹/Quintal)"}
    )
    
    fig.update_geos(
        fitbounds="locations",
        visible=False,
    )
    
    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 30, "l": 0, "b": 0}
    )
    
    return fig

def create_price_heatmap(prices_by_region, commodity):
    """
    Create a heatmap visualization of prices across regions.
    """
    # Create a more detailed matrix including all major regions
    all_regions = ["North India", "South India", "East India", "West India", "Central India"]
    prices_matrix = []
    
    # Get prices for each region, use None if not available
    prices = [prices_by_region.get(region, None) for region in all_regions]
    
    # Create normalized prices for better visualization
    max_price = max(p for p in prices if p is not None) if any(p is not None for p in prices) else 0
    normalized_prices = [p/max_price if p is not None else 0 for p in prices]
    
    fig = go.Figure(data=go.Heatmap(
        z=[normalized_prices],
        x=all_regions,
        y=[commodity],
        colorscale='RdBu_r',
        text=[[f"₹{p:,.2f}" if p is not None else "N/A" for p in prices]],
        texttemplate="%{text}",
        textfont={"size": 14},
        colorbar=dict(
            title="Price Index",
            tickformat=".0%"
        )
    ))
    
    fig.update_layout(
        title=f"Regional Price Distribution - {commodity}",
        xaxis_title="Region",
        yaxis_title="Commodity",
        height=400,
        font=dict(size=12),
        margin=dict(t=50, b=50)
    )
    
    return fig

def create_price_spread_analysis(prices_by_region, commodity):
    """
    Create price spread analysis visualization.
    """
    avg_price = sum(prices_by_region.values()) / len(prices_by_region)
    spreads = {region: ((price - avg_price) / avg_price) * 100 
              for region, price in prices_by_region.items()}
    
    df = pd.DataFrame([
        {"region": region, "spread": spread}
        for region, spread in spreads.items()
    ])
    
    fig = px.bar(
        df,
        x="region",
        y="spread",
        title=f"Price Spread Analysis - {commodity}",
        labels={"spread": "Spread from Average (%)", "region": "Region"},
        color="spread",
        color_continuous_scale="RdBu"
    )
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray"
    )
    
    fig.update_layout(height=400)
    
    return fig
