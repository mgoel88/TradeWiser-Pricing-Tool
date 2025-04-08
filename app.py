"""
WIZX Agricultural Commodity Pricing Platform

A comprehensive platform for agricultural commodity pricing,
quality analysis, and market insights.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import logging
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core functionality
from database_sql import (
    initialize_database,
    get_all_commodities,
    get_commodity_prices,
    get_regions_for_commodity,
    get_price_history
)
from wizx_index import (
    calculate_composite_index,
    calculate_all_indices
)
from quality_analyzer import analyze_commodity_quality
from pricing_engine import calculate_price

# Import UI components
from ui.components.header import render_header, render_subheader
from ui.components.sidebar import render_complete_sidebar
from ui.components.stats_card import render_stats_grid
from ui.components.price_card import render_price_card
from ui.components.quality_analyzer_card import render_quality_analysis_card

# Import pages
from ui.pages.dashboard import render as render_dashboard
from ui.pages.price_calculator import render as render_price_calculator
from ui.pages.quality_analysis import render as render_quality_analysis

# Custom CSS
def load_css():
    """Load custom CSS styles."""
    st.markdown(
        """
        <style>
        /* General styling */
        .main {
            background-color: #F5F7F9;
        }
        
        /* Hide default Streamlit developer menu */
        #MainMenu, footer {
            visibility: hidden;
        }
        
        /* Custom card styling */
        .card {
            background-color: white;
            border-radius: 5px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Custom header styling */
        h1, h2, h3 {
            color: #1E88E5;
        }
        
        /* Custom button styling */
        .stButton > button {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def create_logo():
    """Create a simple logo if it doesn't exist."""
    # Create directory if it doesn't exist
    os.makedirs("assets/img", exist_ok=True)
    
    # Check if logo exists
    logo_path = "assets/img/logo.png"
    if os.path.exists(logo_path):
        return logo_path
    
    # Create a simple logo using Pillow
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create base image (200x200 pixels with white background)
        img = Image.new('RGBA', (200, 200), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a blue circle
        draw.ellipse((10, 10, 190, 190), fill=(30, 136, 229, 255))
        
        # Draw a green leaf-like shape
        draw.pieslice([(40, 40), (160, 160)], 45, 135, fill=(76, 175, 80, 255))
        
        # Add text
        draw.text((70, 120), "WIZX", fill=(255, 255, 255, 255))
        
        # Save the image
        img.save(logo_path)
        
        logger.info(f"Created logo at {logo_path}")
        return logo_path
    except Exception as e:
        logger.error(f"Error creating logo: {e}")
        return None


def initialize_app():
    """Initialize the application."""
    # Check for database
    try:
        # Try getting commodities to check database connection
        get_all_commodities()
        return True
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        
        # Show initialization screen
        st.title("WIZX Platform Initialization")
        
        st.warning("Database not initialized. Please click the button below to initialize the platform.")
        
        if st.button("Initialize Platform", use_container_width=True):
            with st.spinner("Initializing database..."):
                initialize_database()
                st.success("Database initialized successfully!")
                time.sleep(2)
                st.rerun()
        
        return False


def main():
    """Main application entry point."""
    # Set page config
    st.set_page_config(
        page_title="WIZX Agricultural Commodity Platform",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Ensure logo exists
    create_logo()
    
    # Initialize the application if needed
    if not initialize_app():
        return
    
    # Render header
    render_header()
    
    # Render sidebar and get navigation state
    nav = render_complete_sidebar()
    
    # Route to appropriate page
    if nav["page"] == "Dashboard":
        render_dashboard()
    elif nav["page"] == "Price Calculator":
        render_price_calculator()
    elif nav["page"] == "Quality Analysis":
        render_quality_analysis()
    else:
        # Default to dashboard
        render_dashboard()


if __name__ == "__main__":
    main()