"""
Quality analysis page for the WIZX application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
from PIL import Image
import io
import tempfile

# Import components
from ui.components.header import render_subheader
from ui.components.quality_analyzer_card import render_quality_analysis_card

# Import backend functionality
from database_sql import get_all_commodities
from ai_vision import analyze_commodity_image, save_uploaded_image
from quality_analyzer import analyze_quality_from_image
from pricing_engine import calculate_price


def render():
    """Render the quality analysis page."""
    render_subheader(
        title="Agricultural Quality Analysis",
        description="Analyze commodity quality from images using advanced AI",
        icon="microscope"
    )
    
    # Main content
    render_image_analysis()


def render_image_analysis():
    """Render the image analysis section."""
    # Top panel - File upload and selection
    st.markdown("### Upload Commodity Images")
    
    # Get list of commodities
    commodities = get_all_commodities()
    
    # Select commodity for analysis
    selected_commodity = st.selectbox(
        "Select Commodity",
        options=commodities,
        index=0 if commodities else None,
        key="quality_analysis_commodity"
    )
    
    # Analysis Method Selection
    analysis_method = st.radio(
        "Analysis Method",
        options=["AI Vision (OpenAI)", "Computer Vision", "Manual Parameters"],
        index=0,
        horizontal=True,
        key="analysis_method"
    )
    
    # Analysis Type Selection
    analysis_type = st.select_slider(
        "Analysis Detail Level",
        options=["General", "Detailed", "Defects", "Grading"],
        value="Detailed",
        key="analysis_type"
    )
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload commodity images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key="quality_images"
    )
    
    # Use AI checkbox
    use_ai = st.checkbox("Use AI for quality assessment", value=True)
    
    # Analysis button
    if st.button("Analyze Quality", type="primary", use_container_width=True):
        if uploaded_files and selected_commodity:
            all_results = []
            
            with st.spinner("Analyzing images..."):
                for file in uploaded_files:
                    # Save and analyze the image
                    analysis_result = analyze_quality_from_image(
                        file, 
                        selected_commodity, 
                        use_ai=use_ai,
                        analysis_type=analysis_type.lower()
                    )
                    
                    if analysis_result and "quality_params" in analysis_result:
                        quality_params = analysis_result["quality_params"]
                        all_results.append(quality_params)
                        
                        # Save image path for display
                        if "image_path" in analysis_result:
                            image_display = analysis_result["image_path"]
                        else:
                            # Save uploaded file to display
                            image_bytes = file.getvalue()
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                                tmp.write(image_bytes)
                                image_display = tmp.name
                        
                        # Show the analysis result card
                        show_image = True
                        render_quality_analysis_card(
                            image_path=image_display,
                            quality_params=quality_params,
                            analysis_summary=quality_params.get("ai_summary"),
                            quality_score=quality_params.get("quality_score")
                        )
                        
                        # Price Estimation Section
                        st.markdown("### Price Estimation")
                        
                        # Show a select box for regions
                        from database_sql import get_regions
                        regions = get_regions(selected_commodity)
                        
                        selected_region = st.selectbox(
                            "Select Region for Price Calculation",
                            options=regions,
                            index=0 if regions else None,
                            key="price_region"
                        )
                        
                        if st.button("Calculate Price Estimate", use_container_width=True):
                            # Calculate price based on quality parameters
                            price_result = calculate_price(
                                selected_commodity,
                                quality_params,
                                selected_region
                            )
                            
                            if price_result:
                                # Extract values for display
                                base_price = price_result.get("base_price", 0)
                                final_price = price_result.get("final_price", 0)
                                quality_delta = price_result.get("quality_delta", 0)
                                location_delta = price_result.get("location_delta", 0)
                                market_delta = price_result.get("market_delta", 0)
                                
                                # Display price summary
                                st.markdown(
                                    f"""
                                    <div style="background-color: white; border-radius: 10px; padding: 20px; 
                                                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
                                        <h3 style="margin-top: 0; color: #1E88E5;">Price Estimate</h3>
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                            <span style="font-weight: 600;">Base Price:</span>
                                            <span>₹{base_price:.2f}</span>
                                        </div>
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                            <span style="font-weight: 600;">Quality Adjustment:</span>
                                            <span style="color: {'#4CAF50' if quality_delta >= 0 else '#F44336'};">
                                                {'+' if quality_delta >= 0 else ''}₹{quality_delta:.2f}
                                            </span>
                                        </div>
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                            <span style="font-weight: 600;">Location Adjustment:</span>
                                            <span style="color: {'#4CAF50' if location_delta >= 0 else '#F44336'};">
                                                {'+' if location_delta >= 0 else ''}₹{location_delta:.2f}
                                            </span>
                                        </div>
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                            <span style="font-weight: 600;">Market Adjustment:</span>
                                            <span style="color: {'#4CAF50' if market_delta >= 0 else '#F44336'};">
                                                {'+' if market_delta >= 0 else ''}₹{market_delta:.2f}
                                            </span>
                                        </div>
                                        <hr style="margin: 10px 0;">
                                        <div style="display: flex; justify-content: space-between; font-size: 1.2rem; font-weight: 600;">
                                            <span>Final Price:</span>
                                            <span>₹{final_price:.2f}</span>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            else:
                                st.error("Error calculating price estimate.")
                    else:
                        st.error("Error analyzing image. Please try again.")
            
            # If multiple images, show aggregate analysis
            if len(all_results) > 1:
                st.markdown("### Aggregate Analysis")
                
                # Calculate average values for parameters
                avg_params = {}
                for result in all_results:
                    for k, v in result.items():
                        if isinstance(v, (int, float)) and not k.startswith("_"):
                            avg_params[k] = avg_params.get(k, 0) + v / len(all_results)
                
                # Display aggregate quality parameters
                st.markdown("#### Average Quality Parameters")
                
                # Convert to DataFrame for display
                avg_df = pd.DataFrame({
                    "Parameter": list(avg_params.keys()),
                    "Average Value": list(avg_params.values())
                })
                
                # Format values
                avg_df["Average Value"] = avg_df["Average Value"].apply(
                    lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x)
                )
                
                st.dataframe(avg_df, use_container_width=True, hide_index=True)
                
                # Calculate aggregate price if a region is selected
                regions = get_regions(selected_commodity)
                if regions:
                    selected_region = st.selectbox(
                        "Select Region for Aggregate Price",
                        options=regions,
                        index=0 if regions else None,
                        key="agg_price_region"
                    )
                    
                    if st.button("Calculate Aggregate Price", use_container_width=True):
                        # Calculate price based on average quality parameters
                        price_result = calculate_price(
                            selected_commodity,
                            avg_params,
                            selected_region
                        )
                        
                        if price_result:
                            st.markdown(
                                f"""
                                <div style="background-color: #f8f9fa; border-radius: 10px; padding: 15px; margin-top: 10px;">
                                    <h4 style="margin-top: 0;">Aggregate Price Estimate</h4>
                                    <div style="font-size: 1.5rem; font-weight: 600; color: #1E88E5; margin: 10px 0;">
                                        ₹{price_result.get('final_price', 0):.2f}
                                    </div>
                                    <div style="font-size: 0.9rem; color: #777;">
                                        Base: ₹{price_result.get('base_price', 0):.2f} | 
                                        Quality Adj: {price_result.get('quality_delta', 0):.2f} | 
                                        Location Adj: {price_result.get('location_delta', 0):.2f}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
        else:
            # Show error if no files are uploaded or no commodity selected
            if not uploaded_files:
                st.error("Please upload at least one image file.")
            if not selected_commodity:
                st.error("Please select a commodity for analysis.")
    else:
        # Show empty state
        st.markdown(
            """
            <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; 
                        text-align: center; height: 200px; display: flex; flex-direction: column; 
                        justify-content: center; align-items: center; margin-top: 20px;">
                <i class="fas fa-microscope" style="font-size: 3rem; color: #1E88E5; margin-bottom: 15px;"></i>
                <h3 style="margin-bottom: 10px;">Upload Images for Analysis</h3>
                <p style="color: #777;">Upload images of commodity samples to analyze quality parameters using AI.</p>
            </div>
            """,
            unsafe_allow_html=True
        )