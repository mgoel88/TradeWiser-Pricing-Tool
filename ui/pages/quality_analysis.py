"""
Quality analysis page for the WIZX application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import io
from PIL import Image

# Import components
from ui.components.header import render_subheader
from ui.components.quality_analyzer_card import render_quality_analysis_card

# Import backend functionality
from database_sql import get_all_commodities
from quality_analyzer import analyze_commodity_quality
from ai_vision import analyze_commodity_image, save_uploaded_image


def render():
    """Render the quality analysis page."""
    render_subheader(
        title="Commodity Quality Analysis",
        description="Analyze commodity quality using AI-powered tools and image recognition",
        icon="magnifying-glass"
    )
    
    # Create tabs for different analysis methods
    tabs = st.tabs(["Parameter-Based Analysis", "Image Analysis"])
    
    # Parameter-based analysis tab
    with tabs[0]:
        render_parameter_analysis()
    
    # Image-based analysis tab
    with tabs[1]:
        render_image_analysis()


def render_parameter_analysis():
    """Render the parameter-based quality analysis section."""
    st.markdown("### Parameter-Based Quality Analysis")
    
    # Create layout with two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Selection inputs
        commodities = get_all_commodities()
        if not commodities:
            st.error("No commodities available. Please initialize the database first.")
            return
        
        selected_commodity = st.selectbox(
            "Select a Commodity",
            commodities,
            key="param_commodity"
        )
        
        # Quality parameters input
        st.markdown("#### Quality Parameters")
        
        # Set default quality parameters based on commodity
        quality_params = {}
        if selected_commodity == "Wheat":
            quality_params = {
                "protein_content": st.slider("Protein Content (%)", 8.0, 16.0, 12.5, 0.1, key="wheat_protein"),
                "moisture_content": st.slider("Moisture Content (%)", 8.0, 15.0, 11.2, 0.1, key="wheat_moisture"),
                "test_weight": st.slider("Test Weight (kg/hl)", 70.0, 85.0, 79.5, 0.1, key="wheat_test_weight"),
                "damaged_kernels": st.slider("Damaged Kernels (%)", 0.0, 5.0, 0.8, 0.1, key="wheat_damaged")
            }
        elif selected_commodity == "Rice":
            quality_params = {
                "broken_percentage": st.slider("Broken Rice (%)", 0.0, 20.0, 7.2, 0.1, key="rice_broken"),
                "moisture_content": st.slider("Moisture Content (%)", 8.0, 15.0, 12.0, 0.1, key="rice_moisture"),
                "foreign_matter": st.slider("Foreign Matter (%)", 0.0, 3.0, 0.5, 0.1, key="rice_foreign"),
                "head_rice_recovery": st.slider("Head Rice Recovery (%)", 60.0, 90.0, 78.0, 0.1, key="rice_head")
            }
        else:
            # Generic parameters for other commodities
            quality_params = {
                "moisture_content": st.slider("Moisture Content (%)", 8.0, 15.0, 12.0, 0.1, key="gen_moisture"),
                "foreign_matter": st.slider("Foreign Matter (%)", 0.0, 3.0, 0.8, 0.1, key="gen_foreign")
            }
        
        # Add analyze button
        analyze_button = st.button(
            "Analyze Quality",
            use_container_width=True,
            type="primary",
            key="param_analyze"
        )
    
    # Results column
    with col2:
        st.markdown("### Analysis Results")
        
        if analyze_button:
            # Call the quality analyzer
            try:
                analysis_result = analyze_commodity_quality(
                    commodity=selected_commodity,
                    quality_params=quality_params
                )
                
                if analysis_result:
                    # Extract overall score and grade
                    quality_score = analysis_result.get("quality_score", 0)
                    quality_grade = analysis_result.get("quality_grade", "C")
                    
                    # Display quality analysis card
                    render_quality_analysis_card(
                        commodity_name=selected_commodity,
                        quality_params=quality_params,
                        quality_score=quality_score,
                        quality_grade=quality_grade
                    )
                    
                    # Display additional analysis if available
                    if "analysis_summary" in analysis_result:
                        st.markdown("#### Analysis Summary")
                        st.markdown(analysis_result["analysis_summary"])
                else:
                    st.error("Failed to analyze quality. Please try again.")
            except Exception as e:
                st.error(f"Error during quality analysis: {str(e)}")
        else:
            # Placeholder content when no analysis is performed
            st.info(
                """
                Adjust the quality parameters and click 'Analyze Quality' to 
                see a detailed analysis of the commodity quality.
                """
            )


def render_image_analysis():
    """Render the image-based quality analysis section."""
    st.markdown("### Image-Based Quality Analysis")
    st.info("Upload an image of your commodity for AI-powered visual quality analysis.")
    
    # Create layout with two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Commodity selection
        commodities = get_all_commodities()
        if not commodities:
            st.error("No commodities available. Please initialize the database first.")
            return
        
        selected_commodity = st.selectbox(
            "Select a Commodity",
            commodities,
            key="img_commodity"
        )
        
        # Analysis type selection
        analysis_types = {
            "general": "General Quality Assessment",
            "detailed": "Detailed Quality Parameters",
            "defects": "Defect Detection",
            "grading": "Quality Grading"
        }
        
        analysis_type = st.selectbox(
            "Analysis Type",
            list(analysis_types.keys()),
            format_func=lambda x: analysis_types[x],
            key="analysis_type"
        )
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload a commodity image",
            type=["jpg", "jpeg", "png"],
            key="commodity_image"
        )
        
        # Example images selector (optional)
        st.markdown("#### Or select an example image")
        example_options = ["None", "Wheat Sample", "Rice Sample", "Cotton Sample"]
        selected_example = st.selectbox("Example Images", example_options, key="example_selector")
        
        # Map selection to file paths
        example_files = {
            "Wheat Sample": "assets/samples/wheat_sample.jpg",
            "Rice Sample": "assets/samples/rice_sample.jpg",
            "Cotton Sample": "assets/samples/cotton_sample.jpg"
        }
        
        # Use the selected example if chosen
        use_example = selected_example != "None"
        example_file = example_files.get(selected_example) if use_example else None
        
        # Add analyze button
        analyze_image_button = st.button(
            "Analyze Image",
            use_container_width=True,
            type="primary",
            key="img_analyze",
            disabled=(not uploaded_file and not example_file)
        )
    
    # Results column
    with col2:
        st.markdown("### Analysis Results")
        
        # Display selected image if available
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Reset the file pointer for later use
            uploaded_file.seek(0)
        elif example_file:
            # Display the example image if selected
            try:
                image = Image.open(example_file)
                st.image(image, caption=f"Example: {selected_example}", use_column_width=True)
            except FileNotFoundError:
                st.error(f"Example file not found: {example_file}")
                example_file = None
        
        # Process the analysis when button is clicked
        if analyze_image_button:
            # Check if we have an image to analyze
            if uploaded_file is not None:
                # Save the uploaded image
                image_path = save_uploaded_image(uploaded_file)
                
                with st.spinner("Analyzing image using AI..."):
                    try:
                        # Analyze the uploaded image
                        analysis_result = analyze_commodity_image(
                            image_data=image_path,
                            commodity=selected_commodity,
                            analysis_type=analysis_type
                        )
                        
                        # Extract quality parameters from analysis results
                        quality_params = analysis_result.get("quality_params", {})
                        quality_score = analysis_result.get("quality_score", 0)
                        quality_grade = analysis_result.get("quality_grade", "C")
                        
                        # Display quality analysis card
                        render_quality_analysis_card(
                            commodity_name=selected_commodity,
                            quality_params=quality_params,
                            quality_score=quality_score,
                            quality_grade=quality_grade
                        )
                        
                        # Display additional analysis
                        st.markdown("#### AI Analysis")
                        st.markdown(analysis_result.get("analysis_summary", "No analysis available."))
                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")
            elif example_file:
                # Analyze the example image
                with st.spinner("Analyzing example image using AI..."):
                    try:
                        # Analyze the example image
                        analysis_result = analyze_commodity_image(
                            image_data=example_file,
                            commodity=selected_commodity,
                            analysis_type=analysis_type
                        )
                        
                        # Extract quality parameters from analysis results
                        quality_params = analysis_result.get("quality_params", {})
                        quality_score = analysis_result.get("quality_score", 0)
                        quality_grade = analysis_result.get("quality_grade", "C")
                        
                        # Display quality analysis card
                        render_quality_analysis_card(
                            commodity_name=selected_commodity,
                            quality_params=quality_params,
                            quality_score=quality_score,
                            quality_grade=quality_grade
                        )
                        
                        # Display additional analysis
                        st.markdown("#### AI Analysis")
                        st.markdown(analysis_result.get("analysis_summary", "No analysis available."))
                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")
            else:
                st.error("Please upload an image or select an example image first.")