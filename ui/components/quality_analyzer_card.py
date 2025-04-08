"""
Quality analyzer card component for displaying image analysis results.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

def render_quality_analysis_card(image_path, quality_params, analysis_summary=None, quality_score=None):
    """
    Render a card showing commodity image and quality analysis results.
    
    Args:
        image_path: Path to the analyzed image
        quality_params: Dictionary of quality parameters and values
        analysis_summary: Optional text summary of the analysis
        quality_score: Optional overall quality score (0-100)
    """
    st.markdown(
        """
        <div style="background-color: white; border-radius: 10px; padding: 20px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="margin-top: 0; margin-bottom: 15px; color: #1E88E5;">Quality Analysis Results</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        # Display the image
        try:
            img = Image.open(image_path)
            st.image(img, use_column_width=True, caption="Analyzed Sample")
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
    
    with col2:
        # If there's a summary, display it first
        if analysis_summary:
            st.markdown(
                f"""
                <div style="background-color: #f8f9fa; border-left: 4px solid #1E88E5; 
                            padding: 10px 15px; margin-bottom: 15px; border-radius: 0 5px 5px 0;">
                    <span style="font-weight: 600;">AI Analysis:</span> {analysis_summary}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Create a DataFrame for quality parameters
        if quality_params:
            # Filter out special keys
            display_params = {k: v for k, v in quality_params.items() 
                             if k not in ['quality_score', 'quality_grade', 'ai_summary', 
                                          'confidence', 'timestamp', 'analysis_method']}
            
            if display_params:
                # Convert to DataFrame
                df = pd.DataFrame(list(display_params.items()), 
                                 columns=['Parameter', 'Value'])
                
                # Format values
                df['Value'] = df['Value'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else str(x))
                
                # Display as a table
                st.markdown("### Quality Parameters")
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Display quality grade if present
        if 'quality_grade' in quality_params:
            grade = quality_params['quality_grade']
            grade_colors = {
                'A': '#4CAF50',  # Green
                'B': '#8BC34A',  # Light Green
                'C': '#FFEB3B',  # Yellow
                'D': '#FF9800',  # Orange
                'E': '#F44336',  # Red
                'Premium': '#4CAF50',
                'Standard': '#8BC34A',
                'Average': '#FFEB3B',
                'Below Average': '#FF9800',
                'Poor': '#F44336'
            }
            
            color = grade_colors.get(grade, '#1E88E5')  # Default blue if grade not in mapping
            
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 20px; margin-bottom: 20px;">
                    <div style="font-size: 0.9rem; color: #777; margin-bottom: 5px;">Quality Grade</div>
                    <div style="font-size: 1.8rem; font-weight: 600; 
                                color: {color}; margin-bottom: 5px;">{grade}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # If quality score is provided, show it with a gauge
    if quality_score is not None or 'quality_score' in quality_params:
        score = quality_score if quality_score is not None else quality_params.get('quality_score', 0)
        
        st.markdown("### Overall Quality Score")
        
        # Create the gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Quality Score", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#1E88E5"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#FF9E80'},
                    {'range': [40, 70], 'color': '#FFECB3'},
                    {'range': [70, 100], 'color': '#C8E6C9'}
                ],
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display confidence if available
        if 'confidence' in quality_params:
            confidence = quality_params['confidence']
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <span style="font-size: 0.9rem; color: #777;">
                        Analysis Confidence: <b>{confidence:.0%}</b>
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )