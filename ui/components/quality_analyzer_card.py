"""
Quality analyzer card component for WIZX agricultural platform.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_quality_analysis_card(commodity_name, quality_params, quality_score=None, quality_grade=None):
    """
    Render a quality analysis card.
    
    Args:
        commodity_name (str): Name of the commodity
        quality_params (dict): Dictionary of quality parameters and values
        quality_score (float, optional): Overall quality score (0-100)
        quality_grade (str, optional): Quality grade (e.g., "A", "B", "C")
    """
    # Set defaults
    if quality_score is None:
        # Calculate a simple average if not provided
        if quality_params:
            quality_score = sum(quality_params.values()) / len(quality_params)
        else:
            quality_score = 0
    
    if quality_grade is None:
        # Determine grade based on score
        if quality_score >= 90:
            quality_grade = "A+"
        elif quality_score >= 80:
            quality_grade = "A"
        elif quality_score >= 70:
            quality_grade = "B+"
        elif quality_score >= 60:
            quality_grade = "B"
        elif quality_score >= 50:
            quality_grade = "C"
        else:
            quality_grade = "D"
    
    # Create a container
    with st.container():
        # Background and styling
        st.markdown(
            """
            <style>
            .quality-card {
                background-color: white;
                padding: 1rem;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .quality-meter {
                display: flex;
                height: 20px;
                background-color: #f0f0f0;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Header
        st.markdown(
            f"""
            <div class="quality-card">
                <h3 style="margin-top: 0; color: #1E88E5;">{commodity_name} Quality Analysis</h3>
            """,
            unsafe_allow_html=True
        )
        
        # Quality score display
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Convert score to 0-100 scale
            normalized_score = min(100, max(0, quality_score))
            
            # Determine color based on score
            if normalized_score >= 80:
                color = "#4CAF50"  # Green
            elif normalized_score >= 60:
                color = "#FFC107"  # Yellow/Amber
            else:
                color = "#F44336"  # Red
            
            # Display quality meter
            st.markdown(
                f"""
                <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Quality Score: <strong>{normalized_score:.1f}/100</strong>
                </div>
                <div class="quality-meter">
                    <div style="width: {normalized_score}%; background-color: {color}"></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col2:
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="color: #666; font-size: 0.8rem;">Grade</div>
                    <div style="font-size: 2rem; font-weight: bold; color: {color};">
                        {quality_grade}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Quality parameters table
        if quality_params:
            st.markdown("#### Quality Parameters")
            
            # Create a DataFrame for the parameters
            params_df = pd.DataFrame({
                "Parameter": [p.replace("_", " ").title() for p in quality_params.keys()],
                "Value": list(quality_params.values())
            })
            
            # Display the DataFrame
            st.dataframe(params_df, use_container_width=True, hide_index=True)
            
            # Create radar chart for quality parameters
            categories = list(params_df["Parameter"])
            values = list(params_df["Value"])
            
            # Normalize values to 0-1 scale for the radar chart
            max_val = max(values) if values else 1
            normalized_values = [v / max_val for v in values]
            
            # Add the first value again to close the loop
            categories.append(categories[0])
            normalized_values.append(normalized_values[0])
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=categories,
                fill="toself",
                name=commodity_name,
                line_color="#1E88E5",
                fillcolor="rgba(30, 136, 229, 0.2)"
            ))
            
            # Update layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                height=300,
                margin=dict(t=10, b=10, l=40, r=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No quality parameters available for analysis.")
        
        # Close the container div
        st.markdown("</div>", unsafe_allow_html=True)