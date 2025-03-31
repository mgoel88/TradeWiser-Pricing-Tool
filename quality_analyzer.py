"""
Quality analyzer module for analyzing quality from images, lab reports, and other data.
"""

import os
import io
import logging
import json
import base64
import numpy as np
import pandas as pd
from PIL import Image
import re

from models import assess_quality_parameters
from database import get_commodity_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_quality_from_image(image_file, commodity):
    """
    Analyze quality parameters from an image.
    
    Args:
        image_file: Image file object
        commodity (str): The commodity
        
    Returns:
        dict: Extracted quality parameters
    """
    logger.info(f"Analyzing quality from image for {commodity}")
    
    try:
        # Read image data
        image_data = image_file.read()
        
        # In a real implementation, this would use computer vision models
        # to analyze the image and extract quality parameters
        # For demonstration, we'll use the models module
        
        # Reset file pointer for potential future use
        image_file.seek(0)
        
        # Get quality parameters
        quality_params = assess_quality_parameters(commodity, image_data=image_data)
        
        # Get commodity data for validation
        commodity_data = get_commodity_data(commodity)
        
        if not commodity_data or 'quality_parameters' not in commodity_data:
            logger.warning(f"No quality parameters found for commodity: {commodity}")
            return {}
        
        # Validate parameters against known ranges
        validated_params = {}
        
        for param, value in quality_params.items():
            if param in commodity_data['quality_parameters']:
                param_data = commodity_data['quality_parameters'][param]
                min_val = param_data.get('min', 0)
                max_val = param_data.get('max', 100)
                
                # Ensure value is within range
                value = max(min_val, min(max_val, value))
                
                validated_params[param] = value
        
        return validated_params
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        return {}

def analyze_report(report_file):
    """
    Analyze a lab report to extract commodity and quality parameters.
    
    Args:
        report_file: Lab report file object
        
    Returns:
        tuple: (commodity, quality_parameters)
    """
    logger.info("Analyzing lab report")
    
    try:
        # In a real implementation, this would use OCR and text analysis
        # to extract information from the lab report
        # For demonstration, we'll generate plausible data
        
        # Try to guess the commodity from the filename
        filename = report_file.name if hasattr(report_file, 'name') else "unknown"
        
        # Simple keyword matching
        commodity_keywords = {
            "wheat": "Wheat",
            "rice": "Rice",
            "dal": "Tur Dal",
            "tur": "Tur Dal",
            "soya": "Soyabean",
            "mustard": "Mustard"
        }
        
        detected_commodity = None
        
        for keyword, commodity in commodity_keywords.items():
            if keyword.lower() in filename.lower():
                detected_commodity = commodity
                break
        
        # If no commodity detected, use a default
        if not detected_commodity:
            detected_commodity = "Wheat"  # Default
        
        # Get quality parameters for this commodity
        quality_params = assess_quality_parameters(detected_commodity)
        
        return (detected_commodity, quality_params)
    except Exception as e:
        logger.error(f"Error analyzing report: {e}")
        return (None, {})

def analyze_multiple_samples(image_files, commodity):
    """
    Analyze multiple samples of the same commodity and calculate average quality.
    
    Args:
        image_files (list): List of image file objects
        commodity (str): The commodity
        
    Returns:
        dict: Average quality parameters and analysis details
    """
    logger.info(f"Analyzing multiple samples for {commodity}")
    
    if not image_files:
        logger.warning("No image files provided")
        return {
            'status': 'error',
            'message': 'No image files provided'
        }
    
    all_params = []
    
    for i, image_file in enumerate(image_files):
        logger.info(f"Analyzing sample {i+1}")
        
        # Analyze each sample
        sample_params = analyze_quality_from_image(image_file, commodity)
        
        if sample_params:
            all_params.append(sample_params)
    
    if not all_params:
        logger.warning("No valid quality parameters extracted")
        return {
            'status': 'error',
            'message': 'No valid quality parameters extracted'
        }
    
    # Calculate average parameters
    avg_params = {}
    
    # Get all parameter names
    param_names = set()
    for params in all_params:
        param_names.update(params.keys())
    
    # Calculate average for each parameter
    for param in param_names:
        values = [params.get(param, 0) for params in all_params if param in params]
        if values:
            avg_params[param] = sum(values) / len(values)
    
    # Calculate standard deviation for confidence
    std_devs = {}
    
    for param in avg_params:
        values = [params.get(param, 0) for params in all_params if param in params]
        if len(values) > 1:
            std_devs[param] = np.std(values)
        else:
            std_devs[param] = 0
    
    # Calculate coefficient of variation as a measure of reliability
    cv = {}
    
    for param, avg in avg_params.items():
        if avg > 0 and param in std_devs:
            cv[param] = (std_devs[param] / avg) * 100
        else:
            cv[param] = 0
    
    # Determine overall reliability
    if all_params:
        reliability = {
            'samples_analyzed': len(all_params),
            'parameter_cv': cv,
            'overall_cv': sum(cv.values()) / len(cv) if cv else 0
        }
        
        if reliability['overall_cv'] < 10:
            reliability['rating'] = 'high'
        elif reliability['overall_cv'] < 20:
            reliability['rating'] = 'medium'
        else:
            reliability['rating'] = 'low'
    else:
        reliability = {
            'samples_analyzed': 0,
            'rating': 'unknown'
        }
    
    return {
        'status': 'success',
        'average_quality': avg_params,
        'standard_deviations': std_devs,
        'coefficient_of_variation': cv,
        'reliability': reliability,
        'samples_analyzed': len(all_params)
    }

def compare_to_standard_grade(quality_params, commodity):
    """
    Compare quality parameters to standard grade.
    
    Args:
        quality_params (dict): Quality parameters
        commodity (str): The commodity
        
    Returns:
        dict: Comparison results
    """
    logger.info(f"Comparing quality to standard grade for {commodity}")
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data or 'quality_parameters' not in commodity_data:
        logger.warning(f"No quality parameters found for commodity: {commodity}")
        return {
            'status': 'error',
            'message': 'No quality parameters found'
        }
    
    # Get standard quality parameters
    standard_params = {
        param: details.get('standard_value', (details.get('min', 0) + details.get('max', 100)) / 2)
        for param, details in commodity_data['quality_parameters'].items()
    }
    
    # Compare each parameter
    comparison = {}
    
    for param, std_value in standard_params.items():
        if param in quality_params:
            actual_value = quality_params[param]
            
            # Calculate deviation
            deviation = actual_value - std_value
            
            # Calculate percentage deviation
            if std_value != 0:
                percentage = (deviation / std_value) * 100
            else:
                percentage = 0
            
            # Get parameter details
            param_data = commodity_data['quality_parameters'].get(param, {})
            unit = param_data.get('unit', '')
            
            # Determine impact
            impact_type = param_data.get('impact_type', 'linear')
            impact_data = commodity_data.get('quality_impact', {}).get(param, {})
            
            if impact_type == 'linear':
                impact_factor = impact_data.get('factor', 0)
                price_impact = deviation * impact_factor
            elif impact_type == 'threshold':
                if deviation > 0:
                    impact_factor = impact_data.get('premium_factor', 0)
                    price_impact = deviation * impact_factor
                else:
                    impact_factor = impact_data.get('discount_factor', 0)
                    price_impact = deviation * impact_factor
            else:
                impact_factor = 0
                price_impact = 0
            
            # Determine if better or worse
            if impact_factor > 0:
                # Positive factor means higher is better
                is_better = deviation > 0
            else:
                # Negative factor means lower is better
                is_better = deviation < 0
            
            comparison[param] = {
                'standard': std_value,
                'actual': actual_value,
                'deviation': deviation,
                'percentage': percentage,
                'unit': unit,
                'is_better': is_better,
                'price_impact': price_impact
            }
    
    # Calculate overall quality score
    # Simple approach: normalize deviations and calculate weighted average
    quality_score = 0
    total_weight = 0
    
    for param, details in comparison.items():
        impact_data = commodity_data.get('quality_impact', {}).get(param, {})
        weight = abs(impact_data.get('factor', 0))
        
        if weight == 0:
            continue  # Skip parameters with no impact
        
        # Convert deviation to a score between 0 and 100
        # 0 = worst possible value, 100 = best possible value
        param_data = commodity_data['quality_parameters'].get(param, {})
        min_val = param_data.get('min', 0)
        max_val = param_data.get('max', 100)
        std_val = param_data.get('standard_value', (min_val + max_val) / 2)
        actual_val = details['actual']
        
        # Calculate parameter score
        if weight > 0:
            # Higher is better
            if actual_val >= std_val:
                # Above standard, scale from 50 to 100
                param_score = 50 + 50 * min(actual_val - std_val, max_val - std_val) / (max_val - std_val)
            else:
                # Below standard, scale from 0 to 50
                param_score = 50 * max(actual_val - min_val, 0) / (std_val - min_val)
        else:
            # Lower is better
            if actual_val <= std_val:
                # Below standard, scale from 50 to 100
                param_score = 50 + 50 * min(std_val - actual_val, std_val - min_val) / (std_val - min_val)
            else:
                # Above standard, scale from 0 to 50
                param_score = 50 * max(max_val - actual_val, 0) / (max_val - std_val)
        
        # Add to weighted average
        quality_score += param_score * abs(weight)
        total_weight += abs(weight)
    
    if total_weight > 0:
        quality_score /= total_weight
    else:
        quality_score = 50  # Neutral score if no weights
    
    # Determine quality grade
    if quality_score >= 90:
        quality_grade = "Excellent"
    elif quality_score >= 75:
        quality_grade = "Good"
    elif quality_score >= 50:
        quality_grade = "Standard"
    elif quality_score >= 25:
        quality_grade = "Below Standard"
    else:
        quality_grade = "Poor"
    
    return {
        'status': 'success',
        'comparison': comparison,
        'quality_score': quality_score,
        'quality_grade': quality_grade,
        'total_price_impact': sum(details['price_impact'] for details in comparison.values())
    }

def generate_quality_certificate(quality_params, commodity, analysis_result):
    """
    Generate a quality certificate based on analysis results.
    
    Args:
        quality_params (dict): Quality parameters
        commodity (str): The commodity
        analysis_result (dict): Analysis results
        
    Returns:
        dict: Certificate data
    """
    logger.info(f"Generating quality certificate for {commodity}")
    
    # Get current date
    current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data:
        logger.warning(f"No data found for commodity: {commodity}")
        return {
            'status': 'error',
            'message': 'Commodity data not found'
        }
    
    # Create certificate data
    certificate = {
        'certificate_id': f"QC-{commodity.replace(' ', '')}-{current_date.replace('-', '')}",
        'issue_date': current_date,
        'commodity': commodity,
        'quality_parameters': quality_params,
        'analysis_result': analysis_result,
        'certificate_issuer': 'AgriPrice Engine',
        'validity_period': '30 days'
    }
    
    return {
        'status': 'success',
        'certificate': certificate
    }

if __name__ == "__main__":
    # Test the quality analyzer
    print("Testing quality analyzer module...")
