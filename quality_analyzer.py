"""
Quality analyzer module for analyzing quality from images, lab reports, and other data.
Uses advanced AI models for image analysis.
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
import time

from models import assess_quality_parameters
from database import get_commodity_data

# Import AI vision module
try:
    from ai_vision import (
        analyze_commodity_image, 
        analyze_video_frame, 
        extract_quality_parameters,
        save_uploaded_image
    )
    AI_VISION_AVAILABLE = True
except ImportError:
    logger.warning("AI Vision module not available. Using fallback methods.")
    AI_VISION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directory for storing analyzed images
UPLOADS_DIR = "uploads/images"
os.makedirs(UPLOADS_DIR, exist_ok=True)

def analyze_quality_from_image(image_file, commodity, use_ai=True, analysis_type="detailed"):
    """
    Analyze quality parameters from an image.
    
    Args:
        image_file: Image file object
        commodity (str): The commodity
        use_ai (bool): Whether to use AI vision for analysis
        analysis_type (str): Type of analysis to perform ("general", "detailed", "defects", "grading")
        
    Returns:
        dict: Extracted quality parameters
    """
    logger.info(f"Analyzing quality from image for {commodity} using {'AI' if use_ai else 'traditional'} method")
    
    try:
        # Save uploaded image
        save_path = save_uploaded_image(image_file, UPLOADS_DIR)
        
        # Reset file pointer for potential future use
        image_file.seek(0)
        
        # Get commodity data for parameter ranges and validation
        commodity_data = get_commodity_data(commodity)
        if not commodity_data or 'quality_parameters' not in commodity_data:
            logger.warning(f"No quality parameters found for commodity: {commodity}")
            return {}
        
        # Use AI vision if available and requested
        if use_ai and AI_VISION_AVAILABLE:
            try:
                # Analyze using OpenAI GPT-4V
                logger.info(f"Performing {analysis_type} AI analysis for {commodity}")
                result = analyze_commodity_image(save_path, commodity, analysis_type)
                
                if result["status"] == "success":
                    # Extract structured quality parameters from AI analysis
                    quality_params = extract_quality_parameters(result, commodity)
                    
                    # Add AI summary and confidence score
                    if "analysis" in result:
                        # Extract a summary from the analysis for display
                        analysis_text = result["analysis"]
                        quality_params["ai_summary"] = analysis_text[:500] if len(analysis_text) > 500 else analysis_text
                        
                        # Add confidence level based on certainty in the response
                        certainty_phrases = ["I'm confident", "clearly", "definitely", "certainly", "evident"]
                        uncertainty_phrases = ["may be", "might be", "appears to", "possibly", "uncertain", "unclear", "hard to tell"]
                        
                        confidence_score = 0.7  # Default medium-high confidence
                        
                        # Adjust confidence based on language used in analysis
                        for phrase in certainty_phrases:
                            if phrase.lower() in analysis_text.lower():
                                confidence_score += 0.05
                        
                        for phrase in uncertainty_phrases:
                            if phrase.lower() in analysis_text.lower():
                                confidence_score -= 0.1
                        
                        # Cap confidence between 0.1 and 0.95
                        confidence_score = max(0.1, min(0.95, confidence_score))
                        quality_params["confidence"] = confidence_score
                    
                    # Determine quality grade
                    if "quality_score" in quality_params:
                        score = quality_params["quality_score"]
                    else:
                        # Calculate a quality score based on parameters
                        score = calculate_quality_score(quality_params, commodity_data)
                        quality_params["quality_score"] = score
                    
                    # Add quality grade based on score
                    if score >= 90:
                        quality_params["quality_grade"] = "Excellent"
                    elif score >= 75:
                        quality_params["quality_grade"] = "Good"
                    elif score >= 50:
                        quality_params["quality_grade"] = "Average"
                    elif score >= 25:
                        quality_params["quality_grade"] = "Below Average"
                    else:
                        quality_params["quality_grade"] = "Poor"
                    
                    # Add timestamp and path
                    quality_params["timestamp"] = int(time.time())
                    quality_params["image_path"] = save_path
                    quality_params["analysis_type"] = analysis_type
                    
                    logger.info(f"AI analysis completed successfully with confidence {quality_params.get('confidence', 'unknown')}")
                    return quality_params
                else:
                    logger.warning(f"AI analysis failed: {result.get('message', 'Unknown error')}")
                    # Fall back to traditional method
            except Exception as ai_error:
                logger.error(f"Error in AI analysis: {ai_error}")
                logger.warning("Falling back to traditional method")
        
        # Fallback to traditional method if AI not available or failed
        logger.info(f"Using traditional method for analyzing {commodity}")
        
        # Read image data
        image_file.seek(0)
        image_data = image_file.read()
        
        # Reset file pointer again
        image_file.seek(0)
        
        # Get quality parameters using traditional method
        quality_params = assess_quality_parameters(commodity, image_data=image_data, use_ai=False)
        
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
            elif isinstance(value, (int, float)):
                # Include numeric parameters even if not in commodity data
                validated_params[param] = value
        
        # Add metadata to indicate traditional method
        validated_params["timestamp"] = int(time.time())
        validated_params["image_path"] = save_path
        validated_params["analysis_method"] = "traditional"
        validated_params["confidence"] = 0.5  # Medium confidence for traditional method
        
        # Calculate and add quality score
        score = calculate_quality_score(validated_params, commodity_data)
        validated_params["quality_score"] = score
        
        # Add quality grade based on score
        if score >= 90:
            validated_params["quality_grade"] = "Excellent"
        elif score >= 75:
            validated_params["quality_grade"] = "Good"
        elif score >= 50:
            validated_params["quality_grade"] = "Average"
        elif score >= 25:
            validated_params["quality_grade"] = "Below Average"
        else:
            validated_params["quality_grade"] = "Poor"
        
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

def calculate_quality_score(quality_params, commodity_data):
    """
    Calculate a quality score based on quality parameters and commodity data.
    Implements an enhanced algorithm for accurate quality assessment.
    
    Args:
        quality_params (dict): Quality parameters
        commodity_data (dict): Commodity reference data
        
    Returns:
        float: Quality score from 0-100
    """
    logger.info(f"Calculating quality score with {len(quality_params)} parameters")
    
    if not quality_params or not commodity_data or 'quality_parameters' not in commodity_data:
        logger.warning("Missing quality parameters or commodity data")
        return 50  # Default neutral score
    
    # If quality_score already exists and is within range, prioritize it
    if "quality_score" in quality_params and isinstance(quality_params["quality_score"], (int, float)):
        direct_score = quality_params["quality_score"]
        if 0 <= direct_score <= 100:
            logger.info(f"Using provided quality score: {direct_score}")
            return direct_score
    
    # Calculate overall quality score (0-100)
    quality_score = 0
    total_weight = 0
    param_scores = {}  # Store individual parameter scores for logging/debugging
    
    # Get impact factors for each parameter
    impact_data = commodity_data.get('quality_impact', {})
    
    # Quick check for quality_grade if available - use as a baseline
    base_score = 50  # Default neutral
    if "quality_grade" in quality_params:
        grade = quality_params["quality_grade"].lower() if isinstance(quality_params["quality_grade"], str) else ""
        grade_scores = {
            "excellent": 90,
            "premium": 90,
            "good": 75,
            "average": 50,
            "fair": 50,
            "below average": 35,
            "poor": 20,
            "sub-standard": 20
        }
        for grade_name, score in grade_scores.items():
            if grade_name in grade:
                base_score = score
                logger.info(f"Using quality grade '{quality_params['quality_grade']}' as baseline: {base_score}")
                break
    
    # Process each parameter
    for param, value in quality_params.items():
        # Skip non-numeric values and metadata fields
        if not isinstance(value, (int, float)) or param.startswith('ai_') or param in [
            'timestamp', 'image_path', 'analysis_method', 'analysis_type', 'confidence',
            'quality_grade', 'quality_score', 'computed_quality_score', 'error'
        ]:
            continue
            
        # Get parameter details and impact factor
        param_data = commodity_data['quality_parameters'].get(param, {})
        if not param_data:
            # Fall back to default parameters if not in reference data
            param_data = {
                'min': 0,
                'max': 100,
                'standard_value': 50,
                'unit': '',
                'impact_type': 'linear'
            }
            
        # Determine impact factor and type
        impact_info = impact_data.get(param, {})
        impact_factor = abs(impact_info.get('factor', 1.0))  # Default to weight of 1.0
        impact_type = impact_info.get('impact_type', param_data.get('impact_type', 'linear'))
        
        if impact_factor <= 0:
            impact_factor = 1.0  # Ensure positive weight
        
        # Get parameter range
        min_val = param_data.get('min', 0)
        max_val = param_data.get('max', 100)
        std_val = param_data.get('standard_value', (min_val + max_val) / 2)
        
        # Ensure value is within valid range
        value = max(min_val, min(max_val, value))
        
        # Calculate parameter score based on impact type and direction
        if impact_type == 'threshold':
            # Threshold-based scoring (step function)
            thresholds = impact_info.get('thresholds', [])
            if thresholds:
                # Sort thresholds by value
                thresholds.sort(key=lambda x: x.get('value', 0))
                
                # Find applicable threshold
                param_score = 50  # Default
                for threshold in thresholds:
                    threshold_value = threshold.get('value', 0)
                    threshold_score = threshold.get('score', 50)
                    
                    if value >= threshold_value:
                        param_score = threshold_score
                    else:
                        break
            else:
                # Fallback to linear if no thresholds defined
                param_score = calculate_linear_score(value, min_val, max_val, std_val, 
                                                    impact_info.get('factor', 0) >= 0)
        
        elif impact_type == 'exponential':
            # Exponential scoring - more dramatic impact as values deviate from standard
            exponential_factor = impact_info.get('exponential_factor', 2.0)
            is_higher_better = impact_info.get('factor', 0) >= 0
            
            if is_higher_better:
                # Higher is better
                if value >= std_val:
                    # Above standard
                    normalized = min((value - std_val) / (max_val - std_val) if max_val > std_val else 0, 1)
                    param_score = 50 + 50 * (normalized ** (1/exponential_factor))
                else:
                    # Below standard
                    normalized = min((std_val - value) / (std_val - min_val) if std_val > min_val else 0, 1)
                    param_score = 50 - 50 * (normalized ** exponential_factor)
            else:
                # Lower is better
                if value <= std_val:
                    # Below standard
                    normalized = min((std_val - value) / (std_val - min_val) if std_val > min_val else 0, 1)
                    param_score = 50 + 50 * (normalized ** (1/exponential_factor))
                else:
                    # Above standard
                    normalized = min((value - std_val) / (max_val - std_val) if max_val > std_val else 0, 1)
                    param_score = 50 - 50 * (normalized ** exponential_factor)
        
        else:
            # Default to linear scoring
            param_score = calculate_linear_score(value, min_val, max_val, std_val, 
                                               impact_info.get('factor', 0) >= 0)
        
        # Ensure score is within bounds
        param_score = max(0, min(100, param_score))
        
        # Store individual score for logging
        param_scores[param] = param_score
        
        # Add to weighted average
        quality_score += param_score * impact_factor
        total_weight += impact_factor
    
    # Calculate final weighted score
    final_score = 0
    
    if total_weight > 0:
        # If we have calculated parameters, weight them at 70%
        calculated_score = quality_score / total_weight
        final_score = 0.7 * calculated_score + 0.3 * base_score
    else:
        # Use baseline from quality_grade if no parameters processed
        final_score = base_score
    
    # Ensure final score is within range
    final_score = max(0, min(100, final_score))
    
    logger.info(f"Calculated quality score: {final_score:.2f}")
    logger.debug(f"Parameter scores: {param_scores}")
    
    return final_score

def calculate_linear_score(value, min_val, max_val, std_val, higher_better):
    """
    Helper function to calculate linear score for a parameter.
    
    Args:
        value (float): Parameter value
        min_val (float): Minimum allowed value
        max_val (float): Maximum allowed value
        std_val (float): Standard/reference value
        higher_better (bool): Whether higher values are better
        
    Returns:
        float: Score from 0-100
    """
    if higher_better:
        # Higher is better
        if value >= std_val:
            # Above standard, scale from 50 to 100
            range_factor = max_val - std_val
            if range_factor <= 0:
                return 75  # Default if range is invalid
            else:
                return 50 + 50 * min(value - std_val, max_val - std_val) / range_factor
        else:
            # Below standard, scale from 0 to 50
            range_factor = std_val - min_val
            if range_factor <= 0:
                return 25  # Default if range is invalid
            else:
                return 50 * max(value - min_val, 0) / range_factor
    else:
        # Lower is better
        if value <= std_val:
            # Below standard, scale from 50 to 100
            range_factor = std_val - min_val
            if range_factor <= 0:
                return 75  # Default if range is invalid
            else:
                return 50 + 50 * min(std_val - value, std_val - min_val) / range_factor
        else:
            # Above standard, scale from 0 to 50
            range_factor = max_val - std_val
            if range_factor <= 0:
                return 25  # Default if range is invalid
            else:
                return 50 * max(max_val - value, 0) / range_factor

if __name__ == "__main__":
    # Test the quality analyzer
    print("Testing quality analyzer module...")
