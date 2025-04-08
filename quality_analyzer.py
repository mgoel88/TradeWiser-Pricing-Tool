"""
Quality analyzer module for agricultural commodities.

This module provides functions for analyzing quality parameters of agricultural
commodities and calculating quality scores based on industry standards.
"""

import logging
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import database functionality
from database_sql import get_price_quality_parameters


def analyze_commodity_quality(commodity, quality_params):
    """
    Analyze quality of a commodity based on provided parameters.
    
    Args:
        commodity (str): The commodity type
        quality_params (dict): Dictionary of quality parameters
        
    Returns:
        dict: Analysis results
    """
    try:
        if not commodity or not quality_params:
            return None
        
        # Calculate quality score
        quality_score, quality_grade = calculate_quality_score(commodity, quality_params)
        
        # Generate analysis summary
        analysis_summary = generate_quality_summary(commodity, quality_params, quality_score, quality_grade)
        
        # Return analysis results
        return {
            "commodity": commodity,
            "quality_params": quality_params,
            "quality_score": quality_score,
            "quality_grade": quality_grade,
            "confidence": 0.95,  # High confidence for direct parameter analysis
            "analysis_summary": analysis_summary,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error analyzing commodity quality: {e}")
        return None


def calculate_quality_score(commodity, quality_params):
    """
    Calculate quality score based on commodity parameters.
    
    Args:
        commodity (str): The commodity type
        quality_params (dict): Dictionary of quality parameters
        
    Returns:
        tuple: (quality_score, quality_grade)
    """
    try:
        if not commodity or not quality_params:
            return 70, "C"  # Default average score
        
        # Get standard quality parameters for this commodity
        standard_params = get_price_quality_parameters(commodity)
        
        if not standard_params:
            logger.warning(f"No standard quality parameters found for {commodity}")
            
            # Use a basic scoring algorithm when no standards available
            score = 70  # Default base score
            
            # Adjust for common parameters if available
            if "moisture_content" in quality_params:
                moisture = quality_params["moisture_content"]
                # Lower moisture is generally better (within limits)
                if moisture < 10:
                    score += 5
                elif moisture > 16:
                    score -= 10
                
            if "foreign_matter" in quality_params:
                foreign = quality_params["foreign_matter"]
                # Lower foreign matter is better
                if foreign < 0.5:
                    score += 10
                elif foreign > 2:
                    score -= 15 * (foreign / 2)
            
            # Cap the score between 0 and 100
            score = max(0, min(score, 100))
        else:
            # Calculate score based on standard parameters and their impact factors
            score = 80  # Start with a base score
            
            for param_name, param_value in quality_params.items():
                if param_name in standard_params:
                    std_param = standard_params[param_name]
                    
                    # Skip if necessary fields are missing
                    if "min" not in std_param or "max" not in std_param or "impact_factor" not in std_param:
                        continue
                    
                    # Get parameter range and impact
                    param_min = std_param.get("min", 0)
                    param_max = std_param.get("max", 100)
                    param_std = std_param.get("standard_value", (param_min + param_max) / 2)
                    impact_factor = std_param.get("impact_factor", 1.0)
                    
                    # Calculate where this value falls in the range
                    if param_max == param_min:
                        continue  # Avoid division by zero
                    
                    # Normalize the parameter to 0-1 range
                    param_range = param_max - param_min
                    normalized_value = (param_value - param_min) / param_range
                    normalized_std = (param_std - param_min) / param_range
                    
                    # Calculate the difference from standard value
                    # If impact_factor is positive, higher values are better
                    # If impact_factor is negative, lower values are better
                    if impact_factor > 0:
                        # For positive impact, higher than standard is good
                        if normalized_value > normalized_std:
                            # Bonus for being better than standard
                            score_change = (normalized_value - normalized_std) * impact_factor * 20
                        else:
                            # Penalty for being worse than standard
                            score_change = (normalized_value - normalized_std) * impact_factor * 25
                    else:
                        # For negative impact, lower than standard is good
                        if normalized_value < normalized_std:
                            # Bonus for being better than standard
                            score_change = (normalized_std - normalized_value) * abs(impact_factor) * 20
                        else:
                            # Penalty for being worse than standard
                            score_change = (normalized_std - normalized_value) * abs(impact_factor) * 25
                    
                    # Apply change to score
                    score += score_change
            
            # Cap the score between 0 and 100
            score = max(0, min(score, 100))
        
        # Determine grade based on score
        grade = "C"  # Default grade
        
        if score >= 95:
            grade = "A+"
        elif score >= 90:
            grade = "A"
        elif score >= 85:
            grade = "B+"
        elif score >= 80:
            grade = "B"
        elif score >= 75:
            grade = "C+"
        elif score >= 70:
            grade = "C"
        elif score >= 65:
            grade = "D"
        else:
            grade = "E"
        
        return score, grade
    except Exception as e:
        logger.error(f"Error calculating quality score: {e}")
        return 70, "C"  # Default average score


def generate_quality_summary(commodity, quality_params, quality_score, quality_grade):
    """
    Generate a text summary of quality analysis.
    
    Args:
        commodity (str): The commodity type
        quality_params (dict): Quality parameters
        quality_score (float): Calculated quality score
        quality_grade (str): Assigned quality grade
        
    Returns:
        str: Analysis summary
    """
    try:
        # Base summary based on grade
        if quality_grade == "A+":
            summary = f"Premium quality {commodity} meeting the highest industry standards."
        elif quality_grade == "A":
            summary = f"Excellent quality {commodity} meeting strict quality standards."
        elif quality_grade == "B+":
            summary = f"Very good quality {commodity} with few minor quality issues."
        elif quality_grade == "B":
            summary = f"Good quality {commodity} meeting most quality standards."
        elif quality_grade == "C+":
            summary = f"Above average quality {commodity} with some notable quality issues."
        elif quality_grade == "C":
            summary = f"Average quality {commodity} meeting basic quality standards."
        elif quality_grade == "D":
            summary = f"Below average quality {commodity} with significant quality issues."
        else:  # Grade E
            summary = f"Poor quality {commodity} with major quality issues."
        
        # Add details based on specific parameters
        details = []
        
        if "moisture_content" in quality_params:
            moisture = quality_params["moisture_content"]
            if moisture < 12:
                details.append(f"Low moisture content ({moisture:.1f}%) is ideal for storage.")
            elif moisture > 15:
                details.append(f"High moisture content ({moisture:.1f}%) may cause storage issues.")
        
        if "foreign_matter" in quality_params:
            foreign = quality_params["foreign_matter"]
            if foreign < 0.5:
                details.append(f"Minimal foreign matter ({foreign:.1f}%) indicates good cleaning practices.")
            elif foreign > 2:
                details.append(f"High foreign matter content ({foreign:.1f}%) requires additional cleaning.")
        
        # Add commodity-specific parameter summaries
        if commodity == "Rice":
            if "broken_percentage" in quality_params:
                broken = quality_params["broken_percentage"]
                if broken < 5:
                    details.append(f"Low broken percentage ({broken:.1f}%) indicates careful processing.")
                elif broken > 15:
                    details.append(f"High broken percentage ({broken:.1f}%) may affect consumer acceptance.")
            
            if "head_rice_recovery" in quality_params:
                head_rice = quality_params["head_rice_recovery"]
                if head_rice > 80:
                    details.append(f"Excellent head rice recovery ({head_rice:.1f}%).")
                elif head_rice < 65:
                    details.append(f"Low head rice recovery ({head_rice:.1f}%) indicates processing issues.")
        
        elif commodity == "Wheat":
            if "protein_content" in quality_params:
                protein = quality_params["protein_content"]
                if protein > 12:
                    details.append(f"High protein content ({protein:.1f}%) is excellent for bread making.")
                elif protein < 10:
                    details.append(f"Low protein content ({protein:.1f}%) may be better for pastry products.")
            
            if "gluten" in quality_params:
                gluten = quality_params["gluten"]
                if gluten > 28:
                    details.append(f"Strong gluten content ({gluten:.1f}%) indicates good baking quality.")
                elif gluten < 24:
                    details.append(f"Weak gluten content ({gluten:.1f}%) may limit bread-making applications.")
        
        elif commodity == "Maize":
            if "broken_kernels" in quality_params:
                broken = quality_params["broken_kernels"]
                if broken < 3:
                    details.append(f"Low broken kernels ({broken:.1f}%) indicates careful handling.")
                elif broken > 8:
                    details.append(f"High broken kernels ({broken:.1f}%) may affect industrial use quality.")
            
            if "aflatoxin" in quality_params:
                aflatoxin = quality_params["aflatoxin"]
                if aflatoxin < 5:
                    details.append(f"Low aflatoxin level ({aflatoxin:.1f} ppb) indicates good storage conditions.")
                elif aflatoxin > 15:
                    details.append(f"High aflatoxin level ({aflatoxin:.1f} ppb) may pose health concerns.")
        
        # Combine summary and details
        if details:
            summary += f" {' '.join(details)}"
        
        # Add recommendation based on grade
        if quality_grade in ["A+", "A", "B+"]:
            summary += " Suitable for premium markets and export."
        elif quality_grade in ["B", "C+"]:
            summary += " Suitable for domestic markets with standard requirements."
        elif quality_grade in ["C", "D"]:
            summary += " May require processing or blending before commercial use."
        else:  # Grade E
            summary += " Recommended for lower grade applications."
        
        return summary
    except Exception as e:
        logger.error(f"Error generating quality summary: {e}")
        return f"Analysis of {commodity} quality parameters indicates grade {quality_grade} quality."


def compare_quality_parameters(commodity, params1, params2):
    """
    Compare two sets of quality parameters for the same commodity.
    
    Args:
        commodity (str): The commodity type
        params1 (dict): First set of quality parameters
        params2 (dict): Second set of quality parameters
        
    Returns:
        dict: Comparison results
    """
    try:
        if not commodity or not params1 or not params2:
            return None
        
        # Get standard quality parameters
        standard_params = get_price_quality_parameters(commodity)
        
        # Calculate scores for both parameter sets
        score1, grade1 = calculate_quality_score(commodity, params1)
        score2, grade2 = calculate_quality_score(commodity, params2)
        
        # Compare parameters
        comparison = {
            "commodity": commodity,
            "score_difference": score1 - score2,
            "grade_difference": f"{grade1} vs {grade2}",
            "parameter_differences": {}
        }
        
        # Add detailed parameter differences
        all_params = set(list(params1.keys()) + list(params2.keys()))
        
        for param in all_params:
            val1 = params1.get(param)
            val2 = params2.get(param)
            
            if val1 is not None and val2 is not None:
                # Parameter exists in both sets
                diff = val1 - val2
                diff_percent = (diff / val2) * 100 if val2 != 0 else 0
                
                # Check if this parameter has an impact factor in standards
                impact = 0
                if standard_params and param in standard_params:
                    impact = standard_params[param].get("impact_factor", 0)
                
                better = None
                if impact > 0:
                    # Higher values are better
                    better = 1 if diff > 0 else (2 if diff < 0 else None)
                elif impact < 0:
                    # Lower values are better
                    better = 1 if diff < 0 else (2 if diff > 0 else None)
                
                comparison["parameter_differences"][param] = {
                    "value1": val1,
                    "value2": val2,
                    "difference": diff,
                    "difference_percent": diff_percent,
                    "better": better  # 1 = first set is better, 2 = second set is better, None = equal
                }
            elif val1 is not None:
                # Parameter exists only in first set
                comparison["parameter_differences"][param] = {
                    "value1": val1,
                    "value2": None,
                    "difference": "Only in set 1"
                }
            elif val2 is not None:
                # Parameter exists only in second set
                comparison["parameter_differences"][param] = {
                    "value1": None,
                    "value2": val2,
                    "difference": "Only in set 2"
                }
        
        # Determine which set is better overall
        if score1 > score2:
            comparison["better_set"] = 1
            comparison["difference_summary"] = f"Set 1 is better by {score1 - score2:.1f} points (Grade {grade1} vs {grade2})."
        elif score2 > score1:
            comparison["better_set"] = 2
            comparison["difference_summary"] = f"Set 2 is better by {score2 - score1:.1f} points (Grade {grade2} vs {grade1})."
        else:
            comparison["better_set"] = None
            comparison["difference_summary"] = f"Both sets have equal quality scores (Grade {grade1})."
        
        return comparison
    except Exception as e:
        logger.error(f"Error comparing quality parameters: {e}")
        return None


def get_threshold_violation(commodity, quality_params):
    """
    Check if any quality parameters exceed critical thresholds.
    
    Args:
        commodity (str): The commodity type
        quality_params (dict): Quality parameters
        
    Returns:
        dict: Violations found
    """
    try:
        if not commodity or not quality_params:
            return {}
        
        # Define critical thresholds for different commodities
        thresholds = {
            "Rice": {
                "moisture_content": {"max": 18, "critical": 20, "message": "High moisture content leads to spoilage"},
                "broken_percentage": {"max": 25, "critical": 35, "message": "High broken percentage reduces quality grade"},
                "foreign_matter": {"max": 2, "critical": 5, "message": "High foreign matter content is a food safety concern"}
            },
            "Wheat": {
                "moisture_content": {"max": 15, "critical": 18, "message": "High moisture content leads to spoilage"},
                "foreign_matter": {"max": 1.5, "critical": 3, "message": "High foreign matter content is a food safety concern"},
                "damaged_kernels": {"max": 5, "critical": 10, "message": "High damaged kernel percentage may indicate disease"}
            },
            "Maize": {
                "moisture_content": {"max": 16, "critical": 19, "message": "High moisture content leads to spoilage"},
                "aflatoxin": {"max": 20, "critical": 30, "message": "Aflatoxin levels exceed food safety standards"},
                "insect_damage": {"max": 5, "critical": 10, "message": "High insect damage indicates poor storage"}
            },
            # Generic thresholds for other commodities
            "DEFAULT": {
                "moisture_content": {"max": 17, "critical": 20, "message": "High moisture content leads to spoilage"},
                "foreign_matter": {"max": 2, "critical": 5, "message": "High foreign matter content is a food safety concern"},
                "damage": {"max": 7, "critical": 12, "message": "High damage percentage reduces quality grade"}
            }
        }
        
        # Get appropriate thresholds
        commodity_thresholds = thresholds.get(commodity, thresholds["DEFAULT"])
        
        # Check for violations
        violations = {
            "critical": [],
            "warning": []
        }
        
        for param, value in quality_params.items():
            if param in commodity_thresholds:
                threshold = commodity_thresholds[param]
                
                # Check for critical violation
                if "critical" in threshold and value > threshold["critical"]:
                    violations["critical"].append({
                        "parameter": param,
                        "value": value,
                        "threshold": threshold["critical"],
                        "message": threshold.get("message", f"Critical {param} level")
                    })
                # Check for warning violation
                elif "max" in threshold and value > threshold["max"]:
                    violations["warning"].append({
                        "parameter": param,
                        "value": value,
                        "threshold": threshold["max"],
                        "message": threshold.get("message", f"High {param} level")
                    })
        
        return violations
    except Exception as e:
        logger.error(f"Error checking threshold violations: {e}")
        return {"critical": [], "warning": []}


if __name__ == "__main__":
    # Sample usage
    rice_params = {
        "moisture_content": 14.2,
        "broken_percentage": 7.5,
        "foreign_matter": 0.8,
        "discoloration": 2.1,
        "head_rice_recovery": 75.0
    }
    
    result = analyze_commodity_quality("Rice", rice_params)
    if result:
        print(f"Quality Score: {result['quality_score']:.1f}")
        print(f"Quality Grade: {result['quality_grade']}")
        print(f"Analysis: {result['analysis_summary']}")