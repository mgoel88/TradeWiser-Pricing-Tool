"""
AI Vision module for analyzing agricultural commodity images and videos.
Uses OpenAI's GPT-4V model for advanced visual analysis.
"""

import os
import io
import logging
import json
import base64
import time
from PIL import Image
import numpy as np
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def encode_image_to_base64(image_path):
    """
    Encode image to base64 for API submission.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_bytes(image_bytes):
    """
    Encode image bytes to base64 for API submission.
    
    Args:
        image_bytes (bytes): Image binary data
        
    Returns:
        str: Base64 encoded image string
    """
    return base64.b64encode(image_bytes).decode('utf-8')

def save_uploaded_image(uploaded_file, save_dir="uploads/images"):
    """
    Save an uploaded file to the specified directory.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        save_dir (str): Directory to save the file
        
    Returns:
        str: Path to the saved file
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a unique filename based on timestamp
    timestamp = int(time.time())
    filename = f"{timestamp}_{uploaded_file.name}"
    filepath = os.path.join(save_dir, filename)
    
    # Save the file
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    logger.info(f"Saved uploaded file to {filepath}")
    return filepath

def analyze_commodity_image(image_data, commodity, analysis_type="general"):
    """
    Analyze a commodity image using OpenAI's GPT-4V model.
    
    Args:
        image_data: Image data (file path or bytes)
        commodity (str): The commodity type (e.g., "Rice", "Wheat")
        analysis_type (str): Type of analysis to perform
            - "general": General quality assessment
            - "detailed": Detailed quality parameters
            - "defects": Focus on defects and issues
            - "grading": Grade the commodity according to standards
    
    Returns:
        dict: Analysis results
    """
    logger.info(f"Analyzing {commodity} image with {analysis_type} analysis")
    
    try:
        # Check if API key is configured
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable.")
            return {
                "status": "error",
                "message": "OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable.",
                "commodity": commodity,
                "analysis_type": analysis_type
            }
        
        # Convert image to base64
        if isinstance(image_data, str):
            # It's a file path
            base64_image = encode_image_to_base64(image_data)
        elif isinstance(image_data, bytes):
            # It's image bytes
            base64_image = encode_image_from_bytes(image_data)
        else:
            # It's a file-like object (e.g., uploaded by Streamlit)
            image_bytes = image_data.read()
            image_data.seek(0)  # Reset file pointer
            base64_image = encode_image_from_bytes(image_bytes)
        
        # Define quality parameters specific to the commodity for more structured extraction
        commodity_specific_params = {}
        if commodity.lower() == "rice":
            commodity_specific_params = {
                "broken_percentage": "Percentage (0-100) of broken grains",
                "moisture_content": "Moisture content percentage (optimal 12-14%)",
                "foreign_matter": "Percentage (0-100) of foreign materials",
                "discoloration": "Percentage (0-100) of discolored grains",
                "aroma": "Qualitative assessment of aroma (1-10 scale)"
            }
        elif commodity.lower() == "wheat":
            commodity_specific_params = {
                "protein_content": "Estimated protein content (8-15%)",
                "test_weight": "Test weight in kg/hl (75-85 is standard)",
                "moisture_content": "Moisture content percentage (optimal 12-14%)",
                "damaged_kernels": "Percentage (0-100) of damaged kernels",
                "foreign_material": "Percentage (0-100) of foreign materials" 
            }
        elif commodity.lower() in ["tur dal", "tur", "pigeon pea"]:
            commodity_specific_params = {
                "color_uniformity": "Color uniformity score (1-10)",
                "moisture_content": "Moisture content percentage (optimal 10-12%)",
                "foreign_matter": "Percentage (0-100) of foreign materials",
                "damaged_grains": "Percentage (0-100) of damaged grains",
                "weeviled_grains": "Percentage (0-100) of infested grains"
            }
        elif commodity.lower() in ["soyabean", "soybean"]:
            commodity_specific_params = {
                "oil_content": "Estimated oil content percentage (18-23%)",
                "moisture_content": "Moisture content percentage (optimal 12-14%)",
                "foreign_matter": "Percentage (0-100) of foreign materials",
                "damaged_seeds": "Percentage (0-100) of damaged seeds",
                "seed_size": "Average seed size score (1-10)"
            }
        
        # Prepare the prompt based on analysis type
        if analysis_type == "general":
            system_content = f"You are an agricultural expert specializing in {commodity} quality assessment. Analyze the image and provide a general assessment of the quality."
            user_content = f"Please analyze this {commodity} sample and provide a general quality assessment. Include: overall appearance, any obvious defects, and an estimated quality grade (Excellent, Good, Average, Below Average, Poor)."
            
        elif analysis_type == "detailed":
            param_list = "\n".join([f"- {param}: {desc}" for param, desc in commodity_specific_params.items()])
            system_content = f"You are an agricultural expert specializing in {commodity} quality assessment. Analyze the image and provide detailed quality parameters."
            user_content = f"""Please analyze this {commodity} sample in detail. Extract the following quality parameters where visible:

{param_list}

For each parameter, provide a numeric value when possible and estimate a confidence level (0.1-1.0) for each assessment. When exact measurement is not possible, provide an estimated range and educated guess based on visual inspection."""
            
        elif analysis_type == "defects":
            system_content = f"You are an agricultural expert specializing in {commodity} defect detection. Analyze the image and identify any defects or issues."
            user_content = f"""Please analyze this {commodity} sample and identify all visible defects or issues. For each defect, provide:
1. Type of defect (e.g., discoloration, mold, insect damage)
2. Severity (Low, Medium, High)
3. Approximate percentage of sample affected (0-100%)
4. Potential causes
5. Impact on quality and marketability

Also provide an overall defect score from 0-100 where 0 means completely defect-free and 100 means completely defective."""
            
        elif analysis_type == "grading":
            system_content = f"You are an agricultural grading expert specializing in {commodity} standards. Analyze the image and assign a grade according to standard classifications."
            user_content = f"""Please analyze this {commodity} sample and assign a grade according to standard industry classifications.

Provide the following:
1. Assigned grade (A/Premium, B/Standard, C/Fair, D/Sub-standard, or similar appropriate grading scale)
2. Quality score (0-100)
3. Reasoning for the grade based on visible attributes
4. Major quality determinants
5. Confidence score for your grading assessment (0.1-1.0)

Also suggest a fair market value relative to standard grade (e.g., 95% of standard grade price)."""
            
        else:
            # Default to general analysis
            system_content = f"You are an agricultural expert specializing in {commodity} quality assessment. Analyze the image and provide a general assessment of the quality."
            user_content = f"Please analyze this {commodity} sample and provide a general quality assessment. Include: overall appearance, any obvious defects, and an estimated quality grade (Excellent, Good, Average, Below Average, Poor)."
        
        # Add instruction to format response as a structured JSON
        user_content += "\n\nPlease format your response as a JSON object with appropriate fields for each aspect of your analysis. Include a 'summary' field with your overall assessment and a 'quality_score' field with a numeric score from 0-100 where 100 is perfect quality."
        
        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_content
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500,
            temperature=0.2,  # Lower temperature for more factual responses
            response_format={"type": "json_object"}
        )
        
        # Parse and return the result
        result = json.loads(response.choices[0].message.content)
        
        # Add metadata about the analysis
        result["metadata"] = {
            "commodity": commodity,
            "analysis_type": analysis_type,
            "timestamp": int(time.time()),
            "model": "gpt-4o"
        }
        
        return {
            "status": "success",
            "analysis": result,
            "commodity": commodity,
            "analysis_type": analysis_type,
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to analyze image: {str(e)}",
            "commodity": commodity,
            "analysis_type": analysis_type,
            "timestamp": int(time.time())
        }

def analyze_video_frame(video_path, frame_interval=5, commodity=None, max_frames=5):
    """
    Extract and analyze frames from a video.
    
    Args:
        video_path (str): Path to the video file
        frame_interval (int): Interval between frames to extract (in seconds)
        commodity (str): The commodity type
        max_frames (int): Maximum number of frames to extract and analyze
        
    Returns:
        dict: Analysis results for each frame
    """
    # Note: This is a placeholder. Full implementation would require
    # additional libraries like OpenCV to extract video frames.
    # For now, we'll return a placeholder response
    
    logger.warning("Video analysis not fully implemented - requires OpenCV")
    
    return {
        "status": "error",
        "message": "Video analysis not yet implemented. Please upload image files instead."
    }

def extract_quality_parameters(analysis_result, commodity):
    """
    Extract structured quality parameters from AI analysis result.
    
    Args:
        analysis_result (dict): The AI analysis result
        commodity (str): The commodity type
        
    Returns:
        dict: Structured quality parameters
    """
    try:
        logger.info(f"Extracting quality parameters for {commodity}")
        
        # Get analysis data
        analysis = analysis_result.get("analysis", {})
        
        # Default parameters
        quality_params = {}
        
        # Try to extract quality score first 
        quality_score = extract_numeric_param(analysis, ["quality_score", "overall_score", "score", "quality"], 0, 100)
        if quality_score > 0:
            quality_params["quality_score"] = quality_score
            logger.info(f"Extracted quality score: {quality_score}")
        
        # Extract quality grade if available
        quality_grade = extract_text_param(analysis, ["grade", "quality_grade", "overall_grade"], None)
        if quality_grade:
            quality_params["quality_grade"] = quality_grade
            logger.info(f"Extracted quality grade: {quality_grade}")
        
        # Extract confidence score if available
        confidence = extract_numeric_param(analysis, ["confidence", "confidence_score", "certainty"], 0, 1)
        if confidence > 0:
            quality_params["confidence"] = confidence
            logger.info(f"Extracted confidence score: {confidence}")
        
        # Extract summary or analysis text if available
        if "summary" in analysis and isinstance(analysis["summary"], str):
            quality_params["ai_summary"] = analysis["summary"]
        
        # Add analysis type and timestamp
        quality_params["analysis_type"] = analysis_result.get("analysis_type", "general")
        quality_params["timestamp"] = analysis_result.get("timestamp", int(time.time()))
        
        # Extract parameters based on commodity type
        commodity_lower = commodity.lower()
        
        # Generic parameters that apply to all commodities
        common_params = {
            "moisture_content": (["moisture", "moisture_content", "moisture_percentage"], 0, 30),
            "foreign_matter": (["foreign_matter", "foreign_material", "impurities"], 0, 20),
            "damaged_percentage": (["damaged", "damaged_grains", "damage", "damaged_percentage"], 0, 100),
            "color": (["color", "color_score", "color_quality"], 0, 10),
            "purity": (["purity", "purity_percentage", "purity_score"], 0, 100),
            "cleanliness": (["cleanliness", "cleanliness_score"], 0, 10),
            "odor": (["odor", "smell", "aroma", "odor_score"], 0, 10)
        }
        
        # Extract common parameters
        for param_name, (keys, min_val, max_val) in common_params.items():
            value = extract_numeric_param(analysis, keys, min_val, max_val)
            if param_name not in quality_params:  # Don't overwrite if already set
                quality_params[param_name] = value
        
        # Specific commodity parameters
        if commodity_lower == "rice":
            # Try to extract common rice parameters
            rice_params = {
                "broken_percentage": (["broken_percentage", "broken", "broken_grains"], 0, 100),
                "discoloration": (["discoloration", "color_defects", "discolored_grains"], 0, 100),
                "maturity": (["maturity", "maturity_score", "maturity_level"], 0, 10),
                "grain_size": (["grain_size", "size", "kernel_size"], 0, 10),
                "chalkiness": (["chalkiness", "chalky_percentage"], 0, 100),
                "head_rice": (["head_rice", "head_rice_percentage", "whole_grains"], 0, 100),
                "aroma_quality": (["aroma_quality", "fragrance_quality"], 0, 10)
            }
            
            for param_name, (keys, min_val, max_val) in rice_params.items():
                value = extract_numeric_param(analysis, keys, min_val, max_val)
                quality_params[param_name] = value
            
        elif commodity_lower == "wheat":
            # Try to extract common wheat parameters
            wheat_params = {
                "protein_content": (["protein", "protein_content", "protein_percentage"], 0, 20),
                "test_weight": (["test_weight", "weight", "density", "hectoliter_weight"], 60, 85),
                "hardness": (["hardness", "kernel_hardness", "hardness_score"], 0, 10),
                "shriveled_grains": (["shriveled", "shriveled_grains", "shriveled_percentage"], 0, 100),
                "gluten_content": (["gluten", "gluten_content", "gluten_percentage"], 0, 15),
                "falling_number": (["falling_number", "falling_number_score"], 200, 400),
                "dockage": (["dockage", "dockage_percentage"], 0, 20)
            }
            
            for param_name, (keys, min_val, max_val) in wheat_params.items():
                value = extract_numeric_param(analysis, keys, min_val, max_val)
                quality_params[param_name] = value
            
        elif commodity_lower in ["tur dal", "tur", "pigeon pea"]:
            # Try to extract common tur dal parameters
            if "detailed" in analysis_result.get("analysis_type", ""):
                quality_params["foreign_matter"] = extract_numeric_param(analysis, ["foreign_matter", "impurities"], 0, 20)
                quality_params["moisture_content"] = extract_numeric_param(analysis, ["moisture", "moisture_content"], 0, 25)
                quality_params["damaged_grains"] = extract_numeric_param(analysis, ["damaged", "damaged_grains"], 0, 100)
                quality_params["weeviled_grains"] = extract_numeric_param(analysis, ["weeviled", "infested", "insects"], 0, 100)
                quality_params["color_uniformity"] = extract_numeric_param(analysis, ["color_uniformity", "uniformity"], 0, 10)
                quality_params["split_percentage"] = extract_numeric_param(analysis, ["split", "split_percentage"], 0, 100)
            
            quality_grade = extract_text_param(analysis, ["grade", "quality_grade", "overall_grade"], "Average")
            quality_params["quality_grade"] = quality_grade
            
        elif commodity_lower in ["soyabean", "soybean"]:
            # Try to extract common soybean parameters
            if "detailed" in analysis_result.get("analysis_type", ""):
                quality_params["oil_content"] = extract_numeric_param(analysis, ["oil_content", "oil", "oil_percentage"], 0, 30)
                quality_params["moisture_content"] = extract_numeric_param(analysis, ["moisture", "moisture_content"], 0, 20)
                quality_params["foreign_matter"] = extract_numeric_param(analysis, ["foreign_matter", "impurities"], 0, 20)
                quality_params["damaged_seeds"] = extract_numeric_param(analysis, ["damaged", "damaged_seeds"], 0, 100)
                quality_params["seed_size"] = extract_numeric_param(analysis, ["seed_size", "size"], 0, 10)
                quality_params["protein_content"] = extract_numeric_param(analysis, ["protein", "protein_content"], 30, 50)
            
            quality_grade = extract_text_param(analysis, ["grade", "quality_grade", "overall_grade"], "Average")
            quality_params["quality_grade"] = quality_grade
            
        else:
            # Generic parameters for other commodities
            if "detailed" in analysis_result.get("analysis_type", ""):
                quality_params["appearance_score"] = extract_numeric_param(analysis, ["appearance", "appearance_score"], 0, 10)
                quality_params["moisture_content"] = extract_numeric_param(analysis, ["moisture", "moisture_content"], 0, 30)
                quality_params["foreign_matter"] = extract_numeric_param(analysis, ["foreign_matter", "impurities"], 0, 100)
                quality_params["damage_percentage"] = extract_numeric_param(analysis, ["damage", "damage_percentage"], 0, 100)
                quality_params["color_uniformity"] = extract_numeric_param(analysis, ["color", "color_uniformity", "uniformity"], 0, 10)
            
            quality_grade = extract_text_param(analysis, ["grade", "quality_grade", "overall_grade"], "Average")
            quality_params["quality_grade"] = quality_grade
        
        # Add summary from AI
        quality_params["ai_summary"] = extract_text_param(analysis, ["summary", "overall", "conclusion"], "Analysis not available")
        
        # Add market value assessment if available
        market_value = extract_text_param(analysis, ["market_value", "price_estimate", "value_assessment"], "")
        if market_value:
            quality_params["market_value_assessment"] = market_value
            
        # Add relative market value if available (percentage of standard price)
        relative_value = extract_numeric_param(analysis, ["relative_value", "price_percentage", "relative_market_value"], 0, 200)
        if relative_value > 0 and relative_value != 100:  # Only add if not default
            quality_params["relative_market_value"] = relative_value
            
        # Add confidence level if available
        confidence = extract_numeric_param(analysis, ["confidence", "confidence_level", "certainty"], 0, 1)
        if confidence > 0:
            quality_params["confidence"] = confidence
            
        # Calculate quality score if not already present
        if "quality_score" not in quality_params:
            from quality_analyzer import calculate_quality_score
            # Get commodity reference data (placeholder - actual implementation would get this from database)
            commodity_data = {
                "quality_parameters": {param: {"min": 0, "max": 100, "standard_value": 50} for param in quality_params if param not in ["quality_grade", "ai_summary", "confidence"]},
                "quality_impact": {param: {"factor": 1} for param in quality_params if param not in ["quality_grade", "ai_summary", "confidence"]}
            }
            computed_score = calculate_quality_score(quality_params, commodity_data)
            quality_params["computed_quality_score"] = computed_score
        
        # Add timestamp for the analysis
        quality_params["timestamp"] = int(time.time())
        quality_params["analysis_method"] = "ai_vision"
        
        return quality_params
        
    except Exception as e:
        logger.error(f"Error extracting quality parameters: {str(e)}")
        return {"error": str(e), "ai_summary": "Error processing analysis", "quality_grade": "Unknown"}

def extract_numeric_param(data, possible_keys, min_val=0, max_val=100):
    """
    Extract a numeric parameter from AI analysis result.
    
    Args:
        data (dict): AI analysis data
        possible_keys (list): List of possible keys for the parameter
        min_val (float): Minimum acceptable value
        max_val (float): Maximum acceptable value
        
    Returns:
        float: Extracted parameter value or None
    """
    # Try direct keys first
    for key in possible_keys:
        if key in data and isinstance(data[key], (int, float)):
            return max(min_val, min(max_val, float(data[key])))
    
    # Try nested objects
    for obj_key in data:
        if isinstance(data[obj_key], dict):
            for key in possible_keys:
                if key in data[obj_key] and isinstance(data[obj_key][key], (int, float)):
                    return max(min_val, min(max_val, float(data[obj_key][key])))
    
    # Try to extract from strings using regex
    import re
    for obj_key in data:
        if isinstance(data[obj_key], str):
            for key in possible_keys:
                if key.lower() in data[obj_key].lower():
                    # Extract numbers from the string
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", data[obj_key])
                    if numbers:
                        return max(min_val, min(max_val, float(numbers[0])))
    
    # Default to middle of range if not found
    return (min_val + max_val) / 2

def extract_text_param(data, possible_keys, default_value="N/A"):
    """
    Extract a text parameter from AI analysis result.
    
    Args:
        data (dict): AI analysis data
        possible_keys (list): List of possible keys for the parameter
        default_value (str): Default value if not found
        
    Returns:
        str: Extracted parameter value
    """
    # Try direct keys first
    for key in possible_keys:
        if key in data and isinstance(data[key], str):
            return data[key]
    
    # Try nested objects
    for obj_key in data:
        if isinstance(data[obj_key], dict):
            for key in possible_keys:
                if key in data[obj_key] and isinstance(data[obj_key][key], str):
                    return data[obj_key][key]
    
    # Default
    return default_value

if __name__ == "__main__":
    # Test the module
    print("Testing AI Vision module...")