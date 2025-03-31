"""
Database module for storing and retrieving commodity pricing data
with vector capabilities for quality-region-price relationships.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import faiss  # FAISS for vector database functionality
import pickle
from scipy import spatial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
COMMODITY_DB_FILE = os.path.join(DATA_DIR, "commodity_database.json")
USER_INPUT_DB_FILE = os.path.join(DATA_DIR, "user_inputs.json")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# Initialize commodity database with default data if it doesn't exist
def initialize_commodity_database():
    """
    Initialize the commodity database with default data if it doesn't exist.
    """
    if not os.path.exists(COMMODITY_DB_FILE):
        logger.info("Initializing commodity database with default data")
        
        # Default commodity data
        commodity_data = {
            "Wheat": {
                "quality_parameters": {
                    "moisture": {
                        "min": 8.0,
                        "max": 16.0,
                        "standard_value": 12.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "threshold"
                    },
                    "foreign_matter": {
                        "min": 0.0,
                        "max": 5.0,
                        "standard_value": 1.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    },
                    "damaged_grains": {
                        "min": 0.0,
                        "max": 10.0,
                        "standard_value": 2.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    },
                    "shriveled_grains": {
                        "min": 0.0,
                        "max": 8.0,
                        "standard_value": 3.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    }
                },
                "price_range": {
                    "min": 2200,
                    "max": 3500
                },
                "regions": ["North India", "Central India", "West India"],
                "quality_impact": {
                    "moisture": {
                        "factor": -50.0,
                        "premium_factor": -40.0,
                        "discount_factor": -60.0
                    },
                    "foreign_matter": {
                        "factor": -100.0
                    },
                    "damaged_grains": {
                        "factor": -80.0
                    },
                    "shriveled_grains": {
                        "factor": -40.0
                    }
                }
            },
            "Rice": {
                "quality_parameters": {
                    "moisture": {
                        "min": 10.0,
                        "max": 18.0,
                        "standard_value": 14.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "threshold"
                    },
                    "broken_grains": {
                        "min": 0.0,
                        "max": 15.0,
                        "standard_value": 5.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    },
                    "foreign_matter": {
                        "min": 0.0,
                        "max": 3.0,
                        "standard_value": 0.5,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    },
                    "discolored_grains": {
                        "min": 0.0,
                        "max": 5.0,
                        "standard_value": 1.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    }
                },
                "price_range": {
                    "min": 2800,
                    "max": 4200
                },
                "regions": ["North India", "East India", "South India"],
                "quality_impact": {
                    "moisture": {
                        "factor": -40.0,
                        "premium_factor": -30.0,
                        "discount_factor": -50.0
                    },
                    "broken_grains": {
                        "factor": -70.0
                    },
                    "foreign_matter": {
                        "factor": -120.0
                    },
                    "discolored_grains": {
                        "factor": -90.0
                    }
                }
            },
            "Tur Dal": {
                "quality_parameters": {
                    "moisture": {
                        "min": 8.0,
                        "max": 14.0,
                        "standard_value": 10.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "threshold"
                    },
                    "foreign_matter": {
                        "min": 0.0,
                        "max": 3.0,
                        "standard_value": 1.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    },
                    "damaged_grains": {
                        "min": 0.0,
                        "max": 5.0,
                        "standard_value": 2.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    },
                    "weevilled_grains": {
                        "min": 0.0,
                        "max": 3.0,
                        "standard_value": 1.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    }
                },
                "price_range": {
                    "min": 6000,
                    "max": 8500
                },
                "regions": ["South India", "West India", "Central India"],
                "quality_impact": {
                    "moisture": {
                        "factor": -60.0,
                        "premium_factor": -50.0,
                        "discount_factor": -70.0
                    },
                    "foreign_matter": {
                        "factor": -150.0
                    },
                    "damaged_grains": {
                        "factor": -120.0
                    },
                    "weevilled_grains": {
                        "factor": -200.0
                    }
                }
            },
            "Soyabean": {
                "quality_parameters": {
                    "moisture": {
                        "min": 8.0,
                        "max": 14.0,
                        "standard_value": 12.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "threshold"
                    },
                    "foreign_matter": {
                        "min": 0.0,
                        "max": 4.0,
                        "standard_value": 1.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    },
                    "oil_content": {
                        "min": 15.0,
                        "max": 22.0,
                        "standard_value": 18.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    },
                    "damaged_seeds": {
                        "min": 0.0,
                        "max": 5.0,
                        "standard_value": 2.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    }
                },
                "price_range": {
                    "min": 3500,
                    "max": 5000
                },
                "regions": ["Central India", "West India"],
                "quality_impact": {
                    "moisture": {
                        "factor": -45.0,
                        "premium_factor": -35.0,
                        "discount_factor": -55.0
                    },
                    "foreign_matter": {
                        "factor": -110.0
                    },
                    "oil_content": {
                        "factor": 80.0
                    },
                    "damaged_seeds": {
                        "factor": -90.0
                    }
                }
            },
            "Mustard": {
                "quality_parameters": {
                    "moisture": {
                        "min": 6.0,
                        "max": 12.0,
                        "standard_value": 8.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "threshold"
                    },
                    "foreign_matter": {
                        "min": 0.0,
                        "max": 4.0,
                        "standard_value": 2.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    },
                    "oil_content": {
                        "min": 35.0,
                        "max": 45.0,
                        "standard_value": 40.0,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    },
                    "damaged_seeds": {
                        "min": 0.0,
                        "max": 4.0,
                        "standard_value": 1.5,
                        "unit": "%",
                        "step": 0.1,
                        "impact_type": "linear"
                    }
                },
                "price_range": {
                    "min": 4500,
                    "max": 6500
                },
                "regions": ["North India", "East India"],
                "quality_impact": {
                    "moisture": {
                        "factor": -50.0,
                        "premium_factor": -40.0,
                        "discount_factor": -60.0
                    },
                    "foreign_matter": {
                        "factor": -100.0
                    },
                    "oil_content": {
                        "factor": 120.0
                    },
                    "damaged_seeds": {
                        "factor": -85.0
                    }
                }
            }
        }
        
        # Save to database file
        with open(COMMODITY_DB_FILE, 'w') as f:
            json.dump(commodity_data, f, indent=2)
        
        # Initialize the vector database for each commodity
        for commodity in commodity_data:
            initialize_vector_db(commodity, commodity_data[commodity])
    
    # Initialize user input database if it doesn't exist
    if not os.path.exists(USER_INPUT_DB_FILE):
        logger.info("Initializing user input database")
        
        with open(USER_INPUT_DB_FILE, 'w') as f:
            json.dump([], f)

def initialize_vector_db(commodity, commodity_data):
    """
    Initialize vector database for a commodity.
    
    Args:
        commodity (str): The commodity name
        commodity_data (dict): Commodity data
    """
    logger.info(f"Initializing vector database for {commodity}")
    
    # Create vector database directory for this commodity
    commodity_dir = os.path.join(VECTOR_DB_DIR, commodity.lower().replace(' ', '_'))
    os.makedirs(commodity_dir, exist_ok=True)
    
    # Get quality parameters
    quality_params = list(commodity_data.get('quality_parameters', {}).keys())
    
    if not quality_params:
        logger.warning(f"No quality parameters found for {commodity}")
        return
    
    # Create sample vectors for initial database
    # In a real implementation, this would use historical data
    
    # For each region, create sample vectors
    for region in commodity_data.get('regions', []):
        vectors = []
        metadata = []
        
        # Generate sample data points
        for _ in range(100):
            # Create a random quality vector
            quality_vector = []
            quality_values = {}
            
            for param in quality_params:
                param_data = commodity_data['quality_parameters'][param]
                min_val = param_data.get('min', 0)
                max_val = param_data.get('max', 100)
                std_val = param_data.get('standard_value', (min_val + max_val) / 2)
                
                # Generate a value with bias towards standard value
                value = np.random.normal(std_val, (max_val - min_val) / 6)
                value = max(min_val, min(max_val, value))
                
                quality_vector.append(value)
                quality_values[param] = value
            
            # Generate a price for this quality
            price_range = commodity_data.get('price_range', {'min': 2000, 'max': 5000})
            base_price = (price_range['min'] + price_range['max']) / 2
            
            # Calculate quality impact
            quality_impact = 0
            for i, param in enumerate(quality_params):
                if param in commodity_data.get('quality_impact', {}):
                    impact_data = commodity_data['quality_impact'][param]
                    std_val = commodity_data['quality_parameters'][param].get('standard_value')
                    deviation = quality_vector[i] - std_val
                    
                    # Calculate impact based on parameter type
                    param_type = commodity_data['quality_parameters'][param].get('impact_type', 'linear')
                    
                    if param_type == 'linear':
                        param_impact = deviation * impact_data.get('factor', 0)
                    elif param_type == 'threshold':
                        if deviation > 0:
                            param_impact = deviation * impact_data.get('premium_factor', 0)
                        else:
                            param_impact = deviation * impact_data.get('discount_factor', 0)
                    else:
                        param_impact = 0
                    
                    quality_impact += param_impact
            
            # Add regional adjustment
            regional_factors = {
                "North India": 1.02,
                "South India": 0.98,
                "East India": 0.95,
                "West India": 1.05,
                "Central India": 1.0
            }
            
            region_factor = regional_factors.get(region, 1.0)
            
            # Calculate final price
            final_price = base_price + quality_impact
            final_price *= region_factor
            
            # Add to database
            vectors.append(quality_vector)
            metadata.append({
                'quality': quality_values,
                'price': final_price,
                'region': region,
                'date': datetime.now().strftime('%Y-%m-%d')
            })
        
        # Convert to numpy array
        vectors_np = np.array(vectors, dtype='float32')
        
        # Create FAISS index
        d = len(quality_params)  # dimension
        index = faiss.IndexFlatL2(d)  # L2 distance
        index.add(vectors_np)
        
        # Save the index and metadata
        index_path = os.path.join(commodity_dir, f"{region.lower().replace(' ', '_')}_index.faiss")
        metadata_path = os.path.join(commodity_dir, f"{region.lower().replace(' ', '_')}_metadata.pkl")
        
        # Save FAISS index
        faiss.write_index(index, index_path)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Created vector database for {commodity} in {region} with {len(vectors)} samples")

def get_commodity_data(commodity):
    """
    Get data for a specific commodity.
    
    Args:
        commodity (str): The commodity name
        
    Returns:
        dict: Commodity data or None if not found
    """
    # Initialize database if needed
    initialize_commodity_database()
    
    # Load commodity database
    try:
        with open(COMMODITY_DB_FILE, 'r') as f:
            commodity_db = json.load(f)
        
        return commodity_db.get(commodity)
    except Exception as e:
        logger.error(f"Error getting commodity data: {e}")
        return None

def get_all_commodities():
    """
    Get a list of all commodities in the database.
    
    Returns:
        list: List of commodity names
    """
    # Initialize database if needed
    initialize_commodity_database()
    
    # Load commodity database
    try:
        with open(COMMODITY_DB_FILE, 'r') as f:
            commodity_db = json.load(f)
        
        return list(commodity_db.keys())
    except Exception as e:
        logger.error(f"Error getting commodities: {e}")
        return []

def get_regions(commodity):
    """
    Get regions for a specific commodity.
    
    Args:
        commodity (str): The commodity name
        
    Returns:
        list: List of regions or empty list if not found
    """
    commodity_data = get_commodity_data(commodity)
    
    if commodity_data:
        return commodity_data.get('regions', [])
    
    return []

def get_quality_impact(commodity, parameter):
    """
    Get quality impact data for a specific parameter.
    
    Args:
        commodity (str): The commodity name
        parameter (str): The quality parameter
        
    Returns:
        dict: Quality impact data or None if not found
    """
    commodity_data = get_commodity_data(commodity)
    
    if commodity_data and 'quality_impact' in commodity_data:
        return commodity_data['quality_impact'].get(parameter)
    
    return None

def save_user_input(commodity, quality_params, region, price=None):
    """
    Save user input to the database for future reference.
    
    Args:
        commodity (str): The commodity name
        quality_params (dict): Quality parameters
        region (str): The region
        price (float, optional): Price if known
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Saving user input for {commodity} in {region}")
    
    try:
        # Load current user inputs
        with open(USER_INPUT_DB_FILE, 'r') as f:
            user_inputs = json.load(f)
        
        # Add new input
        user_input = {
            'commodity': commodity,
            'quality_params': quality_params,
            'region': region,
            'price': price,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        user_inputs.append(user_input)
        
        # Save back to file
        with open(USER_INPUT_DB_FILE, 'w') as f:
            json.dump(user_inputs, f, indent=2)
        
        # Update vector database if price is known
        if price is not None:
            update_vector_db(commodity, quality_params, region, price)
        
        return True
    except Exception as e:
        logger.error(f"Error saving user input: {e}")
        return False

def update_vector_db(commodity, quality_params, region, price):
    """
    Update the vector database with new data.
    
    Args:
        commodity (str): The commodity name
        quality_params (dict): Quality parameters
        region (str): The region
        price (float): The price
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Updating vector database for {commodity} in {region}")
    
    try:
        # Create vector database directory for this commodity if it doesn't exist
        commodity_dir = os.path.join(VECTOR_DB_DIR, commodity.lower().replace(' ', '_'))
        os.makedirs(commodity_dir, exist_ok=True)
        
        # Paths for this region
        index_path = os.path.join(commodity_dir, f"{region.lower().replace(' ', '_')}_index.faiss")
        metadata_path = os.path.join(commodity_dir, f"{region.lower().replace(' ', '_')}_metadata.pkl")
        
        # Check if files exist
        index_exists = os.path.exists(index_path)
        metadata_exists = os.path.exists(metadata_path)
        
        # Get commodity data for parameter order
        commodity_data = get_commodity_data(commodity)
        
        if not commodity_data or 'quality_parameters' not in commodity_data:
            logger.warning(f"No quality parameters found for {commodity}")
            return False
        
        # Get quality parameters in the correct order
        quality_params_ordered = [quality_params.get(param, 0) for param in commodity_data['quality_parameters']]
        
        # Convert to numpy array
        vector = np.array([quality_params_ordered], dtype='float32')
        
        # Create or load FAISS index
        if index_exists:
            # Load existing index
            index = faiss.read_index(index_path)
            
            # Load existing metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        else:
            # Create new index
            d = len(quality_params_ordered)  # dimension
            index = faiss.IndexFlatL2(d)  # L2 distance
            metadata = []
        
        # Add to index
        index.add(vector)
        
        # Add to metadata
        metadata.append({
            'quality': quality_params,
            'price': price,
            'region': region,
            'date': datetime.now().strftime('%Y-%m-%d')
        })
        
        # Save the index and metadata
        faiss.write_index(index, index_path)
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Updated vector database for {commodity} in {region}")
        
        return True
    except Exception as e:
        logger.error(f"Error updating vector database: {e}")
        return False

def query_similar_qualities(commodity, quality_params, region, k=5):
    """
    Query the vector database for similar quality parameters.
    
    Args:
        commodity (str): The commodity name
        quality_params (dict): Quality parameters to query
        region (str): The region
        k (int): Number of similar items to return
        
    Returns:
        list: Similar items with their prices
    """
    logger.info(f"Querying similar qualities for {commodity} in {region}")
    
    try:
        # Get commodity data for parameter order
        commodity_data = get_commodity_data(commodity)
        
        if not commodity_data or 'quality_parameters' not in commodity_data:
            logger.warning(f"No quality parameters found for {commodity}")
            return []
        
        # Get quality parameters in the correct order
        quality_params_ordered = [quality_params.get(param, 0) for param in commodity_data['quality_parameters']]
        
        # Convert to numpy array
        query_vector = np.array([quality_params_ordered], dtype='float32')
        
        # Paths for this region
        commodity_dir = os.path.join(VECTOR_DB_DIR, commodity.lower().replace(' ', '_'))
        index_path = os.path.join(commodity_dir, f"{region.lower().replace(' ', '_')}_index.faiss")
        metadata_path = os.path.join(commodity_dir, f"{region.lower().replace(' ', '_')}_metadata.pkl")
        
        # Check if files exist
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning(f"Vector database not found for {commodity} in {region}")
            return []
        
        # Load index and metadata
        index = faiss.read_index(index_path)
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Query the index
        D, I = index.search(query_vector, k)  # D: distances, I: indices
        
        # Get results
        results = []
        
        for i, idx in enumerate(I[0]):
            if idx < len(metadata):
                item = metadata[idx].copy()
                item['distance'] = float(D[0][i])
                results.append(item)
        
        return results
    except Exception as e:
        logger.error(f"Error querying similar qualities: {e}")
        return []

def get_price_recommendation(commodity, quality_params, region):
    """
    Get a price recommendation based on similar items in the database.
    
    Args:
        commodity (str): The commodity name
        quality_params (dict): Quality parameters
        region (str): The region
        
    Returns:
        dict: Price recommendation data
    """
    logger.info(f"Getting price recommendation for {commodity} in {region}")
    
    # Query similar items
    similar_items = query_similar_qualities(commodity, quality_params, region, k=10)
    
    if not similar_items:
        # Fallback to calculated price
        from pricing_engine import calculate_price
        
        final_price, base_price, quality_delta, location_delta, market_delta = calculate_price(
            commodity, quality_params, region
        )
        
        return {
            'recommended_price': final_price,
            'source': 'calculated',
            'similar_items': [],
            'confidence': 'medium'
        }
    
    # Calculate weighted average price
    total_weight = 0
    weighted_sum = 0
    
    for item in similar_items:
        # Inverse distance weighting
        weight = 1 / (1 + item['distance'])
        total_weight += weight
        weighted_sum += item['price'] * weight
    
    if total_weight > 0:
        recommended_price = weighted_sum / total_weight
    else:
        # Fallback to simple average
        recommended_price = sum(item['price'] for item in similar_items) / len(similar_items)
    
    # Calculate standard deviation for confidence
    prices = [item['price'] for item in similar_items]
    std_dev = np.std(prices)
    mean_price = np.mean(prices)
    
    # Determine confidence level
    cv = (std_dev / mean_price) * 100  # Coefficient of variation
    
    if cv < 5:
        confidence = 'high'
    elif cv < 15:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return {
        'recommended_price': recommended_price,
        'source': 'similar_items',
        'similar_items': similar_items,
        'confidence': confidence,
        'std_dev': std_dev,
        'cv': cv
    }

if __name__ == "__main__":
    # Test the database functions
    print("Testing database module...")
    
    # Initialize the database
    initialize_commodity_database()
    
    # Get all commodities
    commodities = get_all_commodities()
    print(f"Found {len(commodities)} commodities: {commodities}")
    
    # Get regions for wheat
    regions = get_regions("Wheat")
    print(f"Regions for Wheat: {regions}")
