"""
Supabase Database Module for TradeWiser Pricing Tool
Replaces all in-memory storage with persistent Supabase PostgreSQL + pgvector
"""

import os
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==================== COMMODITY FUNCTIONS ====================

def get_all_commodities() -> List[str]:
    """Get list of all commodity names."""
    try:
        response = supabase.table('commodities').select('name').execute()
        return [row['name'] for row in response.data]
    except Exception as e:
        logger.error(f"Error fetching commodities: {e}")
        return []


def get_commodity_data(commodity: str) -> Optional[Dict[str, Any]]:
    """Get complete commodity data including quality parameters."""
    try:
        # Get commodity base data
        response = supabase.table('commodities')\
            .select('*')\
            .eq('name', commodity)\
            .maybe_single()\
            .execute()
        
        if not response.data:
            logger.warning(f"Commodity {commodity} not found")
            return None
        
        commodity_data = response.data
        commodity_id = commodity_data['id']
        
        # Get quality parameters
        params_response = supabase.table('quality_parameters')\
            .select('*')\
            .eq('commodity_id', commodity_id)\
            .execute()
        
        quality_parameters = {}
        quality_impact = {}
        
        for param in params_response.data:
            param_name = param['parameter_name']
            quality_parameters[param_name] = {
                'min': float(param['min_value']) if param['min_value'] else 0,
                'max': float(param['max_value']) if param['max_value'] else 100,
                'standard_value': float(param['standard_value']) if param['standard_value'] else 50,
                'unit': param['unit'] or '%',
                'step': float(param['step_size']) if param['step_size'] else 0.1,
                'impact_type': param['impact_type'] or 'linear'
            }
            
            quality_impact[param_name] = {
                'factor': float(param['impact_factor']) if param['impact_factor'] else 0,
                'premium_factor': float(param['premium_factor']) if param['premium_factor'] else 0,
                'discount_factor': float(param['discount_factor']) if param['discount_factor'] else 0
            }
        
        # Get regions
        regions_response = supabase.table('commodity_regions')\
            .select('regions(name)')\
            .eq('commodity_id', commodity_id)\
            .eq('is_active', True)\
            .execute()
        
        regions = [r['regions']['name'] for r in regions_response.data if r.get('regions')]
        
        return {
            'name': commodity_data['name'],
            'description': commodity_data.get('description'),
            'trading_unit': commodity_data.get('trading_unit', 'kg'),
            'price_range': {
                'min': float(commodity_data['price_min']) if commodity_data.get('price_min') else 1000,
                'max': float(commodity_data['price_max']) if commodity_data.get('price_max') else 10000
            },
            'quality_parameters': quality_parameters,
            'quality_impact': quality_impact,
            'regions': regions
        }
    except Exception as e:
        logger.error(f"Error fetching commodity data for {commodity}: {e}")
        return None


def get_regions(commodity: str) -> List[str]:
    """Get all regions for a commodity."""
    try:
        commodity_data = get_commodity_data(commodity)
        if commodity_data:
            return commodity_data.get('regions', [])
        return []
    except Exception as e:
        logger.error(f"Error fetching regions for {commodity}: {e}")
        return []


# ==================== PRICE FUNCTIONS ====================

def get_average_market_price(commodity: str, region: str, days_back: int = 30) -> float:
    """Get average market price for commodity and region."""
    try:
        # Get commodity and region IDs
        commodity_response = supabase.table('commodities')\
            .select('id')\
            .eq('name', commodity)\
            .maybe_single()\
            .execute()
        
        if not commodity_response.data:
            return 0
        
        commodity_id = commodity_response.data['id']
        
        region_response = supabase.table('regions')\
            .select('id')\
            .eq('name', region)\
            .maybe_single()\
            .execute()
        
        if not region_response.data:
            return 0
        
        region_id = region_response.data['id']
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        # Get average price
        response = supabase.table('market_prices')\
            .select('price')\
            .eq('commodity_id', commodity_id)\
            .eq('region_id', region_id)\
            .gte('date', start_date.isoformat())\
            .lte('date', end_date.isoformat())\
            .execute()
        
        if response.data:
            prices = [float(row['price']) for row in response.data]
            return sum(prices) / len(prices)
        
        return 0
    except Exception as e:
        logger.error(f"Error getting average market price: {e}")
        return 0


def get_price_history(commodity: str, region: str, days: int = 30) -> List[Dict[str, Any]]:
    """Get price history for commodity and region."""
    try:
        # Get commodity and region IDs
        commodity_response = supabase.table('commodities')\
            .select('id')\
            .eq('name', commodity)\
            .maybe_single()\
            .execute()
        
        if not commodity_response.data:
            return []
        
        commodity_id = commodity_response.data['id']
        
        region_response = supabase.table('regions')\
            .select('id')\
            .eq('name', region)\
            .maybe_single()\
            .execute()
        
        if not region_response.data:
            return []
        
        region_id = region_response.data['id']
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get price history
        response = supabase.table('market_prices')\
            .select('date, price, volume, market_name')\
            .eq('commodity_id', commodity_id)\
            .eq('region_id', region_id)\
            .gte('date', start_date.isoformat())\
            .lte('date', end_date.isoformat())\
            .order('date', desc=True)\
            .execute()
        
        return response.data
    except Exception as e:
        logger.error(f"Error getting price history: {e}")
        return []


# ==================== VECTOR FUNCTIONS ====================

def store_quality_price_vector(
    commodity: str,
    region: str,
    quality_params: Dict[str, float],
    price: float,
    source: str = 'user_input',
    confidence: float = 0.8
) -> bool:
    """Store quality-price vector for ML-based pricing."""
    try:
        # Get commodity and region IDs
        commodity_response = supabase.table('commodities')\
            .select('id')\
            .eq('name', commodity)\
            .maybe_single()\
            .execute()
        
        if not commodity_response.data:
            logger.warning(f"Commodity {commodity} not found")
            return False
        
        commodity_id = commodity_response.data['id']
        
        region_response = supabase.table('regions')\
            .select('id')\
            .eq('name', region)\
            .maybe_single()\
            .execute()
        
        if not region_response.data:
            logger.warning(f"Region {region} not found")
            return False
        
        region_id = region_response.data['id']
        
        # Create vector from quality parameters (pad to 20 dimensions)
        vector_values = list(quality_params.values())[:20]
        while len(vector_values) < 20:
            vector_values.append(0.0)
        
        # Store vector
        supabase.table('quality_price_vectors').insert({
            'commodity_id': commodity_id,
            'region_id': region_id,
            'quality_vector': vector_values,
            'quality_params': quality_params,
            'price': price,
            'confidence': confidence,
            'source': source,
            'date': date.today().isoformat()
        }).execute()
        
        return True
    except Exception as e:
        logger.error(f"Error storing quality-price vector: {e}")
        return False


def find_similar_quality_vectors(
    commodity: str,
    region: str,
    quality_params: Dict[str, float],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Find similar quality vectors using pgvector similarity search."""
    try:
        # Get commodity and region IDs
        commodity_response = supabase.table('commodities')\
            .select('id')\
            .eq('name', commodity)\
            .maybe_single()\
            .execute()
        
        if not commodity_response.data:
            return []
        
        commodity_id = commodity_response.data['id']
        
        region_response = supabase.table('regions')\
            .select('id')\
            .eq('name', region)\
            .maybe_single()\
            .execute()
        
        if not region_response.data:
            return []
        
        region_id = region_response.data['id']
        
        # Create query vector
        query_vector = list(quality_params.values())[:20]
        while len(query_vector) < 20:
            query_vector.append(0.0)
        
        # Call RPC function for vector similarity search
        response = supabase.rpc(
            'match_quality_vectors',
            {
                'query_embedding': query_vector,
                'match_commodity_id': commodity_id,
                'match_region_id': region_id,
                'match_count': limit
            }
        ).execute()
        
        return response.data
    except Exception as e:
        logger.error(f"Error finding similar quality vectors: {e}")
        return []


# ==================== LOGGING FUNCTIONS ====================

def log_api_request(
    user_id: Optional[str],
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: int,
    ip_address: Optional[str] = None,
    error_message: Optional[str] = None
) -> bool:
    """Log API request for monitoring."""
    try:
        supabase.table('api_logs').insert({
            'user_id': user_id,
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_time_ms': response_time_ms,
            'ip_address': ip_address,
            'error_message': error_message
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Error logging API request: {e}")
        return False


def log_crawler_job(
    source: str,
    status: str,
    records_processed: int = 0,
    error_message: Optional[str] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None
) -> bool:
    """Log crawler job execution."""
    try:
        duration_seconds = None
        if started_at and completed_at:
            duration_seconds = int((completed_at - started_at).total_seconds())
        
        supabase.table('crawler_logs').insert({
            'source': source,
            'status': status,
            'records_processed': records_processed,
            'error_message': error_message,
            'started_at': (started_at or datetime.now()).isoformat(),
            'completed_at': completed_at.isoformat() if completed_at else None,
            'duration_seconds': duration_seconds
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Error logging crawler job: {e}")
        return False


# ==================== INITIALIZATION ====================

def initialize_database():
    """Initialize database with sample data if empty."""
    try:
        # Check if commodities exist
        response = supabase.table('commodities').select('id').limit(1).execute()
        
        if not response.data:
            logger.info("Database is empty, initializing with sample data...")
            # Add sample commodity
            commodity_response = supabase.table('commodities').insert({
                'name': 'Wheat',
                'description': 'Common wheat grain',
                'trading_unit': 'kg',
                'price_min': 2000,
                'price_max': 3000
            }).execute()
            
            logger.info("Sample data initialized successfully")
        else:
            logger.info("Database already initialized")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False
