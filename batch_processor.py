
"""
Batch processor module for handling bulk price calculations
"""
import pandas as pd
import logging
from datetime import datetime
from pricing_engine import calculate_price

logger = logging.getLogger(__name__)

def process_batch_file(file_path):
    """
    Process a batch file (CSV/Excel) containing multiple samples
    
    Args:
        file_path (str): Path to the input file
        
    Returns:
        dict: Processing results with calculated prices
    """
    try:
        # Read file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return {"success": False, "message": "Unsupported file format"}

        # Validate required columns
        required_cols = ['commodity', 'region']
        if not all(col in df.columns for col in required_cols):
            return {"success": False, "message": "Missing required columns: commodity, region"}

        # Process each row
        results = []
        for _, row in df.iterrows():
            # Extract quality parameters from columns starting with 'quality_'
            quality_params = {col.replace('quality_', ''): row[col] 
                            for col in df.columns if col.startswith('quality_')}
            
            # Calculate price
            final_price, base_price, quality_delta, location_delta, market_delta = calculate_price(
                row['commodity'],
                quality_params,
                row['region']
            )
            
            results.append({
                "commodity": row['commodity'],
                "region": row['region'],
                "quality_parameters": quality_params,
                "base_price": base_price,
                "quality_adjustment": quality_delta,
                "location_adjustment": location_delta,
                "market_adjustment": market_delta,
                "final_price": final_price
            })

        # Create output DataFrame
        output_df = pd.DataFrame(results)
        
        # Generate export filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = f"data/batch_results_{timestamp}.csv"
        output_df.to_csv(export_path, index=False)

        return {
            "success": True,
            "processed_count": len(results),
            "results": results,
            "export_path": export_path
        }

    except Exception as e:
        logger.error(f"Error processing batch file: {e}")
        return {"success": False, "message": str(e)}
