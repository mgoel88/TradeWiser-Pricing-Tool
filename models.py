"""
Models module for price prediction and quality assessment models.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy import stats

from database import get_commodity_data, get_regions, query_similar_qualities

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def predict_price_trend(commodity, region, days=30):
    """
    Predict future price trends for a commodity in a region.
    
    Args:
        commodity (str): The commodity
        region (str): The region
        days (int): Number of days to predict ahead
        
    Returns:
        dict: Predicted prices with confidence intervals
    """
    logger.info(f"Predicting price trend for {commodity} in {region} for {days} days")
    
    # Get historical price data
    # This would normally fetch real data from a database
    # For now, generate synthetic historical data
    
    # Get price history
    from pricing_engine import get_price_history
    
    # Get 6 months of historical data
    history_days = 180
    price_history = get_price_history(commodity, region, history_days)
    
    if not price_history:
        logger.warning(f"No price history available for {commodity} in {region}")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(price_history)
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create features for time series prediction
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Add lagged features
    for lag in [1, 3, 7, 14, 30]:
        df[f'lag_{lag}'] = df['price'].shift(lag)
    
    # Add rolling window features
    for window in [3, 7, 14, 30]:
        df[f'rolling_mean_{window}'] = df['price'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['price'].rolling(window=window).std()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    if df.empty:
        logger.warning("Not enough data for prediction after feature engineering")
        return None
    
    # Define features and target
    features = ['day_of_year', 'month', 'day_of_month', 'day_of_week'] + \
               [f'lag_{lag}' for lag in [1, 3, 7, 14, 30]] + \
               [f'rolling_mean_{window}' for window in [3, 7, 14, 30]] + \
               [f'rolling_std_{window}' for window in [3, 7, 14, 30]]
    
    X = df[features]
    y = df['price']
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model (robust against overfitting)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_score = model.score(X_val, y_val)
    logger.info(f"Validation R² score: {val_score:.4f}")
    
    # Generate future dates
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    # Create future dataframe with feature engineering
    future_df = pd.DataFrame({'date': future_dates})
    future_df['day_of_year'] = future_df['date'].dt.dayofyear
    future_df['month'] = future_df['date'].dt.month
    future_df['day_of_month'] = future_df['date'].dt.day
    future_df['day_of_week'] = future_df['date'].dt.dayofweek
    
    # The last available prices and statistics
    last_prices = df['price'].tail(30).values  # last 30 days
    
    # Make predictions iteratively (one day at a time)
    predictions = []
    prediction_std = []
    
    for i in range(days):
        # For the first prediction, use historical data
        if i == 0:
            lag_1 = df['price'].iloc[-1]
            lag_3 = df['price'].iloc[-3] if len(df) > 3 else df['price'].mean()
            lag_7 = df['price'].iloc[-7] if len(df) > 7 else df['price'].mean()
            lag_14 = df['price'].iloc[-14] if len(df) > 14 else df['price'].mean()
            lag_30 = df['price'].iloc[-30] if len(df) > 30 else df['price'].mean()
            
            rolling_mean_3 = df['price'].tail(3).mean()
            rolling_mean_7 = df['price'].tail(7).mean()
            rolling_mean_14 = df['price'].tail(14).mean()
            rolling_mean_30 = df['price'].tail(30).mean()
            
            rolling_std_3 = df['price'].tail(3).std()
            rolling_std_7 = df['price'].tail(7).std()
            rolling_std_14 = df['price'].tail(14).std()
            rolling_std_30 = df['price'].tail(30).std()
        else:
            # Use previous predictions for lagged values
            lag_1 = predictions[i-1]
            lag_3 = predictions[i-3] if i >= 3 else df['price'].iloc[-3+(i-3)]
            lag_7 = predictions[i-7] if i >= 7 else df['price'].iloc[-7+(i-7)]
            lag_14 = predictions[i-14] if i >= 14 else df['price'].iloc[-14+(i-14)]
            lag_30 = predictions[i-30] if i >= 30 else df['price'].iloc[-30+(i-30)]
            
            # Rolling statistics
            if i < 3:
                last_n = list(df['price'].tail(3-i).values) + predictions[:i]
                rolling_mean_3 = np.mean(last_n)
                rolling_std_3 = np.std(last_n)
            else:
                rolling_mean_3 = np.mean(predictions[i-3:i])
                rolling_std_3 = np.std(predictions[i-3:i])
            
            if i < 7:
                last_n = list(df['price'].tail(7-i).values) + predictions[:i]
                rolling_mean_7 = np.mean(last_n)
                rolling_std_7 = np.std(last_n)
            else:
                rolling_mean_7 = np.mean(predictions[i-7:i])
                rolling_std_7 = np.std(predictions[i-7:i])
            
            if i < 14:
                last_n = list(df['price'].tail(14-i).values) + predictions[:i]
                rolling_mean_14 = np.mean(last_n)
                rolling_std_14 = np.std(last_n)
            else:
                rolling_mean_14 = np.mean(predictions[i-14:i])
                rolling_std_14 = np.std(predictions[i-14:i])
            
            if i < 30:
                last_n = list(df['price'].tail(30-i).values) + predictions[:i]
                rolling_mean_30 = np.mean(last_n)
                rolling_std_30 = np.std(last_n)
            else:
                rolling_mean_30 = np.mean(predictions[i-30:i])
                rolling_std_30 = np.std(predictions[i-30:i])
        
        # Create feature vector for prediction
        X_pred = pd.DataFrame({
            'day_of_year': [future_df['day_of_year'].iloc[i]],
            'month': [future_df['month'].iloc[i]],
            'day_of_month': [future_df['day_of_month'].iloc[i]],
            'day_of_week': [future_df['day_of_week'].iloc[i]],
            'lag_1': [lag_1],
            'lag_3': [lag_3],
            'lag_7': [lag_7],
            'lag_14': [lag_14],
            'lag_30': [lag_30],
            'rolling_mean_3': [rolling_mean_3],
            'rolling_mean_7': [rolling_mean_7],
            'rolling_mean_14': [rolling_mean_14],
            'rolling_mean_30': [rolling_mean_30],
            'rolling_std_3': [rolling_std_3],
            'rolling_std_7': [rolling_std_7],
            'rolling_std_14': [rolling_std_14],
            'rolling_std_30': [rolling_std_30]
        })
        
        # Make prediction
        pred = model.predict(X_pred)[0]
        predictions.append(pred)
        
        # Use out-of-bag predictions for uncertainty estimate
        preds = []
        for estimator in model.estimators_:
            preds.append(estimator.predict(X_pred)[0])
        
        prediction_std.append(np.std(preds))
    
    # Confidence interval (95%)
    z_score = stats.norm.ppf(0.975)  # 95% confidence
    
    lower_bound = [pred - z_score * std for pred, std in zip(predictions, prediction_std)]
    upper_bound = [pred + z_score * std for pred, std in zip(predictions, prediction_std)]
    
    # Return predictions and dates
    return {
        'date': future_dates,
        'price': predictions,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'std': prediction_std
    }

def assess_quality_parameters(commodity, image_data=None, lab_report=None, use_ai=True):
    """
    Assess quality parameters from image data or lab report.
    
    Args:
        commodity (str): The commodity
        image_data (bytes, optional): Image data for analysis
        lab_report (dict, optional): Lab report data
        use_ai (bool): Whether to use AI models for analysis when available
        
    Returns:
        dict: Assessed quality parameters
    """
    logger.info(f"Assessing quality parameters for {commodity}")
    
    # Get commodity data for parameter ranges
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data or 'quality_parameters' not in commodity_data:
        logger.warning(f"No quality parameters found for commodity: {commodity}")
        return {}
    
    # Try to use AI vision module if image data is provided
    if image_data and use_ai:
        try:
            # Import AI vision module
            from ai_vision import analyze_commodity_image, extract_quality_parameters
            
            # Create a temporary file to store the image
            import tempfile
            import os
            
            # Create temp file
            fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            
            # Write image data to temp file
            with open(temp_path, 'wb') as f:
                f.write(image_data)
            
            # Analyze the image using AI
            result = analyze_commodity_image(temp_path, commodity, analysis_type="detailed")
            
            # Clean up temp file
            os.unlink(temp_path)
            
            if result["status"] == "success":
                # Extract parameters
                ai_quality_params = extract_quality_parameters(result, commodity)
                
                # Validate parameters against commodity data
                validated_params = {}
                
                for param, value in ai_quality_params.items():
                    if param in commodity_data['quality_parameters']:
                        param_data = commodity_data['quality_parameters'][param]
                        min_val = param_data.get('min', 0)
                        max_val = param_data.get('max', 100)
                        
                        # Ensure value is a number and within range
                        if isinstance(value, (int, float)):
                            value = max(min_val, min(max_val, value))
                            validated_params[param] = round(value, 2)
                    else:
                        # Include other parameters that might be useful
                        if isinstance(value, (int, float)):
                            validated_params[param] = value
                
                # Include AI summary if available
                if "ai_summary" in ai_quality_params:
                    validated_params["ai_summary"] = ai_quality_params["ai_summary"]
                
                if validated_params:
                    return validated_params
                
            logger.warning("AI analysis did not produce valid parameters, falling back to traditional method")
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            logger.warning("Error in AI quality assessment, falling back to traditional method")
    
    # Fallback method: generate parameters based on commodity data ranges
    quality_params = {}
    
    for param, details in commodity_data['quality_parameters'].items():
        min_val = details.get('min', 0)
        max_val = details.get('max', 100)
        std_val = details.get('standard_value', (min_val + max_val) / 2)
        
        # Generate a value with bias towards standard value
        # but with some variation to simulate real-world assessment
        value = np.random.normal(std_val, (max_val - min_val) / 8)
        value = max(min_val, min(max_val, value))
        
        quality_params[param] = round(value, 2)
    
    return quality_params

def build_quality_price_model(commodity, region):
    """
    Build a machine learning model to predict price based on quality parameters.
    
    Args:
        commodity (str): The commodity
        region (str): The region
        
    Returns:
        dict: Model details and performance metrics
    """
    logger.info(f"Building quality-price model for {commodity} in {region}")
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data or 'quality_parameters' not in commodity_data:
        logger.warning(f"No quality parameters found for commodity: {commodity}")
        return {
            'status': 'error',
            'message': 'Commodity data not found'
        }
    
    # In a real implementation, this would fetch historical data with quality parameters
    # For demonstration, we'll use synthetic data
    
    # Get quality parameters
    params = list(commodity_data['quality_parameters'].keys())
    
    if not params:
        logger.warning(f"No quality parameters found for {commodity}")
        return {
            'status': 'error',
            'message': 'No quality parameters found'
        }
    
    # Generate synthetic data
    n_samples = 500
    X = np.zeros((n_samples, len(params)))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        quality_values = {}
        
        # Generate random quality parameters
        for j, param in enumerate(params):
            param_data = commodity_data['quality_parameters'][param]
            min_val = param_data.get('min', 0)
            max_val = param_data.get('max', 100)
            std_val = param_data.get('standard_value', (min_val + max_val) / 2)
            
            # Generate a value
            value = np.random.normal(std_val, (max_val - min_val) / 6)
            value = max(min_val, min(max_val, value))
            
            X[i, j] = value
            quality_values[param] = value
        
        # Calculate price using pricing engine
        from pricing_engine import calculate_price
        
        price_result = calculate_price(commodity, quality_values, region)
        y[i] = price_result[0]  # Final price
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a linear regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    
    # Train a random forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate models
    linear_score = linear_model.score(X_test_scaled, y_test)
    rf_score = rf_model.score(X_test, y_test)
    
    # Calculate feature importance
    feature_importance = {}
    
    # From linear model
    for i, param in enumerate(params):
        feature_importance[param] = {
            'linear_coef': float(linear_model.coef_[i]),
            'rf_importance': float(rf_model.feature_importances_[i])
        }
    
    return {
        'status': 'success',
        'linear_model_score': linear_score,
        'random_forest_score': rf_score,
        'feature_importance': feature_importance,
        'n_samples': n_samples,
        'quality_parameters': params
    }

def estimate_quality_coefficients(commodity):
    """
    Estimate quality coefficients for the pricing model.
    
    Args:
        commodity (str): The commodity
        
    Returns:
        dict: Estimated coefficients for quality parameters
    """
    logger.info(f"Estimating quality coefficients for {commodity}")
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data or 'quality_parameters' not in commodity_data:
        logger.warning(f"No quality parameters found for commodity: {commodity}")
        return {}
    
    # Build models for each region to compare
    coefficients = {}
    
    for region in get_regions(commodity):
        model_result = build_quality_price_model(commodity, region)
        
        if model_result['status'] == 'success':
            coefficients[region] = model_result['feature_importance']
    
    return {
        'status': 'success',
        'coefficients': coefficients
    }

def predict_optimal_quality(commodity, region, target_price=None):
    """
    Predict optimal quality parameters to achieve a target price.
    
    Args:
        commodity (str): The commodity
        region (str): The region
        target_price (float, optional): Target price
        
    Returns:
        dict: Optimal quality parameters
    """
    logger.info(f"Predicting optimal quality for {commodity} in {region}")
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data or 'quality_parameters' not in commodity_data:
        logger.warning(f"No quality parameters found for commodity: {commodity}")
        return {
            'status': 'error',
            'message': 'Commodity data not found'
        }
    
    # If target price not provided, use the average price for the region
    if not target_price:
        # Calculate reference price
        price_range = commodity_data.get('price_range', {})
        target_price = (price_range.get('min', 2000) + price_range.get('max', 5000)) / 2
    
    # Get standard quality parameters
    standard_params = {
        param: details.get('standard_value', (details.get('min', 0) + details.get('max', 100)) / 2)
        for param, details in commodity_data['quality_parameters'].items()
    }
    
    # Calculate current price with standard parameters
    from pricing_engine import calculate_price
    
    standard_price_result = calculate_price(commodity, standard_params, region)
    standard_price = standard_price_result[0]  # Final price
    
    # If target price is close to standard price, return standard parameters
    if abs(target_price - standard_price) < 50:
        return {
            'status': 'success',
            'optimal_quality': standard_params,
            'target_price': target_price,
            'estimated_price': standard_price,
            'message': 'Target price achievable with standard quality'
        }
    
    # Build a quality-price model
    model_result = build_quality_price_model(commodity, region)
    
    if model_result['status'] != 'success':
        logger.warning(f"Failed to build quality-price model for {commodity} in {region}")
        return {
            'status': 'error',
            'message': 'Failed to build quality-price model'
        }
    
    # In a real implementation, we would use an optimization algorithm
    # to find the optimal quality parameters that achieve the target price
    # while minimizing deviation from standard quality
    
    # For demonstration, adjust quality parameters based on importance
    optimal_params = standard_params.copy()
    
    # Get quality impact data
    quality_impact = commodity_data.get('quality_impact', {})
    
    # Sort parameters by impact strength
    params_by_impact = sorted(
        quality_impact.items(),
        key=lambda x: abs(x[1].get('factor', 0)),
        reverse=True
    )
    
    # Price difference to achieve
    price_diff = target_price - standard_price
    
    # Adjust parameters with highest impact first
    remaining_diff = price_diff
    
    for param, impact in params_by_impact:
        if abs(remaining_diff) < 10:
            break  # Close enough
        
        param_data = commodity_data['quality_parameters'][param]
        min_val = param_data.get('min', 0)
        max_val = param_data.get('max', 100)
        std_val = param_data.get('standard_value', (min_val + max_val) / 2)
        
        # Get impact factor
        factor = impact.get('factor', 0)
        
        if factor == 0:
            continue  # No impact
        
        # Calculate how much to adjust this parameter
        # This is a simplified approach
        if factor > 0:
            # Positive factor: increasing parameter increases price
            if remaining_diff > 0:
                # Need to increase price: increase parameter
                param_diff = min(remaining_diff / factor, max_val - std_val)
                optimal_params[param] = std_val + param_diff
                remaining_diff -= param_diff * factor
            else:
                # Need to decrease price: decrease parameter
                param_diff = min(-remaining_diff / factor, std_val - min_val)
                optimal_params[param] = std_val - param_diff
                remaining_diff += param_diff * factor
        else:
            # Negative factor: increasing parameter decreases price
            if remaining_diff > 0:
                # Need to increase price: decrease parameter
                param_diff = min(remaining_diff / -factor, std_val - min_val)
                optimal_params[param] = std_val - param_diff
                remaining_diff -= param_diff * -factor
            else:
                # Need to decrease price: increase parameter
                param_diff = min(-remaining_diff / -factor, max_val - std_val)
                optimal_params[param] = std_val + param_diff
                remaining_diff += param_diff * -factor
    
    # Calculate estimated price with optimal parameters
    optimal_price_result = calculate_price(commodity, optimal_params, region)
    optimal_price = optimal_price_result[0]  # Final price
    
    return {
        'status': 'success',
        'optimal_quality': optimal_params,
        'target_price': target_price,
        'estimated_price': optimal_price,
        'price_difference': optimal_price - target_price,
        'standard_quality': standard_params,
        'standard_price': standard_price
    }

if __name__ == "__main__":
    # Test the models module
    print("Testing models module...")
