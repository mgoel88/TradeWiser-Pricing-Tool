
"""
Price forecasting module using machine learning and time series analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from models import predict_price_trend
from pricing_engine import get_price_history
from visualization import create_price_trend_chart

def generate_forecast_features(df):
    """Generate time-based features for forecasting"""
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    return df

def train_forecast_model(price_history, forecast_days=30):
    """Train ML model for price forecasting"""
    df = pd.DataFrame(price_history)
    df['date'] = pd.to_datetime(df['date'])
    
    # Generate features
    df = generate_forecast_features(df)
    
    # Create training data
    features = ['day_of_week', 'month', 'quarter', 'year']
    X = df[features]
    y = df['price']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Generate future dates
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
    
    # Create future features
    future_df = pd.DataFrame({'date': future_dates})
    future_df = generate_forecast_features(future_df)
    
    # Make predictions
    predictions = model.predict(future_df[features])
    
    return {
        'dates': future_dates,
        'predictions': predictions,
        'model': model,
        'last_price': df['price'].iloc[-1]
    }

def analyze_seasonality(price_history):
    """Analyze seasonal patterns in price data"""
    df = pd.DataFrame(price_history)
    df['date'] = pd.to_datetime(df['date'])
    
    # Monthly patterns
    monthly_avg = df.groupby(df['date'].dt.month)['price'].mean()
    monthly_std = df.groupby(df['date'].dt.month)['price'].std()
    
    # Quarterly patterns
    quarterly_avg = df.groupby(df['date'].dt.quarter)['price'].mean()
    
    return {
        'monthly_avg': monthly_avg,
        'monthly_std': monthly_std,
        'quarterly_avg': quarterly_avg
    }
