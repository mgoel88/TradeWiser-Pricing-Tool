"""
Data cleaning module for agricultural commodity pricing data.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

from database_sql import add_data_cleaning_rule, apply_data_cleaning_rules, Session, PricePoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


def initialize_data_cleaning_rules():
    """
    Initialize default data cleaning rules.
    """
    # Z-score outlier detection
    add_data_cleaning_rule(
        name="Z-Score Outlier Detection",
        rule_type="outlier",
        description="Detects outliers using Z-score method",
        parameters={
            "method": "z_score",
            "threshold": 3.0  # Standard 3-sigma rule
        }
    )
    
    # IQR outlier detection
    add_data_cleaning_rule(
        name="IQR Outlier Detection",
        rule_type="outlier",
        description="Detects outliers using Interquartile Range method",
        parameters={
            "method": "iqr",
            "factor": 1.5  # Standard IQR factor
        }
    )
    
    # Linear interpolation for missing values
    add_data_cleaning_rule(
        name="Linear Interpolation",
        rule_type="missing_value",
        description="Fill missing values using linear interpolation",
        parameters={
            "method": "interpolate"
        }
    )
    
    logger.info("Initialized default data cleaning rules")


def detect_price_anomalies(commodity, region=None, days=30, method="isolation_forest"):
    """
    Detect price anomalies using machine learning methods.
    
    Args:
        commodity (str): Commodity name
        region (str, optional): Region name (if None, check all regions)
        days (int): Number of days to analyze
        method (str): Detection method ('isolation_forest', 'dbscan')
        
    Returns:
        dict: Anomaly detection results
    """
    session = Session()
    
    try:
        # Get data
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Build query
        query = session.query(PricePoint).\
            filter(PricePoint.date >= start_date, PricePoint.date <= end_date)
            
        # Join with commodity and region if needed
        if commodity:
            from database_sql import Commodity
            query = query.join(Commodity).filter(Commodity.name == commodity)
            
        if region:
            from database_sql import Region
            query = query.join(Region).filter(Region.name == region)
        
        # Get price points
        price_points = query.all()
        
        if not price_points:
            logger.warning(f"No price data found for the specified parameters")
            return {"anomalies": [], "total": 0}
        
        # Convert to dataframe
        df = pd.DataFrame([
            {
                "id": pp.id,
                "commodity_id": pp.commodity_id,
                "region_id": pp.region_id,
                "date": pp.date,
                "price": pp.price,
                "data_reliability": pp.data_reliability
            }
            for pp in price_points
        ])
        
        # Group by commodity and region
        results = []
        
        for (c_id, r_id), group_df in df.groupby(['commodity_id', 'region_id']):
            if len(group_df) < 5:  # Need enough data points
                continue
            
            # Prepare features (using price and date as numerical feature)
            X = np.array([
                [row.price, (row.date - start_date).days]
                for _, row in group_df.iterrows()
            ])
            
            anomalies = []
            
            if method == "isolation_forest":
                # Use Isolation Forest for anomaly detection
                clf = IsolationForest(contamination=0.05, random_state=42)
                y_pred = clf.fit_predict(X)
                
                # -1 for anomalies, 1 for normal points
                anomaly_indices = np.where(y_pred == -1)[0]
                
                # Get anomalies
                for idx in anomaly_indices:
                    price_point_id = group_df.iloc[idx].id
                    price_point = session.query(PricePoint).filter(PricePoint.id == price_point_id).first()
                    
                    if price_point:
                        # Update reliability score
                        price_point.data_reliability = 0.5
                        
                        # Add to results
                        anomalies.append({
                            "id": price_point.id,
                            "date": price_point.date,
                            "price": price_point.price,
                            "expected_range": (
                                group_df['price'].mean() - group_df['price'].std() * 2,
                                group_df['price'].mean() + group_df['price'].std() * 2
                            )
                        })
            
            elif method == "dbscan":
                # Use DBSCAN for anomaly detection
                # Normalize features
                X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
                
                # Apply DBSCAN
                clustering = DBSCAN(eps=0.5, min_samples=3).fit(X_norm)
                labels = clustering.labels_
                
                # -1 for noise points (anomalies)
                anomaly_indices = np.where(labels == -1)[0]
                
                # Get anomalies
                for idx in anomaly_indices:
                    price_point_id = group_df.iloc[idx].id
                    price_point = session.query(PricePoint).filter(PricePoint.id == price_point_id).first()
                    
                    if price_point:
                        # Update reliability score
                        price_point.data_reliability = 0.5
                        
                        # Add to results
                        anomalies.append({
                            "id": price_point.id,
                            "date": price_point.date,
                            "price": price_point.price,
                            "expected_range": (
                                group_df['price'].mean() - group_df['price'].std() * 2,
                                group_df['price'].mean() + group_df['price'].std() * 2
                            )
                        })
            
            # Add to results
            if anomalies:
                from database_sql import Commodity, Region
                commodity_name = session.query(Commodity.name).filter(Commodity.id == c_id).scalar()
                region_name = session.query(Region.name).filter(Region.id == r_id).scalar()
                
                results.append({
                    "commodity": commodity_name,
                    "region": region_name,
                    "anomalies": anomalies,
                    "total_points": len(group_df),
                    "anomaly_percentage": len(anomalies) / len(group_df) * 100
                })
        
        session.commit()
        
        return {
            "results": results,
            "total_anomalies": sum(len(r["anomalies"]) for r in results),
            "total_points": len(df)
        }
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error detecting price anomalies: {e}")
        return {"error": str(e)}
    finally:
        session.close()


def fix_missing_data(commodity=None, days=30, method="interpolation"):
    """
    Fix missing data in price series.
    
    Args:
        commodity (str, optional): Commodity name (if None, fix all commodities)
        days (int): Number of days to analyze and fix
        method (str): Fixing method ('interpolation', 'backfill', 'average')
        
    Returns:
        dict: Results of the operation
    """
    session = Session()
    
    try:
        # Get data
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get all commodities and regions
        from database_sql import Commodity, Region
        
        if commodity:
            commodity_list = session.query(Commodity).filter(Commodity.name == commodity).all()
        else:
            commodity_list = session.query(Commodity).all()
        
        if not commodity_list:
            logger.warning(f"No commodities found")
            return {"fixed": 0}
        
        fixed_count = 0
        
        for commodity_obj in commodity_list:
            # Get regions for this commodity
            regions = session.query(Region).filter(Region.commodity_id == commodity_obj.id).all()
            
            for region in regions:
                # Get existing price points
                price_points = session.query(PricePoint).\
                    filter(
                        PricePoint.commodity_id == commodity_obj.id,
                        PricePoint.region_id == region.id,
                        PricePoint.date >= start_date,
                        PricePoint.date <= end_date
                    ).\
                    order_by(PricePoint.date).\
                    all()
                
                if not price_points:
                    continue
                
                # Create a dataframe
                df = pd.DataFrame([
                    {"date": pp.date, "price": pp.price, "source_id": pp.source_id}
                    for pp in price_points
                ])
                
                # Set date as index
                df.set_index("date", inplace=True)
                
                # Create a complete date range
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                
                # Reindex to include all dates
                df_reindexed = df.reindex(date_range)
                
                # Count missing dates
                missing_count = df_reindexed['price'].isna().sum()
                
                if missing_count == 0:
                    continue
                
                # Fix missing data
                if method == "interpolation":
                    # Linear interpolation
                    df_reindexed['price'] = df_reindexed['price'].interpolate(method='linear')
                
                elif method == "backfill":
                    # Fill with the previous value
                    df_reindexed['price'] = df_reindexed['price'].fillna(method='ffill')
                
                elif method == "average":
                    # Fill with the average price
                    avg_price = df['price'].mean()
                    df_reindexed['price'] = df_reindexed['price'].fillna(avg_price)
                
                # Add missing price points to the database
                for date_idx, row in df_reindexed.iterrows():
                    if date_idx not in df.index and not pd.isna(row['price']):
                        # Create new price point
                        price_point = PricePoint(
                            commodity_id=commodity_obj.id,
                            region_id=region.id,
                            date=date_idx.date(),
                            price=row['price'],
                            is_verified=False,
                            data_reliability=0.7  # Lower reliability for interpolated data
                        )
                        
                        session.add(price_point)
                        fixed_count += 1
        
        session.commit()
        
        return {
            "fixed": fixed_count
        }
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error fixing missing data: {e}")
        return {"error": str(e)}
    finally:
        session.close()


def validate_price_curve(commodity, region=None, days=30):
    """
    Validate price curves for consistency and continuity.
    
    Args:
        commodity (str): Commodity name
        region (str, optional): Region name (if None, validate all regions)
        days (int): Number of days to analyze
        
    Returns:
        dict: Validation results
    """
    session = Session()
    
    try:
        # Get data
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        from database_sql import Commodity, Region
        
        # Get commodity
        commodity_obj = session.query(Commodity).filter(Commodity.name == commodity).first()
        
        if not commodity_obj:
            logger.warning(f"Commodity {commodity} not found")
            return {"valid": False, "issues": ["Commodity not found"]}
        
        # Get regions
        if region:
            regions = session.query(Region).filter(
                Region.commodity_id == commodity_obj.id,
                Region.name == region
            ).all()
        else:
            regions = session.query(Region).filter(Region.commodity_id == commodity_obj.id).all()
        
        if not regions:
            logger.warning(f"No regions found for commodity {commodity}")
            return {"valid": False, "issues": ["No regions found"]}
        
        results = []
        
        for region_obj in regions:
            # Get price points
            price_points = session.query(PricePoint).\
                filter(
                    PricePoint.commodity_id == commodity_obj.id,
                    PricePoint.region_id == region_obj.id,
                    PricePoint.date >= start_date,
                    PricePoint.date <= end_date
                ).\
                order_by(PricePoint.date).\
                all()
            
            if not price_points:
                results.append({
                    "region": region_obj.name,
                    "valid": False,
                    "issues": ["No price data available"]
                })
                continue
            
            # Create a dataframe
            df = pd.DataFrame([
                {"date": pp.date, "price": pp.price}
                for pp in price_points
            ])
            
            # Check for issues
            issues = []
            
            # Check for large jumps (more than 10% change in a day)
            df['price_change'] = df['price'].pct_change() * 100
            large_jumps = df[abs(df['price_change']) > 10]
            
            if not large_jumps.empty:
                for _, jump in large_jumps.iterrows():
                    issues.append({
                        "type": "large_jump",
                        "date": jump['date'],
                        "change": jump['price_change'],
                        "price": jump['price']
                    })
            
            # Check for long periods of constant prices (more than 7 days)
            df['price_diff'] = df['price'].diff()
            constant_periods = []
            current_period = None
            
            for i, row in df.iterrows():
                if pd.isna(row['price_diff']) or abs(row['price_diff']) < 0.01:
                    if current_period is None:
                        current_period = {
                            "start_date": row['date'],
                            "price": row['price'],
                            "count": 1
                        }
                    else:
                        current_period["count"] += 1
                else:
                    if current_period is not None and current_period["count"] >= 7:
                        constant_periods.append(current_period)
                    current_period = None
            
            if current_period is not None and current_period["count"] >= 7:
                constant_periods.append(current_period)
            
            for period in constant_periods:
                issues.append({
                    "type": "constant_price",
                    "start_date": period["start_date"],
                    "price": period["price"],
                    "days": period["count"]
                })
            
            # Check for missing dates
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            missing_dates = [d.date() for d in date_range if d.date() not in df['date'].values]
            
            if missing_dates:
                issues.append({
                    "type": "missing_dates",
                    "dates": missing_dates,
                    "count": len(missing_dates)
                })
            
            results.append({
                "region": region_obj.name,
                "valid": len(issues) == 0,
                "issues": issues,
                "data_points": len(df),
                "date_range": [start_date, end_date]
            })
        
        return {
            "commodity": commodity,
            "results": results,
            "valid": all(r["valid"] for r in results)
        }
        
    except Exception as e:
        logger.error(f"Error validating price curve: {e}")
        return {"valid": False, "error": str(e)}
    finally:
        session.close()


def clean_data_pipeline(commodity=None, days=30):
    """
    Run the complete data cleaning pipeline.
    
    Args:
        commodity (str, optional): Commodity name (if None, clean all commodities)
        days (int): Number of days to clean
        
    Returns:
        dict: Results of the cleaning operation
    """
    results = {
        "anomalies_detected": 0,
        "missing_data_fixed": 0,
        "validation_issues": 0,
        "rules_applied": 0
    }
    
    try:
        # 1. Detect and handle anomalies
        anomalies = detect_price_anomalies(commodity, days=days)
        results["anomalies_detected"] = anomalies.get("total_anomalies", 0)
        
        # 2. Fix missing data
        missing_fix = fix_missing_data(commodity, days=days)
        results["missing_data_fixed"] = missing_fix.get("fixed", 0)
        
        # 3. Apply data cleaning rules
        if commodity:
            rules_result = apply_data_cleaning_rules(commodity, end_date=date.today(), start_date=date.today()-timedelta(days=days))
            results["rules_applied"] = rules_result.get("cleaned", 0)
        else:
            # Apply to all commodities
            from database_sql import get_all_commodities
            commodities = get_all_commodities()
            
            rules_applied = 0
            for comm in commodities:
                rules_result = apply_data_cleaning_rules(comm, end_date=date.today(), start_date=date.today()-timedelta(days=days))
                rules_applied += rules_result.get("cleaned", 0)
            
            results["rules_applied"] = rules_applied
        
        # 4. Validate price curves
        validation_issues = 0
        
        if commodity:
            validation = validate_price_curve(commodity, days=days)
            for region_result in validation.get("results", []):
                validation_issues += len(region_result.get("issues", []))
        else:
            # Validate all commodities
            from database_sql import get_all_commodities
            commodities = get_all_commodities()
            
            for comm in commodities:
                validation = validate_price_curve(comm, days=days)
                for region_result in validation.get("results", []):
                    validation_issues += len(region_result.get("issues", []))
        
        results["validation_issues"] = validation_issues
        
        # Add summary
        results["summary"] = f"Cleaned {days} days of data, detected {int(results['anomalies_detected'])} anomalies, " \
                             f"fixed {int(results['missing_data_fixed'])} missing data points, " \
                             f"applied rules to {int(results['rules_applied'])} points, " \
                             f"found {int(results['validation_issues'])} validation issues."
        
        logger.info(f"Data cleaning pipeline completed: {results['summary']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in data cleaning pipeline: {e}")
        return {"error": str(e)}


# Initialize data cleaning rules when module is imported
try:
    initialize_data_cleaning_rules()
except Exception as e:
    logger.error(f"Error initializing data cleaning rules: {e}")