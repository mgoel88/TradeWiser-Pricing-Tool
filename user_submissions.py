"""
User submissions module for handling user-contributed data and rewards.
"""

import os
import logging
import json
import hashlib
import pandas as pd
from datetime import datetime, date, timedelta

from database_sql import (
    Session, UserSubmission, Commodity, Region, PricePoint, DataSource,
    save_user_input, verify_user_submission, get_user_rewards
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
USER_SUBMISSIONS_DIR = os.path.join(DATA_DIR, "user_submissions")

# Ensure directories exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(USER_SUBMISSIONS_DIR):
    os.makedirs(USER_SUBMISSIONS_DIR)


def generate_user_id(email):
    """
    Generate a user ID from an email address.
    
    Args:
        email (str): Email address
        
    Returns:
        str: User ID
    """
    # Use SHA-256 for hashing
    hash_obj = hashlib.sha256(email.lower().encode())
    # Return first 12 characters of the hash
    return hash_obj.hexdigest()[:12]


def submit_price_data(
    user_email, commodity, region, price, date_val=None, 
    quality_params=None, market=None, source_details=None
):
    """
    Submit price data from a user.
    
    Args:
        user_email (str): User email for identification and rewards
        commodity (str): Commodity name
        region (str): Region name
        price (float): Price value
        date_val (date, optional): Date of the price data
        quality_params (dict, optional): Quality parameters
        market (str, optional): Specific market or mandi
        source_details (dict, optional): Additional source details
        
    Returns:
        dict: Submission result with submission ID
    """
    # Generate user ID
    user_id = generate_user_id(user_email)
    
    # Default date to today
    if date_val is None:
        date_val = date.today()
    
    session = Session()
    
    try:
        # Get commodity
        commodity_obj = session.query(Commodity).filter(Commodity.name == commodity).first()
        
        if not commodity_obj:
            logger.warning(f"Commodity {commodity} not found")
            return {"success": False, "message": f"Commodity {commodity} not found"}
        
        # Prepare submission data
        submission_data = {
            "market": market,
            "source_details": source_details or {},
            "submission_time": datetime.now().isoformat(),
            "user_email_hash": hashlib.sha256(user_email.lower().encode()).hexdigest()
        }
        
        # Create submission
        submission = UserSubmission(
            user_id=user_id,
            commodity_id=commodity_obj.id,
            region_name=region,
            date=date_val if isinstance(date_val, date) else datetime.strptime(date_val, "%Y-%m-%d").date(),
            price=float(price),
            quality_parameters=quality_params or {},
            is_verified=False,
            verification_score=0.0,
            submission_data=submission_data
        )
        
        session.add(submission)
        session.flush()  # To get the ID
        
        # Save submission to file for backup
        file_path = os.path.join(USER_SUBMISSIONS_DIR, f"submission_{submission.id}.json")
        
        with open(file_path, 'w') as f:
            json.dump({
                "id": submission.id,
                "user_id": user_id,
                "commodity": commodity,
                "region": region,
                "date": date_val.isoformat() if isinstance(date_val, date) else date_val,
                "price": price,
                "quality_parameters": quality_params or {},
                "submission_data": submission_data
            }, f, indent=2)
        
        session.commit()
        
        logger.info(f"User submission created: ID {submission.id}, commodity {commodity}, region {region}")
        
        return {
            "success": True,
            "submission_id": submission.id,
            "message": "Thank you for your submission. It will be verified shortly."
        }
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error creating user submission: {e}")
        return {"success": False, "message": f"Submission failed: {str(e)}"}
    finally:
        session.close()


def get_pending_submissions(limit=20):
    """
    Get pending submissions for verification.
    
    Args:
        limit (int): Maximum number of submissions to retrieve
        
    Returns:
        list: Pending submissions
    """
    session = Session()
    
    try:
        # Get unverified submissions
        submissions = session.query(UserSubmission).\
            filter(UserSubmission.is_verified == False).\
            order_by(UserSubmission.created_at).\
            limit(limit).\
            all()
        
        results = []
        
        for sub in submissions:
            # Get commodity name
            commodity_name = session.query(Commodity.name).\
                filter(Commodity.id == sub.commodity_id).\
                scalar()
            
            # Get similar prices for comparison
            similar_prices = session.query(PricePoint).\
                join(Region, Region.id == PricePoint.region_id).\
                filter(
                    PricePoint.commodity_id == sub.commodity_id,
                    Region.name == sub.region_name,
                    PricePoint.date >= sub.date - timedelta(days=7),
                    PricePoint.date <= sub.date + timedelta(days=7)
                ).\
                all()
            
            # Calculate median price and range
            if similar_prices:
                prices = [pp.price for pp in similar_prices]
                median_price = pd.Series(prices).median()
                min_price = min(prices)
                max_price = max(prices)
                price_diff_pct = abs(sub.price - median_price) / median_price * 100 if median_price else 0
            else:
                median_price = None
                min_price = None
                max_price = None
                price_diff_pct = None
            
            results.append({
                "id": sub.id,
                "user_id": sub.user_id,
                "commodity": commodity_name,
                "region": sub.region_name,
                "date": sub.date,
                "price": sub.price,
                "quality_parameters": sub.quality_parameters,
                "submission_data": sub.submission_data,
                "created_at": sub.created_at,
                "comparison": {
                    "median_price": median_price,
                    "min_price": min_price,
                    "max_price": max_price,
                    "price_diff_pct": price_diff_pct,
                    "similar_prices_count": len(similar_prices)
                }
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting pending submissions: {e}")
        return []
    finally:
        session.close()


def get_user_submission_status(user_id_or_email):
    """
    Get status of a user's submissions.
    
    Args:
        user_id_or_email (str): User ID or email address
        
    Returns:
        dict: User submission status
    """
    # Convert email to user ID if needed
    if '@' in user_id_or_email:
        user_id = generate_user_id(user_id_or_email)
    else:
        user_id = user_id_or_email
    
    session = Session()
    
    try:
        # Get user submissions
        submissions = session.query(UserSubmission).\
            filter(UserSubmission.user_id == user_id).\
            order_by(UserSubmission.created_at.desc()).\
            all()
        
        if not submissions:
            return {
                "user_id": user_id,
                "total_submissions": 0,
                "message": "No submissions found for this user"
            }
        
        # Get rewards info
        rewards = get_user_rewards(user_id)
        
        # Format submission data
        submissions_data = []
        
        for sub in submissions:
            # Get commodity name
            commodity_name = session.query(Commodity.name).\
                filter(Commodity.id == sub.commodity_id).\
                scalar()
            
            submissions_data.append({
                "id": sub.id,
                "commodity": commodity_name,
                "region": sub.region_name,
                "date": sub.date,
                "price": sub.price,
                "is_verified": sub.is_verified,
                "verification_score": sub.verification_score,
                "reward_points": sub.reward_points,
                "created_at": sub.created_at,
                "verification_notes": sub.verification_notes
            })
        
        # Calculate statistics
        verified_count = sum(1 for s in submissions_data if s["is_verified"])
        rejected_count = sum(1 for s in submissions_data if not s["is_verified"] and s["verification_score"] is not None)
        pending_count = sum(1 for s in submissions_data if not s["is_verified"] and s["verification_score"] is None)
        
        return {
            "user_id": user_id,
            "total_submissions": len(submissions),
            "verified_submissions": verified_count,
            "rejected_submissions": rejected_count,
            "pending_submissions": pending_count,
            "verification_rate": verified_count / len(submissions) if len(submissions) > 0 else 0,
            "total_reward_points": rewards.get("total_points", 0),
            "recent_submissions": submissions_data[:10],  # Show only the 10 most recent
            "commodity_breakdown": rewards.get("commodity_breakdown", {})
        }
        
    except Exception as e:
        logger.error(f"Error getting user submission status: {e}")
        return {"error": str(e)}
    finally:
        session.close()


def auto_verify_submission(submission_id):
    """
    Automatically verify a submission based on existing price data.
    
    Args:
        submission_id (int): Submission ID
        
    Returns:
        dict: Verification result
    """
    session = Session()
    
    try:
        # Get submission
        submission = session.query(UserSubmission).filter(UserSubmission.id == submission_id).first()
        
        if not submission:
            logger.warning(f"Submission {submission_id} not found")
            return {"success": False, "message": "Submission not found"}
        
        # Get similar prices for comparison
        similar_prices = session.query(PricePoint).\
            join(Region, Region.id == PricePoint.region_id).\
            filter(
                PricePoint.commodity_id == submission.commodity_id,
                Region.name == submission.region_name,
                PricePoint.date >= submission.date - timedelta(days=7),
                PricePoint.date <= submission.date + timedelta(days=7),
                PricePoint.is_verified == True
            ).\
            all()
        
        if not similar_prices:
            # Not enough data for automatic verification
            return {
                "success": False,
                "message": "Not enough recent verified price data for comparison",
                "verification_score": 0.5,  # Neutral score
                "requires_manual_review": True
            }
        
        # Calculate median price and range
        prices = [pp.price for pp in similar_prices]
        median_price = pd.Series(prices).median()
        price_range = max(prices) - min(prices)
        
        # Calculate price difference percentage
        price_diff_pct = abs(submission.price - median_price) / median_price * 100
        
        # Determine verification score and result
        if price_diff_pct <= 5:  # Within 5% of median
            verification_score = 0.9
            verification_result = True
            verification_notes = "Auto-verified: Price within 5% of median recent prices"
        elif price_diff_pct <= 10:  # Within 10% of median
            verification_score = 0.7
            verification_result = True
            verification_notes = "Auto-verified: Price within 10% of median recent prices"
        elif price_diff_pct <= 20:  # Within 20% of median
            verification_score = 0.5
            verification_result = False  # Needs manual review
            verification_notes = "Requires review: Price deviation between 10-20% from median"
        else:  # More than 20% difference
            verification_score = 0.2
            verification_result = False
            verification_notes = f"Requires review: Price deviates {price_diff_pct:.1f}% from median"
        
        # Save verification result if we're auto-approving
        if verification_result and verification_score >= 0.7:
            verify_user_submission(
                submission_id=submission_id,
                verification_result=verification_result,
                verification_score=verification_score,
                notes=verification_notes
            )
            
            return {
                "success": True,
                "verified": verification_result,
                "verification_score": verification_score,
                "message": verification_notes,
                "comparison": {
                    "submission_price": submission.price,
                    "median_price": median_price,
                    "price_diff_pct": price_diff_pct,
                    "similar_prices_count": len(similar_prices)
                }
            }
        else:
            # Update score but don't auto-verify
            submission.verification_score = verification_score
            submission.verification_notes = verification_notes
            session.commit()
            
            return {
                "success": True,
                "verified": False,
                "verification_score": verification_score,
                "message": verification_notes,
                "requires_manual_review": True,
                "comparison": {
                    "submission_price": submission.price,
                    "median_price": median_price,
                    "price_diff_pct": price_diff_pct,
                    "similar_prices_count": len(similar_prices)
                }
            }
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error auto-verifying submission: {e}")
        return {"success": False, "message": f"Verification failed: {str(e)}"}
    finally:
        session.close()


def calculate_reward_points(submission):
    """
    Calculate reward points for a verified submission.
    
    Args:
        submission (UserSubmission): The submission object
        
    Returns:
        int: Reward points
    """
    # Base points for any submission
    points = 10
    
    # Additional points for completeness
    if submission.quality_parameters and len(submission.quality_parameters) >= 3:
        points += 5
    
    if submission.submission_data and submission.submission_data.get("market"):
        points += 3
    
    if submission.submission_data and submission.submission_data.get("source_details"):
        points += 2
    
    # Additional points for rare commodities/regions
    session = Session()
    
    try:
        # Count submissions for this commodity-region in the last 30 days
        count = session.query(func.count(UserSubmission.id)).\
            filter(
                UserSubmission.commodity_id == submission.commodity_id,
                UserSubmission.region_name == submission.region_name,
                UserSubmission.date >= submission.date - timedelta(days=30),
                UserSubmission.is_verified == True
            ).\
            scalar() or 0
        
        if count <= 1:
            # First submission for this commodity-region in last 30 days
            points += 10
        elif count <= 5:
            # Relatively rare
            points += 5
    finally:
        session.close()
    
    return points


def get_leaderboard(days=30, limit=20):
    """
    Get leaderboard of top contributors.
    
    Args:
        days (int): Number of days to include
        limit (int): Maximum number of users to return
        
    Returns:
        list: Leaderboard data
    """
    session = Session()
    
    try:
        # Get start date
        start_date = date.today() - timedelta(days=days)
        
        # Get top users by points
        results = session.query(
            UserSubmission.user_id,
            func.count(UserSubmission.id).label("submissions"),
            func.sum(UserSubmission.reward_points).label("points")
        ).\
            filter(
                UserSubmission.date >= start_date,
                UserSubmission.is_verified == True
            ).\
            group_by(UserSubmission.user_id).\
            order_by(func.sum(UserSubmission.reward_points).desc()).\
            limit(limit).\
            all()
        
        leaderboard = []
        
        for i, (user_id, submissions, points) in enumerate(results):
            # Get user's top commodities
            top_commodities = session.query(
                Commodity.name,
                func.count(UserSubmission.id).label("count")
            ).\
                join(UserSubmission, UserSubmission.commodity_id == Commodity.id).\
                filter(
                    UserSubmission.user_id == user_id,
                    UserSubmission.date >= start_date,
                    UserSubmission.is_verified == True
                ).\
                group_by(Commodity.name).\
                order_by(func.count(UserSubmission.id).desc()).\
                limit(3).\
                all()
            
            leaderboard.append({
                "rank": i + 1,
                "user_id": user_id,
                "submissions": submissions,
                "points": points or 0,
                "top_commodities": [{"name": name, "submissions": count} for name, count in top_commodities]
            })
        
        return {
            "time_period": f"Last {days} days",
            "leaderboard": leaderboard,
            "total_submissions": sum(item["submissions"] for item in leaderboard),
            "total_points": sum(item["points"] for item in leaderboard)
        }
        
    except Exception as e:
        logger.error(f"Error getting leaderboard: {e}")
        return {"error": str(e)}
    finally:
        session.close()


def bulk_import_submissions(file_path, source_name="Imported Data"):
    """
    Bulk import submissions from a CSV or Excel file.
    
    Args:
        file_path (str): Path to the file
        source_name (str): Name of the source
        
    Returns:
        dict: Import results
    """
    # Determine file type
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        data = pd.read_excel(file_path)
    else:
        return {"success": False, "message": "Unsupported file format"}
    
    # Validate required columns
    required_columns = ['commodity', 'region', 'date', 'price']
    
    for col in required_columns:
        if col not in data.columns:
            return {"success": False, "message": f"Missing required column: {col}"}
    
    # Get admin user
    admin_user_id = "admin_import"
    
    # Process each row
    processed = 0
    errors = []
    
    session = Session()
    
    try:
        # Get or create data source
        source = session.query(DataSource).filter(DataSource.name == source_name).first()
        
        if not source:
            source = DataSource(
                name=source_name,
                description=f"Imported data from {file_path}",
                source_type="import"
            )
            session.add(source)
            session.flush()
        
        for i, row in data.iterrows():
            try:
                # Parse date
                if isinstance(row['date'], str):
                    try:
                        date_val = datetime.strptime(row['date'], "%Y-%m-%d").date()
                    except ValueError:
                        try:
                            date_val = datetime.strptime(row['date'], "%d/%m/%Y").date()
                        except ValueError:
                            errors.append(f"Row {i+1}: Invalid date format")
                            continue
                else:
                    date_val = row['date'].date() if hasattr(row['date'], 'date') else row['date']
                
                # Get or validate commodity
                commodity = session.query(Commodity).filter(Commodity.name == row['commodity']).first()
                
                if not commodity:
                    errors.append(f"Row {i+1}: Commodity '{row['commodity']}' not found")
                    continue
                
                # Extract quality parameters if available
                quality_params = {}
                for col in data.columns:
                    if col.startswith('quality_'):
                        param_name = col.replace('quality_', '')
                        if not pd.isna(row[col]):
                            quality_params[param_name] = row[col]
                
                # Create submission
                submission = UserSubmission(
                    user_id=admin_user_id,
                    commodity_id=commodity.id,
                    region_name=row['region'],
                    date=date_val,
                    price=float(row['price']),
                    quality_parameters=quality_params,
                    is_verified=True,  # Auto-verify imported data
                    verification_score=0.9,
                    submission_data={"source_file": file_path, "row": i+1}
                )
                
                session.add(submission)
                processed += 1
                
            except Exception as e:
                errors.append(f"Row {i+1}: {str(e)}")
        
        session.commit()
        
        return {
            "success": True,
            "processed": processed,
            "errors": errors,
            "message": f"Imported {processed} submissions from {file_path}"
        }
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error importing submissions: {e}")
        return {"success": False, "message": f"Import failed: {str(e)}"}
    finally:
        session.close()


def export_submissions(start_date=None, end_date=None, commodity=None, region=None, file_path=None):
    """
    Export submissions to a CSV file.
    
    Args:
        start_date (date, optional): Start date
        end_date (date, optional): End date
        commodity (str, optional): Commodity name
        region (str, optional): Region name
        file_path (str, optional): Output file path
        
    Returns:
        str: Path to the exported file
    """
    if end_date is None:
        end_date = date.today()
    
    if start_date is None:
        start_date = end_date - timedelta(days=30)
    
    session = Session()
    
    try:
        # Build query
        query = session.query(
            UserSubmission, Commodity.name.label('commodity_name')
        ).\
            join(Commodity, Commodity.id == UserSubmission.commodity_id).\
            filter(
                UserSubmission.date >= start_date,
                UserSubmission.date <= end_date
            )
        
        if commodity:
            query = query.filter(Commodity.name == commodity)
        
        if region:
            query = query.filter(UserSubmission.region_name == region)
        
        # Get submissions
        results = query.all()
        
        if not results:
            return {"success": False, "message": "No submissions found matching the criteria"}
        
        # Prepare data for export
        data = []
        
        for sub, commodity_name in results:
            row = {
                "id": sub.id,
                "user_id": sub.user_id,
                "commodity": commodity_name,
                "region": sub.region_name,
                "date": sub.date,
                "price": sub.price,
                "is_verified": sub.is_verified,
                "verification_score": sub.verification_score,
                "reward_points": sub.reward_points,
                "created_at": sub.created_at
            }
            
            # Add quality parameters
            if sub.quality_parameters:
                for param, value in sub.quality_parameters.items():
                    row[f"quality_{param}"] = value
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate file path if not provided
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            commodity_str = commodity or "all"
            region_str = region or "all"
            file_path = os.path.join(
                DATA_DIR, 
                f"export_{commodity_str}_{region_str}_{timestamp}.csv"
            )
        
        # Export to CSV
        df.to_csv(file_path, index=False)
        
        return {
            "success": True,
            "file_path": file_path,
            "record_count": len(data),
            "message": f"Exported {len(data)} records to {file_path}"
        }
        
    except Exception as e:
        logger.error(f"Error exporting submissions: {e}")
        return {"success": False, "message": f"Export failed: {str(e)}"}
    finally:
        session.close()