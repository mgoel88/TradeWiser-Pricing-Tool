"""
Notification service for WIZX Agricultural Commodity Platform.

This module handles real-time price alerts and notifications for commodity price changes.
It supports in-app notifications and SMS alerts via Twilio.
"""

import os
import json
import logging
from datetime import datetime, timedelta
import time
import threading
import queue
import pandas as pd

# Database imports
from database_sql import get_all_commodities, get_commodity_prices, get_price_history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global notification queue
notification_queue = queue.Queue()

# For WhatsApp notifications, we'd normally need an API key or use
# a WhatsApp Business API solution. For now, we'll use a mock implementation.
WHATSAPP_ENABLED = False  # Set to True when WhatsApp credentials are configured

# Track the last notification time for each commodity+region to avoid duplicates
last_notification_times = {}

# Notification settings
DEFAULT_THRESHOLD_PERCENTAGE = 5.0  # Default threshold for price change alerts (5%)
CHECK_INTERVAL = 60 * 5  # Check for price changes every 5 minutes
NOTIFICATION_EXPIRY = 60 * 60 * 24  # Notifications expire after 24 hours

# Subscription storage (in-memory for now, can be moved to database)
# Format: {user_id: {commodity: [regions]}}
user_subscriptions = {}

# User contact information (would be stored in database in production)
# Format: {user_id: {phone: "phone_number", email: "email"}}
user_contacts = {}


class PriceAlert:
    """Class representing a price alert notification."""
    
    def __init__(self, commodity, region, old_price, new_price, change_percentage, timestamp=None):
        """Initialize a price alert."""
        self.commodity = commodity
        self.region = region
        self.old_price = old_price
        self.new_price = new_price
        self.change_percentage = change_percentage
        self.timestamp = timestamp or datetime.now()
        self.id = f"{commodity}_{region}_{self.timestamp.strftime('%Y%m%d%H%M%S')}"
        self.read = False
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "commodity": self.commodity,
            "region": self.region,
            "old_price": self.old_price,
            "new_price": self.new_price,
            "change_percentage": self.change_percentage,
            "timestamp": self.timestamp.isoformat(),
            "read": self.read
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create alert from dictionary."""
        alert = cls(
            commodity=data["commodity"],
            region=data["region"],
            old_price=data["old_price"],
            new_price=data["new_price"],
            change_percentage=data["change_percentage"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        alert.id = data["id"]
        alert.read = data.get("read", False)
        return alert


def send_whatsapp_notification(phone_number, message):
    """
    Send WhatsApp notification.
    
    Args:
        phone_number (str): The recipient's phone number
        message (str): The notification message
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not WHATSAPP_ENABLED:
        logger.warning("WhatsApp integration not enabled. Using in-app notifications only.")
        return False
    
    # In a real implementation, this would use a WhatsApp Business API 
    # or a third-party service that provides WhatsApp integration
    logger.info(f"Would send WhatsApp message to {phone_number}: {message}")
    return True


def subscribe_to_price_alerts(user_id, commodity, regions=None, phone=None, email=None):
    """
    Subscribe a user to price alerts for a commodity and regions.
    
    Args:
        user_id (str): User identifier
        commodity (str): Commodity name
        regions (list, optional): List of regions to monitor. If None, all regions.
        phone (str, optional): User's phone number for SMS alerts
        email (str, optional): User's email for email alerts
        
    Returns:
        bool: True if subscription added successfully
    """
    try:
        # Initialize user subscription if not exists
        if user_id not in user_subscriptions:
            user_subscriptions[user_id] = {}
        
        # Add commodity subscription
        if commodity not in user_subscriptions[user_id]:
            user_subscriptions[user_id][commodity] = regions or []
        elif regions:
            # Add regions if provided
            existing_regions = user_subscriptions[user_id][commodity]
            for region in regions:
                if region not in existing_regions:
                    existing_regions.append(region)
        
        # Store contact info if provided
        if phone or email:
            if user_id not in user_contacts:
                user_contacts[user_id] = {}
            
            if phone:
                user_contacts[user_id]["phone"] = phone
            
            if email:
                user_contacts[user_id]["email"] = email
        
        logger.info(f"User {user_id} subscribed to {commodity} alerts in regions: {regions or 'all'}")
        return True
    
    except Exception as e:
        logger.error(f"Error subscribing to price alerts: {e}")
        return False


def unsubscribe_from_price_alerts(user_id, commodity=None, region=None):
    """
    Unsubscribe a user from price alerts.
    
    Args:
        user_id (str): User identifier
        commodity (str, optional): Commodity name. If None, unsubscribe from all.
        region (str, optional): Region name. If None, unsubscribe from all regions for the commodity.
        
    Returns:
        bool: True if unsubscribed successfully
    """
    try:
        # Check if user has subscriptions
        if user_id not in user_subscriptions:
            return False
        
        # If no commodity specified, remove all subscriptions
        if commodity is None:
            user_subscriptions[user_id] = {}
            logger.info(f"User {user_id} unsubscribed from all price alerts")
            return True
        
        # If commodity not in subscriptions, nothing to do
        if commodity not in user_subscriptions[user_id]:
            return False
        
        # If no region specified, remove all regions for the commodity
        if region is None:
            del user_subscriptions[user_id][commodity]
            logger.info(f"User {user_id} unsubscribed from {commodity} price alerts")
            return True
        
        # Remove specific region
        regions = user_subscriptions[user_id][commodity]
        if region in regions:
            regions.remove(region)
            logger.info(f"User {user_id} unsubscribed from {commodity} price alerts in {region}")
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"Error unsubscribing from price alerts: {e}")
        return False


def get_user_subscriptions(user_id):
    """
    Get a user's price alert subscriptions.
    
    Args:
        user_id (str): User identifier
        
    Returns:
        dict: User subscriptions
    """
    return user_subscriptions.get(user_id, {})


def check_price_changes(threshold_percentage=DEFAULT_THRESHOLD_PERCENTAGE):
    """
    Check for significant price changes and generate alerts.
    
    Args:
        threshold_percentage (float): Percentage change threshold for alerts
        
    Returns:
        list: New price alerts generated
    """
    try:
        alerts = []
        
        # Get all commodities
        commodities = get_all_commodities()
        
        for commodity in commodities:
            # Get regions for this commodity
            from database_sql import get_regions_for_commodity
            regions = get_regions_for_commodity(commodity) or []
            
            # For each region, check price history
            for region in regions:
                # Get price history for last 2 days
                history = get_price_history(commodity, region, days=2)
                
                # Need at least 2 data points
                if not history or len(history) < 2:
                    continue
                
                # Get the two most recent prices
                sorted_history = sorted(history, key=lambda x: x.get("date", ""))
                
                if len(sorted_history) >= 2:
                    latest = sorted_history[-1]
                    previous = sorted_history[-2]
                    
                    # Calculate percentage change
                    new_price = latest.get("price", 0)
                    old_price = previous.get("price", 0)
                    
                    if old_price > 0:
                        change_percentage = ((new_price - old_price) / old_price) * 100
                        
                        # Check if change exceeds threshold
                        if abs(change_percentage) >= threshold_percentage:
                            # Check if we've already notified for this price change recently
                            key = f"{commodity}_{region}"
                            last_time = last_notification_times.get(key)
                            
                            current_time = datetime.now()
                            if not last_time or (current_time - last_time).total_seconds() > 3600:  # 1 hour cooldown
                                # Create alert
                                alert = PriceAlert(
                                    commodity=commodity,
                                    region=region,
                                    old_price=old_price,
                                    new_price=new_price,
                                    change_percentage=change_percentage
                                )
                                
                                # Add to results
                                alerts.append(alert)
                                
                                # Update last notification time
                                last_notification_times[key] = current_time
                                
                                # Add to notification queue
                                notification_queue.put(alert)
                                
                                logger.info(f"Price alert generated: {commodity} in {region} changed by {change_percentage:.2f}%")
        
        return alerts
    
    except Exception as e:
        logger.error(f"Error checking price changes: {e}")
        return []


def get_notifications(user_id=None, mark_as_read=False, limit=10):
    """
    Get recent notifications from the queue.
    
    Args:
        user_id (str, optional): User ID to filter notifications
        mark_as_read (bool): Whether to mark retrieved notifications as read
        limit (int): Maximum number of notifications to return
        
    Returns:
        list: Recent notifications
    """
    # In a real implementation, this would read from a database
    # For simplicity, we're just reading from the queue
    notifications = []
    
    # Copy items from queue without removing them
    with notification_queue.mutex:
        queue_items = list(notification_queue.queue)
    
    # Filter and limit
    for item in queue_items[:limit]:
        # If user_id provided, check if user is subscribed to this alert
        if user_id:
            subscriptions = get_user_subscriptions(user_id)
            commodity = item.commodity
            region = item.region
            
            # Check if user is subscribed to this commodity and region
            if commodity in subscriptions:
                regions = subscriptions[commodity]
                if not regions or region in regions:
                    notifications.append(item)
                    if mark_as_read:
                        item.read = True
        else:
            # No user filtering
            notifications.append(item)
            if mark_as_read:
                item.read = True
    
    return notifications


def notification_monitor():
    """Background thread to periodically check for price changes."""
    logger.info("Starting notification monitoring thread")
    
    while True:
        try:
            # Check for price changes
            check_price_changes()
            
            # Process notifications
            process_notifications()
            
            # Sleep until next check
            time.sleep(CHECK_INTERVAL)
        
        except Exception as e:
            logger.error(f"Error in notification monitor: {e}")
            time.sleep(60)  # Sleep for a minute on error


def process_notifications():
    """Process notifications in the queue and send alerts to subscribed users."""
    try:
        # Get all items without removing from queue
        with notification_queue.mutex:
            notifications = list(notification_queue.queue)
        
        # Process each notification
        for notification in notifications:
            # Check expiry
            age = (datetime.now() - notification.timestamp).total_seconds()
            if age > NOTIFICATION_EXPIRY:
                # Remove expired notification
                notification_queue.queue.remove(notification)
                continue
            
            # Find users subscribed to this commodity and region
            commodity = notification.commodity
            region = notification.region
            
            for user_id, subscriptions in user_subscriptions.items():
                if commodity in subscriptions:
                    regions = subscriptions[commodity]
                    if not regions or region in regions:
                        # User is subscribed to this alert
                        try:
                            # Get user contact info
                            contact_info = user_contacts.get(user_id, {})
                            phone = contact_info.get("phone")
                            
                            # Create message
                            direction = "increased" if notification.change_percentage > 0 else "decreased"
                            message = (
                                f"WIZX Price Alert: {commodity} prices in {region} have {direction} "
                                f"by {abs(notification.change_percentage):.2f}%. "
                                f"New price: ₹{notification.new_price:.2f}."
                            )
                            
                            # Send WhatsApp message if phone number available
                            if phone:
                                send_whatsapp_notification(phone, message)
                        
                        except Exception as e:
                            logger.error(f"Error sending notification to user {user_id}: {e}")
    
    except Exception as e:
        logger.error(f"Error processing notifications: {e}")


# Start notification monitoring thread
def start_monitoring():
    """Start the notification monitoring thread."""
    thread = threading.Thread(target=notification_monitor, daemon=True)
    thread.start()
    logger.info("Notification monitoring started")


# Example function to set up some test subscriptions
def setup_test_subscriptions():
    """Set up test subscriptions for demonstration."""
    # Test user
    test_user = "user123"
    
    # Subscribe to wheat in all regions
    subscribe_to_price_alerts(test_user, "Wheat")
    
    # Subscribe to rice in specific regions
    subscribe_to_price_alerts(test_user, "Rice", ["Karnataka", "Maharashtra"])
    
    # Add contact info
    user_contacts[test_user] = {
        "phone": "+1234567890",  # Test phone number
        "email": "user@example.com"
    }
    
    logger.info(f"Test subscriptions set up for user {test_user}")


# Initialize when imported
if __name__ == "__main__":
    # For testing
    setup_test_subscriptions()
    start_monitoring()
else:
    # Import-time initialization
    logger.info("Initializing notification service")