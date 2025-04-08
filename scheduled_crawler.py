"""
Scheduled Data Crawler for Agricultural Commodity Pricing

This module provides functionality to automatically update commodity data
from public sources at regular intervals through a background process.
"""

import time
import logging
import threading
import json
import os
from datetime import datetime, timedelta

# Local imports
from data_crawler import (
    fetch_agmarknet_data, fetch_enam_data, crawl_global_price_indices,
    fetch_commodity_list, get_markets_for_state, fetch_historic_price_data,
    get_website_text_content
)
from database_sql import store_crawled_data, update_price_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
CRAWL_LOG_FILE = os.path.join(DATA_DIR, "crawl_log.json")
CONFIG_FILE = os.path.join(DATA_DIR, "crawler_config.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Thread-safe lock for coordinating crawl jobs
crawl_lock = threading.Lock()

def load_config():
    """Load crawler configuration from file."""
    if not os.path.exists(CONFIG_FILE):
        # Default configuration
        config = {
            "enabled": True,
            "sources": {
                "agmarknet": {
                    "enabled": True,
                    "schedule": "daily",
                    "interval_hours": 24,
                    "commodities": [],  # Empty means all available
                    "regions": [],      # Empty means all available
                    "days_to_fetch": 7
                },
                "enam": {
                    "enabled": True,
                    "schedule": "daily",
                    "interval_hours": 24,
                    "commodities": [],
                    "markets": []
                },
                "global_indices": {
                    "enabled": True,
                    "schedule": "daily",
                    "interval_hours": 24
                }
            },
            "historical_data": {
                "enabled": True,
                "schedule": "weekly",
                "years_back": 5
            }
        }
        
        # Save default config
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info("Created default crawler configuration")
        return config
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        logger.info("Loaded crawler configuration")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

def save_config(config):
    """Save crawler configuration to file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info("Saved crawler configuration")
        return True
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return False

def log_crawl_activity(source, status, details=None):
    """Log crawl activity to file for tracking purposes."""
    if not os.path.exists(CRAWL_LOG_FILE):
        log_data = []
    else:
        try:
            with open(CRAWL_LOG_FILE, 'r') as f:
                log_data = json.load(f)
        except:
            log_data = []
    
    # Add new log entry
    log_entry = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "source": source,
        "status": status
    }
    
    if details:
        log_entry["details"] = details
    
    log_data.append(log_entry)
    
    # Keep only the most recent 1000 log entries
    if len(log_data) > 1000:
        log_data = log_data[-1000:]
    
    # Save log data
    try:
        with open(CRAWL_LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving crawl log: {e}")

def crawl_agmarknet(config=None):
    """Crawl Agmarknet data based on configuration."""
    logger.info("Starting Agmarknet data crawl")
    
    if config is None:
        config = load_config()
    
    if config is None or not config.get("enabled", False) or not config.get("sources", {}).get("agmarknet", {}).get("enabled", False):
        logger.info("Agmarknet crawling is disabled in configuration")
        return False
    
    agmarknet_config = config["sources"]["agmarknet"]
    days = agmarknet_config.get("days_to_fetch", 7)
    commodities = agmarknet_config.get("commodities", [])
    regions = agmarknet_config.get("regions", [])
    
    try:
        with crawl_lock:
            if not commodities:
                # Fetch all available commodities
                commodities = fetch_commodity_list(source="agmarknet")
            
            all_data = []
            
            # For each commodity and region, fetch data
            for commodity in commodities:
                for region in regions if regions else [None]:
                    logger.info(f"Fetching Agmarknet data for {commodity} in {region if region else 'all regions'}")
                    
                    data = fetch_agmarknet_data(
                        commodity=commodity,
                        start_date=datetime.now() - timedelta(days=days),
                        end_date=datetime.now(),
                        state=region
                    )
                    
                    if data:
                        all_data.extend(data)
            
            # Store data in the database
            if all_data:
                records_stored = store_crawled_data(all_data, source="agmarknet")
                logger.info(f"Stored {records_stored} records from Agmarknet")
                log_crawl_activity("agmarknet", "success", {"records": records_stored})
                return records_stored
            else:
                logger.warning("No data retrieved from Agmarknet")
                log_crawl_activity("agmarknet", "empty")
                return 0
    
    except Exception as e:
        logger.error(f"Error crawling Agmarknet data: {e}")
        log_crawl_activity("agmarknet", "error", {"error": str(e)})
        return False

def crawl_enam(config=None):
    """Crawl eNAM data based on configuration."""
    logger.info("Starting eNAM data crawl")
    
    if config is None:
        config = load_config()
    
    if config is None or not config.get("enabled", False) or not config.get("sources", {}).get("enam", {}).get("enabled", False):
        logger.info("eNAM crawling is disabled in configuration")
        return False
    
    enam_config = config["sources"]["enam"]
    commodities = enam_config.get("commodities", [])
    markets = enam_config.get("markets", [])
    
    try:
        with crawl_lock:
            if not commodities:
                # Fetch all available commodities
                commodities = fetch_commodity_list(source="enam")
            
            all_data = []
            
            # For each commodity and market, fetch data
            for commodity in commodities:
                for market in markets if markets else [None]:
                    logger.info(f"Fetching eNAM data for {commodity} in {market if market else 'all markets'}")
                    
                    data = fetch_enam_data(
                        commodity=commodity,
                        market=market,
                        date=datetime.now() - timedelta(days=1)  # Yesterday's data
                    )
                    
                    if data:
                        all_data.extend(data)
            
            # Store data in the database
            if all_data:
                records_stored = store_crawled_data(all_data, source="enam")
                logger.info(f"Stored {records_stored} records from eNAM")
                log_crawl_activity("enam", "success", {"records": records_stored})
                return records_stored
            else:
                logger.warning("No data retrieved from eNAM")
                log_crawl_activity("enam", "empty")
                return 0
    
    except Exception as e:
        logger.error(f"Error crawling eNAM data: {e}")
        log_crawl_activity("enam", "error", {"error": str(e)})
        return False

def crawl_global_indices(config=None):
    """Crawl global commodity indices based on configuration."""
    logger.info("Starting global indices crawl")
    
    if config is None:
        config = load_config()
    
    if config is None or not config.get("enabled", False) or not config.get("sources", {}).get("global_indices", {}).get("enabled", False):
        logger.info("Global indices crawling is disabled in configuration")
        return False
    
    try:
        with crawl_lock:
            # Fetch global indices data
            data = crawl_global_price_indices()
            
            if data:
                # Process and store the data
                # This would normally involve transforming the data and
                # storing it in the database
                logger.info(f"Retrieved global indices data: {len(data)} indices")
                log_crawl_activity("global_indices", "success", {"indices": list(data.keys())})
                return True
            else:
                logger.warning("No global indices data retrieved")
                log_crawl_activity("global_indices", "empty")
                return False
    
    except Exception as e:
        logger.error(f"Error crawling global indices: {e}")
        log_crawl_activity("global_indices", "error", {"error": str(e)})
        return False

def update_historical_data(config=None):
    """Update historical price data for all commodities."""
    logger.info("Starting historical data update")
    
    if config is None:
        config = load_config()
    
    if config is None or not config.get("enabled", False) or not config.get("historical_data", {}).get("enabled", False):
        logger.info("Historical data update is disabled in configuration")
        return False
    
    years_back = config["historical_data"].get("years_back", 5)
    
    try:
        with crawl_lock:
            # Get all commodities
            commodities = fetch_commodity_list()
            
            all_data = []
            
            # For each commodity, update historical data
            for commodity in commodities:
                logger.info(f"Updating historical data for {commodity}")
                
                # Get historical data
                data = fetch_historic_price_data(
                    commodity=commodity,
                    years_back=years_back
                )
                
                if data:
                    all_data.extend(data)
            
            # Store historical data
            if all_data:
                records_stored = update_price_data(all_data, data_type="historical")
                logger.info(f"Updated {records_stored} historical price records")
                log_crawl_activity("historical", "success", {"records": records_stored})
                return records_stored
            else:
                logger.warning("No historical data retrieved")
                log_crawl_activity("historical", "empty")
                return 0
    
    except Exception as e:
        logger.error(f"Error updating historical data: {e}")
        log_crawl_activity("historical", "error", {"error": str(e)})
        return False

def crawl_all_sources():
    """Run all crawl jobs based on configuration."""
    logger.info("Starting complete data crawl for all sources")
    
    config = load_config()
    if config is None or not config.get("enabled", False):
        logger.info("Data crawling is disabled in configuration")
        return False
    
    results = {
        "agmarknet": crawl_agmarknet(config),
        "enam": crawl_enam(config),
        "global_indices": crawl_global_indices(config),
        "historical": update_historical_data(config) if config.get("historical_data", {}).get("enabled", False) else "skipped"
    }
    
    logger.info(f"Complete data crawl finished with results: {results}")
    return results

class CrawlerScheduler:
    """Class to handle scheduling and running recurring crawls."""
    
    def __init__(self):
        self.running = False
        self.config = load_config()
        self.jobs = []
        self.thread = None
    
    def setup_jobs(self):
        """Set up scheduled jobs based on configuration."""
        self.jobs = []
        
        # Make sure we have a valid configuration
        if self.config is None:
            logger.warning("No valid configuration found, cannot setup jobs")
            return
        
        # Agmarknet schedule
        if self.config.get("sources", {}).get("agmarknet", {}).get("enabled", False):
            interval_hours = self.config.get("sources", {}).get("agmarknet", {}).get("interval_hours", 24)
            self.jobs.append({
                "name": "agmarknet",
                "interval": interval_hours * 3600,  # Convert to seconds
                "last_run": None,
                "func": crawl_agmarknet
            })
            logger.info(f"Scheduled Agmarknet crawl every {interval_hours} hours")
        
        # eNAM schedule
        if self.config.get("sources", {}).get("enam", {}).get("enabled", False):
            interval_hours = self.config.get("sources", {}).get("enam", {}).get("interval_hours", 24)
            self.jobs.append({
                "name": "enam",
                "interval": interval_hours * 3600,
                "last_run": None,
                "func": crawl_enam
            })
            logger.info(f"Scheduled eNAM crawl every {interval_hours} hours")
        
        # Global indices schedule
        if self.config.get("sources", {}).get("global_indices", {}).get("enabled", False):
            interval_hours = self.config.get("sources", {}).get("global_indices", {}).get("interval_hours", 24)
            self.jobs.append({
                "name": "global_indices",
                "interval": interval_hours * 3600,
                "last_run": None,
                "func": crawl_global_indices
            })
            logger.info(f"Scheduled global indices crawl every {interval_hours} hours")
        
        # Historical data schedule - weekly
        if self.config.get("historical_data", {}).get("enabled", False):
            if self.config.get("historical_data", {}).get("schedule") == "weekly":
                self.jobs.append({
                    "name": "historical",
                    "interval": 7 * 24 * 3600,  # 7 days in seconds
                    "last_run": None,
                    "func": update_historical_data
                })
                logger.info("Scheduled historical data update weekly")
            else:
                # Default to monthly
                self.jobs.append({
                    "name": "historical",
                    "interval": 30 * 24 * 3600,  # 30 days in seconds
                    "last_run": None,
                    "func": update_historical_data
                })
                logger.info("Scheduled historical data update monthly")
    
    def run_pending_jobs(self):
        """Run any jobs that are due to be executed."""
        now = time.time()
        
        for job in self.jobs:
            if job["last_run"] is None or (now - job["last_run"]) >= job["interval"]:
                logger.info(f"Running scheduled job: {job['name']}")
                try:
                    job["func"](self.config)
                    job["last_run"] = now
                except Exception as e:
                    logger.error(f"Error running job {job['name']}: {e}")
    
    def scheduler_loop(self):
        """Main scheduler loop to run jobs at their scheduled times."""
        while self.running:
            try:
                self.run_pending_jobs()
                time.sleep(60)  # Check for pending jobs every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def start(self, run_initial=True):
        """Start the scheduler."""
        if self.thread and self.thread.is_alive():
            logger.warning("Scheduler is already running")
            return False
        
        self.running = True
        self.setup_jobs()
        
        # Run initial crawl if requested
        if run_initial:
            logger.info("Running initial data crawl")
            threading.Thread(target=crawl_all_sources).start()
        
        # Start scheduler thread
        self.thread = threading.Thread(target=self.scheduler_loop, daemon=True)
        self.thread.start()
        
        logger.info("Scheduler started successfully")
        return True
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        
        if self.thread:
            try:
                self.thread.join(timeout=10)
                logger.info("Scheduler stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping scheduler: {e}")
        
        return not self.running

    def update_config(self, new_config):
        """Update the configuration and restart scheduler."""
        if self.config is None:
            self.config = {}
            
        for key, value in new_config.items():
            if key in self.config:
                if isinstance(value, dict) and isinstance(self.config[key], dict):
                    # Recursively update nested dictionaries
                    for subkey, subvalue in value.items():
                        self.config[key][subkey] = subvalue
                else:
                    self.config[key] = value
            else:
                self.config[key] = value
        
        # Save the updated configuration
        save_config(self.config)
        
        # Restart with new configuration
        was_running = self.running
        if was_running:
            self.stop()
        
        if was_running:
            self.start(run_initial=False)
        
        logger.info("Configuration updated and scheduler reconfigured")
        return self.config
    
    def get_status(self):
        """Get the current status of the crawler scheduler."""
        status = {
            "running": self.running,
            "enabled": False,
            "jobs": []
        }
        
        if self.config is not None:
            status["enabled"] = self.config.get("enabled", False)
        
        for job in self.jobs:
            next_run = "never"
            if job["last_run"]:
                next_run_time = job["last_run"] + job["interval"]
                next_run = datetime.fromtimestamp(next_run_time).strftime('%Y-%m-%d %H:%M:%S')
            
            status["jobs"].append({
                "name": job["name"],
                "interval_seconds": job["interval"],
                "last_run": datetime.fromtimestamp(job["last_run"]).strftime('%Y-%m-%d %H:%M:%S') if job["last_run"] else "never",
                "next_run": next_run
            })
        
        return status

# Global scheduler instance
scheduler = CrawlerScheduler()

def start_crawler(run_initial=True):
    """Start the scheduled data crawler system."""
    return scheduler.start(run_initial)

def stop_crawler():
    """Stop the scheduled data crawler system."""
    return scheduler.stop()

def update_crawler_config(new_config):
    """Update the crawler configuration."""
    return scheduler.update_config(new_config)

def get_crawler_status():
    """Get the current status of the data crawler system."""
    return scheduler.get_status()

if __name__ == "__main__":
    # When run directly, start the crawler system
    logger.info("Starting scheduled crawler")
    start_crawler(run_initial=True)
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Scheduled crawler shutting down")
        stop_crawler()