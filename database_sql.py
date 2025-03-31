"""
Database module for storing and retrieving commodity pricing data
with SQL and vector capabilities for quality-region-price relationships.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import faiss
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Date, DateTime, JSON, Text, ForeignKey, Index, func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, scoped_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "./data"
COMMODITY_DB_FILE = os.path.join(DATA_DIR, "commodity_db.json")

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Get database URL from environment variable
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    # Use SQLite as fallback
    DATABASE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'wizx.db')}"
    logger.warning(f"No DATABASE_URL found, using SQLite: {DATABASE_URL}")

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Session = scoped_session(sessionmaker(bind=engine))
Base = declarative_base()

# Vector database storage (in-memory)
vector_dbs = {}

#----------------------
# Database Models
#----------------------

class Commodity(Base):
    __tablename__ = 'commodities'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Quality parameters stored as JSON
    quality_parameters = Column(JSON)
    price_range = Column(JSON)
    
    # Relationships
    regions = relationship("Region", back_populates="commodity", cascade="all, delete-orphan")
    varieties = relationship("Variety", back_populates="commodity", cascade="all, delete-orphan")
    price_points = relationship("PricePoint", back_populates="commodity", cascade="all, delete-orphan")
    wizx_indices = relationship("WIZXIndex", back_populates="commodity", cascade="all, delete-orphan")
    user_submissions = relationship("UserSubmission", back_populates="commodity", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Commodity(name='{self.name}')>"


class Variety(Base):
    __tablename__ = 'varieties'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    commodity_id = Column(Integer, ForeignKey('commodities.id'), nullable=False)
    description = Column(Text)
    
    # Relationships
    commodity = relationship("Commodity", back_populates="varieties")
    
    # Create a unique constraint on name and commodity_id
    __table_args__ = (
        sa.UniqueConstraint('name', 'commodity_id', name='unique_variety_per_commodity'),
    )
    
    def __repr__(self):
        return f"<Variety(name='{self.name}')>"


class Region(Base):
    __tablename__ = 'regions'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    commodity_id = Column(Integer, ForeignKey('commodities.id'), nullable=False)
    state = Column(String(100))
    country = Column(String(100), default="India")
    base_price = Column(Float)
    location_factor = Column(Float, default=1.0)
    
    # Relationships
    commodity = relationship("Commodity", back_populates="regions")
    price_points = relationship("PricePoint", back_populates="region", cascade="all, delete-orphan")
    
    # Create a unique constraint on name and commodity_id
    __table_args__ = (
        sa.UniqueConstraint('name', 'commodity_id', name='unique_region_per_commodity'),
    )
    
    def __repr__(self):
        return f"<Region(name='{self.name}', commodity='{self.commodity.name if self.commodity else None}')>"


class DataSource(Base):
    __tablename__ = 'data_sources'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    url = Column(String(255))
    description = Column(Text)
    source_type = Column(String(50))  # API, website, user, etc.
    api_key = Column(String(255))  # Store encrypted
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    price_points = relationship("PricePoint", back_populates="source")
    
    def __repr__(self):
        return f"<DataSource(name='{self.name}')>"


class PricePoint(Base):
    __tablename__ = 'price_points'
    
    id = Column(Integer, primary_key=True)
    commodity_id = Column(Integer, ForeignKey('commodities.id'), nullable=False)
    region_id = Column(Integer, ForeignKey('regions.id'), nullable=False)
    source_id = Column(Integer, ForeignKey('data_sources.id'))
    
    date = Column(Date, nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Float)  # Optional trading volume
    quality_parameters = Column(JSON)  # Actual quality parameters
    
    # Verification
    is_verified = Column(Boolean, default=False)
    verification_score = Column(Float, default=0.0)  # Higher score = more verified
    data_reliability = Column(Float, default=1.0)  # Weight for WIZX calculation
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    commodity = relationship("Commodity", back_populates="price_points")
    region = relationship("Region", back_populates="price_points")
    source = relationship("DataSource", back_populates="price_points")
    
    # Indices
    __table_args__ = (
        Index('idx_price_points_date', date),
        Index('idx_price_points_commodity_region_date', commodity_id, region_id, date),
    )
    
    def __repr__(self):
        return f"<PricePoint(commodity='{self.commodity.name if self.commodity else None}', region='{self.region.name if self.region else None}', date='{self.date}', price={self.price})>"


class WIZXIndex(Base):
    __tablename__ = 'wizx_indices'
    
    id = Column(Integer, primary_key=True)
    commodity_id = Column(Integer, ForeignKey('commodities.id'), nullable=False)
    date = Column(Date, nullable=False)
    
    index_value = Column(Float, nullable=False)
    previous_value = Column(Float)
    change = Column(Float)
    change_percentage = Column(Float)
    
    # Component data
    components = Column(JSON)  # Store price components used to calculate the index
    weighting_scheme = Column(JSON)  # How different regions are weighted
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    commodity = relationship("Commodity", back_populates="wizx_indices")
    
    # Indices
    __table_args__ = (
        Index('idx_wizx_indices_date', date),
        Index('idx_wizx_indices_commodity_date', commodity_id, date),
        sa.UniqueConstraint('commodity_id', 'date', name='unique_wizx_per_commodity_date'),
    )
    
    def __repr__(self):
        return f"<WIZXIndex(commodity='{self.commodity.name if self.commodity else None}', date='{self.date}', value={self.index_value})>"


class UserSubmission(Base):
    __tablename__ = 'user_submissions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(100))  # Can link to an auth system later
    commodity_id = Column(Integer, ForeignKey('commodities.id'), nullable=False)
    region_name = Column(String(100), nullable=False)
    
    date = Column(Date, nullable=False)
    price = Column(Float, nullable=False)
    quality_parameters = Column(JSON)
    
    # Verification
    is_verified = Column(Boolean, default=False)
    verification_score = Column(Float, default=0.0)
    verification_notes = Column(Text)
    
    # User reward and reputation
    reward_points = Column(Integer, default=0)
    
    submission_data = Column(JSON)  # Additional data like images, documents
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    commodity = relationship("Commodity", back_populates="user_submissions")
    
    # Indices
    __table_args__ = (
        Index('idx_user_submissions_date', date),
        Index('idx_user_submissions_user', user_id),
        Index('idx_user_submissions_commodity_region_date', commodity_id, region_name, date),
    )
    
    def __repr__(self):
        return f"<UserSubmission(user='{self.user_id}', commodity='{self.commodity.name if self.commodity else None}', date='{self.date}', price={self.price})>"


class DataCleaningRule(Base):
    __tablename__ = 'data_cleaning_rules'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    rule_type = Column(String(50))  # outlier, missing value, transformation
    description = Column(Text)
    
    # Rule parameters
    parameters = Column(JSON)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<DataCleaningRule(name='{self.name}', type='{self.rule_type}')>"


#----------------------
# Database Functions
#----------------------

def initialize_database():
    """Create all tables if they don't exist"""
    # Create tables
    Base.metadata.create_all(engine)
    logger.info("Database tables created")
    
    # Initialize default data sources
    initialize_data_sources()
    
    # Initialize commodity database
    initialize_commodity_database()


def initialize_data_sources():
    """Initialize default data sources"""
    session = Session()
    
    # Default data sources
    default_sources = [
        {
            "name": "Agmarknet",
            "url": "https://agmarknet.gov.in/",
            "description": "Agricultural Marketing Information Network by Government of India",
            "source_type": "government"
        },
        {
            "name": "eNAM",
            "url": "https://enam.gov.in/web/",
            "description": "Electronic National Agriculture Market by Government of India",
            "source_type": "government"
        },
        {
            "name": "WIZX User Community",
            "description": "Data submitted by verified WIZX platform users",
            "source_type": "user"
        },
        {
            "name": "Reuters",
            "url": "https://www.reuters.com/markets/commodities/",
            "description": "Reuters commodity market data",
            "source_type": "commercial"
        }
    ]
    
    try:
        for source_data in default_sources:
            existing_source = session.query(DataSource).filter_by(name=source_data["name"]).first()
            if not existing_source:
                new_source = DataSource(**source_data)
                session.add(new_source)
                logger.info(f"Added data source: {source_data['name']}")
        
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Error initializing data sources: {e}")
    finally:
        session.close()


def initialize_commodity_database():
    """
    Initialize the commodity database with default data if it doesn't exist.
    """
    # Check if we have any commodities in the database
    session = Session()
    commodity_count = session.query(func.count(Commodity.id)).scalar()
    
    if commodity_count > 0:
        logger.info(f"Database already contains {commodity_count} commodities")
        session.close()
        return
    
    logger.info("Initializing commodity database with default data")
    
    # Load data from JSON file if it exists, otherwise use defaults
    commodity_data = {}
    if os.path.exists(COMMODITY_DB_FILE):
        try:
            with open(COMMODITY_DB_FILE, 'r') as f:
                commodity_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading commodity database file: {e}")
    
    # If no data in file, use sample data
    if not commodity_data:
        commodity_data = {
            "Wheat": {
                "description": "Wheat is a cereal grain, originally from the Levant region of the Near East and Ethiopian Highlands, but now cultivated worldwide.",
                "varieties": ["Durum", "Common", "Spelt", "Emmer"],
                "quality_parameters": {
                    "Protein Content": {
                        "min": 8.0,
                        "max": 16.0,
                        "standard_value": 12.0,
                        "unit": "%",
                        "impact_factor": 0.8
                    },
                    "Moisture Content": {
                        "min": 8.0,
                        "max": 18.0,
                        "standard_value": 12.0,
                        "unit": "%",
                        "impact_factor": -0.5
                    },
                    "Test Weight": {
                        "min": 72.0,
                        "max": 84.0,
                        "standard_value": 78.0,
                        "unit": "kg/hl",
                        "impact_factor": 0.4
                    },
                    "Falling Number": {
                        "min": 250,
                        "max": 400,
                        "standard_value": 325,
                        "unit": "seconds",
                        "impact_factor": 0.3
                    },
                    "Damaged Kernels": {
                        "min": 0.0,
                        "max": 5.0,
                        "standard_value": 1.0,
                        "unit": "%",
                        "impact_factor": -0.7
                    }
                },
                "price_range": {
                    "min": 1800,
                    "max": 2400
                },
                "regions": {
                    "Madhya Pradesh": {
                        "base_price": 2000,
                        "location_factor": 1.0
                    },
                    "Punjab": {
                        "base_price": 2100,
                        "location_factor": 1.05
                    },
                    "Uttar Pradesh": {
                        "base_price": 1950,
                        "location_factor": 0.975
                    },
                    "Haryana": {
                        "base_price": 2050,
                        "location_factor": 1.025
                    },
                    "Rajasthan": {
                        "base_price": 1900,
                        "location_factor": 0.95
                    }
                }
            },
            "Rice": {
                "description": "Rice is the seed of the grass species Oryza sativa (Asian rice) or Oryza glaberrima (African rice).",
                "varieties": ["Basmati", "Jasmine", "Sona Masuri", "Ponni"],
                "quality_parameters": {
                    "Head Rice": {
                        "min": 55.0,
                        "max": 95.0,
                        "standard_value": 75.0,
                        "unit": "%",
                        "impact_factor": 0.9
                    },
                    "Moisture Content": {
                        "min": 10.0,
                        "max": 16.0,
                        "standard_value": 13.0,
                        "unit": "%",
                        "impact_factor": -0.4
                    },
                    "Chalkiness": {
                        "min": 0.0,
                        "max": 20.0,
                        "standard_value": 5.0,
                        "unit": "%",
                        "impact_factor": -0.6
                    },
                    "Broken Grains": {
                        "min": 0.0,
                        "max": 20.0,
                        "standard_value": 8.0,
                        "unit": "%",
                        "impact_factor": -0.8
                    },
                    "Foreign Matter": {
                        "min": 0.0,
                        "max": 5.0,
                        "standard_value": 1.0,
                        "unit": "%",
                        "impact_factor": -0.7
                    }
                },
                "price_range": {
                    "min": 2500,
                    "max": 6000
                },
                "regions": {
                    "Tamil Nadu": {
                        "base_price": 3000,
                        "location_factor": 1.0
                    },
                    "Punjab": {
                        "base_price": 4000,
                        "location_factor": 1.33
                    },
                    "West Bengal": {
                        "base_price": 2800,
                        "location_factor": 0.93
                    },
                    "Andhra Pradesh": {
                        "base_price": 3200,
                        "location_factor": 1.07
                    },
                    "Telangana": {
                        "base_price": 3100,
                        "location_factor": 1.03
                    }
                }
            },
            "Maize": {
                "description": "Maize, also known as corn, is a cereal grain first domesticated by indigenous peoples in southern Mexico about 10,000 years ago.",
                "varieties": ["Dent Corn", "Flint Corn", "Sweet Corn", "Popcorn"],
                "quality_parameters": {
                    "Moisture Content": {
                        "min": 10.0,
                        "max": 18.0,
                        "standard_value": 14.0,
                        "unit": "%",
                        "impact_factor": -0.5
                    },
                    "Test Weight": {
                        "min": 65.0,
                        "max": 78.0,
                        "standard_value": 72.0,
                        "unit": "kg/hl",
                        "impact_factor": 0.4
                    },
                    "Broken Kernels": {
                        "min": 0.0,
                        "max": 10.0,
                        "standard_value": 3.0,
                        "unit": "%",
                        "impact_factor": -0.6
                    },
                    "Foreign Matter": {
                        "min": 0.0,
                        "max": 5.0,
                        "standard_value": 1.0,
                        "unit": "%",
                        "impact_factor": -0.7
                    },
                    "Damaged Kernels": {
                        "min": 0.0,
                        "max": 7.0,
                        "standard_value": 2.0,
                        "unit": "%",
                        "impact_factor": -0.8
                    }
                },
                "price_range": {
                    "min": 1400,
                    "max": 2200
                },
                "regions": {
                    "Karnataka": {
                        "base_price": 1800,
                        "location_factor": 1.0
                    },
                    "Bihar": {
                        "base_price": 1700,
                        "location_factor": 0.94
                    },
                    "Madhya Pradesh": {
                        "base_price": 1750,
                        "location_factor": 0.97
                    },
                    "Rajasthan": {
                        "base_price": 1650,
                        "location_factor": 0.92
                    },
                    "Maharashtra": {
                        "base_price": 1850,
                        "location_factor": 1.03
                    }
                }
            },
            "Tur Dal": {
                "description": "Tur Dal, also known as Pigeon Pea or Arhar Dal, is a common pulse in India and other South Asian countries.",
                "varieties": ["BSMR-736", "BDN-711", "ICPL-87119", "TS-3R"],
                "quality_parameters": {
                    "Foreign Matter": {
                        "min": 0.0,
                        "max": 3.0,
                        "standard_value": 1.0,
                        "unit": "%",
                        "impact_factor": -0.7
                    },
                    "Moisture Content": {
                        "min": 8.0,
                        "max": 14.0,
                        "standard_value": 11.0,
                        "unit": "%",
                        "impact_factor": -0.5
                    },
                    "Weevilled Grains": {
                        "min": 0.0,
                        "max": 5.0,
                        "standard_value": 1.0,
                        "unit": "%",
                        "impact_factor": -0.8
                    },
                    "Damaged Grains": {
                        "min": 0.0,
                        "max": 5.0,
                        "standard_value": 1.0,
                        "unit": "%",
                        "impact_factor": -0.7
                    },
                    "Admixture": {
                        "min": 0.0,
                        "max": 3.0,
                        "standard_value": 0.5,
                        "unit": "%",
                        "impact_factor": -0.6
                    }
                },
                "price_range": {
                    "min": 5000,
                    "max": 8000
                },
                "regions": {
                    "Maharashtra": {
                        "base_price": 6500,
                        "location_factor": 1.0
                    },
                    "Karnataka": {
                        "base_price": 6300,
                        "location_factor": 0.97
                    },
                    "Telangana": {
                        "base_price": 6200,
                        "location_factor": 0.95
                    },
                    "Madhya Pradesh": {
                        "base_price": 6100,
                        "location_factor": 0.94
                    },
                    "Gujarat": {
                        "base_price": 6700,
                        "location_factor": 1.03
                    }
                }
            },
            "Soyabean": {
                "description": "Soybean is a species of legume widely grown for its edible bean, which has numerous uses.",
                "varieties": ["JS-335", "JS-9560", "NRC-86", "NRC-7"],
                "quality_parameters": {
                    "Oil Content": {
                        "min": 15.0,
                        "max": 25.0,
                        "standard_value": 20.0,
                        "unit": "%",
                        "impact_factor": 0.9
                    },
                    "Moisture Content": {
                        "min": 8.0,
                        "max": 14.0,
                        "standard_value": 10.0,
                        "unit": "%",
                        "impact_factor": -0.5
                    },
                    "Foreign Matter": {
                        "min": 0.0,
                        "max": 5.0,
                        "standard_value": 1.0,
                        "unit": "%",
                        "impact_factor": -0.6
                    },
                    "Damaged Grains": {
                        "min": 0.0,
                        "max": 5.0,
                        "standard_value": 1.0,
                        "unit": "%",
                        "impact_factor": -0.7
                    },
                    "Immature Grains": {
                        "min": 0.0,
                        "max": 10.0,
                        "standard_value": 3.0,
                        "unit": "%",
                        "impact_factor": -0.5
                    }
                },
                "price_range": {
                    "min": 3000,
                    "max": 5000
                },
                "regions": {
                    "Madhya Pradesh": {
                        "base_price": 3800,
                        "location_factor": 1.0
                    },
                    "Maharashtra": {
                        "base_price": 3700,
                        "location_factor": 0.97
                    },
                    "Rajasthan": {
                        "base_price": 3600,
                        "location_factor": 0.95
                    },
                    "Karnataka": {
                        "base_price": 3900,
                        "location_factor": 1.03
                    },
                    "Gujarat": {
                        "base_price": 3750,
                        "location_factor": 0.99
                    }
                }
            }
        }
    
    # Insert data into database
    try:
        # Get source ID for initial data
        agmarknet_source = session.query(DataSource).filter_by(name="Agmarknet").first()
        source_id = agmarknet_source.id if agmarknet_source else None
        
        # Current date for price points
        current_date = date.today()
        
        for commodity_name, data in commodity_data.items():
            # Create commodity
            commodity = Commodity(
                name=commodity_name,
                description=data.get("description", ""),
                quality_parameters=data.get("quality_parameters", {}),
                price_range=data.get("price_range", {})
            )
            session.add(commodity)
            session.flush()  # To get the ID
            
            # Add varieties
            for variety_name in data.get("varieties", []):
                variety = Variety(
                    name=variety_name,
                    commodity_id=commodity.id
                )
                session.add(variety)
            
            # Add regions
            for region_name, region_data in data.get("regions", {}).items():
                region = Region(
                    name=region_name,
                    commodity_id=commodity.id,
                    base_price=region_data.get("base_price", 0),
                    location_factor=region_data.get("location_factor", 1.0)
                )
                session.add(region)
                session.flush()  # To get the ID
                
                # Add a price point for today
                if source_id:
                    price_point = PricePoint(
                        commodity_id=commodity.id,
                        region_id=region.id,
                        source_id=source_id,
                        date=current_date,
                        price=region_data.get("base_price", 0),
                        is_verified=True,
                        data_reliability=1.0
                    )
                    session.add(price_point)
            
            # Create initial WIZX index
            index_value = 1000.0  # Base value
            wizx_index = WIZXIndex(
                commodity_id=commodity.id,
                date=current_date,
                index_value=index_value,
                components={},
                weighting_scheme={}
            )
            session.add(wizx_index)
            
        session.commit()
        logger.info("Initialized commodity database with default data")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error initializing commodity database: {e}")
    finally:
        session.close()
    
    # Initialize vector databases
    initialize_vector_dbs()


def initialize_vector_dbs():
    """Initialize vector databases for all commodities"""
    global vector_dbs
    
    session = Session()
    commodities = session.query(Commodity).all()
    
    for commodity in commodities:
        initialize_vector_db(commodity.name, {
            "quality_parameters": commodity.quality_parameters,
            "regions": {region.name: {"base_price": region.base_price} for region in commodity.regions}
        })
    
    session.close()


def initialize_vector_db(commodity, commodity_data):
    """
    Initialize vector database for a commodity.
    
    Args:
        commodity (str): The commodity name
        commodity_data (dict): Commodity data
    """
    global vector_dbs
    
    if commodity in vector_dbs:
        # Already initialized
        return
    
    logger.info(f"Initializing vector database for {commodity}")
    
    if not commodity_data or 'quality_parameters' not in commodity_data:
        logger.warning(f"No quality parameters found for {commodity}")
        return
    
    # Get dimension of quality vectors
    quality_params = commodity_data['quality_parameters']
    dimension = len(quality_params)
    
    # Initialize FAISS index
    index = faiss.IndexFlatL2(dimension)
    
    # Initialize vectors and metadata
    vectors = []
    metadata = []
    
    # Add some initial data if available
    if 'regions' in commodity_data:
        for region, region_data in commodity_data['regions'].items():
            # Use standard values for quality parameters
            standard_values = [
                param.get('standard_value', (param.get('min', 0) + param.get('max', 100)) / 2)
                for param_name, param in quality_params.items()
            ]
            
            # Normalize values to [0, 1] range
            normalized_values = []
            for i, (param_name, param) in enumerate(quality_params.items()):
                min_val = param.get('min', 0)
                max_val = param.get('max', 100)
                range_val = max_val - min_val
                
                if range_val > 0:
                    normalized_values.append((standard_values[i] - min_val) / range_val)
                else:
                    normalized_values.append(0.5)  # Default to middle if range is zero
            
            # Add to vectors
            vectors.append(normalized_values)
            
            # Add metadata
            metadata.append({
                'region': region,
                'price': region_data.get('base_price', 0),
                'quality_params': dict(zip(quality_params.keys(), standard_values))
            })
    
    # If we have vectors, add them to the index
    if vectors:
        # Convert to numpy array
        vectors_np = np.array(vectors).astype('float32')
        
        # Add to index
        index.add(vectors_np)
        
        # Store index and metadata
        vector_dbs[commodity] = {
            'index': index,
            'metadata': metadata,
            'param_names': list(quality_params.keys()),
            'param_details': quality_params
        }
        
        logger.info(f"Initialized vector database for {commodity} with {len(vectors)} vectors")


def get_commodity_data(commodity):
    """
    Get data for a specific commodity.
    
    Args:
        commodity (str): The commodity name
        
    Returns:
        dict: Commodity data or None if not found
    """
    # Make sure database is initialized
    session = Session()
    
    try:
        commodity_obj = session.query(Commodity).filter(Commodity.name == commodity).first()
        
        if not commodity_obj:
            logger.warning(f"Commodity {commodity} not found in database")
            return None
        
        # Convert to dictionary
        result = {
            "id": commodity_obj.id,
            "name": commodity_obj.name,
            "description": commodity_obj.description,
            "quality_parameters": commodity_obj.quality_parameters,
            "price_range": commodity_obj.price_range,
            "regions": {}
        }
        
        # Add regions
        for region in commodity_obj.regions:
            result["regions"][region.name] = {
                "base_price": region.base_price,
                "location_factor": region.location_factor
            }
        
        # Add varieties
        result["varieties"] = [v.name for v in commodity_obj.varieties]
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting commodity data: {e}")
        return None
    finally:
        session.close()


def get_all_commodities():
    """
    Get a list of all commodities in the database.
    
    Returns:
        list: List of commodity names
    """
    session = Session()
    
    try:
        commodities = session.query(Commodity.name).all()
        return [c[0] for c in commodities]
    except Exception as e:
        logger.error(f"Error getting commodities: {e}")
        return []
    finally:
        session.close()


def get_regions(commodity):
    """
    Get regions for a specific commodity.
    
    Args:
        commodity (str): The commodity name
        
    Returns:
        list: List of regions or empty list if not found
    """
    session = Session()
    
    try:
        regions = session.query(Region.name).\
            join(Commodity).\
            filter(Commodity.name == commodity).\
            all()
        
        return [r[0] for r in regions]
    except Exception as e:
        logger.error(f"Error getting regions: {e}")
        return []
    finally:
        session.close()


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
    
    if commodity_data and 'quality_parameters' in commodity_data:
        quality_params = commodity_data['quality_parameters']
        
        if parameter in quality_params:
            return quality_params[parameter]
    
    return None


def save_user_input(commodity, quality_params, region, price=None, user_id=None):
    """
    Save user input to the database for future reference.
    
    Args:
        commodity (str): The commodity name
        quality_params (dict): Quality parameters
        region (str): The region
        price (float, optional): Price if known
        user_id (str, optional): User identifier
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Validate inputs
    if not commodity or not quality_params or not region:
        logger.warning("Invalid inputs for save_user_input")
        return False
    
    session = Session()
    
    try:
        # Get commodity ID
        commodity_obj = session.query(Commodity).filter(Commodity.name == commodity).first()
        
        if not commodity_obj:
            logger.warning(f"Commodity {commodity} not found in database")
            return False
        
        # Create user submission
        submission = UserSubmission(
            user_id=user_id or "anonymous",
            commodity_id=commodity_obj.id,
            region_name=region,
            date=date.today(),
            price=price or 0,
            quality_parameters=quality_params,
            is_verified=False,
            verification_score=0.0,
            submission_data={}
        )
        
        session.add(submission)
        session.commit()
        
        logger.info(f"Saved user input for {commodity} in {region}")
        
        # Update vector database if price is known
        if price is not None:
            update_vector_db(commodity, quality_params, region, price)
        
        return True
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving user input: {e}")
        return False
    finally:
        session.close()


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
    global vector_dbs
    
    # Validate inputs
    if not commodity or not quality_params or not region or price is None:
        logger.warning("Invalid inputs for update_vector_db")
        return False
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data or 'quality_parameters' not in commodity_data:
        logger.warning(f"No quality parameters found for {commodity}")
        return False
    
    # Initialize vector database if needed
    if commodity not in vector_dbs:
        initialize_vector_db(commodity, commodity_data)
    
    # Make sure vector database exists
    if commodity not in vector_dbs:
        logger.warning(f"Failed to initialize vector database for {commodity}")
        return False
    
    vector_db = vector_dbs[commodity]
    param_names = vector_db['param_names']
    param_details = vector_db['param_details']
    
    # Create quality vector
    quality_values = []
    for param_name in param_names:
        if param_name in quality_params:
            quality_values.append(quality_params[param_name])
        else:
            # Use standard value if parameter not provided
            param = param_details[param_name]
            standard_value = param.get('standard_value', (param.get('min', 0) + param.get('max', 100)) / 2)
            quality_values.append(standard_value)
    
    # Normalize values to [0, 1] range
    normalized_values = []
    for i, param_name in enumerate(param_names):
        param = param_details[param_name]
        min_val = param.get('min', 0)
        max_val = param.get('max', 100)
        range_val = max_val - min_val
        
        if range_val > 0:
            normalized_values.append((quality_values[i] - min_val) / range_val)
        else:
            normalized_values.append(0.5)  # Default to middle if range is zero
    
    # Add to vectors
    vector_np = np.array([normalized_values]).astype('float32')
    vector_db['index'].add(vector_np)
    
    # Add metadata
    vector_db['metadata'].append({
        'region': region,
        'price': price,
        'quality_params': dict(zip(param_names, quality_values))
    })
    
    logger.info(f"Updated vector database for {commodity} with new data point")
    
    return True


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
    global vector_dbs
    
    # Validate inputs
    if not commodity or not quality_params:
        logger.warning("Invalid inputs for query_similar_qualities")
        return []
    
    # Get commodity data
    commodity_data = get_commodity_data(commodity)
    
    if not commodity_data or 'quality_parameters' not in commodity_data:
        logger.warning(f"No quality parameters found for {commodity}")
        return []
    
    # Initialize vector database if needed
    if commodity not in vector_dbs:
        initialize_vector_db(commodity, commodity_data)
    
    # Make sure vector database exists
    if commodity not in vector_dbs:
        logger.warning(f"Failed to initialize vector database for {commodity}")
        return []
    
    vector_db = vector_dbs[commodity]
    param_names = vector_db['param_names']
    param_details = vector_db['param_details']
    metadata = vector_db['metadata']
    
    # Create quality vector
    quality_values = []
    for param_name in param_names:
        if param_name in quality_params:
            quality_values.append(quality_params[param_name])
        else:
            # Use standard value if parameter not provided
            param = param_details[param_name]
            standard_value = param.get('standard_value', (param.get('min', 0) + param.get('max', 100)) / 2)
            quality_values.append(standard_value)
    
    # Normalize values to [0, 1] range
    normalized_values = []
    for i, param_name in enumerate(param_names):
        param = param_details[param_name]
        min_val = param.get('min', 0)
        max_val = param.get('max', 100)
        range_val = max_val - min_val
        
        if range_val > 0:
            normalized_values.append((quality_values[i] - min_val) / range_val)
        else:
            normalized_values.append(0.5)  # Default to middle if range is zero
    
    # Query the index
    query_vector = np.array([normalized_values]).astype('float32')
    
    # Limit k to the number of vectors in the index
    k = min(k, vector_db['index'].ntotal)
    
    if k == 0:
        logger.warning(f"No vectors in the index for {commodity}")
        return []
    
    distances, indices = vector_db['index'].search(query_vector, k)
    
    # Get the similar items
    similar_items = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            item = metadata[idx].copy()
            item['distance'] = distances[0][i]
            similar_items.append(item)
    
    # Filter by region if specified
    if region:
        similar_items = [item for item in similar_items if item['region'] == region]
    
    return similar_items


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
    # Query similar qualities
    similar_items = query_similar_qualities(commodity, quality_params, region)
    
    if not similar_items:
        # Get commodity data for baseline price
        commodity_data = get_commodity_data(commodity)
        
        if commodity_data and 'regions' in commodity_data and region in commodity_data['regions']:
            base_price = commodity_data['regions'][region].get('base_price', 0)
            
            return {
                'recommended_price': base_price,
                'confidence': 'low',
                'based_on': 'baseline',
                'similar_items': []
            }
        else:
            return {
                'recommended_price': 0,
                'confidence': 'very_low',
                'based_on': 'none',
                'similar_items': []
            }
    
    # Calculate weights based on distances
    total_weight = 0
    weighted_sum = 0
    
    for item in similar_items:
        # Convert distance to weight (closer items have higher weight)
        weight = 1.0 / (1.0 + item['distance'])
        total_weight += weight
        weighted_sum += weight * item['price']
    
    # Calculate weighted average price
    if total_weight > 0:
        recommended_price = weighted_sum / total_weight
    else:
        recommended_price = similar_items[0]['price']
    
    # Determine confidence level
    if len(similar_items) >= 5:
        confidence = 'high'
    elif len(similar_items) >= 3:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return {
        'recommended_price': recommended_price,
        'confidence': confidence,
        'based_on': 'similar_items',
        'similar_items': similar_items
    }


def get_price_history(commodity, region, days, start_date=None):
    """
    Get historical price data for a commodity in a region.
    
    Args:
        commodity (str): The commodity
        region (str): The region
        days (int): Number of days of history
        start_date (datetime, optional): Start date for history
        
    Returns:
        list: Historical price data
    """
    session = Session()
    
    try:
        if start_date is None:
            start_date = date.today() - timedelta(days=days)
        end_date = start_date + timedelta(days=days)
        
        # Get region ID
        region_obj = session.query(Region).\
            join(Commodity).\
            filter(Commodity.name == commodity, Region.name == region).\
            first()
        
        if not region_obj:
            logger.warning(f"Region {region} not found for commodity {commodity}")
            return []
        
        # Query price points
        price_points = session.query(PricePoint).\
            filter(
                PricePoint.commodity_id == region_obj.commodity_id,
                PricePoint.region_id == region_obj.id,
                PricePoint.date >= start_date,
                PricePoint.date <= end_date
            ).\
            order_by(PricePoint.date).\
            all()
        
        # Convert to list of dicts
        result = []
        for pp in price_points:
            result.append({
                "date": pp.date,
                "price": pp.price,
                "volume": pp.volume,
                "quality_params": pp.quality_parameters,
                "is_verified": pp.is_verified
            })
        
        # Fill in missing dates
        if result:
            all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Create a dict of existing dates
            existing_dates = {item["date"]: item for item in result}
            
            # Fill in missing dates
            result = []
            prev_price = None
            
            for d in all_dates:
                d_date = d.date()
                if d_date in existing_dates:
                    item = existing_dates[d_date]
                    prev_price = item["price"]
                    result.append(item)
                elif prev_price is not None:
                    # Use previous price for missing date
                    result.append({
                        "date": d_date,
                        "price": prev_price,
                        "volume": None,
                        "quality_params": None,
                        "is_verified": False
                    })
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting price history: {e}")
        return []
    finally:
        session.close()


def calculate_wizx_index(commodity, date_val=None):
    """
    Calculate the WIZX index for a commodity on a specific date.
    
    Args:
        commodity (str): The commodity
        date_val (date, optional): The date for calculation
        
    Returns:
        dict: WIZX index data
    """
    if date_val is None:
        date_val = date.today()
    
    session = Session()
    
    try:
        # Get commodity ID
        commodity_obj = session.query(Commodity).filter(Commodity.name == commodity).first()
        
        if not commodity_obj:
            logger.warning(f"Commodity {commodity} not found in database")
            return None
        
        # Get previous index value (from yesterday)
        previous_index = session.query(WIZXIndex).\
            filter(
                WIZXIndex.commodity_id == commodity_obj.id,
                WIZXIndex.date < date_val
            ).\
            order_by(WIZXIndex.date.desc()).\
            first()
        
        previous_value = 1000.0  # Default base value
        if previous_index:
            previous_value = previous_index.index_value
        
        # Get all regions for the commodity
        regions = session.query(Region).filter(Region.commodity_id == commodity_obj.id).all()
        
        if not regions:
            logger.warning(f"No regions found for commodity {commodity}")
            return None
        
        # Get price points for each region
        components = {}
        total_weight = 0
        weighted_sum = 0
        
        for region in regions:
            # Get price for this region and date
            price_point = session.query(PricePoint).\
                filter(
                    PricePoint.commodity_id == commodity_obj.id,
                    PricePoint.region_id == region.id,
                    PricePoint.date == date_val
                ).\
                order_by(PricePoint.data_reliability.desc()).\
                first()
            
            if price_point:
                # Use region's base price as the reference price
                reference_price = region.base_price or 1.0
                
                # Calculate price ratio
                price_ratio = price_point.price / reference_price
                
                # Apply region weight (currently equal weighting)
                weight = 1.0 / len(regions)
                
                # Apply reliability factor
                reliability = price_point.data_reliability or 1.0
                weight *= reliability
                
                components[region.name] = {
                    "price": price_point.price,
                    "reference_price": reference_price,
                    "price_ratio": price_ratio,
                    "weight": weight,
                    "reliability": reliability
                }
                
                total_weight += weight
                weighted_sum += price_ratio * weight
        
        # If we have components, calculate the index
        if components and total_weight > 0:
            # Normalize the weighted sum
            price_index = weighted_sum / total_weight
            
            # Scale to base 1000
            index_value = 1000.0 * price_index
            
            # Calculate change
            change = index_value - previous_value
            change_percentage = (change / previous_value) * 100 if previous_value else 0
            
            # Store index
            wizx_index = WIZXIndex(
                commodity_id=commodity_obj.id,
                date=date_val,
                index_value=index_value,
                previous_value=previous_value,
                change=change,
                change_percentage=change_percentage,
                components=components,
                weighting_scheme={"type": "equal_weighted", "adjusted_for_reliability": True}
            )
            
            session.add(wizx_index)
            session.commit()
            
            return {
                "commodity": commodity,
                "date": date_val,
                "index_value": index_value,
                "previous_value": previous_value,
                "change": change,
                "change_percentage": change_percentage,
                "components": components
            }
        else:
            logger.warning(f"Insufficient data to calculate WIZX index for {commodity} on {date_val}")
            return None
            
    except Exception as e:
        session.rollback()
        logger.error(f"Error calculating WIZX index: {e}")
        return None
    finally:
        session.close()


def get_wizx_indices(commodity=None, start_date=None, end_date=None):
    """
    Get WIZX indices for a period.
    
    Args:
        commodity (str, optional): The commodity (if None, return for all commodities)
        start_date (date, optional): Start date
        end_date (date, optional): End date
        
    Returns:
        dict: WIZX indices by commodity and date
    """
    session = Session()
    
    try:
        if end_date is None:
            end_date = date.today()
        
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # Build query
        query = session.query(WIZXIndex).filter(
            WIZXIndex.date >= start_date,
            WIZXIndex.date <= end_date
        )
        
        if commodity:
            # Get commodity ID
            commodity_obj = session.query(Commodity).filter(Commodity.name == commodity).first()
            
            if not commodity_obj:
                logger.warning(f"Commodity {commodity} not found in database")
                return {}
            
            query = query.filter(WIZXIndex.commodity_id == commodity_obj.id)
        
        indices = query.all()
        
        # Organize by commodity and date
        result = {}
        
        for idx in indices:
            commodity_name = session.query(Commodity.name).filter(Commodity.id == idx.commodity_id).scalar()
            
            if commodity_name not in result:
                result[commodity_name] = []
            
            result[commodity_name].append({
                "date": idx.date,
                "index_value": idx.index_value,
                "previous_value": idx.previous_value,
                "change": idx.change,
                "change_percentage": idx.change_percentage
            })
        
        # Sort by date
        for commodity_name in result:
            result[commodity_name].sort(key=lambda x: x["date"])
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting WIZX indices: {e}")
        return {}
    finally:
        session.close()


def add_data_source(name, url=None, description=None, source_type="other", api_key=None):
    """
    Add a new data source.
    
    Args:
        name (str): Source name
        url (str, optional): Source URL
        description (str, optional): Description
        source_type (str): Source type
        api_key (str, optional): API key for the source
        
    Returns:
        bool: Success or failure
    """
    session = Session()
    
    try:
        # Check if already exists
        existing = session.query(DataSource).filter(DataSource.name == name).first()
        
        if existing:
            logger.warning(f"Data source {name} already exists")
            return False
        
        # Create new data source
        source = DataSource(
            name=name,
            url=url,
            description=description,
            source_type=source_type,
            api_key=api_key
        )
        
        session.add(source)
        session.commit()
        
        logger.info(f"Added data source: {name}")
        return True
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error adding data source: {e}")
        return False
    finally:
        session.close()


def save_price_data(commodity, region, price, date_val=None, source=None, quality_params=None, volume=None, is_verified=False):
    """
    Save price data to the database.
    
    Args:
        commodity (str): Commodity name
        region (str): Region name
        price (float): Price value
        date_val (date, optional): Date of the price data
        source (str, optional): Data source name
        quality_params (dict, optional): Quality parameters
        volume (float, optional): Trading volume
        is_verified (bool): Whether the data is verified
        
    Returns:
        bool: Success or failure
    """
    session = Session()
    
    try:
        # Default date to today
        if date_val is None:
            date_val = date.today()
        
        # Get commodity
        commodity_obj = session.query(Commodity).filter(Commodity.name == commodity).first()
        
        if not commodity_obj:
            logger.warning(f"Commodity {commodity} not found in database")
            return False
        
        # Get region
        region_obj = session.query(Region).\
            filter(Region.commodity_id == commodity_obj.id, Region.name == region).\
            first()
        
        if not region_obj:
            logger.warning(f"Region {region} not found for commodity {commodity}")
            return False
        
        # Get source ID
        source_id = None
        if source:
            source_obj = session.query(DataSource).filter(DataSource.name == source).first()
            
            if source_obj:
                source_id = source_obj.id
            else:
                logger.warning(f"Data source {source} not found")
        
        # Check if entry already exists
        existing = session.query(PricePoint).\
            filter(
                PricePoint.commodity_id == commodity_obj.id,
                PricePoint.region_id == region_obj.id,
                PricePoint.date == date_val,
                PricePoint.source_id == source_id if source_id else True
            ).\
            first()
        
        if existing:
            # Update existing entry
            existing.price = price
            existing.volume = volume
            existing.quality_parameters = quality_params
            existing.is_verified = is_verified
            logger.info(f"Updated price data for {commodity} in {region} on {date_val}")
        else:
            # Create new entry
            price_point = PricePoint(
                commodity_id=commodity_obj.id,
                region_id=region_obj.id,
                source_id=source_id,
                date=date_val,
                price=price,
                volume=volume,
                quality_parameters=quality_params,
                is_verified=is_verified
            )
            session.add(price_point)
            logger.info(f"Added price data for {commodity} in {region} on {date_val}")
        
        session.commit()
        
        # Update vector database
        if quality_params:
            update_vector_db(commodity, quality_params, region, price)
        
        # Calculate WIZX index for this commodity and date
        calculate_wizx_index(commodity, date_val)
        
        return True
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving price data: {e}")
        return False
    finally:
        session.close()


def add_data_cleaning_rule(name, rule_type, description=None, parameters=None):
    """
    Add a data cleaning rule.
    
    Args:
        name (str): Rule name
        rule_type (str): Rule type (outlier, missing_value, transformation)
        description (str, optional): Description
        parameters (dict, optional): Rule parameters
        
    Returns:
        bool: Success or failure
    """
    session = Session()
    
    try:
        # Check if already exists
        existing = session.query(DataCleaningRule).filter(DataCleaningRule.name == name).first()
        
        if existing:
            logger.warning(f"Data cleaning rule {name} already exists")
            return False
        
        # Create new rule
        rule = DataCleaningRule(
            name=name,
            rule_type=rule_type,
            description=description,
            parameters=parameters or {}
        )
        
        session.add(rule)
        session.commit()
        
        logger.info(f"Added data cleaning rule: {name}")
        return True
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error adding data cleaning rule: {e}")
        return False
    finally:
        session.close()


def apply_data_cleaning_rules(commodity, start_date=None, end_date=None):
    """
    Apply data cleaning rules to price data.
    
    Args:
        commodity (str): Commodity name
        start_date (date, optional): Start date
        end_date (date, optional): End date
        
    Returns:
        dict: Cleaning statistics
    """
    session = Session()
    
    try:
        # Default dates
        if end_date is None:
            end_date = date.today()
        
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        # Get active rules
        rules = session.query(DataCleaningRule).filter(DataCleaningRule.is_active == True).all()
        
        if not rules:
            logger.warning("No active data cleaning rules found")
            return {"cleaned": 0, "total": 0}
        
        # Get commodity ID
        commodity_obj = session.query(Commodity).filter(Commodity.name == commodity).first()
        
        if not commodity_obj:
            logger.warning(f"Commodity {commodity} not found in database")
            return {"cleaned": 0, "total": 0}
        
        # Get price data
        price_points = session.query(PricePoint).\
            filter(
                PricePoint.commodity_id == commodity_obj.id,
                PricePoint.date >= start_date,
                PricePoint.date <= end_date
            ).\
            all()
        
        if not price_points:
            logger.info(f"No price data found for {commodity} in the specified date range")
            return {"cleaned": 0, "total": 0}
        
        # Apply each rule
        cleaned_count = 0
        
        for rule in rules:
            logger.info(f"Applying rule: {rule.name}")
            
            if rule.rule_type == "outlier":
                # Detect and handle outliers
                if "method" in rule.parameters:
                    method = rule.parameters["method"]
                    
                    if method == "z_score":
                        # Z-score method
                        threshold = rule.parameters.get("threshold", 3.0)
                        
                        # Get prices by region
                        region_prices = {}
                        for pp in price_points:
                            region_id = pp.region_id
                            if region_id not in region_prices:
                                region_prices[region_id] = []
                            
                            region_prices[region_id].append(pp)
                        
                        # Check each region separately
                        for region_id, pps in region_prices.items():
                            if len(pps) < 2:
                                continue
                            
                            # Calculate mean and std
                            prices = [pp.price for pp in pps]
                            mean_price = sum(prices) / len(prices)
                            std_price = (sum((p - mean_price) ** 2 for p in prices) / len(prices)) ** 0.5
                            
                            if std_price == 0:
                                continue
                            
                            # Check each price
                            for pp in pps:
                                z_score = abs(pp.price - mean_price) / std_price
                                
                                if z_score > threshold:
                                    # Mark as outlier
                                    pp.data_reliability = 0.5
                                    cleaned_count += 1
                        
                    elif method == "iqr":
                        # IQR method
                        factor = rule.parameters.get("factor", 1.5)
                        
                        # Get prices by region
                        region_prices = {}
                        for pp in price_points:
                            region_id = pp.region_id
                            if region_id not in region_prices:
                                region_prices[region_id] = []
                            
                            region_prices[region_id].append(pp)
                        
                        # Check each region separately
                        for region_id, pps in region_prices.items():
                            if len(pps) < 4:  # Need enough data for quartiles
                                continue
                            
                            # Calculate quartiles
                            prices = sorted([pp.price for pp in pps])
                            n = len(prices)
                            q1 = prices[n // 4]
                            q3 = prices[3 * n // 4]
                            iqr = q3 - q1
                            
                            if iqr == 0:
                                continue
                            
                            lower_bound = q1 - factor * iqr
                            upper_bound = q3 + factor * iqr
                            
                            # Check each price
                            for pp in pps:
                                if pp.price < lower_bound or pp.price > upper_bound:
                                    # Mark as outlier
                                    pp.data_reliability = 0.5
                                    cleaned_count += 1
            
            elif rule.rule_type == "missing_value":
                # Handle missing values
                if "method" in rule.parameters:
                    method = rule.parameters["method"]
                    
                    if method == "interpolate":
                        # Group by region
                        region_prices = {}
                        for pp in price_points:
                            region_id = pp.region_id
                            if region_id not in region_prices:
                                region_prices[region_id] = {}
                            
                            region_prices[region_id][pp.date] = pp
                        
                        # For each region, check for missing dates
                        for region_id, date_prices in region_prices.items():
                            dates = sorted(date_prices.keys())
                            
                            if len(dates) < 2:
                                continue
                            
                            # Check for missing dates
                            all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                            
                            for d in all_dates:
                                d_date = d.date()
                                
                                if d_date not in date_prices:
                                    # Find closest dates before and after
                                    before_dates = [dt for dt in dates if dt < d_date]
                                    after_dates = [dt for dt in dates if dt > d_date]
                                    
                                    if before_dates and after_dates:
                                        before_date = max(before_dates)
                                        after_date = min(after_dates)
                                        
                                        before_price = date_prices[before_date].price
                                        after_price = date_prices[after_date].price
                                        
                                        # Linear interpolation
                                        days_between = (after_date - before_date).days
                                        days_from_before = (d_date - before_date).days
                                        
                                        if days_between > 0:
                                            interpolated_price = before_price + (after_price - before_price) * days_from_before / days_between
                                            
                                            # Create new price point with interpolated value
                                            new_pp = PricePoint(
                                                commodity_id=commodity_obj.id,
                                                region_id=region_id,
                                                date=d_date,
                                                price=interpolated_price,
                                                is_verified=False,
                                                data_reliability=0.7  # Lower reliability for interpolated data
                                            )
                                            
                                            session.add(new_pp)
                                            cleaned_count += 1
        
        session.commit()
        
        return {
            "cleaned": cleaned_count,
            "total": len(price_points)
        }
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error applying data cleaning rules: {e}")
        return {"cleaned": 0, "total": 0, "error": str(e)}
    finally:
        session.close()


def verify_user_submission(submission_id, verification_result, verification_score=None, notes=None):
    """
    Verify a user submission and add reward points if applicable.
    
    Args:
        submission_id (int): Submission ID
        verification_result (bool): True if verified, False if rejected
        verification_score (float, optional): Verification score
        notes (str, optional): Verification notes
        
    Returns:
        bool: Success or failure
    """
    session = Session()
    
    try:
        # Get submission
        submission = session.query(UserSubmission).filter(UserSubmission.id == submission_id).first()
        
        if not submission:
            logger.warning(f"Submission {submission_id} not found")
            return False
        
        # Update submission
        submission.is_verified = verification_result
        
        if verification_score is not None:
            submission.verification_score = verification_score
        
        if notes:
            submission.verification_notes = notes
        
        # Add reward points if verified
        if verification_result:
            # Base points for submission
            reward_points = 10
            
            # Bonus points based on verification score
            if verification_score and verification_score > 0.8:
                reward_points += 5
            
            submission.reward_points = reward_points
            
            # Get commodity and region
            commodity_name = session.query(Commodity.name).filter(Commodity.id == submission.commodity_id).scalar()
            
            # Add to price points if verified
            save_price_data(
                commodity=commodity_name,
                region=submission.region_name,
                price=submission.price,
                date_val=submission.date,
                source="WIZX User Community",
                quality_params=submission.quality_parameters,
                is_verified=True
            )
        
        session.commit()
        
        logger.info(f"Verified submission {submission_id}: {verification_result}")
        return True
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error verifying submission: {e}")
        return False
    finally:
        session.close()


def get_user_rewards(user_id):
    """
    Get reward points for a user.
    
    Args:
        user_id (str): User ID
        
    Returns:
        dict: Reward statistics
    """
    session = Session()
    
    try:
        # Get total reward points
        total_points = session.query(func.sum(UserSubmission.reward_points)).\
            filter(UserSubmission.user_id == user_id).\
            scalar() or 0
        
        # Get submission statistics
        total_submissions = session.query(func.count(UserSubmission.id)).\
            filter(UserSubmission.user_id == user_id).\
            scalar() or 0
        
        verified_submissions = session.query(func.count(UserSubmission.id)).\
            filter(UserSubmission.user_id == user_id, UserSubmission.is_verified == True).\
            scalar() or 0
        
        # Get commodity statistics
        commodity_breakdown = {}
        
        commodity_stats = session.query(
            Commodity.name,
            func.count(UserSubmission.id),
            func.sum(UserSubmission.reward_points)
        ).\
            join(UserSubmission, UserSubmission.commodity_id == Commodity.id).\
            filter(UserSubmission.user_id == user_id).\
            group_by(Commodity.name).\
            all()
        
        for commodity, count, points in commodity_stats:
            commodity_breakdown[commodity] = {
                "submissions": count,
                "points": points or 0
            }
        
        return {
            "user_id": user_id,
            "total_points": total_points,
            "total_submissions": total_submissions,
            "verified_submissions": verified_submissions,
            "verification_rate": verified_submissions / total_submissions if total_submissions > 0 else 0,
            "commodity_breakdown": commodity_breakdown
        }
        
    except Exception as e:
        logger.error(f"Error getting user rewards: {e}")
        return None
    finally:
        session.close()


def add_global_price_index(data_list):
    """
    Add global price index data from external sources.
    
    Args:
        data_list (list): List of dictionaries with global price data
        
    Returns:
        int: Number of records added
    """
    # Example data format:
    # [
    #   {
    #     "source": "Reuters",
    #     "commodity": "Wheat",
    #     "date": "2023-05-01",
    #     "price": 350.5,
    #     "currency": "USD",
    #     "unit": "ton",
    #     "exchange": "CBOT",
    #     "contract": "Jul-23"
    #   },
    #   ...
    # ]
    
    session = Session()
    added_count = 0
    
    try:
        # Get data source IDs
        sources = session.query(DataSource).all()
        source_map = {s.name: s.id for s in sources}
        
        # Get commodities
        commodities = session.query(Commodity).all()
        commodity_map = {c.name: c.id for c in commodities}
        
        # Process each data record
        for data in data_list:
            source_name = data.get("source")
            commodity_name = data.get("commodity")
            date_val = data.get("date")
            
            if not source_name or not commodity_name or not date_val:
                continue
            
            # Convert date string to date object if needed
            if isinstance(date_val, str):
                try:
                    date_val = datetime.strptime(date_val, "%Y-%m-%d").date()
                except ValueError:
                    continue
            
            # Skip if source or commodity not in our database
            if source_name not in source_map or commodity_name not in commodity_map:
                continue
            
            # Create or update global index
            # For now, simply save it as a price point with special region "Global"
            
            # Check if global region exists for this commodity
            global_region = session.query(Region).\
                filter(
                    Region.commodity_id == commodity_map[commodity_name],
                    Region.name == "Global"
                ).\
                first()
            
            if not global_region:
                # Create global region
                global_region = Region(
                    commodity_id=commodity_map[commodity_name],
                    name="Global",
                    country="Global",
                    base_price=data.get("price", 0)
                )
                session.add(global_region)
                session.flush()
            
            # Save as price point
            price_point = PricePoint(
                commodity_id=commodity_map[commodity_name],
                region_id=global_region.id,
                source_id=source_map[source_name],
                date=date_val,
                price=data.get("price", 0),
                quality_parameters={
                    "currency": data.get("currency", "USD"),
                    "unit": data.get("unit", "ton"),
                    "exchange": data.get("exchange"),
                    "contract": data.get("contract")
                },
                is_verified=True,
                data_reliability=0.9  # High reliability for global indices
            )
            
            session.add(price_point)
            added_count += 1
        
        session.commit()
        logger.info(f"Added {added_count} global price index records")
        
        return added_count
        
    except Exception as e:
        session.rollback()
        logger.error(f"Error adding global price indices: {e}")
        return 0
    finally:
        session.close()


# Initialize database when module is imported
initialize_database()