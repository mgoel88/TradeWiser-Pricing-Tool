
"""
Transportation cost calculator module for agricultural commodities.
"""

import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Transport mode base rates (₹ per km per quintal)
TRANSPORT_RATES = {
    "truck": {
        "base_rate": 2.5,
        "min_distance": 50,
        "capacity": 200,  # quintals
        "fixed_cost": 2000,  # base booking cost
    },
    "mini_truck": {
        "base_rate": 3.0,
        "min_distance": 25,
        "capacity": 80,
        "fixed_cost": 1200,
    },
    "tempo": {
        "base_rate": 3.5,
        "min_distance": 10,
        "capacity": 40,
        "fixed_cost": 800,
    }
}

def calculate_transport_cost(
    distance: float,
    quantity: float,
    transport_mode: str = "truck",
    fuel_price: Optional[float] = None,
    toll_charges: Optional[float] = None
) -> Dict:
    """
    Calculate transportation cost based on distance and mode.
    
    Args:
        distance (float): Distance in kilometers
        quantity (float): Quantity in quintals
        transport_mode (str): Mode of transport (truck/mini_truck/tempo)
        fuel_price (float, optional): Current fuel price for adjustments
        toll_charges (float, optional): Additional toll charges
        
    Returns:
        dict: Cost breakdown and total
    """
    if transport_mode not in TRANSPORT_RATES:
        raise ValueError(f"Invalid transport mode: {transport_mode}")
    
    # Get base rates for selected mode
    rates = TRANSPORT_RATES[transport_mode]
    
    # Apply minimum distance
    effective_distance = max(distance, rates["min_distance"])
    
    # Calculate number of vehicles needed
    vehicles_needed = (quantity + rates["capacity"] - 1) // rates["capacity"]
    
    # Calculate base transport cost
    base_cost = effective_distance * rates["base_rate"] * quantity
    
    # Add fixed costs
    fixed_costs = rates["fixed_cost"] * vehicles_needed
    
    # Add fuel price adjustment if provided
    fuel_adjustment = 0
    if fuel_price and fuel_price > 80:  # Baseline fuel price of ₹80
        fuel_factor = (fuel_price - 80) / 80  # Calculate percentage increase
        fuel_adjustment = base_cost * fuel_factor * 0.4  # Fuel is ~40% of cost
    
    # Add toll charges if provided
    toll_charges = toll_charges or 0
    
    # Calculate total cost
    total_cost = base_cost + fixed_costs + fuel_adjustment + toll_charges
    
    # Calculate per quintal cost
    per_quintal_cost = total_cost / quantity if quantity > 0 else 0
    
    return {
        "transport_mode": transport_mode,
        "distance": effective_distance,
        "quantity": quantity,
        "vehicles_needed": vehicles_needed,
        "base_cost": round(base_cost, 2),
        "fixed_costs": round(fixed_costs, 2),
        "fuel_adjustment": round(fuel_adjustment, 2),
        "toll_charges": round(toll_charges, 2),
        "total_cost": round(total_cost, 2),
        "per_quintal_cost": round(per_quintal_cost, 2)
    }

def calculate_landed_cost(
    base_price: float,
    transport_cost: float,
    loading_cost: float = 50,  # ₹50 per quintal default loading cost
    unloading_cost: float = 50,  # ₹50 per quintal default unloading cost
    insurance_pct: float = 0.5,  # 0.5% insurance by default
    other_charges: float = 0
) -> Dict:
    """
    Calculate total landed cost including all components.
    
    Args:
        base_price (float): Base price per quintal
        transport_cost (float): Transport cost per quintal
        loading_cost (float): Loading cost per quintal
        unloading_cost (float): Unloading cost per quintal
        insurance_pct (float): Insurance percentage
        other_charges (float): Any other charges per quintal
        
    Returns:
        dict: Cost breakdown and total landed cost
    """
    # Calculate insurance amount
    insurance_amount = (base_price * insurance_pct) / 100
    
    # Calculate total landed cost
    total_cost = (
        base_price +
        transport_cost +
        loading_cost +
        unloading_cost +
        insurance_amount +
        other_charges
    )
    
    return {
        "base_price": round(base_price, 2),
        "transport_cost": round(transport_cost, 2),
        "loading_cost": round(loading_cost, 2),
        "unloading_cost": round(unloading_cost, 2),
        "insurance_amount": round(insurance_amount, 2),
        "other_charges": round(other_charges, 2),
        "total_landed_cost": round(total_cost, 2)
    }
