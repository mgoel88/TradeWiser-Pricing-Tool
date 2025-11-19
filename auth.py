"""
JWT Authentication Module for TradeWiser Pricing Tool
Extends authentication from tradewiser.in main platform
"""

import os
import jwt
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# JWT Configuration - shared with main platform
JWT_SECRET = os.getenv("JWT_SECRET", "your-jwt-secret-key-here")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# Supabase JWT Configuration
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

# Security scheme
security = HTTPBearer()


def create_access_token(user_id: str, email: str, additional_claims: Dict[str, Any] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        user_id: User ID from Supabase auth
        email: User email
        additional_claims: Additional claims to include in the token
    
    Returns:
        Encoded JWT token
    """
    payload = {
        "user_id": user_id,
        "email": email,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iss": "tradewiser-pricing",
        "aud": "tradewiser-platform"
    }
    
    if additional_claims:
        payload.update(additional_claims)
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token to verify
    
    Returns:
        Decoded token payload
    
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        # Try to decode with main platform JWT secret
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            audience="tradewiser-platform"
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        # Try Supabase JWT secret as fallback
        if SUPABASE_JWT_SECRET:
            try:
                payload = jwt.decode(
                    token,
                    SUPABASE_JWT_SECRET,
                    algorithms=[JWT_ALGORITHM],
                    options={"verify_aud": False}
                )
                return payload
            except jwt.InvalidTokenError:
                logger.error(f"Invalid token: {str(e)}")
                raise HTTPException(status_code=401, detail="Invalid token")
        else:
            logger.error(f"Invalid token: {str(e)}")
            raise HTTPException(status_code=401, detail="Invalid token")


def verify_supabase_token(token: str) -> Dict[str, Any]:
    """
    Verify a Supabase JWT token.
    
    Args:
        token: Supabase JWT token
    
    Returns:
        Decoded token payload
    """
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={"verify_aud": False}
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid Supabase token: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """
    Dependency to get the current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Authorization credentials
    
    Returns:
        User information from token payload
    
    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    payload = verify_token(token)
    
    # Extract user information
    user_info = {
        "user_id": payload.get("user_id") or payload.get("sub"),
        "email": payload.get("email"),
        "role": payload.get("role", "user"),
        "app_permissions": payload.get("app_permissions", {})
    }
    
    # Check if user has access to pricing tool
    if not user_info["app_permissions"].get("pricing_tool", True):
        raise HTTPException(
            status_code=403,
            detail="User does not have access to pricing tool"
        )
    
    return user_info


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[Dict[str, Any]]:
    """
    Optional authentication dependency.
    Returns user info if authenticated, None otherwise.
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_role(required_role: str):
    """
    Dependency factory to require a specific user role.
    
    Args:
        required_role: Required role (e.g., 'admin', 'manager')
    
    Returns:
        Dependency function
    """
    async def role_checker(user: Dict[str, Any] = Depends(get_current_user)):
        user_role = user.get("role", "user")
        
        # Define role hierarchy
        role_hierarchy = {
            "user": 0,
            "manager": 1,
            "admin": 2
        }
        
        if role_hierarchy.get(user_role, 0) < role_hierarchy.get(required_role, 0):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )
        
        return user
    
    return role_checker


# API Key authentication (for backward compatibility)
API_KEY_HEADER = "X-API-Key"
VALID_API_KEYS = set(os.getenv("API_KEYS", "").split(","))


async def verify_api_key(api_key: str) -> bool:
    """
    Verify API key for backward compatibility.
    
    Args:
        api_key: API key to verify
    
    Returns:
        True if valid
    
    Raises:
        HTTPException: If API key is invalid
    """
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return True
