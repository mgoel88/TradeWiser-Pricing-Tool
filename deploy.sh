#!/bin/bash

# TradeWiser Pricing Tool - Production Deployment Script
# This script should be run on the production server (159.65.150.124)

set -e  # Exit on error

echo "🚀 Starting TradeWiser Pricing Tool Deployment..."

# Configuration
DEPLOY_DIR="/var/www/pricing.tradewiser.in"
REPO_URL="https://github.com/mgoel88/TradeWiser-Pricing-Tool.git"
SERVICE_NAME="tradewiser-pricing-api"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root or with sudo${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1: Creating deployment directory...${NC}"
mkdir -p $DEPLOY_DIR
mkdir -p /var/log/tradewiser-pricing
chown -R www-data:www-data /var/log/tradewiser-pricing

echo -e "${YELLOW}Step 2: Cloning/updating repository...${NC}"
if [ -d "$DEPLOY_DIR/.git" ]; then
    echo "Repository exists, pulling latest changes..."
    cd $DEPLOY_DIR
    git pull origin main
else
    echo "Cloning repository..."
    git clone $REPO_URL $DEPLOY_DIR
    cd $DEPLOY_DIR
fi

echo -e "${YELLOW}Step 3: Setting up Python virtual environment...${NC}"
if [ ! -d "$DEPLOY_DIR/venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${YELLOW}Step 4: Checking environment variables...${NC}"
if [ ! -f "$DEPLOY_DIR/.env" ]; then
    echo -e "${RED}ERROR: .env file not found!${NC}"
    echo "Please create .env file with required configuration:"
    echo "  - SUPABASE_URL"
    echo "  - SUPABASE_ANON_KEY"
    echo "  - JWT_SECRET"
    echo "  - etc."
    echo ""
    echo "You can use .env.example as a template:"
    echo "  cp .env.example .env"
    echo "  nano .env"
    exit 1
fi

echo -e "${GREEN}✓ Environment file found${NC}"

echo -e "${YELLOW}Step 5: Installing systemd service...${NC}"
cp tradewiser-pricing-api.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable $SERVICE_NAME

echo -e "${YELLOW}Step 6: Configuring Nginx...${NC}"
if [ ! -f "/etc/nginx/sites-available/pricing.tradewiser.in" ]; then
    cp nginx-pricing.conf /etc/nginx/sites-available/pricing.tradewiser.in
    ln -sf /etc/nginx/sites-available/pricing.tradewiser.in /etc/nginx/sites-enabled/
    
    # Test nginx configuration
    nginx -t
    
    echo -e "${GREEN}✓ Nginx configuration installed${NC}"
else
    echo "Nginx configuration already exists"
fi

echo -e "${YELLOW}Step 7: Setting up SSL certificate...${NC}"
if [ ! -d "/etc/letsencrypt/live/pricing.tradewiser.in" ]; then
    echo "Installing certbot..."
    apt-get update
    apt-get install -y certbot python3-certbot-nginx
    
    echo "Obtaining SSL certificate..."
    certbot --nginx -d pricing.tradewiser.in --non-interactive --agree-tos --email admin@tradewiser.in
    
    echo -e "${GREEN}✓ SSL certificate obtained${NC}"
else
    echo "SSL certificate already exists"
fi

echo -e "${YELLOW}Step 8: Setting permissions...${NC}"
chown -R www-data:www-data $DEPLOY_DIR
chmod -R 755 $DEPLOY_DIR

echo -e "${YELLOW}Step 9: Starting services...${NC}"
systemctl restart $SERVICE_NAME
systemctl reload nginx

echo -e "${YELLOW}Step 10: Checking service status...${NC}"
sleep 2
if systemctl is-active --quiet $SERVICE_NAME; then
    echo -e "${GREEN}✓ API service is running${NC}"
else
    echo -e "${RED}✗ API service failed to start${NC}"
    echo "Check logs with: journalctl -u $SERVICE_NAME -n 50"
    exit 1
fi

if systemctl is-active --quiet nginx; then
    echo -e "${GREEN}✓ Nginx is running${NC}"
else
    echo -e "${RED}✗ Nginx failed to start${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 11: Testing API...${NC}"
sleep 2
if curl -f -s https://pricing.tradewiser.in/health > /dev/null; then
    echo -e "${GREEN}✓ API health check passed${NC}"
else
    echo -e "${RED}✗ API health check failed${NC}"
    echo "Check logs with: journalctl -u $SERVICE_NAME -n 50"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "API URL: https://pricing.tradewiser.in"
echo "API Docs: https://pricing.tradewiser.in/docs"
echo "Health Check: https://pricing.tradewiser.in/health"
echo ""
echo "Useful commands:"
echo "  - View API logs: journalctl -u $SERVICE_NAME -f"
echo "  - Restart API: systemctl restart $SERVICE_NAME"
echo "  - Check status: systemctl status $SERVICE_NAME"
echo "  - View nginx logs: tail -f /var/log/nginx/pricing.tradewiser.in-error.log"
echo ""
