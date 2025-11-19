-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- ==================== CORE TABLES ====================

-- Commodities master table
CREATE TABLE IF NOT EXISTS commodities (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  trading_unit TEXT DEFAULT 'kg',
  price_min DECIMAL(10,2),
  price_max DECIMAL(10,2),
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Quality parameters definition
CREATE TABLE IF NOT EXISTS quality_parameters (
  id SERIAL PRIMARY KEY,
  commodity_id INTEGER REFERENCES commodities(id) ON DELETE CASCADE,
  parameter_name TEXT NOT NULL,
  min_value DECIMAL(10,2),
  max_value DECIMAL(10,2),
  standard_value DECIMAL(10,2),
  unit TEXT,
  step_size DECIMAL(10,4),
  impact_type TEXT, -- 'linear', 'threshold', 'exponential'
  impact_factor DECIMAL(10,2),
  premium_factor DECIMAL(10,2),
  discount_factor DECIMAL(10,2),
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(commodity_id, parameter_name)
);

-- Regions table
CREATE TABLE IF NOT EXISTS regions (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  state TEXT,
  region_type TEXT, -- 'North India', 'South India', etc.
  price_adjustment_factor DECIMAL(5,4) DEFAULT 1.0,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Commodity-Region mapping
CREATE TABLE IF NOT EXISTS commodity_regions (
  id SERIAL PRIMARY KEY,
  commodity_id INTEGER REFERENCES commodities(id) ON DELETE CASCADE,
  region_id INTEGER REFERENCES regions(id) ON DELETE CASCADE,
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(commodity_id, region_id)
);

-- Market price data (crawled data)
CREATE TABLE IF NOT EXISTS market_prices (
  id SERIAL PRIMARY KEY,
  commodity_id INTEGER REFERENCES commodities(id),
  region_id INTEGER REFERENCES regions(id),
  market_name TEXT,
  price DECIMAL(10,2) NOT NULL,
  unit TEXT DEFAULT 'kg',
  volume DECIMAL(10,2),
  quality_grade TEXT,
  date DATE NOT NULL,
  source TEXT NOT NULL, -- 'agmarknet', 'enam', 'manual'
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_market_prices_commodity ON market_prices(commodity_id);
CREATE INDEX IF NOT EXISTS idx_market_prices_region ON market_prices(region_id);
CREATE INDEX IF NOT EXISTS idx_market_prices_date ON market_prices(date DESC);
CREATE INDEX IF NOT EXISTS idx_market_prices_source ON market_prices(source);
CREATE INDEX IF NOT EXISTS idx_market_prices_composite ON market_prices(commodity_id, region_id, date DESC);

-- ==================== VECTOR STORAGE ====================

-- Quality-Price vectors for ML-based pricing
CREATE TABLE IF NOT EXISTS quality_price_vectors (
  id SERIAL PRIMARY KEY,
  commodity_id INTEGER REFERENCES commodities(id),
  region_id INTEGER REFERENCES regions(id),
  quality_vector vector(20), -- Supports up to 20 quality parameters
  quality_params JSONB NOT NULL, -- Actual parameter values
  price DECIMAL(10,2) NOT NULL,
  confidence DECIMAL(5,4),
  source TEXT, -- 'user_input', 'crawled', 'calculated'
  date DATE NOT NULL,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vector similarity search index using IVFFlat
CREATE INDEX IF NOT EXISTS idx_quality_vectors_embedding ON quality_price_vectors 
USING ivfflat (quality_vector vector_cosine_ops) 
WITH (lists = 100);

-- ==================== INDICES & ANALYTICS ====================

-- WIZX Index calculations
CREATE TABLE IF NOT EXISTS wizx_indices (
  id SERIAL PRIMARY KEY,
  commodity_id INTEGER REFERENCES commodities(id),
  index_value DECIMAL(10,4) NOT NULL,
  calculation_date DATE NOT NULL,
  base_date DATE NOT NULL,
  base_value DECIMAL(10,4) NOT NULL,
  methodology TEXT,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(commodity_id, calculation_date)
);

CREATE INDEX IF NOT EXISTS idx_wizx_indices_commodity ON wizx_indices(commodity_id, calculation_date DESC);

-- ==================== LOGGING & MONITORING ====================

-- Crawler job logs
CREATE TABLE IF NOT EXISTS crawler_logs (
  id SERIAL PRIMARY KEY,
  source TEXT NOT NULL, -- 'agmarknet', 'enam', 'global_indices', 'historical'
  status TEXT NOT NULL, -- 'success', 'error', 'empty', 'running'
  records_processed INTEGER DEFAULT 0,
  error_message TEXT,
  started_at TIMESTAMPTZ NOT NULL,
  completed_at TIMESTAMPTZ,
  duration_seconds INTEGER,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_crawler_logs_source ON crawler_logs(source);
CREATE INDEX IF NOT EXISTS idx_crawler_logs_status ON crawler_logs(status);
CREATE INDEX IF NOT EXISTS idx_crawler_logs_started ON crawler_logs(started_at DESC);

-- API access logs
CREATE TABLE IF NOT EXISTS api_logs (
  id SERIAL PRIMARY KEY,
  user_id UUID,
  api_key TEXT,
  endpoint TEXT NOT NULL,
  method TEXT NOT NULL,
  status_code INTEGER NOT NULL,
  response_time_ms INTEGER,
  ip_address INET,
  user_agent TEXT,
  request_body JSONB,
  response_body JSONB,
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_api_logs_user ON api_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_api_logs_endpoint ON api_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_api_logs_created ON api_logs(created_at DESC);

-- ==================== USER INTERACTIONS ====================

-- User submissions for price feedback
CREATE TABLE IF NOT EXISTS user_submissions (
  id SERIAL PRIMARY KEY,
  user_id UUID,
  commodity_id INTEGER REFERENCES commodities(id),
  region_id INTEGER REFERENCES regions(id),
  quality_params JSONB NOT NULL,
  submitted_price DECIMAL(10,2),
  calculated_price DECIMAL(10,2),
  price_difference DECIMAL(10,2),
  feedback TEXT,
  status TEXT DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Price alerts for users
CREATE TABLE IF NOT EXISTS price_alerts (
  id SERIAL PRIMARY KEY,
  user_id UUID,
  commodity_id INTEGER REFERENCES commodities(id),
  region_id INTEGER REFERENCES regions(id),
  alert_type TEXT NOT NULL, -- 'above', 'below', 'change_percent'
  threshold_value DECIMAL(10,2),
  is_active BOOLEAN DEFAULT true,
  last_triggered_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ==================== TRIGGERS ====================

-- Function for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers
CREATE TRIGGER update_commodities_updated_at BEFORE UPDATE ON commodities
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_market_prices_updated_at BEFORE UPDATE ON market_prices
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_submissions_updated_at BEFORE UPDATE ON user_submissions
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==================== RPC FUNCTIONS ====================

-- Vector similarity search function
CREATE OR REPLACE FUNCTION match_quality_vectors(
  query_embedding vector(20),
  match_commodity_id int,
  match_region_id int,
  match_count int DEFAULT 10
)
RETURNS TABLE (
  id int,
  commodity_id int,
  region_id int,
  quality_params jsonb,
  price decimal,
  confidence decimal,
  source text,
  date date,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    qpv.id,
    qpv.commodity_id,
    qpv.region_id,
    qpv.quality_params,
    qpv.price,
    qpv.confidence,
    qpv.source,
    qpv.date,
    1 - (qpv.quality_vector <=> query_embedding) AS similarity
  FROM quality_price_vectors qpv
  WHERE qpv.commodity_id = match_commodity_id
    AND qpv.region_id = match_region_id
  ORDER BY qpv.quality_vector <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Get average market price for commodity and region
CREATE OR REPLACE FUNCTION get_average_market_price(
  p_commodity_id int,
  p_region_id int,
  days_back int DEFAULT 30
)
RETURNS DECIMAL
LANGUAGE plpgsql
AS $$
DECLARE
  avg_price DECIMAL;
BEGIN
  SELECT AVG(price) INTO avg_price
  FROM market_prices
  WHERE commodity_id = p_commodity_id
    AND region_id = p_region_id
    AND date >= CURRENT_DATE - days_back
    AND date <= CURRENT_DATE;
  
  RETURN COALESCE(avg_price, 0);
END;
$$;

-- Get price trend for commodity
CREATE OR REPLACE FUNCTION get_price_trend(
  p_commodity_id int,
  p_region_id int,
  days_back int DEFAULT 90
)
RETURNS TABLE (
  date date,
  avg_price decimal,
  min_price decimal,
  max_price decimal,
  volume decimal
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    mp.date,
    AVG(mp.price)::decimal AS avg_price,
    MIN(mp.price)::decimal AS min_price,
    MAX(mp.price)::decimal AS max_price,
    SUM(mp.volume)::decimal AS volume
  FROM market_prices mp
  WHERE mp.commodity_id = p_commodity_id
    AND mp.region_id = p_region_id
    AND mp.date >= CURRENT_DATE - days_back
    AND mp.date <= CURRENT_DATE
  GROUP BY mp.date
  ORDER BY mp.date DESC;
END;
$$;
