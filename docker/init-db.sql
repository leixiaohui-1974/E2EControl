-- E2EControl Database Initialization
-- 智能水网控制系统数据库初始化

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================================
-- System State Tables
-- ============================================================================

-- Pool state history
CREATE TABLE IF NOT EXISTS pool_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pool_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    water_level FLOAT NOT NULL,
    inflow FLOAT,
    outflow FLOAT,
    gate_position FLOAT,
    target_level FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_pool_states_pool_time ON pool_states(pool_id, timestamp DESC);
CREATE INDEX idx_pool_states_timestamp ON pool_states(timestamp DESC);

-- System events
CREATE TABLE IF NOT EXISTS system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    source VARCHAR(100),
    description TEXT,
    severity VARCHAR(20) DEFAULT 'info',
    data JSONB,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_events_type ON system_events(event_type);
CREATE INDEX idx_events_timestamp ON system_events(timestamp DESC);
CREATE INDEX idx_events_severity ON system_events(severity);

-- ============================================================================
-- Fault Management Tables
-- ============================================================================

-- Fault records
CREATE TABLE IF NOT EXISTS faults (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fault_type VARCHAR(50) NOT NULL,
    component VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    description TEXT,
    diagnosis_data JSONB,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_faults_status ON faults(status);
CREATE INDEX idx_faults_component ON faults(component);
CREATE INDEX idx_faults_detected ON faults(detected_at DESC);

-- Recovery actions
CREATE TABLE IF NOT EXISTS recovery_actions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fault_id UUID REFERENCES faults(id),
    action_type VARCHAR(50) NOT NULL,
    target VARCHAR(100),
    parameters JSONB,
    status VARCHAR(20) DEFAULT 'pending',
    result VARCHAR(20),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_recovery_fault ON recovery_actions(fault_id);
CREATE INDEX idx_recovery_status ON recovery_actions(status);

-- ============================================================================
-- Performance Metrics Tables
-- ============================================================================

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_type VARCHAR(50) NOT NULL,
    component VARCHAR(100),
    value FLOAT NOT NULL,
    unit VARCHAR(20),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_metrics_type_time ON performance_metrics(metric_type, timestamp DESC);

-- Performance alerts
CREATE TABLE IF NOT EXISTS performance_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50) NOT NULL,
    level VARCHAR(20) NOT NULL,
    message TEXT,
    metric_value FLOAT,
    threshold FLOAT,
    component VARCHAR(100),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_alerts_level ON performance_alerts(level);
CREATE INDEX idx_alerts_ack ON performance_alerts(acknowledged);

-- ============================================================================
-- Certification Tables
-- ============================================================================

-- Certification runs
CREATE TABLE IF NOT EXISTS certification_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    target_level VARCHAR(10),
    achieved_level VARCHAR(10) NOT NULL,
    total_scenarios INTEGER NOT NULL,
    passed_scenarios INTEGER NOT NULL,
    failed_scenarios INTEGER NOT NULL,
    pass_rate FLOAT NOT NULL,
    human_intervention_rate FLOAT,
    certification_valid BOOLEAN DEFAULT FALSE,
    report_path VARCHAR(255),
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_cert_achieved ON certification_runs(achieved_level);
CREATE INDEX idx_cert_completed ON certification_runs(completed_at DESC);

-- Scenario results
CREATE TABLE IF NOT EXISTS scenario_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    certification_id UUID REFERENCES certification_runs(id),
    scenario_id VARCHAR(50) NOT NULL,
    scenario_name VARCHAR(200),
    category VARCHAR(50),
    difficulty INTEGER,
    required_level VARCHAR(10),
    result VARCHAR(20) NOT NULL,
    score FLOAT,
    duration FLOAT,
    failures TEXT[],
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_scenario_cert ON scenario_results(certification_id);
CREATE INDEX idx_scenario_result ON scenario_results(result);

-- ============================================================================
-- Configuration Tables
-- ============================================================================

-- System configuration
CREATE TABLE IF NOT EXISTS system_config (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default configuration
INSERT INTO system_config (key, value, description) VALUES
    ('num_pools', '3', 'Number of canal pools'),
    ('control_mode', '"normal"', 'Current control mode'),
    ('self_healing_enabled', 'true', 'Enable self-healing system'),
    ('monitoring_interval', '1.0', 'Monitoring interval in seconds'),
    ('mpc_horizon', '10', 'MPC prediction horizon'),
    ('l3_planning_horizon', '24', 'L3 planning horizon in hours'),
    ('l2_control_horizon', '5', 'L2 control horizon in steps')
ON CONFLICT (key) DO NOTHING;

-- ============================================================================
-- Views
-- ============================================================================

-- Current system status view
CREATE OR REPLACE VIEW current_system_status AS
SELECT
    p.pool_id,
    p.water_level,
    p.inflow,
    p.outflow,
    p.gate_position,
    p.target_level,
    p.timestamp as last_update
FROM pool_states p
INNER JOIN (
    SELECT pool_id, MAX(timestamp) as max_ts
    FROM pool_states
    GROUP BY pool_id
) latest ON p.pool_id = latest.pool_id AND p.timestamp = latest.max_ts;

-- Active faults view
CREATE OR REPLACE VIEW active_faults AS
SELECT * FROM faults WHERE status = 'active' ORDER BY detected_at DESC;

-- Recent alerts view
CREATE OR REPLACE VIEW recent_alerts AS
SELECT * FROM performance_alerts
WHERE timestamp > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;

-- ============================================================================
-- Functions
-- ============================================================================

-- Clean up old data function
CREATE OR REPLACE FUNCTION cleanup_old_data(days_to_keep INTEGER DEFAULT 30)
RETURNS void AS $$
BEGIN
    DELETE FROM pool_states WHERE timestamp < NOW() - (days_to_keep || ' days')::INTERVAL;
    DELETE FROM system_events WHERE timestamp < NOW() - (days_to_keep || ' days')::INTERVAL;
    DELETE FROM performance_metrics WHERE timestamp < NOW() - (days_to_keep || ' days')::INTERVAL;
    DELETE FROM performance_alerts WHERE timestamp < NOW() - (days_to_keep * 2 || ' days')::INTERVAL AND acknowledged = true;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO e2econtrol;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO e2econtrol;
