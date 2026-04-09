CREATE TABLE IF NOT EXISTS session_state (
    session_key VARCHAR(255) PRIMARY KEY,
    current_epoch BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS inbound_message (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_key VARCHAR(255) NOT NULL,
    epoch BIGINT NOT NULL,
    content TEXT NOT NULL,
    status VARCHAR(32) NOT NULL,
    delivery_id VARCHAR(255),
    lease_expires_at BIGINT,
    created_at BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS outbound_message (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_key VARCHAR(255) NOT NULL,
    epoch BIGINT NOT NULL,
    inbound_message_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    status VARCHAR(32) NOT NULL,
    created_at BIGINT NOT NULL
);
