CREATE DATABASE IF NOT EXISTS mockmate DEFAULT CHARSET utf8mb4;
USE mockmate;

CREATE TABLE sessions (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  mode VARCHAR(16) NOT NULL DEFAULT 'qa',
  title VARCHAR(255) DEFAULT '',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE messages (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  session_id BIGINT NOT NULL,
  role VARCHAR(16) NOT NULL,
  question TEXT,
  answer TEXT,
  sources JSON,
  latency_ms INT DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE interviews (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  session_id BIGINT NOT NULL,
  question TEXT,
  knowledge TEXT,
  answer TEXT,
  score DECIMAL(3,1),
  comment TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE eval_runs (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  params JSON,
  hit_rate DECIMAL(5,4),
  coverage DECIMAL(5,4),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);