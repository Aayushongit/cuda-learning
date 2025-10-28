# Docker in Production - Best Practices

## Theory

### Production vs Development:

| Aspect | Development | Production |
|--------|-------------|------------|
| Images | Large, with tools | Minimal, optimized |
| Volumes | Bind mounts | Named volumes |
| Ports | All exposed | Only necessary |
| Restart | Manual | Automatic |
| Logging | Stdout | Log aggregation |
| Monitoring | Optional | Required |
| Security | Relaxed | Hardened |
| Updates | Manual | Automated/scheduled |

### Production Architecture:

```
Internet
    ↓
Load Balancer (nginx/HAProxy)
    ↓
    ├─→ App Server 1 (Container)
    ├─→ App Server 2 (Container)
    └─→ App Server 3 (Container)
         ↓
    Database (Container/RDS)
    Cache (Redis Container)
```

## Optimizing Images for Production

### 1. Multi-Stage Builds:

```dockerfile
# Development stage
FROM node:18 AS development
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "run", "dev"]

# Build stage
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

# Production stage
FROM node:18-alpine AS production
WORKDIR /app

# Copy only necessary files
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./

# Security: non-root user
USER node

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD node healthcheck.js

EXPOSE 3000
CMD ["node", "dist/server.js"]
```

### 2. Layer Caching Optimization:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (rarely changes)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (changes occasionally)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (changes frequently)
COPY . .

CMD ["python", "app.py"]
```

### 3. Image Size Reduction:

```dockerfile
# ❌ BAD: 1.2GB
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y python3 python3-pip
COPY . /app
RUN pip3 install -r /app/requirements.txt
CMD ["python3", "/app/app.py"]

# ✅ GOOD: 150MB
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]

# ✅ BETTER: 50MB
FROM python:3.11-alpine
WORKDIR /app
RUN apk add --no-cache gcc musl-dev
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

## Production docker-compose.yml

```yaml
version: '3.8'

services:
  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - nginx-cache:/var/cache/nginx
    networks:
      - frontend
    depends_on:
      - app
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Application (multiple replicas in production)
  app:
    image: myapp:1.0.0  # Specific version, not 'latest'
    restart: always
    deploy:
      replicas: 3  # Docker Compose doesn't use this, use Swarm/K8s
      resources:
        limits:
          cpus: '1'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
    environment:
      - NODE_ENV=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=redis://redis:6379
    secrets:
      - db_password
      - api_key
    networks:
      - frontend
      - backend
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    logging:
      driver: "fluentd"
      options:
        fluentd-address: localhost:24224
        tag: "app"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Database
  db:
    image: postgres:15-alpine
    restart: always
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./backups:/backups
    networks:
      - backend
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
    logging:
      driver: "json-file"
      options:
        max-size: "50m"
        max-file: "5"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis cache
  redis:
    image: redis:7-alpine
    restart: always
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis-data:/data
    networks:
      - backend
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # Monitoring: Prometheus
  prometheus:
    image: prom/prometheus:latest
    restart: always
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    ports:
      - "9090:9090"
    networks:
      - monitoring

  # Monitoring: Grafana
  grafana:
    image: grafana/grafana:latest
    restart: always
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    ports:
      - "3001:3000"
    networks:
      - monitoring
    depends_on:
      - prometheus

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access
  monitoring:
    driver: bridge

volumes:
  postgres-data:
    driver: local
  redis-data:
    driver: local
  nginx-cache:
    driver: local
  prometheus-data:
    driver: local
  grafana-data:
    driver: local

secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_key:
    file: ./secrets/api_key.txt
```

## Health Checks

### Application Health Check:

```python
# app.py
from flask import Flask, jsonify
import psycopg2

app = Flask(__name__)

@app.route('/health')
def health():
    try:
        # Check database connection
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()

        # Check Redis connection
        redis_client.ping()

        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 503

@app.route('/ready')
def ready():
    # Check if app is ready to receive traffic
    if app_initialized:
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "not ready"}), 503
```

### Dockerfile Health Check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1
```

## Logging

### Centralized Logging Setup:

```yaml
# docker-compose.logging.yml
services:
  # Fluentd for log collection
  fluentd:
    image: fluent/fluentd:edge
    volumes:
      - ./fluentd/fluent.conf:/fluentd/etc/fluent.conf
      - fluentd-data:/fluentd/log
    ports:
      - "24224:24224"
      - "24224:24224/udp"

  # Elasticsearch for log storage
  elasticsearch:
    image: elasticsearch:8.10.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data

  # Kibana for log visualization
  kibana:
    image: kibana:8.10.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  fluentd-data:
  elasticsearch-data:
```

## Monitoring

### Prometheus Configuration:

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'docker'
    static_configs:
      - targets: ['cadvisor:8080']

  - job_name: 'app'
    static_configs:
      - targets: ['app:9090']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

### Add Metrics to Application:

```python
# app.py with Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest
import time

# Define metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total request count')
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency')

@app.route('/api/data')
def get_data():
    REQUEST_COUNT.inc()
    start_time = time.time()

    # Your logic here
    data = fetch_data()

    REQUEST_LATENCY.observe(time.time() - start_time)
    return jsonify(data)

@app.route('/metrics')
def metrics():
    return generate_latest()
```

## Backup and Recovery

### Database Backup Script:

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_CONTAINER="myapp-db"

# Create backup
docker exec $DB_CONTAINER pg_dump -U postgres myapp > \
  "$BACKUP_DIR/backup_$DATE.sql"

# Compress
gzip "$BACKUP_DIR/backup_$DATE.sql"

# Keep only last 7 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete

echo "Backup completed: backup_$DATE.sql.gz"
```

### Automated Backup with Cron:

```yaml
services:
  backup:
    image: postgres:15-alpine
    volumes:
      - ./backups:/backups
      - ./scripts/backup.sh:/backup.sh
    environment:
      - PGHOST=db
      - PGUSER=postgres
      - PGPASSWORD=${DB_PASSWORD}
    command: >
      sh -c "
      while true; do
        sleep 86400;  # Run daily
        pg_dump myapp > /backups/backup_$(date +%Y%m%d).sql;
        find /backups -name 'backup_*.sql' -mtime +7 -delete;
      done
      "
    depends_on:
      - db
```

### Restore from Backup:

```bash
# Restore database
docker exec -i myapp-db psql -U postgres myapp < backup_20240101.sql

# Or using docker compose
docker compose exec -T db psql -U postgres myapp < backup_20240101.sql
```

## Zero-Downtime Deployments

### Rolling Update Strategy:

```bash
# 1. Build new version
docker build -t myapp:2.0.0 .

# 2. Tag as latest
docker tag myapp:2.0.0 myapp:latest

# 3. Update one container at a time
docker compose up -d --no-deps --scale app=3 app

# 4. Verify health
curl http://localhost/health

# 5. If successful, update all
docker compose up -d app
```

### Blue-Green Deployment:

```yaml
services:
  # Blue (current production)
  app-blue:
    image: myapp:1.0.0
    # ... configuration

  # Green (new version)
  app-green:
    image: myapp:2.0.0
    # ... configuration

  # Nginx routes to blue or green
  nginx:
    # Update nginx.conf to switch traffic
```

## Environment Management

### .env.production:

```bash
# Database
DB_NAME=production_db
DB_USER=produser
DB_PASSWORD=strong_password_here
DATABASE_URL=postgresql://produser:strong_password@db:5432/production_db

# Redis
REDIS_PASSWORD=redis_strong_password

# Application
NODE_ENV=production
API_URL=https://api.example.com
SECRET_KEY=very_secure_secret_key

# Monitoring
GRAFANA_PASSWORD=admin_password
```

### Managing Secrets:

```bash
# Using Docker Secrets (Swarm mode)
echo "my_db_password" | docker secret create db_password -

# Using environment files (not committed to git)
echo "DB_PASSWORD=secret" > .env.production
echo ".env.production" >> .gitignore

# Using external secret management
# AWS Secrets Manager, HashiCorp Vault, etc.
```

## CI/CD Integration

### GitHub Actions Example:

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Build Docker image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Run tests
        run: docker run myapp:${{ github.sha }} npm test

      - name: Security scan
        run: trivy image myapp:${{ github.sha }}

      - name: Push to registry
        run: |
          docker login -u ${{ secrets.DOCKER_USERNAME }} -p ${{ secrets.DOCKER_PASSWORD }}
          docker push myapp:${{ github.sha }}

      - name: Deploy to production
        run: |
          ssh production-server "cd /app && \
            docker compose pull && \
            docker compose up -d"
```

## Production Checklist

### Before Deploying:

- [ ] Images scanned for vulnerabilities
- [ ] Health checks configured
- [ ] Resource limits set
- [ ] Restart policies defined
- [ ] Logging configured
- [ ] Monitoring set up
- [ ] Backups automated
- [ ] Secrets secured
- [ ] Network isolation implemented
- [ ] SSL/TLS certificates configured
- [ ] Environment variables set
- [ ] .env files not committed to git
- [ ] Documentation updated
- [ ] Rollback plan ready

## Hands-On Exercise

Create a production-ready setup:

```bash
# 1. Create project structure
mkdir -p myapp/{app,nginx,prometheus,grafana,scripts,secrets}
cd myapp

# 2. Create application
# ... (your app code)

# 3. Create production Dockerfile
# ... (optimized, multi-stage)

# 4. Create docker-compose.prod.yml
# ... (with all services)

# 5. Create secrets
echo "prod_password" > secrets/db_password.txt
chmod 600 secrets/db_password.txt

# 6. Build and test locally
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d

# 7. Verify all services
docker compose ps
docker compose logs app

# 8. Run health checks
curl http://localhost/health

# 9. Check monitoring
open http://localhost:3001  # Grafana

# 10. Test backup
./scripts/backup.sh
```

## Best Practices Summary

1. ✅ Use specific image versions
2. ✅ Implement health checks
3. ✅ Set resource limits
4. ✅ Configure restart policies
5. ✅ Use centralized logging
6. ✅ Set up monitoring
7. ✅ Automate backups
8. ✅ Secure secrets
9. ✅ Implement CI/CD
10. ✅ Plan for zero-downtime deployments

## Next Steps
Move to `10-docker-troubleshooting.md` to learn debugging and problem-solving!
