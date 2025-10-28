# Docker Security - Best Practices

## Theory

### Container Security Layers:

```
┌─────────────────────────────────┐
│  1. Image Security               │ ← Secure base images, scan vulnerabilities
├─────────────────────────────────┤
│  2. Container Runtime            │ ← User permissions, capabilities, seccomp
├─────────────────────────────────┤
│  3. Host Security                │ ← OS hardening, updates, firewall
├─────────────────────────────────┤
│  4. Network Security             │ ← Network isolation, TLS
├─────────────────────────────────┤
│  5. Secrets Management           │ ← Environment variables, secret stores
└─────────────────────────────────┘
```

### Security Principles:

1. **Least Privilege**: Minimal permissions required
2. **Defense in Depth**: Multiple security layers
3. **Immutability**: Containers as disposable
4. **Isolation**: Separate concerns and environments
5. **Visibility**: Logging and monitoring

## Image Security

### 1. Use Official Images:

```dockerfile
# ✅ GOOD: Official image
FROM node:18-alpine

# ⚠️ RISKY: Unknown source
FROM randomuser/node
```

### 2. Use Specific Tags:

```dockerfile
# ❌ BAD: Mutable tag
FROM python:latest

# ✅ GOOD: Specific version
FROM python:3.11.6-slim-bookworm

# ✅ BETTER: SHA256 digest (immutable)
FROM python@sha256:abc123...
```

### 3. Minimal Base Images:

```dockerfile
# Larger attack surface (100+ MB)
FROM ubuntu:22.04

# Smaller attack surface (5-10 MB)
FROM alpine:3.18

# Minimal (2-3 MB, no shell)
FROM scratch  # For compiled binaries only
```

### 4. Multi-Stage Builds (Reduce Attack Surface):

```dockerfile
# Stage 1: Build
FROM golang:1.21 AS builder
WORKDIR /app
COPY . .
RUN go build -o main .

# Stage 2: Runtime (minimal)
FROM alpine:3.18
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/main .
USER nobody  # Don't run as root!
CMD ["./main"]
```

### 5. Scan Images for Vulnerabilities:

```bash
# Using Docker Scout
docker scout cve nginx:latest

# Using Trivy
trivy image nginx:latest

# Using Snyk
snyk container test nginx:latest

# Fail build on high vulnerabilities
trivy image --severity HIGH,CRITICAL --exit-code 1 myimage:latest
```

## User Permissions

### Don't Run as Root:

```dockerfile
# ❌ BAD: Runs as root (UID 0)
FROM node:18
WORKDIR /app
COPY . .
CMD ["node", "server.js"]

# ✅ GOOD: Run as non-root user
FROM node:18
WORKDIR /app
COPY . .
RUN chown -R node:node /app
USER node  # Switch to non-root user
CMD ["node", "server.js"]
```

### Create Custom User:

```dockerfile
FROM alpine:3.18

# Create user and group
RUN addgroup -g 1000 appgroup && \
    adduser -D -u 1000 -G appgroup appuser

# Set ownership
WORKDIR /app
COPY --chown=appuser:appgroup . .

# Switch to user
USER appuser

CMD ["./app"]
```

### Runtime User Override:

```bash
# Run as specific user
docker run --user 1000:1000 myimage

# Run as current user (useful for file permissions)
docker run --user $(id -u):$(id -g) myimage
```

## Security Options

### 1. Read-Only Root Filesystem:

```bash
# Make root filesystem read-only
docker run --read-only myimage

# With temporary writable directory
docker run \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /var/run \
  myimage
```

```dockerfile
# In Dockerfile, mark volumes for writable directories
FROM nginx:alpine
VOLUME /var/cache/nginx
VOLUME /var/run
```

### 2. Drop Capabilities:

```bash
# Drop all capabilities, add only what's needed
docker run \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  myimage

# Common capabilities:
# NET_BIND_SERVICE - Bind to ports < 1024
# CHOWN - Change file ownership
# DAC_OVERRIDE - Bypass file permission checks
# SETUID, SETGID - Set user/group ID

# List container capabilities
docker run --rm alpine sh -c 'apk add -q libcap; capsh --print'
```

### 3. Security Profiles:

```bash
# Use AppArmor profile
docker run --security-opt apparmor=docker-default myimage

# Use custom seccomp profile
docker run --security-opt seccomp=/path/to/profile.json myimage

# Disable seccomp (not recommended!)
docker run --security-opt seccomp=unconfined myimage

# SELinux labels
docker run --security-opt label=level:s0:c100,c200 myimage
```

### 4. No New Privileges:

```bash
# Prevent privilege escalation
docker run --security-opt=no-new-privileges:true myimage
```

```yaml
# In docker-compose.yml
services:
  web:
    image: nginx
    security_opt:
      - no-new-privileges:true
```

## Network Security

### 1. Network Isolation:

```bash
# Create isolated networks
docker network create frontend
docker network create backend

# Database only on backend
docker run -d --network backend --name db postgres

# API on both networks
docker run -d --network backend --name api my-api
docker network connect frontend api

# Web only on frontend
docker run -d --network frontend -p 80:80 nginx
```

### 2. Disable Inter-Container Communication:

```bash
# Create network with ICC disabled
docker network create --opt com.docker.network.bridge.enable_icc=false mynet

# Only explicitly linked containers can communicate
```

### 3. Use TLS:

```yaml
services:
  nginx:
    image: nginx
    ports:
      - "443:443"
    volumes:
      - ./ssl/cert.pem:/etc/nginx/cert.pem:ro
      - ./ssl/key.pem:/etc/nginx/key.pem:ro
```

## Secrets Management

### 1. Never Hardcode Secrets:

```dockerfile
# ❌ BAD: Hardcoded secrets
ENV API_KEY="secret123"
ENV DATABASE_PASSWORD="password"

# ✅ GOOD: Use runtime environment variables
ENV API_KEY=""
ENV DATABASE_PASSWORD=""
```

### 2. Use Docker Secrets (Swarm):

```bash
# Create secret
echo "mypassword" | docker secret create db_password -

# Use in service
docker service create \
  --name db \
  --secret db_password \
  postgres

# Access in container at: /run/secrets/db_password
```

### 3. Use Environment Variables:

```bash
# Pass at runtime
docker run -e DATABASE_PASSWORD="$(cat password.txt)" myimage

# Use .env file (don't commit to git!)
docker compose --env-file .env.prod up
```

### 4. Use Secret Management Tools:

```yaml
# docker-compose.yml with HashiCorp Vault
services:
  app:
    image: myapp
    environment:
      - VAULT_ADDR=http://vault:8200
      - VAULT_TOKEN=${VAULT_TOKEN}
    command: sh -c 'vault kv get -field=password secret/db > /tmp/pass && app'
```

## Resource Limits

### Prevent DoS Attacks:

```bash
# Limit memory
docker run -m 512m --memory-reservation 256m myimage

# Limit CPU
docker run --cpus="1.5" myimage

# Limit PIDs (prevent fork bombs)
docker run --pids-limit 100 myimage

# Limit restarts
docker run --restart=on-failure:3 myimage
```

```yaml
# docker-compose.yml
services:
  app:
    image: myapp
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
          pids: 100
        reservations:
          cpus: '0.25'
          memory: 256M
```

## Logging and Monitoring

### Enable Logging:

```bash
# Configure logging driver
docker run \
  --log-driver=json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  myimage

# Send to syslog
docker run --log-driver=syslog myimage

# Send to external system
docker run \
  --log-driver=fluentd \
  --log-opt fluentd-address=localhost:24224 \
  myimage
```

### Monitor Containers:

```bash
# View resource usage
docker stats

# Inspect security settings
docker inspect --format='{{.HostConfig.SecurityOpt}}' mycontainer

# Check running processes
docker top mycontainer

# Audit events
docker events --filter type=container
```

## Practical Security Checklist

### Dockerfile Security Checklist:

```dockerfile
# ✅ Security-hardened Dockerfile

# 1. Use specific base image
FROM python:3.11.6-slim-bookworm

# 2. Update packages and remove cache
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 3. Create non-root user
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -u 1000 appuser

# 4. Set working directory
WORKDIR /app

# 5. Copy with correct ownership
COPY --chown=appuser:appgroup requirements.txt .

# 6. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy application
COPY --chown=appuser:appgroup . .

# 8. Switch to non-root user
USER appuser

# 9. Use non-root port
EXPOSE 8080

# 10. Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8080/health || exit 1

# 11. Run application
CMD ["python", "app.py"]
```

### Runtime Security:

```bash
docker run -d \
  --name secure-app \
  --read-only \
  --tmpfs /tmp \
  --user 1000:1000 \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --security-opt=no-new-privileges:true \
  --memory=512m \
  --cpus="1.0" \
  --pids-limit=100 \
  --restart=on-failure:3 \
  -p 8080:8080 \
  myapp:latest
```

## Hands-On Exercise

### Secure Flask Application:

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -u 1000 -m flaskuser

WORKDIR /app

# Install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app with correct ownership
COPY --chown=flaskuser:flaskuser app.py .

# Switch to non-root user
USER flaskuser

# Health check
HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["python", "app.py"]
```

**Run securely:**
```bash
docker build -t secure-flask .

docker run -d \
  --name flask-secure \
  --read-only \
  --tmpfs /tmp:noexec,nosuid,size=64m \
  --cap-drop=ALL \
  --security-opt=no-new-privileges:true \
  -m 256m \
  --cpus="0.5" \
  -p 8080:8080 \
  secure-flask
```

## Security Tools

```bash
# 1. Docker Bench Security
git clone https://github.com/docker/docker-bench-security.git
cd docker-bench-security
sudo sh docker-bench-security.sh

# 2. Trivy (vulnerability scanner)
trivy image myimage:latest

# 3. Anchore Engine
anchore-cli image add myimage:latest
anchore-cli image vuln myimage:latest all

# 4. Clair
clairctl analyze myimage:latest

# 5. Hadolint (Dockerfile linter)
docker run --rm -i hadolint/hadolint < Dockerfile
```

## Best Practices Summary

1. ✅ Use official, minimal base images
2. ✅ Use specific image tags (not `latest`)
3. ✅ Run as non-root user
4. ✅ Use read-only filesystem where possible
5. ✅ Drop unnecessary capabilities
6. ✅ Set resource limits
7. ✅ Scan images for vulnerabilities
8. ✅ Use secrets management
9. ✅ Enable logging and monitoring
10. ✅ Keep images updated
11. ✅ Use multi-stage builds
12. ✅ Implement health checks
13. ✅ Network isolation
14. ✅ Regular security audits

## Next Steps
Move to `09-docker-production.md` to learn about production deployments!
