# Dockerfile - Creating Custom Images

## Theory

### What is a Dockerfile?

A **Dockerfile** is a text file containing instructions to build a Docker image automatically.

### Common Dockerfile Instructions:

1. **FROM**: Specifies the base image
2. **WORKDIR**: Sets the working directory
3. **COPY**: Copies files from host to container
4. **ADD**: Like COPY, but can extract archives and download URLs
5. **RUN**: Executes commands during image build
6. **CMD**: Default command when container starts
7. **ENTRYPOINT**: Configures container to run as an executable
8. **ENV**: Sets environment variables
9. **EXPOSE**: Documents which ports the container listens on
10. **VOLUME**: Creates a mount point for persistent data
11. **USER**: Sets the user for subsequent instructions
12. **ARG**: Defines build-time variables

### Build Process:

```
Dockerfile → docker build → Image → docker run → Container
```

## Practical Examples

### Example 1: Simple Python Application

Create a file named `Dockerfile`:

```dockerfile
# Use official Python image as base
# FROM instruction must be first (except for ARG used before FROM)
FROM python:3.11-slim

# Set working directory in container
# All subsequent commands run from this directory
WORKDIR /app

# Copy requirements file
# COPY <source-on-host> <destination-in-container>
COPY requirements.txt .

# Install Python dependencies
# RUN executes commands during build time
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# . means current directory (WORKDIR)
COPY . .

# Expose port 5000
# This is documentation; doesn't actually publish the port
EXPOSE 5000

# Set environment variable
ENV FLASK_APP=app.py

# Command to run when container starts
# CMD executes at runtime (when container starts)
CMD ["python", "app.py"]
```

### Example 2: Node.js Application

```dockerfile
# Multi-stage build for smaller final image
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files first (better caching)
# If package.json doesn't change, this layer is cached
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM node:18-alpine

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules

# Use non-root user for security
USER node

EXPOSE 3000

CMD ["node", "dist/index.js"]
```

### Example 3: Multi-stage Build for Go

```dockerfile
# Stage 1: Build
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
# CGO_ENABLED=0 creates a static binary
RUN CGO_ENABLED=0 GOOS=linux go build -o main .

# Stage 2: Runtime
# Use minimal image for smaller size
FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy binary from builder
COPY --from=builder /app/main .

EXPOSE 8080

CMD ["./main"]
```

## Building Images

### Basic Build:

```bash
# Build from Dockerfile in current directory
docker build -t myapp:v1 .

# -t: Tag the image
# .: Build context (current directory)

# Build with a specific Dockerfile
docker build -f Dockerfile.prod -t myapp:prod .

# Build with build arguments
docker build --build-arg VERSION=1.0 -t myapp:v1 .
```

### Using Build Arguments:

```dockerfile
# In Dockerfile
ARG VERSION=latest
FROM python:${VERSION}

ARG APP_DIR=/app
WORKDIR ${APP_DIR}

# Build-time variable (not available at runtime)
ARG BUILD_DATE
RUN echo "Built on ${BUILD_DATE}"

# Convert to runtime variable if needed
ENV VERSION=${VERSION}
```

```bash
# Building with arguments
docker build \
  --build-arg VERSION=3.11 \
  --build-arg BUILD_DATE=$(date) \
  -t myapp:latest .
```

## CMD vs ENTRYPOINT

### CMD: Default command (can be overridden)

```dockerfile
FROM ubuntu
CMD ["echo", "Hello World"]
```

```bash
docker run myimage              # Output: Hello World
docker run myimage echo "Bye"   # Output: Bye (CMD overridden)
```

### ENTRYPOINT: Container as executable (harder to override)

```dockerfile
FROM ubuntu
ENTRYPOINT ["echo"]
CMD ["Hello World"]
```

```bash
docker run myimage              # Output: Hello World
docker run myimage "Bye"        # Output: Bye (CMD replaced, ENTRYPOINT stays)
```

## Best Practices

### 1. Order Matters (Caching):

```dockerfile
# ❌ BAD: Changes to code invalidate all layers
FROM python:3.11
COPY . .
RUN pip install -r requirements.txt

# ✅ GOOD: Dependencies cached unless requirements.txt changes
FROM python:3.11
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

### 2. Use .dockerignore:

Create `.dockerignore` file:

```
# Git files
.git
.gitignore

# Documentation
README.md
docs/

# Development files
node_modules
__pycache__
*.pyc
.pytest_cache

# IDE
.vscode
.idea

# Environment files
.env
*.log
```

### 3. Minimize Layers:

```dockerfile
# ❌ BAD: Multiple layers
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y vim

# ✅ GOOD: Single layer
RUN apt-get update && apt-get install -y \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*
```

### 4. Use Multi-stage Builds:

Reduces final image size by excluding build tools.

## Hands-On Exercise

Create this project structure:

```
my-python-app/
├── Dockerfile
├── requirements.txt
└── app.py
```

**app.py**:
```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello from Docker!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**requirements.txt**:
```
flask==3.0.0
```

**Dockerfile**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 5000
CMD ["python", "app.py"]
```

**Build and Run**:
```bash
# Build
docker build -t my-flask-app:v1 .

# Run
docker run -d -p 5000:5000 --name flask-app my-flask-app:v1

# Test
curl http://localhost:5000

# Check logs
docker logs flask-app

# Stop and remove
docker stop flask-app
docker rm flask-app
```

## Next Steps
Move to `04-docker-containers.md` to learn container management in depth!
