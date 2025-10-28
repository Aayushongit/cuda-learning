# Docker Compose - Multi-Container Applications

## Theory

### What is Docker Compose?

Docker Compose is a tool for defining and running **multi-container applications**.

**Without Compose:**
```bash
docker network create mynet
docker run -d --name db --network mynet postgres
docker run -d --name api --network mynet my-api
docker run -d --name web --network mynet -p 80:80 nginx
```

**With Compose:**
```yaml
# docker-compose.yml
services:
  db:
    image: postgres
  api:
    image: my-api
  web:
    image: nginx
    ports:
      - "80:80"
```
```bash
docker compose up
```

### Key Concepts:

1. **docker-compose.yml**: Configuration file (YAML format)
2. **Services**: Containers defined in the compose file
3. **Networks**: Automatically created for inter-service communication
4. **Volumes**: Persistent data storage
5. **Environment**: Configuration and secrets

### Compose File Structure:

```yaml
version: '3.8'  # Optional in newer versions

services:       # Define containers
  service1:
    # Container configuration
  service2:
    # Container configuration

networks:       # Define networks (optional)
  network1:

volumes:        # Define volumes (optional)
  volume1:
```

## Basic docker-compose.yml

### Simple Example:

```yaml
services:
  # Web service
  web:
    image: nginx:latest
    ports:
      - "8080:80"
    volumes:
      - ./html:/usr/share/nginx/html

  # Database service
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: myapp
    volumes:
      - db-data:/var/lib/postgresql/data

# Named volumes
volumes:
  db-data:
```

### Using Custom Images (Dockerfile):

```yaml
services:
  web:
    build: .  # Build from Dockerfile in current directory
    ports:
      - "3000:3000"

  api:
    build:
      context: ./api  # Path to Dockerfile directory
      dockerfile: Dockerfile.dev  # Custom Dockerfile name
      args:
        - NODE_ENV=development
    depends_on:
      - db

  db:
    image: postgres:15
```

## Docker Compose Commands

### Essential Commands:

```bash
# Start all services (detached mode)
docker compose up -d

# Start services (foreground, see logs)
docker compose up

# Build and start
docker compose up --build

# Start specific services
docker compose up web db

# Stop all services
docker compose down

# Stop and remove volumes
docker compose down -v

# Stop and remove images
docker compose down --rmi all

# View running services
docker compose ps

# View logs
docker compose logs

# Follow logs
docker compose logs -f

# Logs for specific service
docker compose logs -f web

# Execute command in service
docker compose exec web bash

# Run one-off command
docker compose run web npm test

# Restart services
docker compose restart

# Stop services (don't remove)
docker compose stop

# Start stopped services
docker compose start

# Pause services
docker compose pause

# Unpause services
docker compose unpause

# View service configuration
docker compose config

# Validate compose file
docker compose config --quiet
```

## Complete Example: Full-Stack Application

### Project Structure:
```
my-app/
├── docker-compose.yml
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
└── nginx/
    └── nginx.conf
```

### docker-compose.yml:

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  db:
    image: postgres:15-alpine
    container_name: myapp-db
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${DB_USER:-postgres}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-secret}
      POSTGRES_DB: ${DB_NAME:-myapp}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - backend
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: myapp-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - backend
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  # Backend API
  api:
    build:
      context: ./backend
      dockerfile: Dockerfile
      target: development
    container_name: myapp-api
    restart: unless-stopped
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://${DB_USER:-postgres}:${DB_PASSWORD:-secret}@db:5432/${DB_NAME:-myapp}
      - REDIS_URL=redis://redis:6379
      - FLASK_ENV=development
      - SECRET_KEY=${SECRET_KEY:-dev-secret-key}
    volumes:
      - ./backend:/app
      - /app/venv  # Anonymous volume for virtual env
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - frontend
      - backend
    command: python app.py

  # Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      args:
        - REACT_APP_API_URL=http://localhost:5000
    container_name: myapp-frontend
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
    volumes:
      - ./frontend/src:/app/src
      - /app/node_modules
    depends_on:
      - api
    networks:
      - frontend
    stdin_open: true  # Keep stdin open for React
    tty: true

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: myapp-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - frontend
      - api
    networks:
      - frontend

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge

volumes:
  postgres-data:
    driver: local
  redis-data:
    driver: local
```

### .env File:

```bash
# .env (environment variables)
DB_USER=myuser
DB_PASSWORD=mypassword
DB_NAME=production_db
SECRET_KEY=super-secret-key-change-in-production
```

## Advanced Compose Features

### 1. Extending Services:

```yaml
# docker-compose.base.yml
services:
  web:
    image: nginx
    volumes:
      - ./html:/usr/share/nginx/html

# docker-compose.override.yml (automatically used)
services:
  web:
    ports:
      - "8080:80"

# docker-compose.prod.yml
services:
  web:
    ports:
      - "80:80"

# Use:
docker compose -f docker-compose.base.yml -f docker-compose.prod.yml up
```

### 2. Profiles:

```yaml
services:
  web:
    image: nginx

  db:
    image: postgres

  debug:
    image: alpine
    profiles:
      - debug
    command: sleep infinity

# Start without debug:
docker compose up

# Start with debug:
docker compose --profile debug up
```

### 3. Resource Limits:

```yaml
services:
  web:
    image: nginx
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
```

### 4. Health Checks:

```yaml
services:
  web:
    image: nginx
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 5. Dependencies with Conditions:

```yaml
services:
  api:
    image: my-api
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
```

## Development vs Production

### docker-compose.dev.yml:

```yaml
services:
  api:
    build:
      target: development
    volumes:
      - ./backend:/app  # Live reload
    environment:
      - DEBUG=true
    ports:
      - "5000:5000"  # Expose for debugging
```

### docker-compose.prod.yml:

```yaml
services:
  api:
    build:
      target: production
    restart: always
    environment:
      - DEBUG=false
    # No volume mounts
    # No port exposure (behind nginx)
```

## Hands-On Exercise

### Exercise 1: WordPress with MySQL

Create `docker-compose.yml`:

```yaml
services:
  db:
    image: mysql:8
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: rootpass
      MYSQL_DATABASE: wordpress
      MYSQL_USER: wpuser
      MYSQL_PASSWORD: wppass
    volumes:
      - db-data:/var/lib/mysql

  wordpress:
    image: wordpress:latest
    restart: always
    ports:
      - "8080:80"
    environment:
      WORDPRESS_DB_HOST: db
      WORDPRESS_DB_USER: wpuser
      WORDPRESS_DB_PASSWORD: wppass
      WORDPRESS_DB_NAME: wordpress
    volumes:
      - wp-data:/var/www/html
    depends_on:
      - db

volumes:
  db-data:
  wp-data:
```

```bash
# Start
docker compose up -d

# Visit http://localhost:8080

# View logs
docker compose logs -f

# Stop
docker compose down

# Remove everything including volumes
docker compose down -v
```

### Exercise 2: Development Environment

```yaml
services:
  # Node.js development
  app:
    image: node:18
    working_dir: /app
    volumes:
      - ./:/app
    ports:
      - "3000:3000"
    command: npm run dev
    environment:
      - NODE_ENV=development

  # MongoDB
  mongo:
    image: mongo:6
    ports:
      - "27017:27017"
    volumes:
      - mongo-data:/data/db

  # Mongo Express (GUI)
  mongo-express:
    image: mongo-express
    ports:
      - "8081:8081"
    environment:
      ME_CONFIG_MONGODB_URL: mongodb://mongo:27017/
    depends_on:
      - mongo

volumes:
  mongo-data:
```

## Best Practices

1. **Use .env Files**: For environment-specific config
2. **Version Control**: Commit compose files, not .env
3. **Named Volumes**: For persistent data
4. **Health Checks**: Ensure services are ready
5. **Resource Limits**: Prevent resource exhaustion
6. **Networks**: Isolate services (frontend/backend)
7. **Restart Policies**: `unless-stopped` for production
8. **Build Context**: Keep it minimal with .dockerignore
9. **Service Names**: Use descriptive names
10. **Documentation**: Comment complex configurations

## Troubleshooting

```bash
# View full configuration (with variable substitution)
docker compose config

# Check service status
docker compose ps

# View resource usage
docker compose stats

# Recreate containers
docker compose up -d --force-recreate

# Rebuild images
docker compose build --no-cache

# Remove orphaned containers
docker compose down --remove-orphans

# View service logs
docker compose logs service-name

# Follow all logs
docker compose logs -f
```

## Next Steps
Move to `08-docker-security.md` to learn about securing Docker containers!
