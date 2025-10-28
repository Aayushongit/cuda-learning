# Docker Containers - Running and Managing

## Theory

### Container Lifecycle:

```
Created → Running → Paused → Stopped → Removed
   ↓         ↓         ↓         ↓
 start     pause    unpause   restart
```

### Container States:

1. **Created**: Container exists but hasn't started
2. **Running**: Container is actively running
3. **Paused**: Container process is paused (frozen)
4. **Stopped/Exited**: Container has stopped
5. **Removed**: Container deleted from system

### How Containers Work:

Containers use Linux features:
- **Namespaces**: Isolation (PID, network, mount, etc.)
- **Cgroups**: Resource limits (CPU, memory)
- **Union File Systems**: Layered file system

## Container Management Commands

### Creating and Running:

```bash
# Create without starting
docker create --name my-nginx nginx

# Start a created container
docker start my-nginx

# Create and start (run)
docker run nginx

# Run in detached mode (background)
docker run -d nginx

# Run with custom name
docker run -d --name web-server nginx

# Run interactively (attach terminal)
docker run -it ubuntu bash

# Run with automatic removal after exit
docker run --rm alpine echo "Hello"

# Run with restart policy
docker run -d --restart=always nginx
docker run -d --restart=on-failure:3 nginx
docker run -d --restart=unless-stopped nginx
```

### Listing Containers:

```bash
# Show running containers
docker ps

# Show all containers (including stopped)
docker ps -a

# Show only container IDs
docker ps -q

# Show last created container
docker ps -l

# Show containers with custom format
docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"

# Show container size
docker ps -s

# Filter containers
docker ps --filter "status=exited"
docker ps --filter "name=web"
```

### Stopping and Starting:

```bash
# Stop a running container (graceful, sends SIGTERM)
docker stop my-container

# Stop with timeout (default 10 seconds)
docker stop -t 30 my-container

# Kill a container (forceful, sends SIGKILL)
docker kill my-container

# Start a stopped container
docker start my-container

# Restart a container
docker restart my-container

# Pause a running container
docker pause my-container

# Unpause a paused container
docker unpause my-container
```

### Removing Containers:

```bash
# Remove a stopped container
docker rm my-container

# Force remove a running container
docker rm -f my-container

# Remove all stopped containers
docker container prune

# Remove specific containers
docker rm container1 container2 container3

# Remove all containers (dangerous!)
docker rm -f $(docker ps -aq)
```

## Interacting with Containers

### Executing Commands:

```bash
# Execute command in running container
docker exec my-container ls /app

# Execute interactive command
docker exec -it my-container bash

# Execute as specific user
docker exec -u root my-container whoami

# Execute with environment variable
docker exec -e MY_VAR=value my-container env

# Execute in specific directory
docker exec -w /app my-container pwd
```

### Viewing Logs:

```bash
# View container logs
docker logs my-container

# Follow logs (like tail -f)
docker logs -f my-container

# Show timestamps
docker logs -t my-container

# Show last N lines
docker logs --tail 100 my-container

# Show logs since timestamp
docker logs --since 2024-01-01T00:00:00 my-container

# Show logs in last duration
docker logs --since 10m my-container
```

### Inspecting Containers:

```bash
# Detailed container information (JSON)
docker inspect my-container

# Get specific field
docker inspect --format='{{.State.Status}}' my-container
docker inspect --format='{{.NetworkSettings.IPAddress}}' my-container

# View container processes
docker top my-container

# View container resource usage (live)
docker stats my-container

# View all containers stats
docker stats

# View port mappings
docker port my-container

# View container changes from image
docker diff my-container
```

## Port Mapping

### Publishing Ports:

```bash
# Map container port to host port
# -p <host-port>:<container-port>
docker run -d -p 8080:80 nginx

# Map to random host port
docker run -d -p 80 nginx

# Map specific host IP
docker run -d -p 127.0.0.1:8080:80 nginx

# Map multiple ports
docker run -d -p 8080:80 -p 8443:443 nginx

# Map all exposed ports to random host ports
docker run -d -P nginx

# Check port mappings
docker port my-container
```

## Volume Mounting

### Bind Mounts (Host Directory):

```bash
# Mount host directory to container
# -v <host-path>:<container-path>
docker run -d -v /home/user/data:/app/data nginx

# Mount as read-only
docker run -d -v /home/user/data:/app/data:ro nginx

# Mount current directory
docker run -d -v $(pwd):/app nginx

# Multiple mounts
docker run -d \
  -v /home/user/data:/app/data \
  -v /home/user/config:/app/config \
  nginx
```

### Named Volumes:

```bash
# Create a volume
docker volume create my-volume

# Use volume in container
docker run -d -v my-volume:/app/data nginx

# List volumes
docker volume ls

# Inspect volume
docker volume inspect my-volume

# Remove volume
docker volume rm my-volume

# Remove all unused volumes
docker volume prune
```

## Resource Constraints

### Limiting Resources:

```bash
# Limit memory
docker run -d --memory="512m" nginx

# Limit CPU (0.5 = 50% of one CPU core)
docker run -d --cpus="0.5" nginx

# Limit CPU shares (relative weight)
docker run -d --cpu-shares=512 nginx

# Set CPU affinity (which cores to use)
docker run -d --cpuset-cpus="0,1" nginx

# Limit disk I/O
docker run -d --device-write-bps /dev/sda:1mb nginx
```

## Networking

### Basic Networking:

```bash
# Run with default bridge network
docker run -d nginx

# Run with host network (use host's network stack)
docker run -d --network host nginx

# Run without network
docker run -d --network none nginx

# List networks
docker network ls

# Create custom network
docker network create my-network

# Run container in custom network
docker run -d --network my-network --name web1 nginx

# Connect running container to network
docker network connect my-network web2

# Disconnect from network
docker network disconnect my-network web2
```

## Hands-On Exercise

### Exercise 1: Run a Web Server

```bash
# 1. Run nginx
docker run -d \
  --name my-web \
  -p 8080:80 \
  -v $(pwd)/html:/usr/share/nginx/html:ro \
  nginx

# 2. Create content
mkdir html
echo "<h1>Hello Docker!</h1>" > html/index.html

# 3. Test
curl http://localhost:8080

# 4. View logs
docker logs -f my-web

# 5. Execute command inside
docker exec my-web cat /etc/nginx/nginx.conf

# 6. Check stats
docker stats my-web

# 7. Stop and remove
docker stop my-web
docker rm my-web
```

### Exercise 2: Container Communication

```bash
# 1. Create network
docker network create my-app-network

# 2. Run database
docker run -d \
  --name postgres-db \
  --network my-app-network \
  -e POSTGRES_PASSWORD=secret \
  postgres:15

# 3. Run application (can access postgres-db by name)
docker run -d \
  --name my-app \
  --network my-app-network \
  -e DATABASE_URL=postgresql://postgres:secret@postgres-db:5432/mydb \
  my-app-image

# 4. Containers can communicate using container names as hostnames
docker exec my-app ping postgres-db
```

## Container Best Practices

1. **One Process per Container**: Each container should do one thing
2. **Use .dockerignore**: Exclude unnecessary files
3. **Don't Run as Root**: Use USER instruction in Dockerfile
4. **Use Named Volumes**: For persistent data
5. **Health Checks**: Define health check commands
6. **Resource Limits**: Set memory and CPU limits
7. **Clean Up**: Remove stopped containers regularly
8. **Use Restart Policies**: For production containers

## Next Steps
Move to `05-docker-volumes.md` to learn about persistent data storage!
