# Docker Troubleshooting - Debugging and Problem Solving

## Common Issues and Solutions

### 1. Container Won't Start

#### Problem: Container exits immediately

```bash
# Check container status
docker ps -a

# View logs
docker logs my-container

# Inspect exit code
docker inspect my-container --format='{{.State.ExitCode}}'

# Common exit codes:
# 0   - Success
# 1   - Application error
# 126 - Command cannot execute
# 127 - Command not found
# 137 - Killed (OOM or SIGKILL)
# 139 - Segmentation fault
```

#### Solutions:

```bash
# Solution 1: Check logs
docker logs --tail 50 my-container

# Solution 2: Run interactively to debug
docker run -it --entrypoint /bin/sh my-image

# Solution 3: Check if command exists
docker run --rm my-image which python

# Solution 4: Override entrypoint
docker run -it --entrypoint bash my-container
```

### 2. Out of Memory (OOM) Errors

#### Problem: Container killed due to memory

```bash
# Check if container was OOM killed
docker inspect my-container --format='{{.State.OOMKilled}}'

# Check memory usage
docker stats my-container
```

#### Solutions:

```bash
# Solution 1: Increase memory limit
docker run -m 1g my-image

# Solution 2: Add swap limit
docker run -m 512m --memory-swap 1g my-image

# Solution 3: Find memory leaks in application
docker stats --no-stream

# Solution 4: Check which process uses memory
docker exec my-container ps aux --sort=-rss | head -5
```

### 3. Networking Issues

#### Problem: Can't connect to container

```bash
# Check if port is published
docker port my-container

# Check if process is listening
docker exec my-container netstat -tulpn

# Check container IP
docker inspect my-container --format='{{.NetworkSettings.IPAddress}}'
```

#### Solutions:

```bash
# Solution 1: Verify port mapping
docker run -d -p 8080:80 nginx
curl http://localhost:8080

# Solution 2: Check firewall
sudo iptables -L
sudo ufw status

# Solution 3: Test from inside container
docker exec my-container curl localhost:80

# Solution 4: Check network connectivity
docker exec my-container ping google.com

# Solution 5: Inspect network
docker network inspect bridge
```

#### Problem: Containers can't communicate

```bash
# Check if on same network
docker network inspect my-network

# Test DNS resolution
docker exec container1 nslookup container2
docker exec container1 ping container2
```

#### Solutions:

```bash
# Solution 1: Use custom network (not default bridge)
docker network create my-net
docker run -d --network my-net --name c1 alpine
docker run -d --network my-net --name c2 alpine
docker exec c1 ping c2  # Should work

# Solution 2: Check network connectivity
docker exec c1 ip route
docker exec c1 cat /etc/resolv.conf

# Solution 3: Connect container to network
docker network connect my-net my-container
```

### 4. Volume/Permission Issues

#### Problem: Permission denied errors

```bash
# Check file ownership
docker exec my-container ls -la /app/data

# Check container user
docker exec my-container whoami
docker exec my-container id
```

#### Solutions:

```bash
# Solution 1: Match user IDs
docker run -u $(id -u):$(id -g) -v $(pwd):/data my-image

# Solution 2: Change ownership in Dockerfile
# Dockerfile:
COPY --chown=appuser:appgroup . .

# Solution 3: Fix permissions on host
sudo chown -R $(id -u):$(id -g) ./data

# Solution 4: Use named volume (managed by Docker)
docker volume create my-data
docker run -v my-data:/data my-image
```

#### Problem: Data not persisting

```bash
# Check if volume is mounted
docker inspect my-container --format='{{.Mounts}}'

# List volumes
docker volume ls

# Inspect volume
docker volume inspect my-volume
```

#### Solutions:

```bash
# Solution 1: Verify volume mount
docker run -v my-volume:/data my-image
docker inspect my-container --format='{{json .Mounts}}' | jq

# Solution 2: Check if writing to correct path
docker exec my-container ls -la /data

# Solution 3: Ensure volume exists
docker volume create my-volume
```

### 5. Image Build Failures

#### Problem: Build fails

```bash
# Build with detailed output
docker build --no-cache --progress=plain .

# Check available space
df -h

# Check Docker disk usage
docker system df
```

#### Solutions:

```bash
# Solution 1: Clean up
docker system prune -a --volumes

# Solution 2: Increase build resources
docker build --memory 4g --cpu-shares 1024 .

# Solution 3: Fix Dockerfile syntax
docker build --file Dockerfile.debug .

# Solution 4: Check build context size
du -sh .

# Solution 5: Use .dockerignore
cat > .dockerignore << EOF
node_modules
.git
*.log
EOF
```

#### Problem: Layer caching issues

```bash
# Disable cache
docker build --no-cache .

# Build from specific stage
docker build --target production .
```

### 6. Container Performance Issues

#### Problem: Slow container

```bash
# Check resource usage
docker stats my-container

# Check processes
docker top my-container

# Check disk I/O
docker exec my-container iostat

# Check system calls
docker run --rm -it --pid=container:my-container nicolaka/netshoot
```

#### Solutions:

```bash
# Solution 1: Check resource limits
docker inspect my-container --format='{{.HostConfig.Memory}}'
docker inspect my-container --format='{{.HostConfig.CpuShares}}'

# Solution 2: Profile application
docker exec my-container top -b -n 1

# Solution 3: Use proper base image (Alpine for smaller size)
FROM python:3.11-alpine  # vs python:3.11

# Solution 4: Optimize layers
# Combine RUN commands to reduce layers
```

### 7. Docker Daemon Issues

#### Problem: Docker daemon not responding

```bash
# Check Docker service status
sudo systemctl status docker

# Check Docker daemon logs
sudo journalctl -u docker -n 50

# Check Docker socket
ls -la /var/run/docker.sock
```

#### Solutions:

```bash
# Solution 1: Restart Docker daemon
sudo systemctl restart docker

# Solution 2: Check disk space
df -h /var/lib/docker

# Solution 3: Clean up
docker system prune -a

# Solution 4: Check daemon config
cat /etc/docker/daemon.json
```

### 8. DNS Resolution Issues

#### Problem: Can't resolve hostnames

```bash
# Check DNS settings
docker exec my-container cat /etc/resolv.conf

# Test DNS
docker exec my-container nslookup google.com
```

#### Solutions:

```bash
# Solution 1: Set custom DNS
docker run --dns 8.8.8.8 --dns 8.8.4.4 my-image

# Solution 2: Configure daemon DNS
# /etc/docker/daemon.json
{
  "dns": ["8.8.8.8", "1.1.1.1"]
}

# Restart Docker
sudo systemctl restart docker

# Solution 3: Check network driver
docker network inspect bridge --format='{{.Options}}'
```

## Debugging Tools and Techniques

### 1. Interactive Debugging

```bash
# Run container with shell
docker run -it --rm alpine sh

# Execute shell in running container
docker exec -it my-container bash

# Override entrypoint
docker run -it --entrypoint /bin/sh my-image

# Debug multi-stage build
docker build --target builder -t debug .
docker run -it debug sh
```

### 2. Logging

```bash
# View logs
docker logs my-container

# Follow logs
docker logs -f my-container

# Last 100 lines
docker logs --tail 100 my-container

# Logs since timestamp
docker logs --since 2024-01-01T10:00:00 my-container

# Logs with timestamps
docker logs -t my-container

# Save logs to file
docker logs my-container > container.log 2>&1
```

### 3. Inspection Commands

```bash
# Full container details (JSON)
docker inspect my-container

# Specific field
docker inspect --format='{{.State.Status}}' my-container
docker inspect --format='{{.NetworkSettings.IPAddress}}' my-container
docker inspect --format='{{json .Config.Env}}' my-container | jq

# Image details
docker inspect my-image
docker history my-image

# Network details
docker network inspect my-network

# Volume details
docker volume inspect my-volume
```

### 4. Resource Monitoring

```bash
# Real-time stats
docker stats

# Stats for specific containers
docker stats my-container1 my-container2

# Stats once (no streaming)
docker stats --no-stream

# Custom format
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

### 5. Process Inspection

```bash
# View container processes
docker top my-container

# View with full command
docker top my-container aux

# View process tree
docker exec my-container ps auxf
```

### 6. Network Debugging

```bash
# Install debugging tools in container
docker exec -it my-container sh -c '
  apt-get update &&
  apt-get install -y curl iputils-ping dnsutils netcat
'

# Or use dedicated debug container
docker run -it --rm --network container:my-container nicolaka/netshoot

# Inside debug container:
ping google.com
nslookup container-name
netstat -tulpn
tcpdump -i any port 80
curl http://other-container:8080
```

### 7. Filesystem Debugging

```bash
# View file changes
docker diff my-container

# Copy files from container
docker cp my-container:/app/logs/error.log ./

# Copy files to container
docker cp ./config.json my-container:/app/

# Check disk usage
docker exec my-container du -sh /*
docker exec my-container df -h
```

### 8. Advanced Debugging with nsenter

```bash
# Get container PID
PID=$(docker inspect --format '{{.State.Pid}}' my-container)

# Enter container namespace
sudo nsenter -t $PID -n -m -p

# Now you're inside container with full access
ps aux
netstat -tulpn
```

## Debugging Docker Compose

```bash
# Validate compose file
docker compose config

# Check which files are used
docker compose config --files

# View resolved configuration
docker compose config --services

# Debug specific service
docker compose up --build --force-recreate web

# View logs for all services
docker compose logs

# View logs for specific service
docker compose logs -f api

# Check service dependencies
docker compose config --resolve-image-digests
```

## Common Error Messages

### "No space left on device"

```bash
# Check disk usage
docker system df

# Clean up
docker system prune -a --volumes

# Remove specific items
docker volume prune
docker image prune -a
docker container prune
```

### "port is already allocated"

```bash
# Find what's using the port
sudo lsof -i :8080
sudo netstat -tulpn | grep 8080

# Stop container using port
docker ps -a | grep 8080
docker stop <container-id>

# Use different port
docker run -p 8081:80 nginx
```

### "conflict: unable to remove repository reference"

```bash
# Check what's using the image
docker ps -a --filter ancestor=my-image

# Stop and remove containers
docker rm -f $(docker ps -aq --filter ancestor=my-image)

# Force remove image
docker rmi -f my-image
```

### "driver failed programming external connectivity"

```bash
# Restart Docker daemon
sudo systemctl restart docker

# Check iptables
sudo iptables -L

# Flush Docker iptables rules
sudo iptables -t nat -F DOCKER
```

## Best Practices for Troubleshooting

1. **Check Logs First**: `docker logs` is your friend
2. **Use --rm Flag**: For testing, auto-remove containers
3. **Run Interactively**: Use `-it` for debugging
4. **Small Changes**: Test one change at a time
5. **Check Basics**: Ports, networks, volumes
6. **Isolate Issues**: Test components separately
7. **Use Debug Images**: Include debugging tools
8. **Monitor Resources**: Use `docker stats`
9. **Read Error Messages**: They usually tell you what's wrong
10. **Clean Up Regularly**: `docker system prune`

## Debugging Checklist

When something doesn't work:

- [ ] Check container status: `docker ps -a`
- [ ] View logs: `docker logs container-name`
- [ ] Inspect container: `docker inspect container-name`
- [ ] Check resources: `docker stats`
- [ ] Verify network: `docker network inspect`
- [ ] Check volumes: `docker volume inspect`
- [ ] Test connectivity: `docker exec ... ping/curl`
- [ ] Check DNS: `docker exec ... nslookup`
- [ ] View processes: `docker top container-name`
- [ ] Check filesystem: `docker diff container-name`

## Hands-On Exercise

Debug a broken application:

```bash
# 1. Run intentionally broken container
docker run -d --name broken-app \
  -e DATABASE_URL=wrong_url \
  -p 8080:3000 \
  my-broken-app

# 2. Container exits - debug it!
docker ps -a  # Check status
docker logs broken-app  # Check logs
docker inspect broken-app  # Check config

# 3. Fix and restart
docker rm broken-app
docker run -d --name fixed-app \
  -e DATABASE_URL=postgresql://user:pass@db:5432/mydb \
  -p 8080:3000 \
  my-app

# 4. Verify it works
curl http://localhost:8080/health
```

## Helpful Debugging One-Liners

```bash
# Find container using most CPU
docker stats --no-stream | sort -k3 -h | tail -1

# Find container using most memory
docker stats --no-stream | sort -k4 -h | tail -1

# List all container IPs
docker ps -q | xargs docker inspect --format='{{.Name}} {{.NetworkSettings.IPAddress}}'

# Remove all stopped containers
docker rm $(docker ps -aq -f status=exited)

# Remove all dangling images
docker rmi $(docker images -qf dangling=true)

# Get container environment variables
docker exec my-container env

# Check if port is accessible from host
docker run --rm --network host alpine nc -zv localhost 8080
```

## Congratulations!

You've completed the Docker learning path! You now know:
- Docker basics and concepts
- Images and Dockerfiles
- Container management
- Volumes and data persistence
- Networking
- Docker Compose
- Security best practices
- Production deployments
- Troubleshooting and debugging

**Next**: Move to the Kubernetes section to learn container orchestration!
