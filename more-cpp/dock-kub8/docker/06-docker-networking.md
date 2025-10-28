# Docker Networking - Container Communication

## Theory

### Docker Network Architecture:

Docker creates isolated network environments for containers using:
- **Virtual Ethernet Bridges**: Connect containers
- **iptables**: Firewall rules for port forwarding
- **Network Namespaces**: Isolated network stacks

### Network Drivers:

```
1. BRIDGE (default)
   - Containers on same host
   - Private network with NAT

2. HOST
   - Uses host's network directly
   - No isolation

3. NONE
   - No networking
   - Completely isolated

4. OVERLAY
   - Multi-host networking
   - Docker Swarm

5. MACVLAN
   - Assign MAC address to container
   - Appears as physical device
```

## Bridge Networks (Default)

### How Bridge Networks Work:

```
Host Network Stack
    ↓
docker0 bridge (172.17.0.1)
    ├─→ Container A (172.17.0.2)
    ├─→ Container B (172.17.0.3)
    └─→ Container C (172.17.0.4)
```

### Default Bridge Network:

```bash
# Run container on default bridge
docker run -d --name web1 nginx

# Inspect default bridge
docker network inspect bridge

# Containers can communicate via IP
docker exec web1 ping 172.17.0.3

# BUT: Cannot use container names as hostnames on default bridge
docker exec web1 ping web2  # ❌ Won't work
```

### Custom Bridge Networks (Better!):

```bash
# Create custom bridge network
docker network create my-network

# Run containers on custom network
docker run -d --name web1 --network my-network nginx
docker run -d --name web2 --network my-network nginx

# Containers can communicate using names (DNS!)
docker exec web1 ping web2  # ✅ Works!

# Inspect network
docker network inspect my-network

# See connected containers
docker network inspect -f '{{range .Containers}}{{.Name}} {{end}}' my-network
```

### Bridge Network Options:

```bash
# Create with specific subnet
docker network create \
  --subnet=192.168.100.0/24 \
  --gateway=192.168.100.1 \
  my-network

# Create with IP range
docker network create \
  --subnet=172.20.0.0/16 \
  --ip-range=172.20.240.0/20 \
  my-network

# Assign static IP to container
docker run -d \
  --network my-network \
  --ip 192.168.100.10 \
  --name web \
  nginx
```

## Network Commands

### Basic Network Operations:

```bash
# List networks
docker network ls

# Create network
docker network create my-net

# Remove network
docker network rm my-net

# Remove all unused networks
docker network prune

# Inspect network
docker network inspect my-net

# Connect running container to network
docker network connect my-net my-container

# Disconnect from network
docker network disconnect my-net my-container
```

### Multi-Network Containers:

```bash
# Create two networks
docker network create frontend
docker network create backend

# Run database on backend only
docker run -d \
  --name postgres \
  --network backend \
  postgres:15

# Run API on both networks (can access DB and be accessed by web)
docker run -d \
  --name api \
  --network backend \
  my-api

docker network connect frontend api

# Run web server on frontend only
docker run -d \
  --name web \
  --network frontend \
  nginx

# Result:
# web → api → postgres
# web cannot directly access postgres (security!)
```

## Host Network

### Using Host Network:

```bash
# Container uses host's network stack directly
docker run -d \
  --network host \
  nginx

# nginx is now accessible on host's port 80
# No port mapping needed
# No isolation!

# Check from host
curl http://localhost:80

# Use cases:
# - Performance (no NAT overhead)
# - Legacy applications
# - Network debugging
```

**Note**: Host network doesn't work well on Docker Desktop (Mac/Windows)

## None Network

### Isolated Containers:

```bash
# Container has no network access
docker run -d \
  --network none \
  --name isolated \
  alpine

# Container can't reach internet
docker exec isolated ping google.com  # Fails

# Use cases:
# - Maximum security
# - Batch processing
# - Testing isolation
```

## Container DNS

### Name Resolution:

```bash
# Create network
docker network create mynet

# Run containers
docker run -d --name db --network mynet postgres:15
docker run -d --name api --network mynet my-api

# DNS resolution works automatically
docker exec api ping db          # ✅ Works
docker exec api nslookup db      # ✅ Works
docker exec api curl http://db:5432  # ✅ Works

# Docker's embedded DNS server: 127.0.0.11
docker exec api cat /etc/resolv.conf
# nameserver 127.0.0.11
```

### Network Aliases:

```bash
# One container, multiple hostnames
docker run -d \
  --network mynet \
  --network-alias database \
  --network-alias db \
  --network-alias postgres \
  --name pg \
  postgres:15

# All these work:
docker exec api ping pg
docker exec api ping database
docker exec api ping db
docker exec api ping postgres
```

## Port Publishing

### Publishing Ports:

```bash
# Publish single port
# -p <host-port>:<container-port>
docker run -d -p 8080:80 nginx
# Access: http://localhost:8080

# Publish on specific interface
docker run -d -p 127.0.0.1:8080:80 nginx
# Only accessible from localhost

# Publish on random port
docker run -d -p 80 nginx
docker port <container-id>  # Check assigned port

# Publish multiple ports
docker run -d \
  -p 8080:80 \
  -p 8443:443 \
  nginx

# Publish all exposed ports
docker run -d -P nginx
# Docker assigns random ports to all EXPOSE ports in Dockerfile

# UDP ports
docker run -d -p 53:53/udp dns-server
```

### Port Mapping Best Practices:

```bash
# ❌ DON'T: Use same port for multiple containers
docker run -d -p 80:80 nginx1  # ✅
docker run -d -p 80:80 nginx2  # ❌ Fails: Port already in use

# ✅ DO: Use different host ports
docker run -d -p 8081:80 nginx1
docker run -d -p 8082:80 nginx2

# ✅ DO: Use load balancer for same service
docker network create web
docker run -d --network web --name web1 nginx
docker run -d --network web --name web2 nginx
docker run -d -p 80:80 --network web load-balancer
```

## Container Linking (Legacy)

```bash
# Old way (deprecated, but still works)
docker run -d --name db postgres:15
docker run -d --name api --link db:database my-api

# Don't use this! Use custom networks instead.
```

## Overlay Networks (Multi-Host)

### For Docker Swarm:

```bash
# Initialize swarm
docker swarm init

# Create overlay network
docker network create \
  --driver overlay \
  --attachable \
  my-overlay

# Deploy services on overlay
docker service create \
  --name web \
  --network my-overlay \
  --replicas 3 \
  nginx

# Containers can communicate across hosts
```

## Hands-On Exercises

### Exercise 1: Multi-Tier Application

```bash
# 1. Create networks
docker network create frontend
docker network create backend

# 2. Run database (backend only)
docker run -d \
  --name postgres \
  --network backend \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=myapp \
  postgres:15

# 3. Run API (both networks)
docker run -d \
  --name api \
  --network backend \
  -e DATABASE_URL=postgresql://postgres:secret@postgres:5432/myapp \
  my-api-image

docker network connect frontend api

# 4. Run web server (frontend only)
docker run -d \
  --name web \
  --network frontend \
  -p 80:80 \
  -e API_URL=http://api:3000 \
  nginx

# Network topology:
# Internet ←→ web (frontend) ←→ api (frontend+backend) ←→ postgres (backend)
```

### Exercise 2: Service Discovery

```bash
# 1. Create network
docker network create app-net

# 2. Run multiple backend instances
docker run -d --name api1 --network app-net --network-alias api my-api
docker run -d --name api2 --network app-net --network-alias api my-api
docker run -d --name api3 --network app-net --network-alias api my-api

# 3. Run load balancer
docker run -it --rm \
  --network app-net \
  alpine sh

# Inside container, try:
# nslookup api
# Shows all 3 IPs! Docker does DNS round-robin
# ping api
# Hits different instances
```

### Exercise 3: Network Debugging

```bash
# 1. Create test network
docker network create test-net

# 2. Run two containers
docker run -d --name c1 --network test-net alpine sleep 3600
docker run -d --name c2 --network test-net alpine sleep 3600

# 3. Install network tools
docker exec c1 apk add --no-cache bind-tools curl

# 4. Test connectivity
docker exec c1 ping c2
docker exec c1 nslookup c2
docker exec c1 traceroute c2

# 5. Check network interfaces
docker exec c1 ip addr
docker exec c1 ip route

# 6. Check DNS
docker exec c1 cat /etc/resolv.conf

# 7. Test external connectivity
docker exec c1 ping google.com
docker exec c1 nslookup google.com
```

### Exercise 4: Port Mapping

```bash
# 1. Run multiple instances of same app
docker run -d --name web1 -p 8081:80 nginx
docker run -d --name web2 -p 8082:80 nginx
docker run -d --name web3 -p 8083:80 nginx

# 2. Customize content
docker exec web1 sh -c 'echo "Server 1" > /usr/share/nginx/html/index.html'
docker exec web2 sh -c 'echo "Server 2" > /usr/share/nginx/html/index.html'
docker exec web3 sh -c 'echo "Server 3" > /usr/share/nginx/html/index.html'

# 3. Test
curl http://localhost:8081  # Server 1
curl http://localhost:8082  # Server 2
curl http://localhost:8083  # Server 3

# 4. Check all port mappings
docker ps --format "{{.Names}}: {{.Ports}}"
```

## Network Troubleshooting

### Common Issues:

```bash
# 1. Container can't resolve names
docker exec mycontainer cat /etc/resolv.conf
# Fix: Use custom bridge network (not default bridge)

# 2. Can't connect to container port
docker port mycontainer
# Fix: Make sure port is published (-p flag)

# 3. Containers on different networks
docker network inspect network1
docker network inspect network2
# Fix: Connect container to both networks

# 4. Network conflicts
docker network inspect bridge
# Fix: Use different subnets

# 5. DNS not working
docker exec mycontainer nslookup containername
# Fix: Ensure both containers on same custom network
```

## Network Best Practices

1. **Use Custom Bridge Networks**: For name resolution
2. **Separate Networks**: Frontend/backend isolation
3. **Minimal Port Publishing**: Only publish what's necessary
4. **Use Network Aliases**: For service discovery
5. **Document Networks**: In docker-compose.yml
6. **Security**: Isolate sensitive services
7. **Monitor**: Use `docker network inspect`
8. **Clean Up**: Remove unused networks

## Next Steps
Move to `07-docker-compose.md` to learn about multi-container orchestration!
