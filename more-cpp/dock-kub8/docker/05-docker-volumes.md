# Docker Volumes - Persistent Data Storage

## Theory

### The Data Persistence Problem:

**Container File System = Temporary**
- When a container is removed, all data inside is lost
- Changes to files are lost when container stops
- Cannot share data between containers easily

**Solution: Volumes and Bind Mounts**

### Three Types of Mounts:

```
1. VOLUMES (Managed by Docker)
   Host: /var/lib/docker/volumes/my-volume/_data
   Container: /app/data

2. BIND MOUNTS (Direct host path)
   Host: /home/user/mydata
   Container: /app/data

3. TMPFS MOUNTS (In memory, temporary)
   Memory: RAM only
   Container: /app/cache
```

### Volumes vs Bind Mounts:

| Feature | Volumes | Bind Mounts |
|---------|---------|-------------|
| Managed by Docker | ✅ Yes | ❌ No |
| Path | Docker controls | You specify |
| Portability | ✅ Better | ⚠️ Host-dependent |
| Performance | ✅ Better on Linux | ⚠️ Can be slower |
| Backup | ✅ Easier | ⚠️ Manual |
| Use Case | Production data | Development |

## Working with Volumes

### Creating Volumes:

```bash
# Create a named volume
docker volume create my-data

# Create with driver options
docker volume create \
  --driver local \
  --opt type=none \
  --opt device=/mnt/external \
  --opt o=bind \
  my-external-volume

# List all volumes
docker volume ls

# Inspect volume details
docker volume inspect my-data

# Output shows:
# - CreatedAt
# - Driver
# - Mountpoint (actual location on host)
# - Name
```

### Using Volumes in Containers:

```bash
# Run with named volume
docker run -d \
  --name mysql-db \
  -v my-data:/var/lib/mysql \
  -e MYSQL_ROOT_PASSWORD=secret \
  mysql:8

# Run with anonymous volume (Docker generates name)
docker run -d \
  -v /var/lib/mysql \
  -e MYSQL_ROOT_PASSWORD=secret \
  mysql:8

# Multiple volumes
docker run -d \
  -v data-vol:/app/data \
  -v logs-vol:/app/logs \
  -v cache-vol:/app/cache \
  my-app

# Read-only volume
docker run -d \
  -v config-vol:/app/config:ro \
  my-app
```

### Volume Lifecycle:

```bash
# Create volume
docker volume create app-data

# Use in container
docker run -d -v app-data:/data --name app1 alpine sleep 1000

# Data persists even after container is removed
docker stop app1
docker rm app1

# Use same volume in new container (data is still there!)
docker run -d -v app-data:/data --name app2 alpine sleep 1000

# List volumes
docker volume ls

# Remove volume (only when no containers use it)
docker volume rm app-data

# Remove all unused volumes
docker volume prune

# Force remove (dangerous - may lose data!)
docker volume rm -f app-data
```

## Bind Mounts

### Using Bind Mounts:

```bash
# Mount current directory
docker run -d \
  -v $(pwd):/app \
  node:18 npm start

# Mount specific host path
docker run -d \
  -v /home/user/project:/app \
  -v /home/user/logs:/var/log \
  my-app

# Mount with read-only flag
docker run -d \
  -v /home/user/config:/etc/myapp:ro \
  my-app

# Mount with propagation flags
docker run -d \
  -v /host/path:/container/path:rshared \
  my-app

# Windows style
docker run -d \
  -v C:/Users/username/data:/app/data \
  my-app
```

### Bind Mount Use Cases:

```bash
# 1. Development: Live code updates
docker run -d \
  -v $(pwd):/app \
  -w /app \
  node:18 \
  nodemon server.js

# 2. Configuration files
docker run -d \
  -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro \
  nginx

# 3. Host log collection
docker run -d \
  -v /var/log/myapp:/app/logs \
  my-app

# 4. Sharing data between host and container
docker run -it \
  -v $(pwd)/shared:/shared \
  ubuntu bash
```

## tmpfs Mounts (Temporary, In-Memory)

### Using tmpfs:

```bash
# Mount tmpfs (data stored in RAM, lost on stop)
docker run -d \
  --tmpfs /app/cache \
  my-app

# With size limit
docker run -d \
  --tmpfs /app/cache:size=512M \
  my-app

# Use case: Sensitive temporary data
docker run -d \
  --tmpfs /app/temp:rw,size=100M,mode=1777 \
  my-app

# Multiple tmpfs mounts
docker run -d \
  --tmpfs /cache:size=64M \
  --tmpfs /tmp:size=128M \
  my-app
```

## Volume Drivers

### Using Different Drivers:

```bash
# Local driver (default)
docker volume create --driver local my-vol

# NFS volume
docker volume create \
  --driver local \
  --opt type=nfs \
  --opt o=addr=192.168.1.100,rw \
  --opt device=:/path/to/share \
  nfs-volume

# CIFS/SMB volume
docker volume create \
  --driver local \
  --opt type=cifs \
  --opt o=username=user,password=pass \
  --opt device=//server/share \
  smb-volume

# Using plugin drivers (example: rexray)
docker volume create \
  --driver rexray/ebs \
  --opt size=20 \
  aws-ebs-vol
```

## Backing Up and Restoring Volumes

### Backup Volume Data:

```bash
# Method 1: Using tar
docker run --rm \
  -v my-data:/data \
  -v $(pwd):/backup \
  ubuntu \
  tar czf /backup/backup.tar.gz -C /data .

# Method 2: Create backup container
docker run --rm \
  --volumes-from app-container \
  -v $(pwd):/backup \
  ubuntu \
  tar czf /backup/app-backup.tar.gz /app/data
```

### Restore Volume Data:

```bash
# Create new volume
docker volume create restored-data

# Restore from backup
docker run --rm \
  -v restored-data:/data \
  -v $(pwd):/backup \
  ubuntu \
  tar xzf /backup/backup.tar.gz -C /data

# Use restored volume
docker run -d \
  -v restored-data:/app/data \
  my-app
```

## Sharing Volumes Between Containers

### Method 1: Named Volume:

```bash
# Create shared volume
docker volume create shared-data

# Container 1: Writer
docker run -d \
  --name writer \
  -v shared-data:/data \
  alpine \
  sh -c "while true; do date >> /data/log.txt; sleep 5; done"

# Container 2: Reader
docker run -d \
  --name reader \
  -v shared-data:/data:ro \
  alpine \
  sh -c "while true; do tail -f /data/log.txt; done"

# Check reader logs
docker logs -f reader
```

### Method 2: --volumes-from:

```bash
# Container with volumes
docker run -d \
  --name data-container \
  -v /data \
  -v /config \
  alpine

# Share volumes from data-container
docker run -d \
  --name app1 \
  --volumes-from data-container \
  my-app

docker run -d \
  --name app2 \
  --volumes-from data-container \
  my-app
```

## Dockerfile VOLUME Instruction

### Declaring Volumes in Dockerfile:

```dockerfile
FROM ubuntu:22.04

# Install application
RUN apt-get update && apt-get install -y myapp

# Declare volume mount point
# Creates anonymous volume if not specified at runtime
VOLUME /app/data
VOLUME /app/logs

# Alternative syntax for multiple volumes
VOLUME ["/app/data", "/app/logs"]

CMD ["myapp"]
```

```bash
# Run container - anonymous volumes created
docker run -d my-app
# Docker creates: /var/lib/docker/volumes/<random-id>/_data

# Run with named volume - uses your volume instead
docker run -d -v my-data:/app/data my-app
```

## Hands-On Exercises

### Exercise 1: Database with Persistent Storage

```bash
# 1. Create volume for PostgreSQL
docker volume create postgres-data

# 2. Run PostgreSQL with volume
docker run -d \
  --name my-postgres \
  -v postgres-data:/var/lib/postgresql/data \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=myapp \
  postgres:15

# 3. Connect and create data
docker exec -it my-postgres psql -U postgres -d myapp
# Inside psql:
# CREATE TABLE users (id SERIAL, name VARCHAR(50));
# INSERT INTO users (name) VALUES ('Alice'), ('Bob');
# SELECT * FROM users;
# \q

# 4. Stop and remove container
docker stop my-postgres
docker rm my-postgres

# 5. Create new container with same volume
docker run -d \
  --name new-postgres \
  -v postgres-data:/var/lib/postgresql/data \
  -e POSTGRES_PASSWORD=secret \
  postgres:15

# 6. Verify data persisted
docker exec -it new-postgres psql -U postgres -d myapp
# SELECT * FROM users;  -- Data is still there!
```

### Exercise 2: Development Environment

```bash
# 1. Create project directory
mkdir my-node-app
cd my-node-app
echo 'console.log("Hello from Node!");' > index.js

# 2. Run with bind mount for live reload
docker run -it --rm \
  -v $(pwd):/app \
  -w /app \
  node:18 \
  node index.js

# 3. Edit index.js in your editor, run again - changes reflected!

# 4. Add package.json
cat > package.json << EOF
{
  "name": "my-app",
  "dependencies": {
    "express": "^4.18.0"
  }
}
EOF

# 5. Install dependencies using volume
docker run --rm \
  -v $(pwd):/app \
  -w /app \
  node:18 \
  npm install

# 6. node_modules now on host filesystem
ls -la node_modules
```

### Exercise 3: Backup and Restore

```bash
# 1. Create volume with data
docker volume create important-data
docker run --rm \
  -v important-data:/data \
  alpine \
  sh -c "echo 'Critical data' > /data/file.txt"

# 2. Backup
docker run --rm \
  -v important-data:/data \
  -v $(pwd):/backup \
  alpine \
  tar czf /backup/data-backup.tar.gz -C /data .

# 3. Remove volume
docker volume rm important-data

# 4. Restore
docker volume create restored-data
docker run --rm \
  -v restored-data:/data \
  -v $(pwd):/backup \
  alpine \
  tar xzf /backup/data-backup.tar.gz -C /data

# 5. Verify
docker run --rm \
  -v restored-data:/data \
  alpine \
  cat /data/file.txt
```

## Best Practices

1. **Use Named Volumes** for production data
2. **Use Bind Mounts** for development
3. **Backup Regularly** - automate volume backups
4. **Use Volume Drivers** for cloud/network storage
5. **Set Permissions** correctly on bind mounts
6. **Document Volumes** in your Dockerfile
7. **Clean Up** unused volumes regularly
8. **Use Read-Only** when container shouldn't modify data

## Next Steps
Move to `06-docker-networking.md` to learn about container networking!
