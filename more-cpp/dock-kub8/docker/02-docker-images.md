# Docker Images - Building Blocks of Containers

## Theory

### What is a Docker Image?

A Docker image is a **read-only template** that contains:
- Application code
- Dependencies
- Libraries
- Environment variables
- Configuration files

### Image Layers:

Docker images are made up of **layers**. Each instruction in a Dockerfile creates a new layer.

```
Image Structure:
┌──────────────────────┐
│  Layer 4: App Code   │  ← Your application
├──────────────────────┤
│  Layer 3: Python     │  ← Runtime
├──────────────────────┤
│  Layer 2: Packages   │  ← Dependencies
├──────────────────────┤
│  Layer 1: Base OS    │  ← Ubuntu/Alpine
└──────────────────────┘
```

**Benefits of Layers:**
- **Caching**: Unchanged layers are reused
- **Efficiency**: Only changed layers need to be rebuilt
- **Storage**: Common layers are shared between images

### Image Naming Convention:

```
[registry]/[username]/[repository]:[tag]

Examples:
- ubuntu:22.04
- nginx:latest
- docker.io/library/python:3.11-slim
- myregistry.com/myteam/myapp:v1.0
```

## Practical Exercise

### Working with Images:

```bash
# Pull an image from Docker Hub
docker pull nginx

# Pull a specific version (tag)
docker pull nginx:1.25

# List all local images
docker images

# Search for images on Docker Hub
docker search python

# Inspect an image (detailed info)
docker image inspect nginx

# View image history (layers)
docker history nginx

# Remove an image
docker rmi nginx:1.25

# Remove all unused images
docker image prune

# Force remove an image (even if containers are using it)
docker rmi -f nginx
```

### Running Containers from Images:

```bash
# Run nginx in detached mode (background)
docker run -d nginx

# Run with a custom name
docker run -d --name my-nginx nginx

# Run with port mapping (host:container)
docker run -d -p 8080:80 nginx
# Now access nginx at http://localhost:8080

# Run with environment variables
docker run -d -e MYSQL_ROOT_PASSWORD=secret mysql

# Run and remove container after exit (--rm)
docker run --rm nginx echo "Hello Docker"
```

### Understanding Tags:

```bash
# Pull different versions
docker pull python:3.11
docker pull python:3.11-slim
docker pull python:3.11-alpine

# 'latest' is the default tag
docker pull python        # Same as python:latest
docker pull python:latest
```

### Saving and Loading Images:

```bash
# Save an image to a tar file
docker save nginx:latest > nginx.tar
docker save -o nginx.tar nginx:latest

# Load an image from a tar file
docker load < nginx.tar
docker load -i nginx.tar

# Export a container as an image
docker export my-container > my-container.tar

# Import a container as an image
docker import my-container.tar my-image:latest
```

## Hands-On Exercise

Try this sequence:

```bash
# 1. Pull Ubuntu image
docker pull ubuntu:22.04

# 2. Run it interactively
docker run -it --name my-ubuntu ubuntu:22.04 bash

# Inside the container, run:
apt-get update
apt-get install -y curl
exit

# 3. Check the container
docker ps -a

# 4. Create a new image from the modified container
docker commit my-ubuntu my-ubuntu-with-curl:v1

# 5. Verify the new image
docker images

# 6. Run your custom image
docker run -it my-ubuntu-with-curl:v1 bash
# Inside: curl --version should work now!
```

## Image Best Practices

1. **Use Official Images**: Start with official images from Docker Hub
2. **Use Specific Tags**: Don't rely on `latest` in production
3. **Keep Images Small**: Use alpine variants when possible
4. **Layer Wisely**: Put frequently changing code in later layers
5. **Clean Up**: Remove unused images regularly

## Next Steps
Move to `03-dockerfile.md` to learn how to create your own images!
