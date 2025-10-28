# Docker Basics - What is Docker?

## Theory

### What is Docker?
Docker is a platform for developing, shipping, and running applications in **containers**.

### Key Concepts:

1. **Container**: A lightweight, standalone, executable package that includes everything needed to run software:
   - Code
   - Runtime
   - System tools
   - System libraries
   - Settings

2. **Image**: A read-only template used to create containers. Think of it as a "snapshot" or "blueprint".

3. **Dockerfile**: A text file with instructions to build a Docker image.

4. **Docker Hub**: A registry where Docker images are stored (like GitHub for Docker images).

### Why Docker?

- **Consistency**: "Works on my machine" → "Works everywhere"
- **Isolation**: Each container runs independently
- **Efficiency**: Containers share the OS kernel (lighter than VMs)
- **Portability**: Run anywhere Docker is installed
- **Scalability**: Easy to scale up/down

### Docker vs Virtual Machines:

```
Virtual Machine:
┌─────────────────────────┐
│   App A   │   App B     │
│  Bins/Libs│  Bins/Libs  │
│  Guest OS │  Guest OS   │
├─────────────────────────┤
│      Hypervisor         │
├─────────────────────────┤
│      Host OS            │
├─────────────────────────┤
│   Infrastructure        │
└─────────────────────────┘

Docker Container:
┌─────────────────────────┐
│   App A   │   App B     │
│  Bins/Libs│  Bins/Libs  │
├─────────────────────────┤
│    Docker Engine        │
├─────────────────────────┤
│      Host OS            │
├─────────────────────────┤
│   Infrastructure        │
└─────────────────────────┘
```

**Containers are lighter and faster because they share the host OS kernel!**

## Practical Exercise

### Installing Docker (Linux):
```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify installation
docker --version
```

### Your First Docker Command:
```bash
# Run a hello-world container
docker run hello-world

# This command:
# 1. Downloads the 'hello-world' image from Docker Hub
# 2. Creates a container from that image
# 3. Runs the container
# 4. Shows output
# 5. Exits
```

### Basic Docker Commands:
```bash
# Check Docker version
docker --version

# View system information
docker info

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# List downloaded images
docker images

# Pull an image from Docker Hub
docker pull ubuntu

# Run a container interactively
docker run -it ubuntu bash
```

## Next Steps
Move to `02-docker-images.md` to learn about Docker images in detail!
