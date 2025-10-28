# Docker and Kubernetes - Zero to Hero Learning Path

Welcome to your comprehensive Docker and Kubernetes learning journey! This repository contains 20 hands-on lessons that will take you from complete beginner to production-ready container orchestration expert.

## 📚 Course Structure

### Part 1: Docker (Container Technology)
Learn how to containerize applications with Docker

### Part 2: Kubernetes (Container Orchestration)
Master running containers at scale with Kubernetes

## 🐳 Docker Learning Path

### Module 1: Docker Fundamentals
- **[01-docker-basics.md](./docker/01-docker-basics.md)** - What is Docker? Installation, basic concepts
- **[02-docker-images.md](./docker/02-docker-images.md)** - Working with images, tags, and Docker Hub
- **[03-dockerfile.md](./docker/03-dockerfile.md)** - Creating custom images, multi-stage builds

### Module 2: Container Management
- **[04-docker-containers.md](./docker/04-docker-containers.md)** - Running and managing containers
- **[05-docker-volumes.md](./docker/05-docker-volumes.md)** - Persistent data storage
- **[06-docker-networking.md](./docker/06-docker-networking.md)** - Container networking and communication

### Module 3: Production Docker
- **[07-docker-compose.md](./docker/07-docker-compose.md)** - Multi-container applications
- **[08-docker-security.md](./docker/08-docker-security.md)** - Security best practices
- **[09-docker-production.md](./docker/09-docker-production.md)** - Production deployments
- **[10-docker-troubleshooting.md](./docker/10-docker-troubleshooting.md)** - Debugging and problem-solving

## ☸️ Kubernetes Learning Path

### Module 1: Kubernetes Fundamentals
- **[01-kubernetes-basics.md](./kubernetes/01-kubernetes-basics.md)** - What is Kubernetes? Architecture, installation
- **[02-kubernetes-pods.md](./kubernetes/02-kubernetes-pods.md)** - Pods, containers, lifecycle
- **[03-kubernetes-deployments.md](./kubernetes/03-kubernetes-deployments.md)** - Managing replicas, scaling, updates

### Module 2: Networking & Configuration
- **[04-kubernetes-services.md](./kubernetes/04-kubernetes-services.md)** - Service discovery, load balancing
- **[05-kubernetes-configmaps-secrets.md](./kubernetes/05-kubernetes-configmaps-secrets.md)** - Configuration management
- **[06-kubernetes-volumes.md](./kubernetes/06-kubernetes-volumes.md)** - Persistent storage

### Module 3: Advanced Topics
- **[07-kubernetes-ingress.md](./kubernetes/07-kubernetes-ingress.md)** - HTTP/HTTPS routing
- **[08-kubernetes-namespaces.md](./kubernetes/08-kubernetes-namespaces.md)** - Resource isolation
- **[09-kubernetes-helm.md](./kubernetes/09-kubernetes-helm.md)** - Package management
- **[10-kubernetes-production.md](./kubernetes/10-kubernetes-production.md)** - Production best practices

## 🎯 Learning Approach

Each lesson contains:
1. **Theory**: Concepts explained with diagrams
2. **Practical Code**: Real-world examples with comments
3. **Hands-On Exercises**: Practice what you learned
4. **Best Practices**: Industry-standard recommendations
5. **Commands Reference**: Quick command cheat sheets

## 🚀 Getting Started

### Prerequisites
- Linux/Mac/Windows with WSL2
- Basic command-line knowledge
- Text editor (VS Code recommended)
- At least 8GB RAM, 20GB free disk space

### Recommended Learning Path

**Week 1-2: Docker Fundamentals**
```bash
# Start here
cd docker/
# Follow files 01 → 10 in order
```

**Week 3-4: Kubernetes Fundamentals**
```bash
cd kubernetes/
# Follow files 01 → 10 in order
```

### Quick Start

#### Docker
```bash
# Install Docker (Linux)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Verify
docker --version
docker run hello-world
```

#### Kubernetes
```bash
# Install Minikube (Local K8s)
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install kubectl /usr/local/bin/

# Start cluster
minikube start
kubectl cluster-info
```

## 📖 How to Use This Course

### For Beginners
1. Start with Docker lesson 01
2. Read theory section carefully
3. Run all code examples
4. Complete hands-on exercises
5. Move to next lesson only after understanding current one
6. After Docker 10, move to Kubernetes 01

### For Experienced Users
- Use as reference guide
- Jump to specific topics
- Focus on production lessons (09-10)
- Use best practices sections

## 🛠️ Practical Projects

Apply your knowledge with these projects:

### Docker Projects
1. **Containerize a web application**
   - Dockerfile with multi-stage build
   - Docker Compose with database
   - Volume for persistent data

2. **Microservices architecture**
   - Multiple containers
   - Custom network
   - Service discovery

### Kubernetes Projects
1. **Deploy 3-tier application**
   - Frontend (React/Vue)
   - Backend API (Node/Python)
   - Database (PostgreSQL)
   - Ingress for routing

2. **Production-ready setup**
   - Deployments with replicas
   - ConfigMaps and Secrets
   - Persistent volumes
   - Monitoring with Prometheus
   - Auto-scaling

## 📊 Learning Milestones

Track your progress:

### Docker Milestones
- [ ] Run your first container
- [ ] Build custom Docker image
- [ ] Use Docker Compose
- [ ] Implement multi-stage builds
- [ ] Deploy production-ready container

### Kubernetes Milestones
- [ ] Deploy your first pod
- [ ] Create a deployment
- [ ] Expose service
- [ ] Use ConfigMaps and Secrets
- [ ] Set up Ingress
- [ ] Deploy production application

## 🔧 Useful Commands Cheat Sheet

### Docker
```bash
# Images
docker pull IMAGE
docker build -t NAME .
docker images

# Containers
docker run IMAGE
docker ps
docker logs CONTAINER
docker exec -it CONTAINER bash

# Cleanup
docker system prune -a
```

### Kubernetes
```bash
# Cluster
kubectl cluster-info
kubectl get nodes

# Pods
kubectl get pods
kubectl describe pod NAME
kubectl logs POD

# Deployments
kubectl create deployment NAME --image=IMAGE
kubectl scale deployment NAME --replicas=3

# Services
kubectl expose deployment NAME --port=80
kubectl get svc
```

## 📚 Additional Resources

### Official Documentation
- [Docker Docs](https://docs.docker.com/)
- [Kubernetes Docs](https://kubernetes.io/docs/)
- [Docker Hub](https://hub.docker.com/)

### Tools
- [Minikube](https://minikube.sigs.k8s.io/) - Local Kubernetes
- [Kind](https://kind.sigs.k8s.io/) - Kubernetes in Docker
- [Helm](https://helm.sh/) - Kubernetes package manager
- [k9s](https://k9scli.io/) - Terminal UI for Kubernetes

### Certification Paths
- **Docker**: Docker Certified Associate (DCA)
- **Kubernetes**:
  - CKAD (Certified Kubernetes Application Developer)
  - CKA (Certified Kubernetes Administrator)
  - CKS (Certified Kubernetes Security Specialist)

## 🎓 Next Steps After Completion

1. **Service Mesh**: Istio, Linkerd
2. **GitOps**: ArgoCD, Flux
3. **Monitoring**: Prometheus, Grafana, ELK Stack
4. **Security**: OPA, Falco, Vault
5. **Cloud Platforms**: EKS (AWS), GKE (Google), AKS (Azure)

## 💡 Tips for Success

1. **Practice Daily**: Spend at least 1 hour coding
2. **Type Everything**: Don't copy-paste, understand by typing
3. **Break Things**: Learn by experimenting and fixing errors
4. **Build Projects**: Apply concepts to real applications
5. **Join Communities**: Docker/Kubernetes forums, Slack, Reddit

## 🤝 Contributing

Found an error? Have a suggestion? Contributions are welcome!

## 📝 Notes

- All examples use Linux commands
- Adjust paths for Windows (use WSL2 recommended)
- Some features require cloud provider (AWS, GCP, Azure)
- Practice on local cluster before production

## 🏆 Course Completion

After completing all 20 lessons, you will be able to:
- ✅ Containerize any application with Docker
- ✅ Create production-ready Docker images
- ✅ Deploy multi-container applications with Docker Compose
- ✅ Understand Kubernetes architecture
- ✅ Deploy and manage applications on Kubernetes
- ✅ Implement auto-scaling and self-healing
- ✅ Secure containerized applications
- ✅ Monitor and troubleshoot containers
- ✅ Follow industry best practices
- ✅ Deploy production applications confidently

## 📅 Suggested Timeline

- **Week 1**: Docker 01-05 (Basics through Volumes)
- **Week 2**: Docker 06-10 (Networking through Troubleshooting)
- **Week 3**: Kubernetes 01-05 (Basics through ConfigMaps)
- **Week 4**: Kubernetes 06-10 (Volumes through Production)

**Total**: 4 weeks to Zero to Hero! 🚀

---

**Happy Learning! 🎉**

Start with [Docker Basics](./docker/01-docker-basics.md) and begin your container journey today!
