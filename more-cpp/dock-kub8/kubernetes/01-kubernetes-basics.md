# Kubernetes Basics - Introduction to K8s

## Theory

### What is Kubernetes?

**Kubernetes (K8s)** is an open-source **container orchestration platform** that automates:
- Deployment
- Scaling
- Management
- Networking

of containerized applications.

**Think of it as**: Docker runs containers on one machine. Kubernetes runs containers across many machines (cluster).

### Why Kubernetes?

**Problems without K8s:**
- Manual deployment to servers
- No automatic recovery from failures
- Manual scaling (add/remove containers)
- Complex service discovery
- No load balancing
- Difficult updates and rollbacks

**Solutions with K8s:**
- Automated deployment
- Self-healing (restarts failed containers)
- Auto-scaling (based on load)
- Service discovery (DNS)
- Built-in load balancing
- Rolling updates & rollbacks

### Kubernetes Architecture:

```
┌─────────────────────────────────────────────────────┐
│              CONTROL PLANE (Master)                 │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐    │
│  │   API    │  │ Scheduler│  │   Controller  │    │
│  │  Server  │  │          │  │    Manager    │    │
│  └──────────┘  └──────────┘  └───────────────┘    │
│       ┌──────────┐                                  │
│       │   etcd   │  (Key-value store)               │
│       └──────────┘                                  │
└─────────────────────────────────────────────────────┘
           │
           ├──────────────┬──────────────┬───────────
           │              │              │
     ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
     │  NODE 1   │  │  NODE 2   │  │  NODE 3   │
     │ ┌───────┐ │  │ ┌───────┐ │  │ ┌───────┐ │
     │ │kubelet│ │  │ │kubelet│ │  │ │kubelet│ │
     │ └───────┘ │  │ └───────┘ │  │ └───────┘ │
     │ ┌───────┐ │  │ ┌───────┐ │  │ ┌───────┐ │
     │ │ Pods  │ │  │ │ Pods  │ │  │ │ Pods  │ │
     │ └───────┘ │  │ └───────┘ │  │ └───────┘ │
     └───────────┘  └───────────┘  └───────────┘
```

### Control Plane Components:

1. **API Server**: Front-end for K8s (kubectl talks to this)
2. **etcd**: Database storing cluster state
3. **Scheduler**: Assigns pods to nodes
4. **Controller Manager**: Runs controller processes
5. **Cloud Controller Manager**: Cloud-specific logic (optional)

### Node Components:

1. **kubelet**: Agent running on each node
2. **kube-proxy**: Network proxy on each node
3. **Container Runtime**: Docker, containerd, CRI-O

## Key Concepts

### 1. Pod:
- Smallest deployable unit in K8s
- One or more containers that share network/storage
- Usually 1 container per pod

```
Pod
┌──────────────────────┐
│  Container 1         │
│  (nginx)             │
├──────────────────────┤
│  Container 2         │
│  (sidecar logging)   │
└──────────────────────┘
Shared: Network, Volumes
```

### 2. Node:
- Worker machine (VM or physical)
- Runs pods
- Managed by control plane

### 3. Cluster:
- Set of nodes
- Control plane + worker nodes

### 4. Namespace:
- Virtual cluster within cluster
- Isolate resources
- Examples: default, kube-system, development, production

## Installing Kubernetes

### Option 1: Minikube (Local Development)

```bash
# Install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start Minikube
minikube start

# Check status
minikube status

# Stop Minikube
minikube stop

# Delete cluster
minikube delete
```

### Option 2: Kind (Kubernetes in Docker)

```bash
# Install Kind
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Create cluster
kind create cluster

# Create cluster with custom name
kind create cluster --name my-cluster

# List clusters
kind get clusters

# Delete cluster
kind delete cluster
```

### Option 3: K3s (Lightweight Kubernetes)

```bash
# Install K3s
curl -sfL https://get.k3s.io | sh -

# Check status
sudo systemctl status k3s

# Get kubeconfig
sudo cat /etc/rancher/k3s/k3s.yaml
```

### Install kubectl (Kubernetes CLI)

```bash
# Download kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Make executable
chmod +x kubectl

# Move to PATH
sudo mv kubectl /usr/local/bin/

# Verify installation
kubectl version --client

# Check cluster connection
kubectl cluster-info
```

## Basic kubectl Commands

### Cluster Information:

```bash
# View cluster info
kubectl cluster-info

# View nodes
kubectl get nodes

# Node details
kubectl describe node <node-name>

# Cluster version
kubectl version
```

### Namespace Commands:

```bash
# List namespaces
kubectl get namespaces
kubectl get ns

# Create namespace
kubectl create namespace development
kubectl create ns dev

# Delete namespace
kubectl delete namespace development

# Set default namespace
kubectl config set-context --current --namespace=development
```

### Basic Resource Commands:

```bash
# Get resources
kubectl get pods
kubectl get services
kubectl get deployments
kubectl get all

# Get resources in all namespaces
kubectl get pods --all-namespaces
kubectl get pods -A

# Get resources in specific namespace
kubectl get pods -n kube-system

# Describe resource (detailed info)
kubectl describe pod my-pod
kubectl describe service my-service

# Delete resource
kubectl delete pod my-pod
kubectl delete service my-service
```

## Your First Pod

### Method 1: Imperative (Command Line)

```bash
# Run a pod
kubectl run nginx --image=nginx

# Check if running
kubectl get pods

# View pod details
kubectl describe pod nginx

# Get pod logs
kubectl logs nginx

# Execute command in pod
kubectl exec nginx -- ls /usr/share/nginx/html

# Interactive shell
kubectl exec -it nginx -- bash

# Delete pod
kubectl delete pod nginx
```

### Method 2: Declarative (YAML File)

Create `my-first-pod.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
  labels:
    app: nginx
spec:
  containers:
  - name: nginx-container
    image: nginx:latest
    ports:
    - containerPort: 80
```

```bash
# Create pod from YAML
kubectl apply -f my-first-pod.yaml

# View pod
kubectl get pods

# Describe pod
kubectl describe pod nginx-pod

# Delete pod
kubectl delete -f my-first-pod.yaml
```

## Understanding YAML Configuration

### Basic Structure:

```yaml
# Every K8s resource has these fields:
apiVersion: v1              # API version
kind: Pod                   # Resource type
metadata:                   # Metadata about resource
  name: my-pod              # Name (required)
  labels:                   # Labels (key-value pairs)
    app: myapp
    env: production
spec:                       # Specification (desired state)
  containers:
  - name: container-name
    image: nginx:latest
```

### Common apiVersion Values:

- `v1`: Core API (Pod, Service, ConfigMap, etc.)
- `apps/v1`: Deployments, StatefulSets, DaemonSets
- `batch/v1`: Jobs, CronJobs
- `networking.k8s.io/v1`: NetworkPolicies, Ingress

## Hands-On Exercise

### Exercise 1: Run Your First Application

```bash
# 1. Start Minikube (if using Minikube)
minikube start

# 2. Check cluster
kubectl cluster-info
kubectl get nodes

# 3. Create a pod
kubectl run hello-k8s --image=nginx

# 4. Check pod status
kubectl get pods

# 5. Wait for pod to be running
kubectl get pods -w  # Watch mode (Ctrl+C to exit)

# 6. Get pod details
kubectl describe pod hello-k8s

# 7. View logs
kubectl logs hello-k8s

# 8. Execute command
kubectl exec hello-k8s -- nginx -v

# 9. Interactive shell
kubectl exec -it hello-k8s -- bash
# Inside pod:
# curl localhost
# exit

# 10. Delete pod
kubectl delete pod hello-k8s
```

### Exercise 2: Use YAML Configuration

Create `web-app.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: web-app
  labels:
    app: web
    tier: frontend
spec:
  containers:
  - name: nginx
    image: nginx:1.25
    ports:
    - containerPort: 80
      name: http
    env:
    - name: ENVIRONMENT
      value: "development"
```

```bash
# Apply configuration
kubectl apply -f web-app.yaml

# Get pod
kubectl get pod web-app

# View labels
kubectl get pod web-app --show-labels

# Get pods by label
kubectl get pods -l app=web

# View environment variables
kubectl exec web-app -- env | grep ENVIRONMENT

# Delete using file
kubectl delete -f web-app.yaml
```

## kubectl Cheat Sheet

```bash
# Get resources
kubectl get <resource>                  # List resources
kubectl get <resource> -o wide          # More details
kubectl get <resource> -o yaml          # YAML output
kubectl get <resource> -o json          # JSON output

# Create resources
kubectl create -f <file>                # Create from file
kubectl apply -f <file>                 # Apply configuration

# Update resources
kubectl edit <resource> <name>          # Edit in editor
kubectl apply -f <file>                 # Update from file

# Delete resources
kubectl delete <resource> <name>        # Delete by name
kubectl delete -f <file>                # Delete from file

# Describe resources
kubectl describe <resource> <name>      # Detailed info

# Logs and exec
kubectl logs <pod>                      # View logs
kubectl logs -f <pod>                   # Follow logs
kubectl exec <pod> -- <command>         # Execute command
kubectl exec -it <pod> -- bash          # Interactive shell

# Port forwarding
kubectl port-forward <pod> 8080:80      # Forward port

# Copy files
kubectl cp <pod>:/path/to/file ./file   # Copy from pod
kubectl cp ./file <pod>:/path/to/file   # Copy to pod
```

## Important Concepts

### Imperative vs Declarative:

**Imperative** (Commands):
```bash
kubectl run nginx --image=nginx
kubectl create deployment web --image=nginx
kubectl scale deployment web --replicas=3
```

**Declarative** (YAML):
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f .  # Apply all YAML files in directory
```

**Best Practice**: Use declarative approach (YAML) for production!

### Labels and Selectors:

```yaml
# Labels: Key-value pairs to identify resources
metadata:
  labels:
    app: myapp
    environment: production
    tier: frontend

# Selectors: Query resources by labels
kubectl get pods -l app=myapp
kubectl get pods -l environment=production,tier=frontend
kubectl get pods -l 'environment in (dev,staging)'
```

## Kubernetes Dashboard (Optional)

```bash
# Enable dashboard (Minikube)
minikube dashboard

# Or install manually
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml

# Access dashboard
kubectl proxy
# Visit: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
```

## Kubernetes vs Docker

| Feature | Docker | Kubernetes |
|---------|--------|------------|
| Scope | Single host | Multiple hosts (cluster) |
| Scaling | Manual | Automatic |
| Load Balancing | Manual | Built-in |
| Self-healing | No | Yes |
| Updates | Manual | Rolling updates |
| Networking | Simple | Complex, flexible |
| Use Case | Development | Production |

**Docker and Kubernetes work together!**
- Docker: Creates containers
- Kubernetes: Orchestrates containers

## Next Steps

You now understand:
- What Kubernetes is and why we need it
- K8s architecture
- Basic kubectl commands
- How to run pods

Move to `02-kubernetes-pods.md` to dive deeper into Pods!
