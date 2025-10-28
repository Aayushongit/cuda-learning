# Kubernetes Namespaces - Resource Isolation

## Theory

**Namespace** = Virtual cluster within a physical cluster

**Use cases:**
- Multi-tenancy (dev, staging, prod)
- Team isolation
- Resource quotas
- Access control

## Default Namespaces

```bash
kubectl get namespaces

# default: Default namespace for objects
# kube-system: Kubernetes system components
# kube-public: Publicly readable
# kube-node-lease: Node heartbeats
```

## Creating Namespaces

```bash
# Imperative
kubectl create namespace development
kubectl create ns staging

# Declarative
kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    environment: production
EOF
```

## Using Namespaces

```bash
# Create resource in namespace
kubectl run nginx --image=nginx -n development

# Get resources in namespace
kubectl get pods -n development

# Get all namespaces
kubectl get pods --all-namespaces
kubectl get pods -A

# Set default namespace
kubectl config set-context --current --namespace=development

# Get current namespace
kubectl config view --minify | grep namespace:
```

## Resource YAML with Namespace

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
  namespace: development  # Specify namespace
spec:
  containers:
  - name: nginx
    image: nginx
```

## ResourceQuota

Limit resource consumption per namespace:

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: dev-quota
  namespace: development
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    pods: "50"
    services: "20"
    persistentvolumeclaims: "10"
```

```bash
kubectl apply -f quota.yaml
kubectl describe resourcequota -n development
```

## LimitRange

Default limits for containers:

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: dev-limits
  namespace: development
spec:
  limits:
  - max:
      cpu: "2"
      memory: "2Gi"
    min:
      cpu: "100m"
      memory: "128Mi"
    default:
      cpu: "500m"
      memory: "512Mi"
    defaultRequest:
      cpu: "250m"
      memory: "256Mi"
    type: Container
```

## Cross-Namespace Access

```yaml
# Service in namespace "backend"
apiVersion: v1
kind: Service
metadata:
  name: database
  namespace: backend
spec:
  selector:
    app: postgres
  ports:
  - port: 5432

---
# Access from "frontend" namespace
# DNS: database.backend.svc.cluster.local
```

## Commands

```bash
kubectl get ns
kubectl create ns NAME
kubectl delete ns NAME
kubectl describe ns NAME
kubectl get all -n NAME
```

## Best Practices

1. ✅ Use namespaces for environments (dev/staging/prod)
2. ✅ Set resource quotas
3. ✅ Use RBAC per namespace
4. ✅ Label namespaces
5. ✅ Don't use default namespace in production

## Next Steps
Move to `09-kubernetes-helm.md` to learn package management!
