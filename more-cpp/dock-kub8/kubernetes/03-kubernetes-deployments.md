# Kubernetes Deployments - Managing Application Replicas

## Theory

### What is a Deployment?

A **Deployment** provides declarative updates for Pods and ReplicaSets. It's the recommended way to deploy applications in Kubernetes.

**Why not use Pods directly?**
- Pods don't restart if deleted
- No automatic scaling
- No rolling updates
- Manual management

**Deployments provide:**
- Self-healing (recreates failed pods)
- Scaling (replicas)
- Rolling updates
- Rollback capability
- Version history

### Hierarchy:

```
Deployment
    ↓
ReplicaSet (manages replicas)
    ↓
Pods (actual instances)
```

### Deployment Workflow:

```
You → Define Deployment (YAML) → Deployment Controller
                                         ↓
                                 Creates ReplicaSet
                                         ↓
                                 Creates Pods (replicas)
                                         ↓
                                 Monitors & maintains desired state
```

## Basic Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx

spec:
  # Number of pod replicas
  replicas: 3

  # Selector to match pods
  selector:
    matchLabels:
      app: nginx

  # Pod template
  template:
    metadata:
      labels:
        app: nginx  # Must match selector
    spec:
      containers:
      - name: nginx
        image: nginx:1.25
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
```

## Creating and Managing Deployments

```bash
# Create deployment from YAML
kubectl apply -f deployment.yaml

# Create deployment imperatively
kubectl create deployment nginx --image=nginx --replicas=3

# List deployments
kubectl get deployments
kubectl get deploy

# Describe deployment
kubectl describe deployment nginx-deployment

# See deployment details
kubectl get deployment nginx-deployment -o yaml

# Watch deployment rollout
kubectl rollout status deployment/nginx-deployment

# View deployment history
kubectl rollout history deployment/nginx-deployment
```

## Scaling Deployments

```bash
# Scale to 5 replicas
kubectl scale deployment nginx-deployment --replicas=5

# Get current replicas
kubectl get deployment nginx-deployment

# Auto-scale (HorizontalPodAutoscaler)
kubectl autoscale deployment nginx-deployment --min=2 --max=10 --cpu-percent=80

# Check HPA
kubectl get hpa

# View pods (should see 5 replicas)
kubectl get pods -l app=nginx
```

### Scaling in YAML:

```yaml
spec:
  replicas: 5  # Change this number
```

Then: `kubectl apply -f deployment.yaml`

## Rolling Updates

### Update Deployment Image:

```bash
# Update image version
kubectl set image deployment/nginx-deployment nginx=nginx:1.26

# Watch rollout
kubectl rollout status deployment/nginx-deployment

# Check rollout history
kubectl rollout history deployment/nginx-deployment

# See details of a specific revision
kubectl rollout history deployment/nginx-deployment --revision=2
```

### Update Strategy in YAML:

```yaml
spec:
  replicas: 3
  strategy:
    type: RollingUpdate  # or Recreate
    rollingUpdate:
      maxSurge: 1        # Max pods above desired count
      maxUnavailable: 1  # Max pods unavailable during update
```

**RollingUpdate** (default):
- Gradually replaces old pods with new ones
- Zero downtime
- Can have both versions running temporarily

**Recreate**:
- Kills all old pods
- Creates new pods
- Downtime occurs

## Rollback

```bash
# Undo last rollout
kubectl rollout undo deployment/nginx-deployment

# Rollback to specific revision
kubectl rollout undo deployment/nginx-deployment --to-revision=2

# Pause rollout (for canary deployments)
kubectl rollout pause deployment/nginx-deployment

# Resume rollout
kubectl rollout resume deployment/nginx-deployment

# Check rollout status
kubectl rollout status deployment/nginx-deployment
```

## Complete Deployment Example

```yaml
# production-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  namespace: production
  labels:
    app: web
    version: v1
  annotations:
    kubernetes.io/change-cause: "Initial deployment v1.0"

spec:
  # Number of replicas
  replicas: 5

  # Revision history limit
  revisionHistoryLimit: 10

  # Pod selector
  selector:
    matchLabels:
      app: web

  # Update strategy
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1

  # Time before force killing pod
  minReadySeconds: 5

  # Pod template
  template:
    metadata:
      labels:
        app: web
        version: v1

    spec:
      # Init container
      initContainers:
      - name: init-db
        image: busybox
        command: ['sh', '-c', 'until nslookup db; do sleep 2; done']

      # Main containers
      containers:
      - name: web
        image: myapp:1.0
        ports:
        - containerPort: 8080
          name: http

        # Environment variables
        env:
        - name: DB_HOST
          value: "postgres-service"
        - name: DB_PORT
          value: "5432"

        # Resource limits
        resources:
          requests:
            memory: "256Mi"
            cpu: "500m"
          limits:
            memory: "512Mi"
            cpu: "1000m"

        # Health checks
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

        # Volume mounts
        volumeMounts:
        - name: config
          mountPath: /etc/config
        - name: cache
          mountPath: /app/cache

      # Volumes
      volumes:
      - name: config
        configMap:
          name: app-config
      - name: cache
        emptyDir: {}

      # Security
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
```

## Deployment Strategies

### 1. Rolling Update (Default):

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1
    maxUnavailable: 0  # Zero-downtime
```

### 2. Recreate:

```yaml
strategy:
  type: Recreate
```

### 3. Blue-Green (Manual):

```yaml
# Blue deployment (current)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: blue
  template:
    metadata:
      labels:
        app: myapp
        version: blue
    spec:
      containers:
      - name: app
        image: myapp:1.0

---
# Green deployment (new version)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: green
  template:
    metadata:
      labels:
        app: myapp
        version: green
    spec:
      containers:
      - name: app
        image: myapp:2.0

# Service switches between blue and green
```

### 4. Canary Deployment:

```yaml
# Main deployment (90%)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-stable
spec:
  replicas: 9
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
        track: stable
    spec:
      containers:
      - name: app
        image: myapp:1.0

---
# Canary deployment (10%)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
        track: canary
    spec:
      containers:
      - name: app
        image: myapp:2.0
```

## Label Selectors

```yaml
selector:
  # Match exact labels
  matchLabels:
    app: nginx
    environment: production

  # Match expressions
  matchExpressions:
  - key: tier
    operator: In
    values:
    - frontend
    - backend
  - key: version
    operator: NotIn
    values:
    - deprecated
  - key: app
    operator: Exists
```

## Hands-On Exercises

### Exercise 1: Create and Scale Deployment

```bash
# Create deployment
kubectl create deployment web --image=nginx --replicas=2

# Check status
kubectl get deployments
kubectl get pods

# Scale up
kubectl scale deployment web --replicas=5
kubectl get pods -w

# Scale down
kubectl scale deployment web --replicas=3
```

### Exercise 2: Rolling Update

```yaml
# app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 4
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: nginx
        image: nginx:1.24
        ports:
        - containerPort: 80
```

```bash
# Deploy v1
kubectl apply -f app-deployment.yaml

# Watch pods
kubectl get pods -l app=myapp -w

# Update to v2
kubectl set image deployment/app nginx=nginx:1.25 --record

# Watch rollout
kubectl rollout status deployment/app

# Check history
kubectl rollout history deployment/app

# Rollback
kubectl rollout undo deployment/app
```

### Exercise 3: Self-Healing

```bash
# Create deployment
kubectl create deployment test --image=nginx --replicas=3

# Get pods
kubectl get pods

# Delete a pod
kubectl delete pod <pod-name>

# Watch auto-recreation
kubectl get pods -w
# A new pod is automatically created!
```

## Commands Cheat Sheet

```bash
# Create
kubectl create deployment NAME --image=IMAGE
kubectl apply -f deployment.yaml

# List
kubectl get deployments
kubectl get deploy NAME

# Details
kubectl describe deployment NAME
kubectl get deployment NAME -o yaml

# Scale
kubectl scale deployment NAME --replicas=N

# Update
kubectl set image deployment/NAME CONTAINER=IMAGE:TAG
kubectl edit deployment NAME
kubectl apply -f deployment.yaml

# Rollout
kubectl rollout status deployment/NAME
kubectl rollout history deployment/NAME
kubectl rollout undo deployment/NAME
kubectl rollout pause deployment/NAME
kubectl rollout resume deployment/NAME

# Delete
kubectl delete deployment NAME
kubectl delete -f deployment.yaml
```

## Best Practices

1. ✅ Always use Deployments (not bare Pods)
2. ✅ Set resource limits
3. ✅ Use readiness/liveness probes
4. ✅ Use labels for organization
5. ✅ Set `revisionHistoryLimit`
6. ✅ Use `--record` flag for history
7. ✅ Test in staging before production
8. ✅ Use `minReadySeconds` for stability
9. ✅ Monitor rollout status
10. ✅ Have rollback plan ready

## Next Steps

Move to `04-kubernetes-services.md` to learn about exposing applications!
