# Kubernetes Services - Networking and Service Discovery

## Theory

### What is a Service?

A **Service** is an abstract way to expose an application running on a set of Pods as a network service.

**Problem without Services:**
- Pods have ephemeral IPs (change when recreated)
- Multiple pod replicas need load balancing
- External access to pods is difficult

**Services provide:**
- Stable IP address (ClusterIP)
- DNS name
- Load balancing across pods
- Service discovery

### Service Types:

```
1. ClusterIP (default)
   - Internal cluster IP
   - Accessible only within cluster

2. NodePort
   - Exposes on each node's IP at a static port
   - Accessible from outside cluster

3. LoadBalancer
   - Cloud provider load balancer
   - Accessible from internet

4. ExternalName
   - Maps to external DNS name
```

## Service Architecture

```
Service (stable IP: 10.0.0.5)
         ↓
  Load Balancer
    ↓    ↓    ↓
  Pod1 Pod2 Pod3
(10.1.0.2) (10.1.0.3) (10.1.0.4)

Pods can be added/removed/recreated
Service IP stays the same!
```

## ClusterIP Service (Default)

### Basic ClusterIP:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-service
spec:
  # Type: ClusterIP (default, can omit)
  type: ClusterIP

  # Selector to match pods
  selector:
    app: web

  # Ports
  ports:
  - name: http
    protocol: TCP
    port: 80        # Service port
    targetPort: 8080  # Pod port
```

**Usage:**
- Internal communication between services
- Frontend → Backend
- App → Database

```bash
# Create service
kubectl apply -f service.yaml

# Get services
kubectl get services
kubectl get svc

# Describe service
kubectl describe service web-service

# Get service details
kubectl get svc web-service -o yaml

# Access from within cluster
kubectl run test --image=curlimages/curl -it --rm -- curl http://web-service
```

## NodePort Service

### Expose on Node IP:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-nodeport
spec:
  type: NodePort
  selector:
    app: web
  ports:
  - port: 80          # Service port
    targetPort: 8080  # Pod port
    nodePort: 30080   # Node port (30000-32767)
    # If omitted, random port assigned
```

**Access:**
```
http://<NodeIP>:30080
http://<any-node-ip>:30080
```

```bash
# Get node IP (Minikube)
minikube ip

# Get node IP (regular cluster)
kubectl get nodes -o wide

# Access service
curl http://<node-ip>:30080

# Or use port-forward
kubectl port-forward svc/web-nodeport 8080:80
curl http://localhost:8080
```

## LoadBalancer Service

### Cloud Provider Load Balancer:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: web-loadbalancer
spec:
  type: LoadBalancer
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 8080
```

**Note:** Only works on cloud providers (AWS, GCP, Azure)

```bash
# Get external IP
kubectl get svc web-loadbalancer

# Wait for EXTERNAL-IP
kubectl get svc web-loadbalancer -w

# Access service
curl http://<external-ip>
```

## Service Discovery

### 1. Environment Variables:

Kubernetes automatically creates env vars for services:

```bash
# If service "web-service" exists
kubectl exec pod-name -- env | grep WEB_SERVICE

# Shows:
# WEB_SERVICE_SERVICE_HOST=10.0.0.5
# WEB_SERVICE_SERVICE_PORT=80
```

### 2. DNS (Recommended):

Kubernetes DNS format:
```
<service-name>.<namespace>.svc.cluster.local
```

```bash
# Same namespace
curl http://web-service

# Different namespace
curl http://web-service.production.svc.cluster.local

# Short form
curl http://web-service.production
```

## Complete Example

```yaml
# Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80

---
# ClusterIP Service (internal)
apiVersion: v1
kind: Service
metadata:
  name: web-service
spec:
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 80

---
# NodePort Service (external)
apiVersion: v1
kind: Service
metadata:
  name: web-nodeport
spec:
  type: NodePort
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 80
    nodePort: 30080
```

## Multi-Port Services

```yaml
apiVersion: v1
kind: Service
metadata:
  name: multi-port-service
spec:
  selector:
    app: myapp
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: https
    port: 443
    targetPort: 8443
  - name: metrics
    port: 9090
    targetPort: 9090
```

## Headless Service

**No load balancing, returns pod IPs directly**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: headless-service
spec:
  clusterIP: None  # Headless!
  selector:
    app: database
  ports:
  - port: 5432
```

**Use case:** StatefulSets, databases

```bash
# DNS returns all pod IPs
nslookup headless-service
```

## ExternalName Service

**Map to external DNS:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-db
spec:
  type: ExternalName
  externalName: my.database.example.com
```

```bash
# Access external service as if it's internal
curl http://external-db
# Resolves to my.database.example.com
```

## Session Affinity

**Sticky sessions:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: sticky-service
spec:
  selector:
    app: web
  sessionAffinity: ClientIP  # Same client → same pod
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800  # 3 hours
  ports:
  - port: 80
```

## Endpoints

**Services automatically create Endpoints:**

```bash
# View endpoints
kubectl get endpoints web-service

# Shows pod IPs
kubectl describe endpoints web-service
```

**Manual Endpoints (service without selector):**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-service
spec:
  ports:
  - port: 80

---
apiVersion: v1
kind: Endpoints
metadata:
  name: external-service  # Same as service name
subsets:
- addresses:
  - ip: 192.168.1.100
  - ip: 192.168.1.101
  ports:
  - port: 80
```

## Hands-On Exercises

### Exercise 1: Create Service for Deployment

```bash
# Create deployment
kubectl create deployment nginx --image=nginx --replicas=3

# Expose as ClusterIP
kubectl expose deployment nginx --port=80 --target-port=80

# Test from another pod
kubectl run test --image=busybox -it --rm -- wget -O- http://nginx

# Create NodePort service
kubectl expose deployment nginx --port=80 --type=NodePort --name=nginx-nodeport

# Get node port
kubectl get svc nginx-nodeport

# Access (Minikube)
minikube service nginx-nodeport --url
```

### Exercise 2: Multi-Tier Application

```yaml
# database-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_PASSWORD
          value: secret
        ports:
        - containerPort: 5432

---
# database-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432

---
# backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: api
        image: myapi
        env:
        - name: DATABASE_URL
          value: "postgresql://postgres:secret@postgres-service:5432/mydb"
        ports:
        - containerPort: 3000

---
# backend-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: backend
  ports:
  - port: 3000
    targetPort: 3000

---
# frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: web
        image: nginx
        ports:
        - containerPort: 80

---
# frontend-service.yaml (NodePort for external access)
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
spec:
  type: NodePort
  selector:
    app: frontend
  ports:
  - port: 80
    targetPort: 80
    nodePort: 30080
```

```bash
# Apply all
kubectl apply -f .

# Check services
kubectl get svc

# Test DNS
kubectl run test --image=busybox -it --rm -- nslookup postgres-service
kubectl run test --image=busybox -it --rm -- nslookup backend-service
```

## Service Commands

```bash
# Create service
kubectl expose deployment NAME --port=PORT
kubectl create service clusterip NAME --tcp=PORT:TARGET_PORT
kubectl apply -f service.yaml

# List services
kubectl get services
kubectl get svc

# Details
kubectl describe service NAME
kubectl get svc NAME -o yaml

# Get endpoints
kubectl get endpoints NAME

# Delete service
kubectl delete service NAME
kubectl delete svc NAME
```

## Best Practices

1. ✅ Use ClusterIP for internal communication
2. ✅ Use descriptive service names
3. ✅ Match selector labels carefully
4. ✅ Use DNS for service discovery (not env vars)
5. ✅ Name ports in multi-port services
6. ✅ Use readiness probes (unhealthy pods excluded)
7. ✅ Consider session affinity needs
8. ✅ Use LoadBalancer only when needed (costs money)
9. ✅ Monitor endpoints health
10. ✅ Use namespaces for isolation

## Next Steps

Move to `05-kubernetes-configmaps-secrets.md` to learn configuration management!
