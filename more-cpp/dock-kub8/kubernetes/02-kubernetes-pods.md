# Kubernetes Pods - Deep Dive

## Theory

### What is a Pod?

A **Pod** is the smallest deployable unit in Kubernetes. It represents a single instance of a running process in your cluster.

### Key Characteristics:

1. **One or More Containers**: Usually one, sometimes multiple (tightly coupled)
2. **Shared Network**: All containers share same IP and ports
3. **Shared Storage**: Can mount same volumes
4. **Atomic Unit**: Created, scheduled, and deleted together
5. **Ephemeral**: Pods are mortal (not durable)

### Pod Lifecycle:

```
┌──────────┐
│ Pending  │ ← Pod accepted, containers being created
└────┬─────┘
     │
┌────▼─────┐
│ Running  │ ← All containers running
└────┬─────┘
     │
     ├──→ ┌────────────┐
     │    │ Succeeded  │ ← All containers terminated successfully
     │    └────────────┘
     │
     └──→ ┌────────────┐
          │  Failed    │ ← At least one container failed
          └────────────┘

Other states: Unknown, CrashLoopBackOff
```

### Single vs Multi-Container Pods:

```
SINGLE CONTAINER (Most Common):
┌─────────────────┐
│ Pod: web-app    │
│ ┌─────────────┐ │
│ │   nginx     │ │
│ └─────────────┘ │
│ IP: 10.244.0.5  │
└─────────────────┘

MULTI-CONTAINER (Sidecar Pattern):
┌──────────────────────────┐
│ Pod: app-with-logging    │
│ ┌─────────┐ ┌──────────┐ │
│ │ main    │ │ logging  │ │
│ │ app     │ │ sidecar  │ │
│ └─────────┘ └──────────┘ │
│ IP: 10.244.0.6           │
│ Share: Network, Volumes  │
└──────────────────────────┘
```

## Pod YAML Specification

### Basic Pod:

```yaml
apiVersion: v1
kind: Pod
metadata:
  # Name must be unique within namespace
  name: nginx-pod

  # Labels for organization and selection
  labels:
    app: nginx
    environment: development
    tier: frontend

  # Annotations (non-identifying metadata)
  annotations:
    description: "My nginx web server"
    version: "1.0"

spec:
  # List of containers
  containers:
  - name: nginx
    image: nginx:1.25

    # Container ports (documentation, not enforcement)
    ports:
    - containerPort: 80
      name: http
      protocol: TCP

    # Environment variables
    env:
    - name: ENV
      value: "development"

    # Resource requests and limits
    resources:
      requests:
        memory: "64Mi"
        cpu: "250m"
      limits:
        memory: "128Mi"
        cpu: "500m"
```

### Pod with Multiple Containers:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: multi-container-pod
spec:
  # Multiple containers in same pod
  containers:

  # Main application container
  - name: app
    image: myapp:1.0
    ports:
    - containerPort: 8080
    volumeMounts:
    - name: shared-data
      mountPath: /app/data

  # Sidecar: Log aggregator
  - name: log-aggregator
    image: fluentd:latest
    volumeMounts:
    - name: shared-data
      mountPath: /logs

  # Sidecar: Metrics exporter
  - name: metrics
    image: prometheus-exporter:latest
    ports:
    - containerPort: 9090

  # Shared volume between containers
  volumes:
  - name: shared-data
    emptyDir: {}
```

## Container Configuration Options

### 1. Image and Pull Policy:

```yaml
spec:
  containers:
  - name: app
    image: nginx:1.25  # Image with tag

    # Image pull policy
    imagePullPolicy: Always  # or IfNotPresent, Never

    # Pull from private registry
    # (requires imagePullSecrets)

  # For private registries
  imagePullSecrets:
  - name: my-registry-secret
```

### 2. Commands and Arguments:

```yaml
spec:
  containers:
  - name: app
    image: ubuntu

    # Override ENTRYPOINT
    command: ["/bin/sh"]

    # Override CMD
    args: ["-c", "echo Hello && sleep 3600"]

    # Alternative syntax
    # command:
    #   - /bin/sh
    # args:
    #   - -c
    #   - echo Hello && sleep 3600
```

### 3. Environment Variables:

```yaml
spec:
  containers:
  - name: app
    image: myapp
    env:

    # Direct value
    - name: ENV
      value: "production"

    # From pod information (Downward API)
    - name: POD_NAME
      valueFrom:
        fieldRef:
          fieldPath: metadata.name

    - name: POD_IP
      valueFrom:
        fieldRef:
          fieldPath: status.podIP

    # From ConfigMap
    - name: CONFIG_VALUE
      valueFrom:
        configMapKeyRef:
          name: my-config
          key: config.key

    # From Secret
    - name: PASSWORD
      valueFrom:
        secretKeyRef:
          name: my-secret
          key: password

    # Load all keys from ConfigMap
    envFrom:
    - configMapRef:
        name: my-config
```

### 4. Resource Requests and Limits:

```yaml
spec:
  containers:
  - name: app
    image: myapp
    resources:
      # Requests: Guaranteed resources
      requests:
        memory: "256Mi"   # 256 Mebibytes
        cpu: "500m"       # 500 millicores (0.5 CPU)
        ephemeral-storage: "2Gi"

      # Limits: Maximum resources
      limits:
        memory: "512Mi"
        cpu: "1000m"      # 1 CPU
        ephemeral-storage: "4Gi"

# CPU units:
# 1 CPU = 1000m (millicores)
# 0.5 CPU = 500m
# 1 CPU = 1 AWS vCPU = 1 GCP Core

# Memory units:
# Ki = Kibibyte (1024 bytes)
# Mi = Mebibyte (1024 Ki)
# Gi = Gibibyte (1024 Mi)
```

### 5. Probes (Health Checks):

```yaml
spec:
  containers:
  - name: app
    image: myapp

    # Liveness Probe: Restart if fails
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 30  # Wait before first check
      periodSeconds: 10        # Check every 10 seconds
      timeoutSeconds: 5        # Timeout for check
      failureThreshold: 3      # Restart after 3 failures
      successThreshold: 1      # Success after 1 success

    # Readiness Probe: Remove from service if fails
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5

    # Startup Probe: For slow-starting containers
    startupProbe:
      httpGet:
        path: /startup
        port: 8080
      failureThreshold: 30
      periodSeconds: 10
```

### Probe Types:

```yaml
# 1. HTTP GET
livenessProbe:
  httpGet:
    path: /health
    port: 8080
    httpHeaders:
    - name: Custom-Header
      value: Value

# 2. TCP Socket
livenessProbe:
  tcpSocket:
    port: 8080

# 3. Exec Command
livenessProbe:
  exec:
    command:
    - cat
    - /tmp/healthy

# 4. gRPC (Kubernetes 1.24+)
livenessProbe:
  grpc:
    port: 9090
```

### 6. Lifecycle Hooks:

```yaml
spec:
  containers:
  - name: app
    image: myapp
    lifecycle:
      # PostStart: Run after container starts
      postStart:
        exec:
          command: ["/bin/sh", "-c", "echo Started > /tmp/started"]

      # PreStop: Run before container stops
      preStop:
        exec:
          command: ["/bin/sh", "-c", "nginx -s quit"]
```

## Volume Mounts

### EmptyDir (Temporary Storage):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-with-emptydir
spec:
  containers:
  - name: writer
    image: alpine
    command: ["/bin/sh", "-c"]
    args: ["while true; do date >> /data/log.txt; sleep 5; done"]
    volumeMounts:
    - name: cache
      mountPath: /data

  - name: reader
    image: alpine
    command: ["/bin/sh", "-c"]
    args: ["tail -f /data/log.txt"]
    volumeMounts:
    - name: cache
      mountPath: /data

  volumes:
  - name: cache
    emptyDir: {}
    # emptyDir:
    #   sizeLimit: 1Gi
    #   medium: Memory  # Use RAM instead of disk
```

### HostPath (Host Directory):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-with-hostpath
spec:
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - name: host-data
      mountPath: /usr/share/nginx/html

  volumes:
  - name: host-data
    hostPath:
      path: /data/html
      type: Directory  # DirectoryOrCreate, File, FileOrCreate, etc.
```

## Pod Security

### Security Context:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  # Pod-level security context
  securityContext:
    runAsUser: 1000
    runAsGroup: 3000
    fsGroup: 2000

  containers:
  - name: app
    image: nginx

    # Container-level security context (overrides pod-level)
    securityContext:
      runAsUser: 2000
      runAsNonRoot: true
      readOnlyRootFilesystem: true
      allowPrivilegeEscalation: false
      capabilities:
        drop:
        - ALL
        add:
        - NET_BIND_SERVICE
```

## Init Containers

### Initialization Before Main Containers:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-with-init
spec:
  # Init containers run BEFORE main containers
  initContainers:

  # Init 1: Wait for service
  - name: wait-for-db
    image: busybox
    command: ['sh', '-c']
    args:
    - until nslookup db-service; do
        echo waiting for db;
        sleep 2;
      done

  # Init 2: Download configuration
  - name: download-config
    image: busybox
    command: ['sh', '-c']
    args:
    - wget -O /config/app.conf http://config-server/config
    volumeMounts:
    - name: config
      mountPath: /config

  # Main containers start AFTER all init containers succeed
  containers:
  - name: app
    image: myapp
    volumeMounts:
    - name: config
      mountPath: /etc/config

  volumes:
  - name: config
    emptyDir: {}
```

## Common Pod Patterns

### 1. Sidecar Pattern:

```yaml
# Main container + helper container
apiVersion: v1
kind: Pod
metadata:
  name: sidecar-example
spec:
  containers:
  # Main application
  - name: app
    image: myapp
    volumeMounts:
    - name: logs
      mountPath: /var/log/app

  # Sidecar: Log forwarder
  - name: log-forwarder
    image: fluentd
    volumeMounts:
    - name: logs
      mountPath: /var/log/app

  volumes:
  - name: logs
    emptyDir: {}
```

### 2. Adapter Pattern:

```yaml
# Transform data for consumption
apiVersion: v1
kind: Pod
metadata:
  name: adapter-example
spec:
  containers:
  # Main app with custom log format
  - name: app
    image: myapp

  # Adapter: Converts logs to standard format
  - name: log-adapter
    image: log-formatter
```

### 3. Ambassador Pattern:

```yaml
# Proxy connections
apiVersion: v1
kind: Pod
metadata:
  name: ambassador-example
spec:
  containers:
  # Main app
  - name: app
    image: myapp

  # Ambassador: Proxy to external service
  - name: proxy
    image: ambassador-proxy
    ports:
    - containerPort: 8080
```

## Pod Commands

```bash
# Create pod
kubectl run nginx --image=nginx
kubectl create -f pod.yaml
kubectl apply -f pod.yaml

# List pods
kubectl get pods
kubectl get pods -o wide
kubectl get pods --show-labels
kubectl get pods -l app=nginx

# Describe pod
kubectl describe pod nginx

# View logs
kubectl logs nginx
kubectl logs nginx -c container-name  # Multi-container
kubectl logs nginx --previous  # Previous instance
kubectl logs -f nginx  # Follow logs

# Execute commands
kubectl exec nginx -- ls /
kubectl exec -it nginx -- bash

# Port forwarding
kubectl port-forward nginx 8080:80

# Copy files
kubectl cp nginx:/etc/nginx/nginx.conf ./nginx.conf
kubectl cp ./file.txt nginx:/tmp/

# Edit pod (limited changes)
kubectl edit pod nginx

# Delete pod
kubectl delete pod nginx
kubectl delete -f pod.yaml
```

## Hands-On Exercises

### Exercise 1: Create Pod with Environment Variables

```yaml
# pod-env.yaml
apiVersion: v1
kind: Pod
metadata:
  name: env-pod
spec:
  containers:
  - name: app
    image: busybox
    command: ['sh', '-c', 'echo "Hello, $NAME from $LOCATION" && sleep 3600']
    env:
    - name: NAME
      value: "Kubernetes"
    - name: LOCATION
      value: "Cloud"
```

```bash
kubectl apply -f pod-env.yaml
kubectl logs env-pod
kubectl exec env-pod -- env
```

### Exercise 2: Pod with Health Checks

```yaml
# pod-health.yaml
apiVersion: v1
kind: Pod
metadata:
  name: health-pod
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
    livenessProbe:
      httpGet:
        path: /
        port: 80
      initialDelaySeconds: 10
      periodSeconds: 5
    readinessProbe:
      httpGet:
        path: /
        port: 80
      initialDelaySeconds: 5
      periodSeconds: 3
```

```bash
kubectl apply -f pod-health.yaml
kubectl describe pod health-pod  # Check probe status
kubectl get pod health-pod  # Should show READY 1/1
```

### Exercise 3: Multi-Container Pod

```yaml
# multi-container.yaml
apiVersion: v1
kind: Pod
metadata:
  name: multi-container
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
    volumeMounts:
    - name: html
      mountPath: /usr/share/nginx/html

  - name: content-updater
    image: busybox
    command: ["/bin/sh"]
    args:
    - -c
    - while true; do
        date > /html/index.html;
        sleep 10;
      done
    volumeMounts:
    - name: html
      mountPath: /html

  volumes:
  - name: html
    emptyDir: {}
```

```bash
kubectl apply -f multi-container.yaml
kubectl port-forward multi-container 8080:80 &
curl http://localhost:8080  # See date updating every 10s
kubectl logs multi-container -c nginx
kubectl logs multi-container -c content-updater
```

## Pod Best Practices

1. **One Process per Container**: Don't use supervisor/systemd
2. **Use Labels**: For organization and selection
3. **Set Resource Limits**: Prevent resource starvation
4. **Add Health Checks**: Liveness and readiness probes
5. **Use Init Containers**: For initialization logic
6. **Read-Only Root**: Use `readOnlyRootFilesystem: true`
7. **Non-Root User**: Use `runAsNonRoot: true`
8. **Small Images**: Alpine or distroless
9. **Avoid hostPath**: Use PersistentVolumes instead
10. **Don't Manage Pods Directly**: Use Deployments (next lesson)

## Next Steps

You now understand:
- Pod architecture and lifecycle
- Pod configuration options
- Multi-container patterns
- Health checks and probes
- Security contexts

Move to `03-kubernetes-deployments.md` to learn about managing pod replicas!
