# Kubernetes Helm - Package Manager

## Theory

**Helm** = Package manager for Kubernetes (like apt, yum, npm)

**Benefits:**
- Pre-configured applications (Charts)
- Templating and reusability
- Version management
- Easy upgrades/rollbacks
- Dependency management

**Concepts:**
- **Chart**: Package of Kubernetes resources
- **Release**: Instance of a chart running in cluster
- **Repository**: Collection of charts

## Installing Helm

```bash
# Download Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify
helm version
```

## Using Helm Charts

```bash
# Add repository
helm repo add bitnami https://charts.bitnami.com/bitnami

# Update repositories
helm repo update

# Search charts
helm search repo nginx

# Install chart
helm install my-nginx bitnami/nginx

# List releases
helm list

# Upgrade release
helm upgrade my-nginx bitnami/nginx

# Rollback
helm rollback my-nginx 1

# Uninstall
helm uninstall my-nginx
```

## Creating Custom Chart

```bash
# Create new chart
helm create myapp

# Chart structure:
myapp/
├── Chart.yaml          # Chart metadata
├── values.yaml         # Default configuration
├── templates/          # Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
└── charts/            # Dependencies
```

### Chart.yaml
```yaml
apiVersion: v2
name: myapp
description: My Application
version: 1.0.0
appVersion: "1.0"
```

### values.yaml
```yaml
replicaCount: 3
image:
  repository: nginx
  tag: "1.25"
  pullPolicy: IfNotPresent
service:
  type: ClusterIP
  port: 80
ingress:
  enabled: false
resources:
  limits:
    cpu: 100m
    memory: 128Mi
```

### templates/deployment.yaml
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app: {{ .Release.Name }}
    spec:
      containers:
      - name: {{ .Chart.Name }}
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        ports:
        - containerPort: 80
        resources:
          {{- toYaml .Values.resources | nindent 10 }}
```

## Installing Custom Chart

```bash
# Install from directory
helm install my-release ./myapp

# Install with custom values
helm install my-release ./myapp --set replicaCount=5

# Install with values file
helm install my-release ./myapp -f custom-values.yaml

# Dry run (test)
helm install my-release ./myapp --dry-run --debug
```

## Helm Commands

```bash
# Repository management
helm repo add NAME URL
helm repo list
helm repo update

# Chart management
helm search repo KEYWORD
helm show chart CHART
helm show values CHART

# Release management
helm install NAME CHART
helm list
helm status NAME
helm upgrade NAME CHART
helm rollback NAME REVISION
helm uninstall NAME

# Chart creation
helm create NAME
helm lint ./chart
helm package ./chart
```

## Popular Charts

```bash
# Install PostgreSQL
helm install my-db bitnami/postgresql

# Install Redis
helm install my-cache bitnami/redis

# Install Prometheus + Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack
```

## Next Steps
Move to `10-kubernetes-production.md` for production best practices!
