# Kubernetes Production - Best Practices & Operations

## Production Checklist

### 1. High Availability

```yaml
# Multiple replicas
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 3  # Minimum 3 for HA
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0  # Zero-downtime
```

### 2. Resource Management

```yaml
spec:
  containers:
  - name: app
    resources:
      requests:  # Guaranteed
        memory: "256Mi"
        cpu: "500m"
      limits:    # Maximum
        memory: "512Mi"
        cpu: "1000m"
```

### 3. Health Checks

```yaml
spec:
  containers:
  - name: app
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5
    startupProbe:
      httpGet:
        path: /startup
        port: 8080
      failureThreshold: 30
      periodSeconds: 10
```

### 4. Pod Disruption Budget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: web-pdb
spec:
  minAvailable: 2  # or maxUnavailable: 1
  selector:
    matchLabels:
      app: web
```

### 5. Autoscaling

```yaml
# HorizontalPodAutoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 6. Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-policy
spec:
  podSelector:
    matchLabels:
      tier: backend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tier: frontend
    ports:
    - protocol: TCP
      port: 8080
```

### 7. Security

```yaml
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
  containers:
  - name: app
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
```

### 8. Monitoring & Logging

```bash
# Install Prometheus + Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack

# Install EFK Stack (Elasticsearch, Fluentd, Kibana)
helm repo add elastic https://helm.elastic.co
helm install elasticsearch elastic/elasticsearch
helm install kibana elastic/kibana
```

### 9. Backup Strategy

```bash
# Install Velero for cluster backups
velero install \
  --provider aws \
  --bucket my-backup-bucket \
  --backup-location-config region=us-east-1

# Create backup
velero backup create my-backup

# Restore
velero restore create --from-backup my-backup
```

### 10. CI/CD Pipeline

```yaml
# GitLab CI example
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - docker build -t myapp:$CI_COMMIT_SHA .
    - docker push myapp:$CI_COMMIT_SHA

deploy:
  stage: deploy
  script:
    - kubectl set image deployment/myapp myapp=myapp:$CI_COMMIT_SHA
    - kubectl rollout status deployment/myapp
  only:
    - main
```

## Production Architecture Example

```yaml
# Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: production

---
# ResourceQuota
apiVersion: v1
kind: ResourceQuota
metadata:
  name: prod-quota
  namespace: production
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 200Gi
    limits.cpu: "200"
    limits.memory: 400Gi
    pods: "100"

---
# Deployment with all best practices
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  namespace: production
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 0
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - web
            topologyKey: kubernetes.io/hostname
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: web
        image: myapp:1.0
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
# Service
apiVersion: v1
kind: Service
metadata:
  name: web-service
  namespace: production
spec:
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 8080

---
# Ingress with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-ingress
  namespace: production
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - myapp.com
    secretName: web-tls
  rules:
  - host: myapp.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80

---
# HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 5
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

---
# PDB
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: web-pdb
  namespace: production
spec:
  minAvailable: 3
  selector:
    matchLabels:
      app: web
```

## Cluster Maintenance

```bash
# Node maintenance
kubectl cordon node-1      # Mark unschedulable
kubectl drain node-1       # Evict pods
# Perform maintenance
kubectl uncordon node-1    # Mark schedulable

# Cluster upgrade
kubeadm upgrade plan
kubeadm upgrade apply v1.28.0

# Certificate rotation
kubeadm certs renew all
```

## Disaster Recovery

1. **Regular backups**: etcd, PVs, manifests
2. **Multi-region deployment**
3. **Disaster recovery plan**
4. **Automated recovery procedures**
5. **Regular DR drills**

## Monitoring Metrics

**Node metrics:**
- CPU usage
- Memory usage
- Disk I/O
- Network traffic

**Pod metrics:**
- CPU/Memory usage
- Restart count
- Request/Limits ratio

**Application metrics:**
- Request rate
- Error rate
- Latency
- Saturation

## Best Practices Summary

1. ✅ Use namespaces for isolation
2. ✅ Set resource requests/limits
3. ✅ Implement health checks
4. ✅ Use horizontal autoscaling
5. ✅ Enable pod disruption budgets
6. ✅ Use network policies
7. ✅ Implement security contexts
8. ✅ Monitor everything
9. ✅ Automate backups
10. ✅ Test disaster recovery
11. ✅ Use GitOps (ArgoCD, Flux)
12. ✅ Implement CI/CD
13. ✅ Use Helm for deployments
14. ✅ Regular cluster maintenance
15. ✅ Document everything

## Congratulations!

You've completed the Kubernetes learning path! You now know:
- Kubernetes architecture and concepts
- Pods, Deployments, Services
- Configuration management
- Storage and volumes
- Networking and Ingress
- Namespaces and resource quotas
- Helm package manager
- Production best practices

**Continue learning:** CKAD/CKA certifications, service mesh (Istio), GitOps (ArgoCD), advanced networking, security hardening!
