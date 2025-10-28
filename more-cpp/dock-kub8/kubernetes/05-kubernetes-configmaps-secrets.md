# Kubernetes ConfigMaps and Secrets - Configuration Management

## Theory

### Configuration Management Problem:

**Don't hardcode config in images!**
- Environment-specific values (URLs, ports)
- Credentials
- Feature flags
- Application settings

### Solutions:

1. **ConfigMaps**: Non-sensitive configuration data
2. **Secrets**: Sensitive data (passwords, tokens, keys)

## ConfigMaps

### What is a ConfigMap?

Store configuration data as key-value pairs, accessible to pods as:
- Environment variables
- Command-line arguments
- Configuration files in volumes

### Creating ConfigMaps

#### Method 1: From Literal Values

```bash
kubectl create configmap app-config \
  --from-literal=APP_ENV=production \
  --from-literal=APP_PORT=8080 \
  --from-literal=LOG_LEVEL=info
```

#### Method 2: From File

```bash
# Create config file
cat > app.properties << EOF
database.host=postgres-service
database.port=5432
cache.enabled=true
EOF

# Create ConfigMap from file
kubectl create configmap app-config --from-file=app.properties
```

#### Method 3: From Directory

```bash
# Create directory with config files
mkdir config
echo "production" > config/environment
echo "8080" > config/port

# Create ConfigMap from directory
kubectl create configmap app-config --from-file=config/
```

#### Method 4: YAML Definition

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  # Simple key-value pairs
  APP_ENV: "production"
  APP_PORT: "8080"
  LOG_LEVEL: "info"

  # Multi-line configuration file
  app.conf: |
    server {
      listen 80;
      server_name example.com;
      location / {
        proxy_pass http://backend;
      }
    }
```

```bash
kubectl apply -f configmap.yaml
```

### Using ConfigMaps in Pods

#### 1. As Environment Variables

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: myapp
    env:
    # Single value from ConfigMap
    - name: APP_ENV
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: APP_ENV

    - name: LOG_LEVEL
      valueFrom:
        configMapKeyRef:
          name: app-config
          key: LOG_LEVEL

    # Load all keys as env vars
    envFrom:
    - configMapRef:
        name: app-config
```

#### 2. As Volume Mount (Files)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - name: config-volume
      mountPath: /etc/config

  volumes:
  - name: config-volume
    configMap:
      name: app-config
```

```bash
# Files appear in /etc/config/
# /etc/config/APP_ENV
# /etc/config/APP_PORT
# /etc/config/app.conf
```

#### 3. Specific Keys as Files

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
spec:
  containers:
  - name: nginx
    image: nginx
    volumeMounts:
    - name: config
      mountPath: /etc/nginx/nginx.conf
      subPath: nginx.conf  # Mount specific file

  volumes:
  - name: config
    configMap:
      name: nginx-config
      items:
      - key: nginx.conf
        path: nginx.conf
```

## Secrets

### What is a Secret?

Store sensitive data (passwords, tokens, keys) with:
- Base64 encoding (not encryption!)
- RBAC access control
- Encryption at rest (optional, configured on cluster)

### Secret Types:

- `Opaque`: Generic (default)
- `kubernetes.io/dockerconfigjson`: Docker registry credentials
- `kubernetes.io/tls`: TLS certificates
- `kubernetes.io/service-account-token`: Service account token
- `kubernetes.io/basic-auth`: Basic authentication

### Creating Secrets

#### Method 1: From Literal Values

```bash
kubectl create secret generic db-secret \
  --from-literal=username=admin \
  --from-literal=password=secretpass123
```

#### Method 2: From Files

```bash
# Create files
echo -n 'admin' > username.txt
echo -n 'secretpass123' > password.txt

# Create secret from files
kubectl create secret generic db-secret \
  --from-file=username=username.txt \
  --from-file=password=password.txt

# Clean up files
rm username.txt password.txt
```

#### Method 3: YAML Definition

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
data:
  # Base64 encoded values
  username: YWRtaW4=        # admin
  password: c2VjcmV0cGFzczEyMw==  # secretpass123
```

```bash
# Encode values
echo -n 'admin' | base64        # YWRtaW4=
echo -n 'secretpass123' | base64  # c2VjcmV0cGFzczEyMw==

# Apply
kubectl apply -f secret.yaml
```

#### Using stringData (No encoding needed)

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
stringData:  # Kubernetes encodes automatically
  username: admin
  password: secretpass123
```

### Using Secrets in Pods

#### 1. As Environment Variables

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: myapp
    env:
    # Single secret value
    - name: DB_USERNAME
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: username

    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: password

    # Load all keys as env vars
    envFrom:
    - secretRef:
        name: db-secret
```

#### 2. As Volume Mount

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
  - name: app
    image: myapp
    volumeMounts:
    - name: secret-volume
      mountPath: /etc/secrets
      readOnly: true

  volumes:
  - name: secret-volume
    secret:
      secretName: db-secret
```

```bash
# Files in /etc/secrets/
# /etc/secrets/username  (contains: admin)
# /etc/secrets/password  (contains: secretpass123)
```

### Docker Registry Secret

```bash
# Create Docker registry secret
kubectl create secret docker-registry regcred \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=myuser \
  --docker-password=mypass \
  --docker-email=myemail@example.com

# Use in pod
apiVersion: v1
kind: Pod
metadata:
  name: private-image-pod
spec:
  imagePullSecrets:
  - name: regcred
  containers:
  - name: app
    image: myuser/private-image:latest
```

### TLS Secret

```bash
# Create TLS secret
kubectl create secret tls tls-secret \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key
```

```yaml
# Use in Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tls-ingress
spec:
  tls:
  - hosts:
    - example.com
    secretName: tls-secret
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: web-service
            port:
              number: 80
```

## Complete Example

```yaml
# ConfigMap for application settings
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  APP_NAME: "MyApp"
  LOG_LEVEL: "info"
  API_URL: "http://api-service:8080"
  nginx.conf: |
    server {
      listen 80;
      location / {
        proxy_pass http://backend:8080;
      }
    }

---
# Secret for database credentials
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
stringData:
  DB_HOST: "postgres-service"
  DB_PORT: "5432"
  DB_USER: "appuser"
  DB_PASSWORD: "supersecret123"
  DB_NAME: "myappdb"

---
# Deployment using ConfigMap and Secret
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
      - name: app
        image: myapp:1.0
        ports:
        - containerPort: 8080

        # Environment variables from ConfigMap
        env:
        - name: APP_NAME
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: APP_NAME

        # Environment variables from Secret
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: DB_USER
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: DB_PASSWORD

        # Load all keys from ConfigMap
        envFrom:
        - configMapRef:
            name: app-config
        - secretRef:
            name: db-secret

        # Mount nginx config file
        volumeMounts:
        - name: config
          mountPath: /etc/nginx/nginx.conf
          subPath: nginx.conf

      volumes:
      - name: config
        configMap:
          name: app-config
```

## ConfigMap/Secret Commands

```bash
# ConfigMap
kubectl create configmap NAME --from-literal=key=value
kubectl create configmap NAME --from-file=file.txt
kubectl get configmaps
kubectl describe configmap NAME
kubectl edit configmap NAME
kubectl delete configmap NAME

# Secret
kubectl create secret generic NAME --from-literal=key=value
kubectl create secret generic NAME --from-file=file.txt
kubectl get secrets
kubectl describe secret NAME
kubectl edit secret NAME
kubectl delete secret NAME

# View content (base64 decode)
kubectl get secret NAME -o jsonpath='{.data.password}' | base64 --decode
```

## Updating ConfigMaps and Secrets

```bash
# Method 1: Edit directly
kubectl edit configmap app-config

# Method 2: Replace from file
kubectl create configmap app-config --from-file=config.txt --dry-run=client -o yaml | kubectl replace -f -

# Method 3: Update YAML and apply
kubectl apply -f configmap.yaml
```

**Note:** Pods don't automatically reload after ConfigMap/Secret update!

**Solutions:**
1. Restart pods: `kubectl rollout restart deployment/app`
2. Use tools like Reloader
3. Application watches for file changes (volume mounts)

## Hands-On Exercises

### Exercise 1: ConfigMap from Literals

```bash
# Create ConfigMap
kubectl create configmap app-env \
  --from-literal=ENV=production \
  --from-literal=PORT=8080

# Create pod using ConfigMap
kubectl run test-pod --image=busybox --rm -it -- sh
# Inside pod:
# printenv | grep PORT
```

### Exercise 2: Secret for Database

```bash
# Create secret
kubectl create secret generic mysql-secret \
  --from-literal=MYSQL_ROOT_PASSWORD=rootpass \
  --from-literal=MYSQL_DATABASE=mydb

# Create MySQL pod
kubectl run mysql \
  --image=mysql:8 \
  --env-from=secret/mysql-secret

# Verify
kubectl logs mysql
```

### Exercise 3: Config File Volume

```yaml
# nginx-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config
data:
  index.html: |
    <html>
    <body>
      <h1>Hello from ConfigMap!</h1>
    </body>
    </html>

---
apiVersion: v1
kind: Pod
metadata:
  name: nginx-configmap
spec:
  containers:
  - name: nginx
    image: nginx
    volumeMounts:
    - name: html
      mountPath: /usr/share/nginx/html

  volumes:
  - name: html
    configMap:
      name: nginx-config
```

```bash
kubectl apply -f nginx-config.yaml
kubectl port-forward nginx-configmap 8080:80
curl http://localhost:8080
```

## Best Practices

1. ✅ Use ConfigMaps for non-sensitive config
2. ✅ Use Secrets for sensitive data
3. ✅ Don't commit secrets to Git
4. ✅ Use external secret management (Vault, AWS Secrets Manager)
5. ✅ Enable encryption at rest for secrets
6. ✅ Use RBAC to restrict secret access
7. ✅ Mount secrets as volumes (not env vars) when possible
8. ✅ Use tools like sealed-secrets for GitOps
9. ✅ Rotate secrets regularly
10. ✅ Keep ConfigMaps small (<1MB)

## Next Steps

Move to `06-kubernetes-volumes.md` to learn about persistent storage!
