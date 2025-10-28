# Kubernetes Persistent Volumes - Storage Management

## Theory

Pods are ephemeral - data is lost when they restart. For persistent storage, Kubernetes provides:

1. **Volume**: Directory accessible to containers in a pod
2. **PersistentVolume (PV)**: Cluster storage resource
3. **PersistentVolumeClaim (PVC)**: Request for storage
4. **StorageClass**: Dynamic provisioning of PVs

## Volume Types

### 1. emptyDir (Temporary)
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - mountPath: /cache
      name: cache-volume
  volumes:
  - name: cache-volume
    emptyDir: {}  # Deleted when pod deleted
```

### 2. hostPath (Node Storage)
```yaml
volumes:
- name: host-volume
  hostPath:
    path: /data/app
    type: DirectoryOrCreate
```

### 3. configMap and secret
```yaml
volumes:
- name: config
  configMap:
    name: app-config
- name: secret
  secret:
    secretName: app-secret
```

## PersistentVolume (PV)

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-data
spec:
  capacity:
    storage: 10Gi
  volumeMode: Filesystem
  accessModes:
  - ReadWriteOnce  # RWO: Single node
  - ReadOnlyMany   # ROX: Multiple nodes read
  - ReadWriteMany  # RWX: Multiple nodes write
  persistentVolumeReclaimPolicy: Retain  # Retain|Delete|Recycle
  storageClassName: standard
  hostPath:
    path: "/mnt/data"
```

## PersistentVolumeClaim (PVC)

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-data
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: standard
```

## Using PVC in Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-storage
spec:
  containers:
  - name: app
    image: nginx
    volumeMounts:
    - mountPath: /data
      name: storage
  volumes:
  - name: storage
    persistentVolumeClaim:
      claimName: pvc-data
```

## StorageClass (Dynamic Provisioning)

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-storage
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iopsPerGB: "10"
allowVolumeExpansion: true
```

## Complete Example

```yaml
# PersistentVolume
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mysql-pv
spec:
  capacity:
    storage: 20Gi
  accessModes:
  - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: manual
  hostPath:
    path: "/mnt/mysql"

---
# PersistentVolumeClaim
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mysql-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: manual

---
# Deployment using PVC
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mysql
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: mysql:8
        env:
        - name: MYSQL_ROOT_PASSWORD
          value: password
        volumeMounts:
        - name: mysql-storage
          mountPath: /var/lib/mysql
      volumes:
      - name: mysql-storage
        persistentVolumeClaim:
          claimName: mysql-pvc
```

## StatefulSet with Volumes

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web
spec:
  serviceName: "nginx"
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        volumeMounts:
        - name: www
          mountPath: /usr/share/nginx/html
  volumeClaimTemplates:
  - metadata:
      name: www
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 1Gi
```

## Commands

```bash
# PV
kubectl get pv
kubectl describe pv NAME

# PVC
kubectl get pvc
kubectl describe pvc NAME
kubectl delete pvc NAME

# StorageClass
kubectl get storageclass
kubectl describe sc NAME
```

## Next Steps
Move to `07-kubernetes-ingress.md` for HTTP routing!
