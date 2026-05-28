# 3D Reconstruction Deployment Architecture

Tai lieu nay tong hop kien truc van hanh cua du an theo 5 lop: User Traffic,
Kubernetes Runtime, Application Layer, CI/CD + GitOps, Monitoring + Autoscaling.

## Trang Thai Hien Tai Trong Repo

Du an hien co cac thanh phan chay local/dev:

| Thanh phan | Trang thai |
| --- | --- |
| Mobile client | Expo / React Native trong `mobile/`, goi backend qua `EXPO_PUBLIC_API_BASE_URL` hoac default `http://192.168.1.3:8000`. |
| Backend AI | FastAPI trong `server/main.py`, port `8000`, co YOLO segmentation va image reconstruction baseline. |
| ML pipeline | Python/PyTorch trong `project/src`, gom preprocessing, training, evaluation, inference. |
| Docker | Co `server/Dockerfile` va `docker-compose.yml` cho backend + Floci local storage mock. |
| Database | Chua co database application. Artifact hien luu local trong `server/uploads`, `server/models`, `server/segment_outputs`. |
| Kubernetes | Chua co manifest Kubernetes/Helm/Ingress/Service/Deployment trong repo. |
| CI/CD | Chua thay GitHub Actions/GitLab CI/Jenkins config trong repo. |
| GitOps | Chua co Argo CD/Flux manifest. |
| Monitoring | Chua co Prometheus/Grafana/Alertmanager/Loki config. |
| Autoscaling | Chua co HPA/Karpenter/Cluster Autoscaler/Terraform/Ansible config. |

## A. User Traffic

### Hien Tai

| Hang muc | Gia tri hien tai |
| --- | --- |
| Domain/DNS | Chua cau hinh trong repo. |
| Public endpoint | Dev/local backend o `http://<host-ip>:8000`. Mobile dang default `http://192.168.1.3:8000`. |
| Load Balancer | Chua co. Docker Compose expose backend truc tiep `8000:8000`. |
| Gateway/Ingress | Chua co Kubernetes Ingress/Gateway. |
| NodePort | Chua co vi chua co Kubernetes Service. |

### De Xuat Production

| Hang muc | De xuat |
| --- | --- |
| Domain | `api.<domain>` cho backend API; neu co web frontend thi dung `<domain>` hoac `app.<domain>`. |
| DNS provider | Cloudflare neu can proxy/WAF/cache de cau hinh nhanh; Route 53 neu ha tang nam tren AWS va muon quan ly native. |
| Load Balancer | AWS ALB cho EKS; neu self-managed Kubernetes thi dung Nginx/HAProxy ben ngoai. |
| Ingress Controller | Nginx Ingress hoac Traefik. Voi API AI co upload file, Nginx Ingress de cau hinh body size/timeout ro rang. |
| Public ports | `80` redirect sang `443`, API chi phuc vu qua HTTPS. |
| NodePort fallback | Neu khong co cloud LoadBalancer, co the dat `30080` cho HTTP va `30443` cho HTTPS, nhung production nen di qua LB/Ingress. |

Routing de xuat:

| Host/Path | Service dich |
| --- | --- |
| `/api/health` | FastAPI backend `/health`. |
| `/api/detect-frame` | FastAPI backend `/detect-frame`. |
| `/api/segment-object` | FastAPI backend `/segment-object`. |
| `/api/reconstruct-object` | FastAPI backend `/reconstruct-object`. |
| `/api/reconstruct-image` | FastAPI backend `/reconstruct-image`. |
| `/models/*`, `/uploads/*`, `/segments/*` | Artifact static paths cua backend, hoac chuyen sang object storage/S3/CDN. |
| `/` | Web frontend neu build Expo Web/React web. Mobile native khong can route nay. |

## B. Kubernetes Runtime

### Hien Tai

Repo chua co Kubernetes runtime. Docker Compose hien co 2 service:

| Service | Image/Cong nghe | Port | Ghi chu |
| --- | --- | --- | --- |
| `backend` | Build tu `server/Dockerfile`, Python 3.10 + FastAPI + Uvicorn | `8000` | Xu ly YOLO, preprocessing, reconstruction. |
| `floci` | `floci/floci:latest` | `4566` | Storage/AWS local mock, chua thay backend su dung truc tiep trong code luu artifact. |

### De Xuat Namespace

| Namespace | Muc dich |
| --- | --- |
| `ai3d-app` | Backend API, optional web frontend, worker jobs. |
| `ai3d-ingress` | Nginx/Traefik Ingress Controller. |
| `ai3d-observability` | Prometheus, Grafana, Loki, Alertmanager. |
| `ai3d-data` | Neu chay MinIO/PostgreSQL/Redis trong cluster. Production nen uu tien managed service ben ngoai. |

### Deployment/Pods De Xuat

| Service | Cong nghe | Replica toi thieu | Tai nguyen de xuat | Ghi chu |
| --- | --- | --- | --- | --- |
| `ai3d-backend-api` | FastAPI/Uvicorn, Python 3.10, PyTorch, Ultralytics YOLO | `2` | CPU 2-4 core, RAM 4-8GB; GPU neu inference nang | Chay endpoint API, nen tach inference nang thanh worker neu traffic tang. |
| `ai3d-reconstruction-worker` | Python worker dung cung image backend | `1` | GPU optional, RAM cao | Xu ly job reconstruct bat dong bo, tranh block request upload. |
| `ai3d-web` | Expo Web/React static build neu can | `2` | Nho | Co the deploy len CDN thay vi Kubernetes. |
| `redis` | Redis queue | `1` dev, managed production | Nho | Chi can khi tach job queue. |
| `object-storage` | S3/MinIO | Managed production | Theo dung luong | Luu upload, PLY, preview PNG, metadata JSON. |

### Service/Routing Rules De Xuat

| Kubernetes Service | Port | Selector | Duong dan |
| --- | --- | --- | --- |
| `backend-api-svc` | `8000` | `app=ai3d-backend-api` | `/api/*`, `/models/*`, `/uploads/*`, `/segments/*`. |
| `web-svc` | `80` | `app=ai3d-web` | `/` neu co web frontend. |
| `redis-svc` | `6379` | `app=redis` | Chi noi bo namespace. |

### Bao Mat Noi Bo De Xuat

NetworkPolicy nen theo nguyen tac default deny:

| Rule | Y nghia |
| --- | --- |
| Ingress Controller -> Backend API | Cho phep traffic vao backend. |
| Backend API -> Redis/Object Storage | Cho phep backend tao job va luu artifact. |
| Worker -> Redis/Object Storage | Cho phep worker lay job va ghi output. |
| Monitoring -> Backend `/metrics` | Cho phep Prometheus scrape metric neu co. |
| Deny Pod khong lien quan | Chan truy cap ngang khong can thiet. |

Secrets can co:

| Secret | Noi dung |
| --- | --- |
| `registry-credentials` | Credential pull image neu dung private registry. |
| `object-storage-secret` | S3/MinIO endpoint, access key, secret key, bucket. |
| `model-config-secret` | Duong dan checkpoint production, model version neu can an. |
| `app-env-secret` | API token, CORS origin, webhook secret neu them auth/notification. |

ConfigMap nen co:

| ConfigMap | Noi dung |
| --- | --- |
| `backend-config` | `YOLO_DETECTION_CONFIDENCE`, `YOLO_DETECTION_IMAGE_SIZE`, `DETECTION_MAX_OBJECTS`, `RECON_IMAGE_SIZE`, timeout. |
| `routing-config` | Public base URL, artifact base URL. |

## C. Application Layer

### Backend Business Modules Hien Co

| Module | Endpoint/File | Chuc nang |
| --- | --- | --- |
| Health/Runtime status | `GET /health` | Kiem tra checkpoint, YOLO weights, config inference. |
| Object detection | `POST /detect-frame` | Nhan frame anh, chay YOLO segmentation/detection, tra bbox/mask/label/confidence. |
| Mask refinement | `SAM2_ENABLED=true` | SAM2 refine mask tu YOLO bbox truoc khi crop object, fallback ve YOLO neu khong bat buoc. |
| Object segmentation | `POST /segment-object` | Nhan anh + optional bbox, crop/mask object, luu artifact segment. |
| Object reconstruction | `POST /reconstruct-object` | Tu anh va bbox/mask, preprocess object, goi TripoSR, export GLB/OBJ/PNG/JSON. |
| Single image reconstruction | `POST /reconstruct-image` | Reconstruct truc tiep tu anh upload bang backend hien tai. |
| Scan workflow mock | `POST /upload-scan-video`, `GET /scan-status/{job_id}` | Contract mau cho video scan, hien chua co queue/worker thuc. |
| ML preprocessing | `project/src/preprocessing/*` | Lam sach Pix3D, crop/mask/square pad, sample mesh thanh point cloud. |
| Training | `project/src/training/training_pipeline.py` | Train ResNet/decoder sinh point cloud. |
| Evaluation | `project/src/evaluation/*` | Chamfer/F-score/per-sample/worst-case gallery. |
| Inference CLI | `project/src/inference/baseline_inference.py` | Load checkpoint, predict point cloud, save outputs. |

### Database/Storage

| Hang muc | Trang thai |
| --- | --- |
| SQL/NoSQL database | Chua co trong repo. Khong thay PostgreSQL/MySQL/MongoDB/SQL Server dependency hoac config. |
| Artifact storage | Hien luu local trong `server/uploads`, `server/models`, `server/segment_outputs`. |
| Dataset storage | Local folder `project/data/*`; raw data ignored by git. |
| Docker Compose storage mock | `floci` expose `4566`, env AWS mock co trong compose, nhung code backend hien van luu local. |

De xuat production:

| Nhu cau | De xuat |
| --- | --- |
| Metadata job/result/user/session | PostgreSQL managed ngoai cluster. |
| Queue job reconstruct | Redis/RQ hoac Celery + Redis. |
| File upload/output | S3/MinIO + CDN, khong luu lau trong container filesystem. |
| Model checkpoint | S3/EFS/model registry, mount read-only vao pod hoac download khi startup. |

## D. CI/CD + GitOps

### Hien Tai

| Hang muc | Trang thai |
| --- | --- |
| Source code | Local git repo. Remote GitHub/GitLab chua xac dinh trong tai lieu repo. |
| CI | Chua thay `.github/workflows`, `.gitlab-ci.yml`, Jenkinsfile. |
| Docker image | Co build local tu `server/Dockerfile`, chua co pipeline push registry. |
| Registry | Chua cau hinh. |
| Image tag rule | Chua cau hinh. |
| GitOps | Chua co Argo CD/Flux. |
| Kubernetes manifest repo | Chua co. |

### De Xuat CI/CD

GitHub Actions pipeline de xuat:

1. On pull request:
   - Install Python deps.
   - Run lint/type check neu bo sung.
   - Run smoke tests backend.
   - Run `python -m compileall`.
   - Optional: run small CPU smoke inference/eval.
2. On merge to `main`:
   - Build backend Docker image.
   - Tag image:
     - `ghcr.io/<org>/ai3d-backend:<git-sha>`
     - `ghcr.io/<org>/ai3d-backend:main`
     - optional semver tag khi release.
   - Push image to GHCR/ECR.
   - Update Kubernetes manifest image tag.
3. GitOps:
   - Argo CD watches `infra/k8s/overlays/prod`.
   - Sync Deployment khi image tag thay doi.
   - Rollback bang cach revert manifest commit.

Repo layout de xuat:

```text
infra/
  k8s/
    base/
      backend-deployment.yaml
      backend-service.yaml
      backend-configmap.yaml
      backend-hpa.yaml
      ingress.yaml
      networkpolicy.yaml
    overlays/
      dev/
      prod/
.github/
  workflows/
    backend-ci.yml
    backend-image.yml
```

## E. Monitoring & Autoscaling

### Hien Tai

| Hang muc | Trang thai |
| --- | --- |
| Cloud provider | Chua cau hinh. |
| Worker node type | Chua xac dinh. |
| Build server | Chua xac dinh. |
| Terraform/Ansible | Chua co. |
| Metrics/logging | Chua co Prometheus/Grafana/Loki/Datadog config. |
| Alerting | Chua co. |
| HPA/Autoscaling | Chua co. |

### De Xuat Monitoring

| Lop | Cong cu |
| --- | --- |
| Metrics | Prometheus + kube-state-metrics + node-exporter. |
| Dashboard | Grafana. |
| Logs | Loki + Promtail hoac CloudWatch Logs neu AWS. |
| Alert | Alertmanager gui Slack/Email. |
| App metrics | Them `/metrics` cho FastAPI bang `prometheus-fastapi-instrumentator`. |

Canh bao de xuat:

| Alert | Dieu kien |
| --- | --- |
| API high error rate | 5xx > 2-5% trong 5 phut. |
| API high latency | p95 `/reconstruct-*` qua nguong, vi du > 30s. |
| CPU high | CPU pod > 80% trong 10 phut. |
| RAM high | Memory > 85% request/limit. |
| Disk pressure | Node/container disk gan day. |
| Model missing | `/health` bao checkpoint/YOLO weights khong ton tai. |
| Queue backlog | So job cho > nguong neu tach worker queue. |
| GPU saturation | GPU utilization/memory cao neu dung GPU node. |

### De Xuat Autoscaling

Neu dung EKS/AWS:

| Thanh phan | De xuat |
| --- | --- |
| Worker node CPU | EC2 `m6i.large`/`m6i.xlarge` cho API nhe. |
| Worker node GPU | EC2 `g5.xlarge` hoac phu hop chi phi cho inference/training nhe. |
| Cluster autoscaler | Karpenter hoac Cluster Autoscaler. |
| IaC | Terraform tao VPC/EKS/node groups/ALB/IAM/S3/ECR. |
| Build | GitHub Actions hosted runner hoac self-hosted runner rieng. |

Scale up workflow:

1. HPA thay CPU/RAM/queue metric vuot nguong va tang replica cua `backend-api` hoac `reconstruction-worker`.
2. Neu cluster khong du resource, Karpenter/Cluster Autoscaler tao them EC2 worker node.
3. Node moi join cluster qua IAM/bootstrap cua EKS.
4. Pod duoc scheduler dat len node moi.
5. AWS Load Balancer Controller/Ingress tu cap nhat target group, khong can sua HAProxy thu cong neu dung ALB.

Scale down workflow:

1. HPA giam replica khi metric on dinh duoi nguong trong cooldown window.
2. Cluster Autoscaler/Karpenter chon node it tai.
3. Kubernetes drain pod khoi node theo `terminationGracePeriodSeconds`.
4. PodDisruptionBudget giu toi thieu replica de khong mat service.
5. Job dang xu ly can co co che idempotent/resume; artifact ghi vao S3/PostgreSQL/Redis, khong ghi doc nhat tren node local.
6. Node bi xoa sau khi khong con pod quan trong.

Neu self-managed Kubernetes:

| Viec | De xuat |
| --- | --- |
| Add node | Terraform/Ansible tao VM, cai container runtime/kubelet, `kubeadm join`. |
| Update LB | Script cap nhat HAProxy/Nginx upstream, reload zero-downtime. |
| Remove node | `kubectl cordon`, `kubectl drain --ignore-daemonsets --delete-emptydir-data`, xoa khoi HAProxy, terminate VM. |
| Bao ve data | Khong luu data customer tren node local; dung object storage/PV co reclaim policy ro rang. |

## Viec Can Bo Sung Vao Repo

| Muc | File/Folder can them |
| --- | --- |
| Kubernetes manifests | `infra/k8s/base/*.yaml`, `infra/k8s/overlays/dev`, `infra/k8s/overlays/prod`. |
| Docker image CI | `.github/workflows/backend-image.yml`. |
| Test CI | `.github/workflows/backend-ci.yml`. |
| Runtime config | `infra/k8s/base/backend-configmap.yaml`, secret template dung External Secrets/Sealed Secrets. |
| Observability | `infra/monitoring/` hoac Helm values cho kube-prometheus-stack/Loki. |
| Storage | S3/MinIO integration trong backend thay vi chi local filesystem. |
| Queue | Redis/Celery/RQ neu reconstruct can bat dong bo. |
| Auth/CORS | Gioi han CORS origin, them API auth neu expose public. |
