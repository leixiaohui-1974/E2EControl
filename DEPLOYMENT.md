# 🚀 部署指南

## 本地开发环境

### 快速开始

```bash
# 1. 克隆/进入项目目录
cd /workspace

# 2. 安装依赖
pip3 install -r requirements.txt

# 3. 运行快速测试
chmod +x quick_test.sh
./quick_test.sh

# 4. 运行交互式演示
python3 demo.py

# 5. 运行完整仿真
python3 main_enhanced.py

# 6. 启动API服务器
python3 api.py
```

---

## Docker部署

### 方法1: 使用Dockerfile

创建 `Dockerfile`:

```dockerfile
FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 暴露API端口
EXPOSE 5000

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 默认命令（可选择运行API或仿真）
CMD ["python3", "api.py"]
```

构建和运行:

```bash
# 构建镜像
docker build -t smart-pool-agent:v2.0 .

# 运行API服务器
docker run -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  smart-pool-agent:v2.0

# 运行仿真
docker run -v $(pwd)/results:/app \
  smart-pool-agent:v2.0 \
  python3 main_enhanced.py
```

### 方法2: 使用docker-compose

创建 `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
    command: python3 api.py
    restart: unless-stopped
  
  simulator:
    build: .
    volumes:
      - ./results:/app/results
    command: python3 main_enhanced.py
    depends_on:
      - api
```

运行:

```bash
docker-compose up -d
docker-compose logs -f
```

---

## 生产环境部署

### 使用Gunicorn（推荐）

```bash
# 安装Gunicorn
pip3 install gunicorn

# 启动API服务器
gunicorn -w 4 \
  -b 0.0.0.0:5000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  api:app
```

### 使用Systemd服务

创建 `/etc/systemd/system/smart-pool-api.service`:

```ini
[Unit]
Description=Smart Pool Agent API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/smart-pool-agent
Environment="PATH=/opt/smart-pool-agent/venv/bin"
ExecStart=/opt/smart-pool-agent/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 api:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

管理服务:

```bash
# 启用服务
sudo systemctl enable smart-pool-api

# 启动服务
sudo systemctl start smart-pool-api

# 查看状态
sudo systemctl status smart-pool-api

# 查看日志
sudo journalctl -u smart-pool-api -f
```

### 使用Nginx反向代理

创建 `/etc/nginx/sites-available/smart-pool`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时设置（长时间运行的仿真）
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # 静态文件（可选）
    location /static {
        alias /opt/smart-pool-agent/static;
        expires 30d;
    }
}
```

启用站点:

```bash
sudo ln -s /etc/nginx/sites-available/smart-pool \
            /etc/nginx/sites-enabled/

sudo nginx -t
sudo systemctl reload nginx
```

---

## 云平台部署

### AWS部署

#### 使用EC2

```bash
# 1. 启动EC2实例（Ubuntu 20.04 LTS）

# 2. SSH连接
ssh -i key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com

# 3. 安装依赖
sudo apt update
sudo apt install python3-pip python3-venv nginx -y

# 4. 克隆项目
git clone <repo> /opt/smart-pool-agent
cd /opt/smart-pool-agent

# 5. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 6. 安装依赖
pip install -r requirements.txt

# 7. 配置systemd和nginx（见上文）

# 8. 配置安全组
# 允许入站: 80 (HTTP), 443 (HTTPS), 5000 (API)
```

#### 使用ECS (Docker)

1. 推送镜像到ECR:

```bash
# 登录ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com

# 构建并推送
docker build -t smart-pool-agent .
docker tag smart-pool-agent:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/smart-pool-agent:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/smart-pool-agent:latest
```

2. 创建ECS任务定义和服务

### Google Cloud Platform

#### App Engine

创建 `app.yaml`:

```yaml
runtime: python39
entrypoint: gunicorn -b :$PORT api:app

instance_class: F2

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.65

env_variables:
  FLASK_ENV: 'production'
```

部署:

```bash
gcloud app deploy
```

#### Cloud Run

```bash
# 构建镜像
gcloud builds submit --tag gcr.io/PROJECT-ID/smart-pool-agent

# 部署
gcloud run deploy smart-pool-agent \
  --image gcr.io/PROJECT-ID/smart-pool-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 5000
```

### Azure

#### App Service

```bash
# 创建资源组
az group create --name smart-pool-rg --location eastus

# 创建App Service计划
az appservice plan create \
  --name smart-pool-plan \
  --resource-group smart-pool-rg \
  --sku B1 \
  --is-linux

# 创建Web App
az webapp create \
  --resource-group smart-pool-rg \
  --plan smart-pool-plan \
  --name smart-pool-app \
  --runtime "PYTHON|3.9"

# 部署代码
az webapp up \
  --runtime PYTHON:3.9 \
  --sku B1 \
  --resource-group smart-pool-rg
```

---

## Kubernetes部署

### 创建Deployment

`k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smart-pool-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: smart-pool-api
  template:
    metadata:
      labels:
        app: smart-pool-api
    spec:
      containers:
      - name: api
        image: smart-pool-agent:v2.0
        ports:
        - containerPort: 5000
        env:
        - name: FLASK_ENV
          value: "production"
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
          requests:
            memory: "256Mi"
            cpu: "250m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 创建Service

`k8s/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: smart-pool-api
spec:
  selector:
    app: smart-pool-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

部署:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 查看状态
kubectl get pods
kubectl get svc

# 查看日志
kubectl logs -f deployment/smart-pool-api
```

---

## 性能优化

### 1. 数据库优化

```yaml
# config.yaml
database:
  enabled: true
  path: "simulation_data.db"
  
  # 优化选项
  journal_mode: WAL  # Write-Ahead Logging
  synchronous: NORMAL
  cache_size: 10000
```

### 2. API缓存

```python
# 安装Redis
pip install redis flask-caching

# api.py
from flask_caching import Cache

cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0'
})

@app.route('/scenarios')
@cache.cached(timeout=300)  # 缓存5分钟
def list_scenarios():
    # ...
```

### 3. 异步任务队列

```python
# 使用Celery处理长时间仿真
from celery import Celery

celery = Celery('tasks', broker='redis://localhost:6379/0')

@celery.task
def run_simulation_async(script):
    sim = SmartPoolSimulation()
    sim.run(script=script)
    return sim.simulation_id
```

### 4. 负载均衡

使用多个worker:

```bash
gunicorn -w 8 -k gevent --worker-connections 1000 api:app
```

---

## 监控和日志

### 使用Prometheus

```python
# 安装
pip install prometheus-flask-exporter

# api.py
from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app)
```

### 使用ELK Stack

1. Logstash配置:

```conf
input {
  file {
    path => "/var/log/smart-pool/*.log"
    type => "smart-pool"
  }
}

filter {
  json {
    source => "message"
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "smart-pool-%{+YYYY.MM.dd}"
  }
}
```

---

## 安全加固

### 1. API认证

```python
from functools import wraps
from flask import request

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.getenv('API_KEY'):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/simulation/run', methods=['POST'])
@require_api_key
def run_simulation():
    # ...
```

### 2. HTTPS配置

使用Let's Encrypt:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 3. 速率限制

```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per hour"]
)

@app.route('/simulation/run', methods=['POST'])
@limiter.limit("10 per minute")
def run_simulation():
    # ...
```

---

## 备份策略

### 数据库备份

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup"
DB_PATH="/app/simulation_data.db"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份
cp $DB_PATH $BACKUP_DIR/simulation_data_$DATE.db

# 压缩
gzip $BACKUP_DIR/simulation_data_$DATE.db

# 删除7天前的备份
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete
```

设置定时任务:

```bash
# crontab -e
0 2 * * * /opt/smart-pool-agent/backup.sh
```

---

## 故障排查

### 常见问题

#### 1. API服务无法启动

```bash
# 检查端口占用
lsof -i :5000

# 检查日志
tail -f smart_pool.log

# 检查权限
ls -la config.yaml
```

#### 2. 数据库锁定

```python
# config.yaml
database:
  journal_mode: WAL
  timeout: 30
```

#### 3. 内存不足

```bash
# 监控内存使用
python3 -c "
from monitor import MonitoringSystem
import psutil
print(f'内存使用: {psutil.virtual_memory().percent}%')
"

# 限制仿真时长
# config.yaml
simulation:
  total_hours: 24  # 减少时长
```

---

## 健康检查脚本

```bash
#!/bin/bash
# health_check.sh

API_URL="http://localhost:5000"

# 检查API健康
response=$(curl -s -o /dev/null -w "%{http_code}" $API_URL/health)

if [ $response -eq 200 ]; then
    echo "✓ API正常"
    exit 0
else
    echo "✗ API异常 (HTTP $response)"
    
    # 尝试重启服务
    sudo systemctl restart smart-pool-api
    exit 1
fi
```

---

## 性能测试

```bash
# 使用Apache Bench
ab -n 1000 -c 10 http://localhost:5000/health

# 使用wrk
wrk -t4 -c100 -d30s http://localhost:5000/scenarios
```

---

## 总结

本部署指南涵盖了:

✅ 本地开发环境  
✅ Docker容器化  
✅ 生产环境部署  
✅ 云平台部署（AWS/GCP/Azure）  
✅ Kubernetes编排  
✅ 性能优化  
✅ 监控和日志  
✅ 安全加固  
✅ 备份策略  
✅ 故障排查  

根据实际需求选择合适的部署方案。
