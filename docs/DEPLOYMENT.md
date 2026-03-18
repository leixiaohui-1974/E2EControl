# 部署指南

## 本地开发环境

### 快速启动

```bash
cd e2econtrol

# 安装依赖
pip install -r requirements.txt

# 运行快速测试
python quick_test.sh

# 启动API服务器
hydroe2e-api
```

## Docker部署

### 使用Dockerfile

项目根目录已包含 `Dockerfile`：

```bash
# 构建镜像
docker build -t e2econtrol:v1.0 .

# 运行API服务
docker run -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  e2econtrol:v1.0

# 运行仿真（一次性）
docker run -v $(pwd)/results:/app \
  e2econtrol:v1.0 \
  python -m hydroe2e.main
```

### 使用Docker Compose

项目根目录已包含 `docker-compose.yml`，一键启动所有服务：

```bash
# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

Docker Compose配置包含两个服务：
- **api**：REST API服务，端口5000
- **simulator**：仿真计算服务

## 生产环境部署

### 使用Gunicorn（推荐）

```bash
pip install gunicorn

# 启动（4个worker进程）
gunicorn -w 4 \
  -b 0.0.0.0:5000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  "hydroe2e.api:app"
```

### 配置Systemd服务

创建 `/etc/systemd/system/e2econtrol.service`：

```ini
[Unit]
Description=E2EControl API Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/e2econtrol
Environment="PATH=/opt/e2econtrol/venv/bin"
ExecStart=/opt/e2econtrol/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 "hydroe2e.api:app"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable e2econtrol
sudo systemctl start e2econtrol
sudo systemctl status e2econtrol
```

### 配置Nginx反向代理

创建 `/etc/nginx/sites-available/e2econtrol`：

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
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/e2econtrol /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Kubernetes部署

### Deployment配置

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: e2econtrol-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: e2econtrol-api
  template:
    metadata:
      labels:
        app: e2econtrol-api
    spec:
      containers:
      - name: api
        image: e2econtrol:v1.0
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

### Service配置

```yaml
apiVersion: v1
kind: Service
metadata:
  name: e2econtrol-api
spec:
  selector:
    app: e2econtrol-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl get svc
```

## 云平台部署

### AWS EC2

```bash
# SSH连接EC2实例
ssh -i key.pem ubuntu@ec2-instance

# 安装依赖
sudo apt update
sudo apt install python3-pip python3-venv nginx -y

# 部署项目
git clone <repo> /opt/e2econtrol
cd /opt/e2econtrol
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 配置systemd和nginx（见上文）
```

### 阿里云/腾讯云

部署方式与AWS EC2类似，使用对应云平台的ECS实例即可。也可使用各平台的容器服务直接部署Docker镜像。

## 性能优化

### 数据库优化

编辑 `config.yaml`：

```yaml
database:
  enabled: true
  path: "simulation_data.db"
  journal_mode: WAL
  synchronous: NORMAL
  cache_size: 10000
```

### 异步任务队列

对于长时间运行的仿真任务，建议使用Celery + Redis：

```bash
pip install celery redis
```

### 负载均衡

```bash
gunicorn -w 8 -k gevent --worker-connections 1000 "hydroe2e.api:app"
```

## 监控

### 健康检查脚本

```bash
#!/bin/bash
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health)
if [ $response -eq 200 ]; then
    echo "API正常"
else
    echo "API异常 (HTTP $response)"
    sudo systemctl restart e2econtrol
fi
```

### 日志管理

- API访问日志：`logs/access.log`
- 错误日志：`logs/error.log`
- 系统日志：`smart_pool.log`

可接入ELK Stack或Prometheus进行集中监控。

## 安全加固

### HTTPS配置

```bash
# 使用Let's Encrypt免费证书
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### API认证

在请求头中携带API Key：

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:5000/health
```

### 速率限制

建议使用 `flask-limiter` 配置每分钟/每小时请求限额。

## 备份策略

```bash
#!/bin/bash
# 每日凌晨2点备份数据库
BACKUP_DIR="/backup"
DB_PATH="/opt/e2econtrol/simulation_data.db"
DATE=$(date +%Y%m%d_%H%M%S)

cp $DB_PATH $BACKUP_DIR/simulation_data_$DATE.db
gzip $BACKUP_DIR/simulation_data_$DATE.db
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete
```

定时任务：

```bash
# crontab -e
0 2 * * * /opt/e2econtrol/scripts/backup.sh
```
