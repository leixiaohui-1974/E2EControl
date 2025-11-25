# E2EControl Docker 快速部署指南

## 快速开始

### 1. 环境要求

- Docker 20.10+
- Docker Compose 2.0+
- 至少 4GB 可用内存
- 至少 10GB 磁盘空间

### 2. 一键启动

```bash
# 克隆仓库
git clone <repository-url>
cd E2EControl

# 设置权限
chmod +x scripts/docker-start.sh

# 启动服务
./scripts/docker-start.sh start
```

### 3. 访问服务

| 服务 | 地址 | 说明 |
|------|------|------|
| Web仪表板 | http://localhost:8080 | 实时监控界面 |
| API服务 | http://localhost:8000 | REST API |
| Grafana | http://localhost:3000 | 可视化面板 (需启用监控) |
| Prometheus | http://localhost:9090 | 指标采集 (需启用监控) |

---

## 部署模式

### 基础模式

仅启动核心控制服务：

```bash
./scripts/docker-start.sh start
```

包含：
- controller: 主控制器
- dashboard: Web仪表板
- api: API服务
- db: PostgreSQL数据库
- redis: Redis缓存

### 完整监控模式

包含Prometheus和Grafana：

```bash
./scripts/docker-start.sh start monitoring
```

### 认证测试模式

运行完整场景认证：

```bash
./scripts/docker-start.sh certify
```

报告输出到 `reports/` 目录。

---

## 常用命令

```bash
# 启动服务
./scripts/docker-start.sh start

# 停止服务
./scripts/docker-start.sh stop

# 重启服务
./scripts/docker-start.sh restart

# 查看状态
./scripts/docker-start.sh status

# 查看日志
./scripts/docker-start.sh logs              # 所有服务
./scripts/docker-start.sh logs controller   # 指定服务

# 重新构建镜像
./scripts/docker-start.sh build

# 运行认证测试
./scripts/docker-start.sh certify

# 清理所有数据
./scripts/docker-start.sh clean
```

---

## 配置说明

### 环境变量

复制模板并修改：

```bash
cp docker/.env.example .env
```

主要配置项：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `E2E_NUM_POOLS` | 3 | 渠池数量 |
| `E2E_ENABLE_SELF_HEALING` | true | 启用自愈系统 |
| `E2E_WEB_PORT` | 8080 | Web仪表板端口 |
| `E2E_API_PORT` | 8000 | API服务端口 |
| `POSTGRES_PASSWORD` | - | 数据库密码 (建议修改) |

### 数据持久化

数据存储在Docker卷中：
- `postgres_data`: 数据库数据
- `redis_data`: Redis数据
- `prometheus_data`: 监控数据
- `grafana_data`: Grafana配置

本地目录：
- `./logs`: 日志文件
- `./data`: 运行数据
- `./reports`: 报告文件

---

## 架构说明

```
┌─────────────────────────────────────────────────────────┐
│                      用户访问层                          │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│    │ Dashboard   │  │   API       │  │  Grafana    │   │
│    │  :8080      │  │  :8000      │  │  :3000      │   │
│    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘   │
└───────────┼────────────────┼────────────────┼──────────┘
            │                │                │
┌───────────▼────────────────▼────────────────▼──────────┐
│                     控制服务层                          │
│         ┌─────────────────────────────────┐            │
│         │         Controller              │            │
│         │   • Hierarchical MPC            │            │
│         │   • Self-Healing                │            │
│         │   • Performance Monitor         │            │
│         └──────────────┬──────────────────┘            │
└────────────────────────┼───────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────┐
│                      数据层                             │
│    ┌─────────────┐              ┌─────────────┐        │
│    │ PostgreSQL  │              │    Redis    │        │
│    │  :5432      │              │   :6379     │        │
│    └─────────────┘              └─────────────┘        │
└────────────────────────────────────────────────────────┘
```

---

## 故障排除

### 容器启动失败

```bash
# 查看详细日志
docker compose logs controller

# 检查容器状态
docker compose ps -a
```

### 数据库连接失败

确保数据库容器已启动：
```bash
docker compose ps db
docker compose logs db
```

### 端口占用

修改 `.env` 中的端口配置或停止占用端口的服务：
```bash
lsof -i :8080
```

### 内存不足

调整Docker资源限制或减少服务：
```yaml
# docker-compose.yml
services:
  controller:
    deploy:
      resources:
        limits:
          memory: 1G
```

---

## 生产部署建议

1. **安全配置**
   - 修改所有默认密码
   - 启用 HTTPS
   - 配置防火墙

2. **高可用**
   - 使用外部数据库集群
   - 配置负载均衡
   - 设置健康检查

3. **监控告警**
   - 启用 Prometheus + Grafana
   - 配置告警规则
   - 设置日志收集

4. **备份**
   - 定期备份数据库
   - 导出配置文件
