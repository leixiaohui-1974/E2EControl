# API参考文档

## 概述

E2EControl提供基于Flask的REST API服务，支持仿真控制、指令解析、状态查询等功能。

**基础URL：** `http://localhost:5000`

**启动方式：**

```bash
# 开发模式
hydroe2e-api

# 生产模式
gunicorn -w 4 -b 0.0.0.0:5000 "hydroe2e.api:app"
```

## 端点列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 获取API信息 |
| GET | `/health` | 健康检查 |
| GET | `/config` | 获取系统配置 |
| POST | `/interpret` | 解析自然语言指令 |
| GET | `/scenarios` | 获取所有场景定义 |
| POST | `/simulation/run` | 运行仿真（同步/异步） |
| GET | `/simulation/{id}/status` | 查询仿真状态 |
| GET | `/simulation/{id}/history` | 获取仿真历史数据 |
| GET | `/simulation/{id}/alerts` | 获取仿真告警 |
| GET | `/simulation/{id}/report` | 获取仿真报告 |
| GET | `/simulation/{id}/result.png` | 获取仿真图表 |
| GET | `/simulations` | 获取所有仿真列表 |

---

## 端点详情

### 1. 获取API信息

```
GET /
```

**响应示例：**

```json
{
  "name": "Smart Pool Agent API",
  "version": "2.0",
  "description": "智能闸门控制系统REST API",
  "endpoints": { ... }
}
```

### 2. 健康检查

```
GET /health
```

**响应示例：**

```json
{
  "status": "healthy",
  "timestamp": "2025-11-24T10:00:00",
  "version": "2.0"
}
```

### 3. 获取系统配置

```
GET /config
```

**响应示例：**

```json
{
  "success": true,
  "config": {
    "simulation": {
      "total_hours": 50,
      "dt": 3600.0,
      "area": 10000.0
    },
    "mpc": { ... },
    "monitoring": { ... }
  }
}
```

### 4. 解析自然语言指令

```
POST /interpret
Content-Type: application/json
```

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| instruction | string | 是 | 中文自然语言指令 |

**请求示例：**

```json
{
  "instruction": "收到暴雨预警，立刻降低水位！"
}
```

**响应示例：**

```json
{
  "success": true,
  "instruction": "收到暴雨预警，立刻降低水位！",
  "confidence": 0.79,
  "config": {
    "W_level": 100.0,
    "W_smooth": 5.0,
    "Z_ref": 2.0,
    "delta_Q_max": 5.0
  },
  "scenario": "防洪调度"
}
```

### 5. 获取所有场景定义

```
GET /scenarios
```

**响应示例：**

```json
{
  "success": true,
  "count": 5,
  "scenarios": [
    {
      "name": "正常供水",
      "keywords": ["保持水位平稳", "正常供水"],
      "config": { ... }
    }
  ]
}
```

### 6. 运行仿真

```
POST /simulation/run
Content-Type: application/json
```

**请求参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| script | array | 是 | 仿真脚本，格式为 `[[时刻, 指令], ...]` |
| async | boolean | 否 | 是否异步运行，默认 `false` |

**同步请求示例：**

```json
{
  "script": [
    [0, "保持水位平稳，正常供水。"],
    [10, "收到暴雨预警，立刻降低水位！"],
    [20, "恢复正常供水。"]
  ]
}
```

**同步响应示例：**

```json
{
  "success": true,
  "simulation_id": 3,
  "status": "completed",
  "results": {
    "total_hours": 50,
    "final_level": 2.95,
    "alerts_count": 0
  }
}
```

**异步请求示例：**

```json
{
  "script": [
    [0, "保持水位平稳，正常供水。"]
  ],
  "async": true
}
```

**异步响应示例：**

```json
{
  "success": true,
  "simulation_id": 4,
  "status": "running",
  "message": "仿真已启动，使用 /simulation/{id}/status 查询状态"
}
```

### 7. 查询仿真状态

```
GET /simulation/{id}/status
```

**路径参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| id | int | 仿真ID |

**响应示例：**

```json
{
  "success": true,
  "simulation_id": 4,
  "status": "completed",
  "start_time": "2025-11-24T10:00:00",
  "end_time": "2025-11-24T10:01:30"
}
```

`status` 可能的值：`running`、`completed`、`failed`

### 8. 获取仿真历史数据

```
GET /simulation/{id}/history
```

**响应示例：**

```json
{
  "success": true,
  "simulation_id": 3,
  "count": 50,
  "history": [
    {
      "time_step": 0,
      "level": 3.0,
      "q_in": 5.0,
      "q_out": 5.0,
      "target_level": 3.0,
      "instruction": "保持水位平稳，正常供水。"
    }
  ]
}
```

### 9. 获取仿真告警

```
GET /simulation/{id}/alerts
```

**响应示例：**

```json
{
  "success": true,
  "simulation_id": 3,
  "count": 2,
  "alerts": [
    {
      "time_step": 15,
      "level": "WARNING",
      "alert_type": "水位偏高",
      "message": "当前水位 8.5m 超过警戒线",
      "data": { ... }
    }
  ]
}
```

### 10. 获取仿真报告

```
GET /simulation/{id}/report
```

**响应示例：**

```json
{
  "success": true,
  "simulation_id": 3,
  "report": "# 智能闸门仿真报告\n\n..."
}
```

### 11. 获取仿真图表

```
GET /simulation/{id}/result.png
```

返回PNG格式的仿真结果图表，可在浏览器中直接查看或下载。

### 12. 获取所有仿真列表

```
GET /simulations
```

**响应示例：**

```json
{
  "success": true,
  "count": 3,
  "simulations": [
    {
      "id": 1,
      "status": "completed",
      "start_time": "2025-11-24T09:00:00",
      "end_time": "2025-11-24T09:01:30",
      "total_hours": 50,
      "source": "database"
    }
  ]
}
```

---

## 错误处理

所有端点在出错时返回统一格式：

```json
{
  "success": false,
  "error": "错误描述信息"
}
```

常见HTTP状态码：

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

---

## Python客户端示例

```python
import requests
import time

BASE_URL = "http://localhost:5000"

# 健康检查
resp = requests.get(f"{BASE_URL}/health")
print(resp.json())

# 解释指令
resp = requests.post(f"{BASE_URL}/interpret",
    json={"instruction": "收到暴雨预警"})
result = resp.json()
print(f"置信度: {result['confidence']:.2f}")

# 异步运行仿真
script = [
    [0, "保持水位平稳，正常供水。"],
    [10, "收到暴雨预警，立刻降低水位！"]
]
resp = requests.post(f"{BASE_URL}/simulation/run",
    json={"script": script, "async": True})
sim_id = resp.json()["simulation_id"]

# 轮询状态
while True:
    resp = requests.get(f"{BASE_URL}/simulation/{sim_id}/status")
    status = resp.json()["status"]
    if status != "running":
        break
    time.sleep(2)

# 获取历史数据
resp = requests.get(f"{BASE_URL}/simulation/{sim_id}/history")
history = resp.json()["history"]
print(f"数据点数: {len(history)}")
```

---

## 安全配置

### API认证

在生产环境中建议添加API Key认证：

```python
# 请求时携带API Key
headers = {"X-API-Key": "your-secret-key"}
requests.get(f"{BASE_URL}/health", headers=headers)
```

### CORS配置

API默认启用CORS，允许所有来源。生产环境中可在 `api.py` 中限制来源域名。

### 速率限制

建议在生产环境中配置速率限制，防止API被滥用。可使用 `flask-limiter` 库实现。
