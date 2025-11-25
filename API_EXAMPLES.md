# 🌐 API使用示例

## 启动API服务器

### 方法1: 使用配置文件

编辑 `config.yaml`:
```yaml
api:
  enabled: true
  host: "0.0.0.0"
  port: 5000
  debug: false
```

然后运行:
```bash
python3 api.py
```

### 方法2: 直接启动

```bash
# 启动开发服务器
python3 api.py

# 或使用 Flask 命令
export FLASK_APP=api.py
flask run --host=0.0.0.0 --port=5000
```

服务器将在 `http://localhost:5000` 启动

---

## API端点示例

### 1. 获取API信息

```bash
curl http://localhost:5000/
```

**响应**:
```json
{
  "name": "Smart Pool Agent API",
  "version": "2.0",
  "description": "智能闸门控制系统REST API",
  "endpoints": { ... }
}
```

### 2. 健康检查

```bash
curl http://localhost:5000/health
```

**响应**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-24T10:00:00",
  "version": "2.0"
}
```

### 3. 获取配置信息

```bash
curl http://localhost:5000/config
```

**响应**:
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

### 4. 解释自然语言指令

```bash
curl -X POST http://localhost:5000/interpret \
  -H "Content-Type: application/json" \
  -d '{"instruction": "收到暴雨预警，立刻降低水位！"}'
```

**响应**:
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
  "scenario": "模糊匹配"
}
```

### 5. 获取所有场景定义

```bash
curl http://localhost:5000/scenarios
```

**响应**:
```json
{
  "success": true,
  "count": 5,
  "scenarios": [
    {
      "name": "正常供水",
      "keywords": ["保持水位平稳", "正常供水"],
      "config": { ... }
    },
    ...
  ]
}
```

### 6. 运行仿真（同步）

```bash
curl -X POST http://localhost:5000/simulation/run \
  -H "Content-Type: application/json" \
  -d '{
    "script": [
      [0, "保持水位平稳，正常供水。"],
      [10, "收到暴雨预警，立刻降低水位！"],
      [20, "恢复正常供水。"]
    ]
  }'
```

**响应**:
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

### 7. 运行仿真（异步）

```bash
curl -X POST http://localhost:5000/simulation/run \
  -H "Content-Type: application/json" \
  -d '{
    "script": [
      [0, "保持水位平稳，正常供水。"]
    ],
    "async": true
  }'
```

**响应**:
```json
{
  "success": true,
  "simulation_id": 4,
  "status": "running",
  "message": "仿真已启动，使用 /simulation/{id}/status 查询状态"
}
```

### 8. 查询仿真状态

```bash
curl http://localhost:5000/simulation/4/status
```

**响应**:
```json
{
  "success": true,
  "simulation_id": 4,
  "status": "running",  // 或 "completed", "failed"
  "start_time": "2025-11-24T10:00:00",
  "end_time": null
}
```

### 9. 获取仿真历史数据

```bash
curl http://localhost:5000/simulation/3/history
```

**响应**:
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
    },
    ...
  ]
}
```

### 10. 获取仿真告警

```bash
curl http://localhost:5000/simulation/3/alerts
```

**响应**:
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
    },
    ...
  ]
}
```

### 11. 获取仿真报告

```bash
curl http://localhost:5000/simulation/3/report
```

**响应**:
```json
{
  "success": true,
  "simulation_id": 3,
  "report": "# 智能闸门仿真报告\n\n..."
}
```

### 12. 获取仿真图表

```bash
# 在浏览器中打开
open http://localhost:5000/simulation/3/result.png

# 或下载
curl http://localhost:5000/simulation/3/result.png -o result.png
```

### 13. 获取所有仿真列表

```bash
curl http://localhost:5000/simulations
```

**响应**:
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
    },
    ...
  ]
}
```

---

## Python客户端示例

### 基础使用

```python
import requests

BASE_URL = "http://localhost:5000"

# 1. 健康检查
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. 解释指令
data = {"instruction": "收到暴雨预警"}
response = requests.post(f"{BASE_URL}/interpret", json=data)
result = response.json()
print(f"置信度: {result['confidence']:.2f}")
print(f"目标水位: {result['config']['Z_ref']}m")

# 3. 运行仿真（异步）
script = [
    [0, "保持水位平稳，正常供水。"],
    [10, "收到暴雨预警，立刻降低水位！"]
]
response = requests.post(
    f"{BASE_URL}/simulation/run",
    json={"script": script, "async": True}
)
sim_id = response.json()['simulation_id']

# 4. 轮询状态
import time
while True:
    response = requests.get(f"{BASE_URL}/simulation/{sim_id}/status")
    status = response.json()['status']
    print(f"状态: {status}")
    
    if status != 'running':
        break
    time.sleep(2)

# 5. 获取结果
response = requests.get(f"{BASE_URL}/simulation/{sim_id}/history")
history = response.json()['history']
print(f"历史记录数: {len(history)}")

# 6. 下载图表
response = requests.get(f"{BASE_URL}/simulation/{sim_id}/result.png")
with open(f'result_{sim_id}.png', 'wb') as f:
    f.write(response.content)
```

### 完整客户端类

```python
class SmartPoolClient:
    """智能闸门API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
    
    def health_check(self):
        """健康检查"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def interpret(self, instruction: str):
        """解释指令"""
        response = requests.post(
            f"{self.base_url}/interpret",
            json={"instruction": instruction}
        )
        return response.json()
    
    def run_simulation(self, script, async_mode=False):
        """运行仿真"""
        response = requests.post(
            f"{self.base_url}/simulation/run",
            json={"script": script, "async": async_mode}
        )
        return response.json()
    
    def get_status(self, sim_id: int):
        """获取仿真状态"""
        response = requests.get(f"{self.base_url}/simulation/{sim_id}/status")
        return response.json()
    
    def get_history(self, sim_id: int):
        """获取历史数据"""
        response = requests.get(f"{self.base_url}/simulation/{sim_id}/history")
        return response.json()
    
    def wait_for_completion(self, sim_id: int, interval: int = 2):
        """等待仿真完成"""
        while True:
            status = self.get_status(sim_id)
            if status['status'] != 'running':
                return status
            time.sleep(interval)

# 使用示例
client = SmartPoolClient()

# 检查服务
print(client.health_check())

# 运行仿真
result = client.run_simulation(script, async_mode=True)
sim_id = result['simulation_id']

# 等待完成
final_status = client.wait_for_completion(sim_id)
print(f"仿真完成: {final_status}")

# 获取数据
history = client.get_history(sim_id)
print(f"数据点数: {len(history['history'])}")
```

---

## JavaScript/Node.js示例

```javascript
const axios = require('axios');

const BASE_URL = 'http://localhost:5000';

// 解释指令
async function interpretInstruction(instruction) {
  const response = await axios.post(`${BASE_URL}/interpret`, {
    instruction: instruction
  });
  return response.data;
}

// 运行仿真
async function runSimulation(script, asyncMode = false) {
  const response = await axios.post(`${BASE_URL}/simulation/run`, {
    script: script,
    async: asyncMode
  });
  return response.data;
}

// 使用示例
(async () => {
  // 解释指令
  const result = await interpretInstruction('收到暴雨预警');
  console.log(`置信度: ${result.confidence}`);
  
  // 运行仿真
  const script = [
    [0, "保持水位平稳，正常供水。"],
    [10, "收到暴雨预警，立刻降低水位！"]
  ];
  
  const simResult = await runSimulation(script, true);
  console.log(`仿真ID: ${simResult.simulation_id}`);
  
  // 轮询状态
  const simId = simResult.simulation_id;
  while (true) {
    const statusRes = await axios.get(`${BASE_URL}/simulation/${simId}/status`);
    const status = statusRes.data.status;
    console.log(`状态: ${status}`);
    
    if (status !== 'running') break;
    
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  
  console.log('仿真完成！');
})();
```

---

## curl脚本示例

创建 `test_api.sh`:

```bash
#!/bin/bash

BASE_URL="http://localhost:5000"

echo "=== 测试API ==="

# 1. 健康检查
echo -e "\n1. 健康检查"
curl -s $BASE_URL/health | jq

# 2. 解释指令
echo -e "\n2. 解释指令"
curl -s -X POST $BASE_URL/interpret \
  -H "Content-Type: application/json" \
  -d '{"instruction": "收到暴雨预警"}' | jq

# 3. 获取场景
echo -e "\n3. 获取场景"
curl -s $BASE_URL/scenarios | jq '.scenarios[] | .name'

# 4. 获取仿真列表
echo -e "\n4. 获取仿真列表"
curl -s $BASE_URL/simulations | jq

echo -e "\n=== 测试完成 ==="
```

运行:
```bash
chmod +x test_api.sh
./test_api.sh
```

---

## 常见问题

### Q: API服务启动失败？

**A**: 检查端口是否被占用：
```bash
# 检查5000端口
lsof -i :5000

# 或使用其他端口
python3 api.py --port 8000
```

### Q: 跨域问题？

**A**: API已启用CORS，允许所有来源访问。如需限制：

```python
# 在 api.py 中修改
CORS(app, origins=['http://example.com'])
```

### Q: 如何保护API？

**A**: 添加认证中间件：

```python
from functools import wraps
from flask import request

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your-secret-key':
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/simulation/run', methods=['POST'])
@require_api_key
def run_simulation():
    # ...
```

### Q: 性能优化？

**A**: 
1. 使用生产级WSGI服务器（Gunicorn/uWSGI）
2. 启用缓存
3. 使用异步任务队列（Celery）
4. 数据库连接池

---

## 部署建议

### 使用Gunicorn

```bash
pip install gunicorn

# 启动
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

### 使用Docker

创建 `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "api:app"]
```

构建和运行:
```bash
docker build -t smart-pool-api .
docker run -p 5000:5000 smart-pool-api
```

---

**更多信息请参考**: `README.md`
