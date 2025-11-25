# 📖 使用指南

## 快速上手（5分钟）

### 第一步：验证环境

```bash
# 运行快速测试脚本
chmod +x quick_test.sh
./quick_test.sh
```

如果看到 `✓ 所有测试通过！系统正常！`，说明环境配置正确。

### 第二步：选择使用方式

根据您的需求选择：

1. **命令行仿真** → 运行 `python3 main_enhanced.py`
2. **API服务** → 运行 `python3 api.py`
3. **交互式演示** → 运行 `python3 demo.py`
4. **性能测试** → 运行 `python3 benchmark.py`

---

## 📊 方式1：命令行仿真

### 运行默认仿真

```bash
python3 main_enhanced.py
```

**生成文件**:
- `simulation_result_enhanced.png` - 结果图表
- `simulation_enhanced.gif` - 动画演示
- `simulation_report_enhanced.md` - 详细报告
- `simulation_data.db` - 数据库（可选）
- `smart_pool.log` - 运行日志

### 自定义仿真

编辑 `config.yaml`:

```yaml
simulation:
  total_hours: 100    # 改为100小时
  
mpc:
  horizon: 5         # 更快求解

database:
  enabled: false     # 禁用数据库加速

logging:
  level: "WARNING"   # 减少日志输出
```

### 查看结果

```bash
# 查看图表
open simulation_result_enhanced.png

# 查看动画
open simulation_enhanced.gif

# 查看报告
cat simulation_report_enhanced.md

# 查看日志
tail -f smart_pool.log
```

---

## 🌐 方式2：API服务

### 启动服务器

```bash
# 方法1: 直接启动
python3 api.py

# 方法2: 后台运行
nohup python3 api.py > api.log 2>&1 &

# 方法3: 使用Gunicorn (生产环境)
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

服务器将在 `http://localhost:5000` 启动

### 测试API

```bash
# 健康检查
curl http://localhost:5000/health

# 解释指令
curl -X POST http://localhost:5000/interpret \
  -H "Content-Type: application/json" \
  -d '{"instruction": "收到暴雨预警"}'

# 获取场景列表
curl http://localhost:5000/scenarios | jq

# 运行仿真（异步）
curl -X POST http://localhost:5000/simulation/run \
  -H "Content-Type: application/json" \
  -d '{
    "script": [[0, "保持水位平稳"]],
    "async": true
  }'

# 查询状态
curl http://localhost:5000/simulation/1/status
```

### 使用Python客户端

```python
from api_client import SmartPoolAPIClient

# 创建客户端
client = SmartPoolAPIClient("http://localhost:5000")

# 健康检查
if client.is_healthy():
    print("服务正常")

# 解释指令
result = client.interpret("收到暴雨预警")
print(f"置信度: {result['confidence']:.2f}")

# 运行仿真
script = [[0, "保持水位平稳，正常供水。"]]
sim_id = client.run_simulation(script, async_mode=True)

# 等待完成
final_status = client.wait_for_completion(sim_id)

# 获取历史
history = client.get_history(sim_id)
print(f"数据点: {len(history)}")

client.close()
```

---

## 🎬 方式3：交互式演示

### 启动演示

```bash
python3 demo.py
```

### 演示项目

演示程序包含6个模块展示：

1. **配置管理系统** - 展示配置加载和参数
2. **语义解释器** - 测试指令匹配和置信度
3. **监控告警系统** - 展示告警触发机制
4. **数据持久化** - 展示数据库操作
5. **API端点** - 列出所有API接口
6. **完整系统** - 运行完整仿真

每个演示都是交互式的，可以按需选择。

---

## ⚡ 方式4：性能测试

### 运行基准测试

```bash
python3 benchmark.py
```

### 测试项目

1. **语义解释器性能** - 1000次迭代
2. **MPC求解器性能** - 100次迭代
3. **物理模拟性能** - 10000次迭代
4. **监控系统性能** - 1000次迭代
5. **数据库性能** - 1000条记录
6. **内存使用** - 100小时仿真

**预期结果**:
```
语义解释器: ~0.5ms
MPC求解器: ~50ms
物理模拟: ~20μs
监控系统: ~0.3ms
数据库写入: ~0.05ms
```

---

## 🔧 高级使用

### 1. 添加自定义场景

编辑 `config.yaml`，在 `scenarios` 下添加：

```yaml
scenarios:
  - name: "自定义场景"
    keywords: ["关键词1", "关键词2"]
    config:
      W_level: 10.0
      W_smooth: 5.0
      Z_ref: 3.0
      delta_Q_max: 2.0
      constraints: {}
```

### 2. 调整MPC参数

```yaml
mpc:
  horizon: 10        # 预测步数（越大越精确但越慢）
  Q_cap: 20.0       # 最大流量
  Z_min: 0.0        # 最小水位
  Z_max: 10.0       # 最大水位
```

### 3. 配置监控阈值

```yaml
monitoring:
  level_warning_high: 8.0    # 高水位警戒
  level_warning_low: 1.0     # 低水位警戒
  level_critical_high: 9.5   # 高水位危险
  level_critical_low: 0.5    # 低水位危险
```

### 4. 启用/禁用功能

```yaml
database:
  enabled: false    # 禁用数据库（加速10%）

logging:
  level: "WARNING"  # 只显示警告和错误

api:
  enabled: true     # 启用API服务
```

### 5. 使用编程API

```python
from brain_enhanced import EnhancedSemanticInterpreter
from physics import CanalPoolSimulator
from control import UniversalMPCSolver

# 初始化
brain = EnhancedSemanticInterpreter()
physics = CanalPoolSimulator(area=10000.0, dt=3600.0, delay_steps=1)
solver = UniversalMPCSolver(horizon=10, dt=3600.0, area=10000.0, delay_steps=1)

# 解释指令
instruction = "收到暴雨预警"
config, confidence = brain.interpret(instruction)
print(f"置信度: {confidence:.2f}")

# MPC求解
level = physics.get_level()
q_in = solver.solve(level, 5.0, [5.0]*10, config)
print(f"控制指令: {q_in:.2f} m³/s")

# 物理仿真
next_level = physics.step(q_in, 5.0)
print(f"下一步水位: {next_level:.2f} m")
```

---

## 📊 查看结果

### 仿真报告

```bash
cat simulation_report_enhanced.md
```

报告包含：
- 系统配置
- 各阶段分析
- 监控统计
- 告警详情

### 数据库查询

```python
from database import SimulationDatabase

db = SimulationDatabase("simulation_data.db")

# 获取最近的仿真
sims = db.get_recent_simulations(limit=5)
for sim in sims:
    print(f"ID: {sim['id']}, 时长: {sim['total_hours']}h")

# 查看历史
sim_id = sims[0]['id']
history = db.get_simulation_history(sim_id)
print(f"数据点数: {len(history)}")

# 查看告警
alerts = db.get_simulation_alerts(sim_id)
print(f"告警数: {len(alerts)}")

db.close()
```

### 日志分析

```bash
# 查看错误
grep ERROR smart_pool.log

# 查看警告
grep WARNING smart_pool.log

# 查看场景切换
grep "场景切换" smart_pool.log

# 实时监控
tail -f smart_pool.log
```

---

## 🐛 故障排查

### 问题1: 导入模块失败

```bash
# 检查依赖
pip3 install -r requirements.txt

# 验证安装
python3 -c "import numpy, cvxpy, matplotlib, yaml, flask"
```

### 问题2: 优化求解失败

查看日志中的详细错误：
```bash
grep "优化" smart_pool.log
```

常见原因：
- 约束冲突（如Q_in_max=0但Z_min很高）
- delta_Q_max太小
- 预测时域过大

解决方法：
```yaml
mpc:
  horizon: 5  # 减小时域

default_control:
  delta_Q_max: 5.0  # 增大变化率
```

### 问题3: API无法访问

```bash
# 检查端口占用
lsof -i :5000

# 检查进程
ps aux | grep api.py

# 重启服务
pkill -f api.py
python3 api.py
```

### 问题4: 中文显示乱码

```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-microhei

# 或修改配置
# config.yaml
visualization:
  chinese_font: ["DejaVu Sans", "Arial"]
```

### 问题5: 数据库锁定

```yaml
# config.yaml
database:
  journal_mode: WAL
  timeout: 30
```

---

## 💡 使用技巧

### 1. 快速验证

```bash
# 运行5步快速测试
python3 -c "
from main_enhanced import SmartPoolSimulation
sim = SmartPoolSimulation()
sim.total_hours = 5
sim.run()
"
```

### 2. 批量测试场景

```python
from brain_enhanced import EnhancedSemanticInterpreter

brain = EnhancedSemanticInterpreter()

# 测试所有场景
scenarios = brain.list_scenarios()
for scenario_name in scenarios:
    details = brain.get_scenario_details(scenario_name)
    print(f"{scenario_name}: {details['keywords']}")
```

### 3. 导出数据

```python
from database import SimulationDatabase
import json

db = SimulationDatabase()
history = db.get_simulation_history(1)

# 导出为JSON
with open('export.json', 'w') as f:
    json.dump(history, f, indent=2)

db.close()
```

### 4. 自定义脚本

```python
from main_enhanced import SmartPoolSimulation

sim = SmartPoolSimulation()

# 自定义场景
custom_script = [
    (0, "保持水位平稳，正常供水。"),
    (5, "收到暴雨预警，立刻降低水位！"),
    (15, "进入冰期输水模式，严禁扰动冰盖。"),
    (25, "恢复正常供水。")
]

sim.run(script=custom_script)

if sim.db:
    sim.db.close()
```

---

## 📚 相关文档

- **README.md** - 完整项目文档
- **QUICKSTART.md** - 5分钟快速开始
- **API_EXAMPLES.md** - API使用示例
- **DEPLOYMENT.md** - 部署指南
- **FINAL_REPORT.md** - 最终完成报告

---

## 🆘 获取帮助

### 查看内置帮助

```bash
python3 main_enhanced.py --help  # 主程序帮助
python3 api.py --help            # API帮助
```

### 运行示例

```bash
python3 demo.py                  # 交互式演示
python3 api_client.py            # API客户端示例
```

### 检查系统状态

```bash
./quick_test.sh                  # 快速测试
python3 benchmark.py             # 性能测试
```

---

## 📝 常用命令速查

```bash
# 基础操作
python3 main_enhanced.py         # 运行仿真
python3 api.py                   # 启动API
python3 demo.py                  # 交互演示
./quick_test.sh                  # 快速测试

# 测试
python3 test_units.py            # 原始测试
python3 -m unittest test_enhanced.py -v  # 增强测试
python3 -m unittest test_api.py -v       # API测试
python3 benchmark.py                      # 性能测试

# API操作
curl http://localhost:5000/health           # 健康检查
curl http://localhost:5000/scenarios | jq  # 获取场景
curl -X POST http://localhost:5000/interpret \
  -H "Content-Type: application/json" \
  -d '{"instruction": "指令"}'               # 解释指令

# 查看结果
cat simulation_report_enhanced.md    # 查看报告
tail -f smart_pool.log               # 实时日志
sqlite3 simulation_data.db ".tables" # 查看数据库

# 清理
rm -f *.db *.log *.png *.gif *.md    # 清理生成文件
```

---

**祝您使用愉快！如有问题，请参考相关文档或运行演示程序。** 🎉
