# 🚀 快速开始指南

## 5分钟快速上手

### 1. 安装依赖

```bash
pip3 install -r requirements.txt
```

### 2. 运行增强版仿真

```bash
python3 main_enhanced.py
```

这将：
- ✅ 加载配置文件 `config.yaml`
- ✅ 初始化所有模块（语义解释器、MPC求解器、物理模拟器、监控系统）
- ✅ 运行50小时的仿真
- ✅ 生成可视化结果和报告

### 3. 查看结果

运行完成后，将生成以下文件：

- 📊 `simulation_result_enhanced.png` - 静态结果图
- 🎬 `simulation_enhanced.gif` - 动态演示动画
- 📝 `simulation_report_enhanced.md` - 详细分析报告
- 💾 `simulation_data.db` - 数据库（可选）
- 📋 `smart_pool.log` - 运行日志

### 4. 运行测试

```bash
# 运行增强测试套件
python3 -m pytest test_enhanced.py -v

# 运行原始单元测试
python3 test_units.py
```

### 5. 性能测试

```bash
python3 benchmark.py
```

## 📝 简单示例

### 使用默认配置

```python
from main_enhanced import SmartPoolSimulation

# 创建仿真
sim = SmartPoolSimulation()

# 运行（使用默认场景）
sim.run()

# 完成
if sim.db:
    sim.db.close()
```

### 自定义场景

```python
from main_enhanced import SmartPoolSimulation

sim = SmartPoolSimulation()

# 自定义场景脚本
custom_script = [
    (0, "保持水位平稳，正常供水。"),
    (15, "收到暴雨预警，立刻降低水位！"),
    (30, "恢复正常供水。")
]

sim.run(script=custom_script)

if sim.db:
    sim.db.close()
```

### 测试语义解释器

```python
from brain_enhanced import EnhancedSemanticInterpreter
from logger import setup_logging

# 设置日志
setup_logging({'level': 'INFO', 'console_output': True})

# 创建解释器
brain = EnhancedSemanticInterpreter()

# 测试指令
config, confidence = brain.interpret("收到暴雨预警")

print(f"置信度: {confidence:.2f}")
print(f"目标水位: {config['Z_ref']}m")
print(f"最大流量变化: {config['delta_Q_max']}m³/s")
```

### 查询历史数据

```python
from database import SimulationDatabase

# 打开数据库
db = SimulationDatabase("simulation_data.db")

# 获取最近的仿真
sims = db.get_recent_simulations(limit=5)
for sim in sims:
    print(f"ID: {sim['id']}, 时长: {sim['total_hours']}h")

# 查看特定仿真的历史
if sims:
    sim_id = sims[0]['id']
    history = db.get_simulation_history(sim_id)
    print(f"历史记录数: {len(history)}")

db.close()
```

## ⚙️ 配置调整

编辑 `config.yaml` 来调整参数：

```yaml
# 修改仿真时长
simulation:
  total_hours: 100  # 改为100小时

# 调整MPC时域（更快求解）
mpc:
  horizon: 5  # 从10改为5

# 修改日志级别
logging:
  level: "WARNING"  # 只显示警告和错误

# 禁用数据库（更快运行）
database:
  enabled: false
```

## 🔍 调试技巧

### 1. 查看详细日志

```yaml
logging:
  level: "DEBUG"
  console_output: true
```

### 2. 单步调试

```python
from brain_enhanced import EnhancedSemanticInterpreter
from physics import CanalPoolSimulator
from control import UniversalMPCSolver

# 初始化
brain = EnhancedSemanticInterpreter()
physics = CanalPoolSimulator(area=10000.0, dt=3600.0, delay_steps=1)
solver = UniversalMPCSolver(horizon=5, dt=3600.0, area=10000.0, delay_steps=1)

# 单步执行
instruction = "保持水位平稳"
config, confidence = brain.interpret(instruction)
print(f"配置: {config}")

level = physics.get_level()
q_in = solver.solve(level, 5.0, [5.0]*5, config)
print(f"控制指令: {q_in:.2f} m³/s")

next_level = physics.step(q_in, 5.0)
print(f"下一步水位: {next_level:.2f} m")
```

### 3. 监控告警触发

```python
from monitor import MonitoringSystem
from logger import setup_logging

setup_logging({'level': 'INFO', 'console_output': True})
monitor = MonitoringSystem()

# 测试不同水位
test_levels = [3.0, 8.5, 9.8, 0.3]
for level in test_levels:
    alerts = monitor.check_state(0, level, 5.0, 5.0, {'Z_ref': 3.0})
    print(f"水位 {level}m: {len(alerts)} 个告警")
```

## 🆘 常见问题

### Q: 找不到模块？
```bash
# 确保在正确的目录
cd /workspace

# 检查Python版本
python3 --version  # 需要 3.8+

# 重新安装依赖
pip3 install -r requirements.txt
```

### Q: 优化求解失败？
检查日志中的详细错误。常见原因：
- 约束冲突（如Q_in_max=0但Z_min很高）
- delta_Q_max太小
- 解决：放宽约束或调整目标

### Q: 中文显示乱码？
```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-microhei

# 或在config.yaml中指定字体
visualization:
  chinese_font: ["SimHei", "DejaVu Sans"]
```

## 📚 下一步

- 阅读完整文档：`README.md`
- 查看源码注释了解实现细节
- 修改 `config.yaml` 尝试不同参数
- 添加自己的场景定义
- 运行性能测试了解系统瓶颈

## 💡 提示

- 首次运行可能需要编译CVXPY求解器，会稍慢
- 数据库默认启用，可禁用以提升性能
- 调整MPC时域可平衡精度和速度
- 使用WARNING日志级别减少输出

祝您使用愉快！🎉
