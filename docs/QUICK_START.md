# 快速入门指南

本指南帮助你在5分钟内安装并运行E2EControl系统。

## 环境要求

- Python 3.8 或更高版本
- pip 包管理器
- 操作系统：Linux / macOS / Windows

## 第一步：安装依赖

```bash
cd e2econtrol

# 方式一：安装核心依赖（最小安装）
pip install numpy matplotlib cvxpy networkx

# 方式二：安装完整依赖（推荐）
pip install -r requirements.txt
```

**依赖说明：**

| 包名 | 用途 | 必需 |
|------|------|------|
| numpy | 数值计算 | 是 |
| matplotlib | 数据可视化 | 是 |
| cvxpy | 凸优化求解（MPC/ADMM） | 是 |
| networkx | 图论与网络拓扑 | 是 |
| flask | REST API服务 | 否 |
| scikit-learn | 机器学习检测器（Phase 4） | 否 |
| torch | 深度学习检测器（Phase 4） | 否 |
| pyyaml | 配置文件解析 | 否 |

## 第二步：验证安装

```bash
# 运行快速测试
python -c "
import numpy as np
import cvxpy as cp
print('NumPy:', np.__version__)
print('CVXPY:', cp.__version__)
print('安装成功！')
"
```

## 第三步：运行演示

### 演示1：基础MPC控制（Phase 1）

```bash
python -m hydroe2e.main
```

运行完成后生成 `simulation_result.png` 和 `simulation.gif`，展示三个场景：
- 正常供水：平稳水位控制
- 暴雨应急：快速降低水位
- 恢复运行：回到正常状态

### 演示2：数字孪生仿真（Phase 3）

```bash
python digital_twin/scenarios/deep_dive_simulation.py
```

运行100步深度仿真，覆盖6个复杂工况，生成 `digital_twin_dashboard.png`（3x3九维监控面板）。

### 演示3：异常检测（Phase 4.1）

```bash
python phase4/examples/ml_detection_demo.py
```

演示15+种异常检测算法，包括统计方法、机器学习和集成融合，生成 `ml_detection_results.png`。

### 演示4：故障诊断（Phase 4.2）

```bash
python phase4/examples/diagnosis_demo.py
```

演示5大类型故障的自动诊断过程。

### 演示5：自愈系统（Phase 4.3）

```bash
python phase4/self_healing/self_healing_system.py
```

演示5种故障场景的完整自愈闭环：传感器漂移、执行器卡死、控制器异常、通信故障、传感器失效。生成 `self_healing_report.png`。

### 演示6：完整系统集成（Phase 5）

```bash
python phase5/integrated_system.py
```

运行端到端集成演示，生成 `integrated_system_results.png`。

## 第四步：启动API服务

```bash
# 启动API服务
hydroe2e-api

# 服务启动后，测试接口
curl http://localhost:5000/health

# 解释指令
curl -X POST http://localhost:5000/interpret \
  -H "Content-Type: application/json" \
  -d '{"instruction": "收到暴雨预警，立刻降低水位！"}'

# 运行仿真
curl -X POST http://localhost:5000/simulation/run \
  -H "Content-Type: application/json" \
  -d '{
    "script": [
      [0, "保持水位平稳，正常供水。"],
      [10, "收到暴雨预警，立刻降低水位！"]
    ]
  }'
```

## 第五步：自定义场景

编辑 `main.py`，修改仿真脚本：

```python
# 自定义场景脚本
script = [
    (0, "保持水位平稳，正常供水。"),
    (10, "进入冰期输水模式，严禁扰动冰盖。"),
    (20, "下游检测到污染，紧急切断出流！"),
    (30, "恢复正常供水。"),
]
```

调整MPC控制参数，编辑 `config.yaml`：

```yaml
simulation:
  total_hours: 50        # 仿真总时长
  dt: 3600.0             # 时间步长（秒）
  area: 10000.0          # 渠池面积（m2）

mpc:
  horizon: 10            # 预测时域
  W_level: 50.0          # 水位跟踪权重
  W_smooth: 10.0         # 平滑性权重
```

## 查看可视化结果

所有演示运行后会生成PNG图片：

```bash
# 列出所有生成的图片
ls *.png
ls digital_twin_dashboard.png
ls phase4/self_healing/*.png
ls phase5/*.png
```

## 运行测试

```bash
# 运行单元测试
python -m pytest tests/ -v

# 运行综合测试
python comprehensive_test.py

# 运行性能测试
python performance_test.py
```

## 下一步

- 阅读 [架构设计](ARCHITECTURE.md) 了解系统全貌
- 阅读 [API参考](API_REFERENCE.md) 了解完整API
- 阅读 [部署指南](DEPLOYMENT.md) 部署到生产环境
- 阅读 [开发指南](DEVELOPMENT.md) 参与项目开发
