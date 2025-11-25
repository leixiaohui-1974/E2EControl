# Phase 2 快速启动指南

## 🎯 本阶段目标

实现**3-5池级联系统**的**分布式协同控制**，验证多池MPC的可行性和性能。

---

## ⚡ 5分钟快速体验

### 1. 安装依赖（如果还没装）

```bash
pip3 install numpy matplotlib cvxpy networkx
```

### 2. 运行三池级联演示

```bash
cd phase2/examples
python3 three_pool_demo.py
```

**预期输出**:
- 仿真50小时的运行
- 生成水位和流量图表
- 显示性能指标（RMSE、最大偏差）

### 3. 查看结果

```bash
# 查看生成的图表
xdg-open cascaded_system_result.png  # Linux
# open cascaded_system_result.png     # macOS
```

---

## 📚 代码结构说明

### 核心模块

#### 1. `models/cascaded_system.py` - 级联系统物理模型

```python
from phase2.models.cascaded_system import CascadedCanalSystem

# 创建3池系统
system = CascadedCanalSystem(
    num_pools=3,        # 渠池数量
    pool_area=10000.0,  # 渠池面积(m²)
    dt=3600.0           # 时间步长(秒)
)

# 仿真一步
control_actions = [0.5, 0.5, 0.5, 0.5]  # 闸门开度
state = system.step(control_actions, demand=5.0)

# 获取状态
print(state['pools'][0]['level'])  # 第1个池的水位
```

**关键类**:
- `Gate`: 闸门模型（开度→流量）
- `CascadedPool`: 单个渠池（水力学演化）
- `CascadedCanalSystem`: 完整级联系统

#### 2. `controllers/distributed_mpc.py` - 分布式MPC控制器

```python
from phase2.controllers.distributed_mpc import DistributedMPCController

# 创建控制器
controller = DistributedMPCController(
    num_pools=3,   # 池数量
    horizon=10,    # 预测时域
    dt=3600.0      # 时间步长
)

# 求解最优控制
current_levels = [3.0, 3.1, 2.9]      # 当前水位
q_in_prevs = [5.0, 5.0, 5.0]          # 上一步入流
q_out_forecasts = [[5.0]*10]*3        # 出流预测

solutions = controller.solve(current_levels, q_in_prevs, q_out_forecasts)

# 获取最优控制
for i, (q_in, q_out) in enumerate(solutions):
    print(f"池{i}: 入流={q_in:.2f}, 出流={q_out:.2f}")
```

**算法特点**:
- **ADMM算法**: 分布式优化，收敛快
- **本地+协调**: 每个池独立优化，边界协调
- **多目标**: 水位跟踪 + 流量平滑

---

## 🔧 自定义参数

### 修改系统参数

编辑 `cascaded_system.py` 中的参数：

```python
class CascadedPool:
    def __init__(self, pool_id, area=10000.0, dt=3600.0):
        # 修改这些参数
        self.area = area           # 渠池面积
        self.dt = dt               # 时间步长
        self.level = 3.0           # 初始水位
```

### 修改MPC参数

编辑 `distributed_mpc.py` 中的权重：

```python
class LocalMPC:
    def __init__(self, ...):
        self.W_level = 10.0      # 水位跟踪权重（越大越重视水位精度）
        self.W_smooth = 5.0      # 流量平滑权重（越大越平稳）
        
        self.Z_ref = 3.0         # 目标水位(m)
        self.Z_min = 0.5         # 最低水位(m)
        self.Z_max = 8.0         # 最高水位(m)
        
        self.Q_max = 20.0        # 最大流量(m³/s)
        self.delta_Q_max = 2.0   # 最大流量变化率(m³/s/h)
```

---

## 🧪 测试不同场景

### 场景1: 需求突变

```python
# 在 three_pool_demo.py 中修改需求
demands = np.ones(50) * 5.0
demands[20:30] = 10.0  # 第20-30小时需求翻倍

run_cascaded_simulation(total_hours=50)
```

### 场景2: 初始水位偏差

```python
# 修改 cascaded_system.py
self.pools = [
    CascadedPool(pool_id=f"pool_{i}", area=pool_area, dt=dt,
                 initial_level=2.0 + i*0.5)  # 不同初始水位
    for i in range(num_pools)
]
```

### 场景3: 扩展到5池

```python
system = CascadedCanalSystem(num_pools=5, ...)
controller = DistributedMPCController(num_pools=5, ...)
```

---

## 📊 性能指标解读

运行演示后，会看到类似输出：

```
4. 分析结果...
   池0: RMSE=0.0523m, 最大偏差=0.1245m
   池1: RMSE=0.0631m, 最大偏差=0.1532m
   池2: RMSE=0.0714m, 最大偏差=0.1821m
   池0→池1 流量平衡误差: 0.0234 m³/s
   池1→池2 流量平衡误差: 0.0189 m³/s
```

**指标说明**:

| 指标 | 含义 | 理想值 |
|------|------|--------|
| RMSE | 水位跟踪均方根误差 | < 0.1m |
| 最大偏差 | 水位最大偏离目标值 | < 0.2m |
| 流量平衡误差 | 上下游流量不一致程度 | < 0.05 m³/s |

---

## 🐛 常见问题

### Q1: 优化求解失败
```
池0 MPC求解失败: Solver 'ECOS' failed.
```

**解决方法**:
1. 放宽约束（增大水位范围）
2. 减小预测时域（horizon=5）
3. 调整初始条件（避免极端状态）

### Q2: 收敛速度慢
```
ADMM未在10次迭代内收敛
```

**解决方法**:
1. 增加惩罚参数 `rho = 2.0`
2. 增加最大迭代次数 `max_iterations = 20`
3. 放宽收敛容差 `tolerance = 1e-2`

### Q3: 流量波动大

**解决方法**:
1. 增大平滑权重 `W_smooth = 10.0`
2. 减小流量变化率 `delta_Q_max = 1.0`
3. 增大预测时域 `horizon = 15`

---

## 🚀 下一步开发

当前状态: ✅ 基础框架已完成

待完成任务（按优先级）:

1. ⏳ **网络拓扑管理器** (phase2/topology/network.py)
   - 使用NetworkX建图
   - 支持复杂拓扑（支渠、合流）

2. ⏳ **多目标优化** (phase2/controllers/multi_objective.py)
   - Pareto优化
   - 权重自适应

3. ⏳ **前馈补偿** (phase2/controllers/feedforward.py)
   - 需求预测
   - 扰动前馈

4. ⏳ **5池以上测试** (phase2/examples/large_scale_demo.py)
   - 大规模系统验证
   - 性能基准测试

5. ⏳ **可视化工具** (phase2/visualization/realtime_dashboard.py)
   - 实时状态监控
   - 交互式控制面板

---

## 📖 参考资料

### 论文
1. Negenborn et al. (2009) - "Distributed MPC for canal systems"
2. Boyd et al. (2011) - "Distributed optimization via ADMM"
3. Litrico & Fromion (2009) - "Modeling and Control of Hydrosystems"

### 代码示例
- [do-mpc: Model Predictive Control in Python](https://www.do-mpc.com/)
- [pympc: Python MPC Toolbox](https://github.com/TobiaMarcucci/pympc)

### 水利工程
- 《渠道输水自动化控制技术》
- 《输水渠道水力学》

---

## 💬 技术支持

- **问题反馈**: GitHub Issues
- **技术讨论**: Discussions
- **文档首页**: [README.md](../README.md)
- **完整计划**: [NEXT_PHASE_PLAN.md](../NEXT_PHASE_PLAN.md)

---

**Happy Coding! 🎉**

*最后更新: 2025-11-24*
