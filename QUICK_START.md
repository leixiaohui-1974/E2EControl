# 🚀 快速开始指南

## 5分钟上手智能水网控制系统

### 1. 环境准备

```bash
# 安装Python依赖
pip install numpy matplotlib cvxpy networkx
```

### 2. 运行演示

#### 演示1: 基础MPC控制（Phase 1）
```bash
python3 main.py
```
**输出：** `simulation_result.png`, `simulation.gif`

#### 演示2: 数字孪生仿真（Phase 3）
```bash
python3 digital_twin/scenarios/deep_dive_simulation.py
```
**输出：** `digital_twin_dashboard.png`（3×3图表）

#### 演示3: 自愈系统（Phase 4）
```bash
python3 phase4/self_healing/self_healing_system.py
```
**输出：** `phase4/self_healing/self_healing_report.png`

#### 演示4: 完整系统集成（Phase 5）
```bash
python3 phase5/integrated_system.py
```
**输出：** `phase5/integrated_system_results.png`（2×3图表）

### 3. 查看结果

所有演示运行后都会生成PNG图片，可以直接查看：

```bash
ls -lh *.png
ls -lh digital_twin_dashboard.png
ls -lh phase4/self_healing/*.png
ls -lh phase5/*.png
```

### 4. 阅读文档

- **项目总览：** `README.md`
- **项目状态：** `PROJECT_STATUS.md`
- **最终总结：** `FINAL_SUMMARY.md`
- **完成报告：** `PROJECT_COMPLETE.md`
- **各阶段文档：** `phase*/README.md`

### 5. 自定义运行

#### 示例：修改场景

编辑 `main.py`，修改场景脚本：

```python
# 原场景
(0, "保持水位平稳，正常供水。"),
(15, "收到暴雨预警，立刻降低水位腾出库容！安全第一！"),

# 改为
(0, "保持水位平稳，正常供水。"),
(10, "进入冰期输水模式，严禁扰动冰盖。"),
(20, "下游检测到污染，紧急切断出流！"),
```

#### 示例：调整参数

编辑 `control.py`，修改MPC参数：

```python
def __init__(self, horizon=10, dt=3600.0, area=10000.0, delay_steps=1):
    self.N = horizon      # 预测时域（改为5会更快）
    self.dt = dt          # 时间步长（秒）
    self.area = area      # 渠池面积（m²）
    self.tau = delay_steps  # 系统延迟
```

### 6. 故障排查

**Q: 找不到模块？**
```bash
# 确保在项目根目录
cd /workspace
python3 main.py
```

**Q: CVXPY求解失败？**
```bash
# 检查是否安装了求解器
pip install cvxpy
```

**Q: 中文显示乱码？**
```bash
# 安装中文字体（Ubuntu）
sudo apt-get install fonts-wqy-microhei
```

### 7. 下一步

- 阅读 `README.md` 了解详细功能
- 查看 `PROJECT_COMPLETE.md` 了解项目全貌
- 运行 `phase4` 和 `phase5` 的高级功能
- 根据需求定制自己的场景

### 8. 获取帮助

- 查看文档目录下的 `*.md` 文件
- 阅读代码中的注释
- 运行演示程序学习用法

---

**开始探索吧！** 🚀

