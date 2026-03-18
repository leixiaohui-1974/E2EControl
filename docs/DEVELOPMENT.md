# 开发指南

## 开发环境搭建

```bash
# 克隆项目
git clone <repo-url> e2econtrol
cd e2econtrol

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install pytest black flake8
```

## 代码规范

- **格式化工具**：Black（行长度120）
- **代码检查**：Flake8
- **注释密度**：保持在30%以上
- **Docstring**：每个类和公开方法都需要文档字符串
- **命名规范**：类名PascalCase，函数和变量snake_case

```bash
# 格式化代码
black --line-length 120 *.py

# 代码检查
flake8 --max-line-length 120 *.py
```

## 项目开发路线图

### 智能化等级体系

对标无人驾驶L0-L5，系统定义了水网自主运行等级：

| 等级 | 名称 | 人工参与度 | 关键能力 |
|------|------|-----------|---------|
| L0 | 完全人工 | 100% | 基础监控 |
| L1 | 辅助决策 | 80% | 状态感知、建议生成 |
| L2 | 部分自动 | 50% | 场景识别、基本控制 |
| L3 | 条件自动 | 20% | 异常检测、故障诊断 |
| L4 | 高度自动 | 5% | 自愈、自适应优化 |
| L5 | 完全自主 | 0% | 全场景覆盖、自主学习 |

**当前状态**：L2-L3（具备异常检测和基本自愈能力）

### 已完成阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 基础MPC控制 | 已完成 |
| Phase 2 | 分布式DMPC优化 | 已完成 |
| Phase 3 | 数字孪生系统 | 已完成 |
| Phase 4 | 智能决策与自愈 | 已完成 |
| Phase 5 | 系统集成 | 90%完成 |

### 短期计划（1-3个月）

- 性能优化：代码profiling、关键路径优化、C++/Cython加速
- Web界面：Flask/FastAPI后端 + React前端 + 实时监控大屏
- 部署配置：Docker容器化、CI/CD流水线、监控告警

### 中期计划（3-6个月）

- 实际案例接入：南水北调渠道、水库大坝调度
- 实时数据对接：SCADA系统、传感器数据流、天气预报
- 分布式部署：边缘计算节点、云边协同、数据同步

### 长期计划（6-12个月）

- 全网协同：多渠池联合调度、跨区域协调、流域级优化
- AI增强：强化学习优化、预测性维护、持续学习
- 数字孪生云：SaaS服务、多租户支持、API市场

## 在环测试框架

系统设计了完整的在环测试（HIL）框架，覆盖7大场景类别、62个二级工况：

```
场景体系
|-- S1. 正常运行场景 (12个工况)
|-- S2. 防洪调度场景 (10个工况)
|-- S3. 干旱应对场景 (8个工况)
|-- S4. 冰期运行场景 (8个工况)
|-- S5. 污染应急场景 (8个工况)
|-- S6. 设备故障场景 (10个工况)
+-- S7. 安全攻击场景 (6个工况)
```

**测试流程：**

```
初始化 --> 场景加载 --> 工况注入 --> 系统响应 --> 结果评估 --> 报告生成
```

**等级认证标准：**

| 等级 | 通过率要求 | 人工干预率 |
|------|-----------|-----------|
| L1 | > 60% | < 80% |
| L2 | > 75% | < 50% |
| L3 | > 85% | < 20% |
| L4 | > 95% | < 5% |
| L5 | > 99% | ~0% |

## 模块开发指南

### 新增异常检测器

继承 `BaseDetector` 基类：

```python
from hydroe2e.phase4.anomaly_detection.base_detector import BaseDetector

class MyDetector(BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(name="MyDetector", **kwargs)

    def fit(self, data):
        """训练模型"""
        pass

    def detect(self, data_point):
        """检测单个数据点"""
        # 返回 (is_anomaly: bool, score: float)
        pass
```

### 新增故障类型

在故障类型库中添加新条目：

```python
fault_type = {
    "type": "new_fault_type",
    "category": "physical",
    "severity_range": (0.3, 0.9),
    "features": ["feature1", "feature2"],
    "repair_actions": ["action1", "action2"],
}
```

### 新增仿真场景

参照 `digital_twin/scenarios/deep_dive_simulation.py` 编写新场景。

### 新增API端点

在 `hydroe2e/api.py` 中注册新路由：

```python
@app.route('/new_endpoint', methods=['GET'])
def new_endpoint():
    return jsonify({"success": True, "data": ...})
```

## 评价指标体系

### 安全性指标

| 指标 | 目标值 | 权重 |
|------|--------|------|
| 水位越限率 | < 0.1% | 30% |
| 故障响应时间 | < 60s | 20% |
| 事故避免率 | > 99% | 30% |
| 降级运行时间 | < 5% | 20% |

### 可靠性指标

| 指标 | 目标值 | 权重 |
|------|--------|------|
| 系统可用率 | > 99.9% | 40% |
| MTBF | > 720h | 20% |
| MTTR | < 10min | 20% |
| 自愈成功率 | > 90% | 20% |

### 控制性能指标

| 指标 | 目标值 | 权重 |
|------|--------|------|
| 水位跟踪误差 | < 5cm | 30% |
| 流量稳定性 | < 5% | 25% |
| 调节时间 | < 30min | 25% |
| 能效指标 | 优化基准 | 20% |

## 贡献指南

1. Fork本项目
2. 创建功能分支：`git checkout -b feature/my-feature`
3. 编写代码并添加测试
4. 确保所有测试通过：`python -m pytest tests/ -v`
5. 提交代码：`git commit -m "feat: 新增XX功能"`
6. 推送分支：`git push origin feature/my-feature`
7. 创建Pull Request

### Commit消息规范

```
feat: 新增功能
fix: 修复Bug
docs: 文档更新
refactor: 代码重构
test: 测试相关
perf: 性能优化
```
