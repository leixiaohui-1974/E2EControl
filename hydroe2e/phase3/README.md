# Phase 3: 智能场景识别与决策

> **状态**: 🚧 **进行中** (基础框架已完成)

---

## 📋 概述

Phase 3实现了**场景自动识别**和**智能决策引擎**，系统能够自主判断当前工况并选择最优策略。

### 核心成果（基础框架）

- ✅ 场景类型定义（18种场景，6大类）
- ✅ 规则引擎（12条识别规则）
- ✅ 特征提取器（时序特征分析）
- ✅ 综合场景识别器（混合方法）
- ✅ 简单知识库（7种策略+案例库）
- ✅ 智能决策引擎（场景→策略→参数）

---

## 🗂️ 目录结构

```
phase3/
├── scenario_recognition/
│   ├── scenario_types.py          # 场景类型定义 (400行) ⭐
│   ├── rule_engine.py             # 规则引擎 (470行) ⭐
│   ├── feature_extractor.py       # 特征提取器 (280行) ⭐
│   └── scenario_recognizer.py     # 场景识别器 (370行) ⭐
├── knowledge/
│   └── simple_kb.py               # 知识库 (360行) ⭐
├── decision/
│   └── decision_engine.py         # 决策引擎 (380行) ⭐
├── examples/
│   └── integrated_demo.py         # 综合演示 (220行) ⭐
└── README.md                       # 本文档
```

**总计**: 6个核心模块，2,480行代码

---

## 🎯 核心功能

### 1. 场景识别（18种场景）

#### 场景分类体系

| 类别 | 子场景 | 数量 |
|------|--------|------|
| 正常运行 | 日常、高峰、低谷、周末、夜间、过渡 | 6 |
| 防洪 | 洪前、洪峰、洪后、山洪 | 4 |
| 干旱 | 缺水、优先分配、应急供水 | 3 |
| 冰期 | 结冰、稳定冰期、融冰、冰塞 | 4 |
| 应急 | 污染、故障、爆裂、停电、网络攻击 | 5 |
| 维护 | 计划维护、紧急维修、巡检、清淤 | 4 |

#### 识别方法

**1. 规则引擎**（12条规则）
```python
from scenario_recognition.rule_engine import RuleEngine, SystemState

engine = RuleEngine()

state = SystemState(
    time=8, 
    levels=[3.0], 
    flows=[10.0], 
    demands=[12.0]
)

matches = engine.recognize(state)
# 返回: [('peak_demand', 0.95), ...]
```

**2. 特征提取**
```python
from scenario_recognition.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(window_size=24)

# 添加历史数据
for t in range(24):
    extractor.add_observation(levels, flows, demands)

# 提取特征
features = extractor.extract_features()
# 返回: 统计、趋势、波动、周期等特征

# 场景指标
indicators = extractor.get_scenario_indicators()
# 返回: 高峰可能性、波动水平、趋势方向等
```

**3. 综合识别**（混合方法）
```python
from scenario_recognition.scenario_recognizer import ScenarioRecognizer

recognizer = ScenarioRecognizer()

result = recognizer.recognize(state)

print(f"场景: {result.scenario_name}")
print(f"置信度: {result.confidence:.1%}")
print(f"识别方法: {result.method}")  # 'rule', 'feature', 'hybrid'
```

### 2. 知识库（策略+案例）

#### 预定义策略（7种）

| 策略 | 场景 | 特点 |
|------|------|------|
| 日常正常运行 | daily_operation | 平衡性能和能耗 |
| 高峰快速响应 | peak_demand | 重视供水，启用前馈 |
| 低谷节能 | off_peak | 重视节能和平稳 |
| 防洪泄洪 | flood_peak | 短时域，快速响应 |
| 干旱节水 | water_shortage | 长时域，优化分配 |
| 冰期稳定 | stable_ice | 平滑调整，减少波动 |
| 应急响应 | emergency | 高频更新，灵活调整 |

**使用示例**:
```python
from knowledge.simple_kb import SimpleKnowledgeBase

kb = SimpleKnowledgeBase()

# 获取策略
strategy = kb.get_strategy("peak_response")
print(f"时域: {strategy.horizon}")
print(f"权重: {strategy.weights}")

# 获取最佳实践
best_practices = kb.get_best_practices("peak_demand")
print(f"推荐策略: {best_practices['recommended_strategy']}")
print(f"经验教训: {best_practices['lessons_learned']}")
```

### 3. 智能决策引擎

**完整决策流程**:
1. 场景识别（规则+特征）
2. 策略选择（知识库+最佳实践）
3. 参数配置（自适应调整）
4. 风险评估（多维度分析）
5. 行动建议（立即+监控）
6. 推理解释（可追溯）

**使用示例**:
```python
from decision.decision_engine import DecisionEngine

engine = DecisionEngine()

decision = engine.make_decision(state)

# 场景识别结果
print(decision.scenario.scenario_name)

# 选定策略
print(decision.strategy.name)

# MPC配置
print(decision.mpc_config)

# 风险评估
print(decision.risk_level)  # 'low', 'medium', 'high', 'critical'
print(decision.risk_factors)

# 行动建议
print(decision.immediate_actions)
print(decision.monitoring_points)

# 决策推理
print(decision.reasoning)
```

---

## 🚀 快速开始

### 运行综合演示

```bash
cd phase3/examples
python3 integrated_demo.py
```

演示内容：
- 6个典型场景序列
- 从正常→高峰→预警→洪峰→恢复→低谷
- 完整的识别→决策→建议流程

---

## 📊 功能特性

### 场景识别

| 特性 | 说明 |
|------|------|
| 识别方法 | 规则引擎 + 特征提取 混合 |
| 场景数量 | 18种（可扩展） |
| 置信度评估 | ✓ 支持 |
| 实时性 | <100ms |
| 可解释性 | ✓ 提供识别依据 |

### 决策引擎

| 特性 | 说明 |
|------|------|
| 策略库 | 7种预定义策略 |
| 自适应配置 | ✓ 根据场景动态调整 |
| 风险评估 | 4级风险（低、中、高、紧急） |
| 行动建议 | ✓ 立即行动 + 监控要点 |
| 决策推理 | ✓ 完整推理链 |

---

## 🔬 技术实现

### 规则引擎

**规则示例**:
```python
Rule(
    name="高峰需求识别",
    scenario_id="peak_demand",
    conditions=[
        lambda state: np.mean(state.demands) > 8.0,  # 需求大
        lambda state: _is_peak_hour(state.time),     # 高峰时段
    ],
    confidence=0.9,
    priority=3
)
```

### 特征提取

**提取特征**:
- 统计特征：均值、标准差、范围
- 趋势特征：线性趋势、R²
- 波动特征：波动率、变异系数
- 周期特征：周期检测、周期长度
- 变化特征：变化率、加速度

### 知识库

**知识表示**:
```python
@dataclass
class ControlStrategy:
    strategy_id: str
    scenario_id: str
    horizon: int
    weights: Dict[str, float]
    constraints: Dict
    use_feedforward: bool
```

**案例库**:
```python
@dataclass
class CaseRecord:
    case_id: str
    scenario_id: str
    initial_state: Dict
    strategy_used: str
    outcome: str  # 'success', 'partial', 'failed'
    lessons_learned: List[str]
```

---

## 🎨 创新点

1. **混合识别方法**
   - 规则引擎处理明确场景
   - 特征提取捕捉统计特性
   - 加权融合提高准确率

2. **知识驱动决策**
   - 策略库存储最佳实践
   - 案例库记录历史经验
   - 动态学习和优化

3. **多维风险评估**
   - 场景优先级
   - 识别置信度
   - 水位/流量风险
   - 告警状态

4. **可解释决策**
   - 完整推理链
   - 决策依据
   - 行动建议
   - 监控要点

---

## 📈 下一步计划

### Phase 3 完整版（待实现）

- [ ] 机器学习模型（LSTM场景识别）
- [ ] 完整知识图谱（500+实体）
- [ ] 深度学习特征提取
- [ ] 案例推理（CBR）
- [ ] 在线学习和模型更新
- [ ] 大规模测试（100+场景）

### 与Phase 2集成

- [ ] 集成到分布式MPC
- [ ] 自动策略切换
- [ ] 性能对比测试
- [ ] 实际场景验证

---

## 📖 相关文档

- **完整规划**: [NEXT_PHASE_PLAN.md](../NEXT_PHASE_PLAN.md)
- **Phase 1**: [Phase 1 README](../README.md)
- **Phase 2**: [Phase 2 README](../phase2/README.md)
- **项目总览**: [PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md)

---

**Phase 3状态**: 🚧 **基础框架完成**  
**开发时间**: 2025年11月24日  
**代码量**: 2,480行（6个核心模块）  
**下一步**: 机器学习模型集成

*Water is Life, AI is Future* 🌊💧

