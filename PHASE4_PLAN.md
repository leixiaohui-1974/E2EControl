# Phase 4: 异常检测与自愈控制

> **状态**: 🚧 **开发中**  
> **优先级**: 高  
> **预计完成**: Phase 4.1 基础框架

---

## 📋 概述

Phase 4旨在为智能水网控制系统增加**自我诊断**和**自动恢复**能力，使系统能够自动检测异常、诊断故障、并采取自愈措施，大幅提升系统的可靠性和鲁棒性。

---

## 🎯 核心目标

### 主要目标

1. **异常检测系统**
   - 实时监测系统运行状态
   - 自动识别各类异常情况
   - 准确率 >95%，误报率 <5%

2. **故障诊断系统**
   - 快速定位故障源
   - 分析故障原因
   - 给出诊断报告

3. **自愈控制器**
   - 自动隔离故障
   - 启用备用方案
   - 自动恢复正常

4. **预测性维护**
   - 设备寿命预测
   - 故障预警
   - 维护计划优化

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                  Phase 4: 异常检测与自愈层                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │ 异常检测模块  │  │ 故障诊断模块  │  │ 自愈控制模块  │  │
│  ├───────────────┤  ├───────────────┤  ├───────────────┤  │
│  │• 统计检测     │  │• 规则推理     │  │• 故障隔离     │  │
│  │• 机器学习     │  │• 因果分析     │  │• 方案切换     │  │
│  │• 深度学习     │  │• 专家系统     │  │• 自动恢复     │  │
│  │• 多源融合     │  │• 知识库       │  │• 降级运行     │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │               预测性维护模块                          │ │
│  │  • 设备寿命预测 (RUL)                                │ │
│  │  • 故障预警 (Early Warning)                          │ │
│  │  • 维护优化 (Maintenance Scheduling)                 │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              集成层: 自适应MPC (Phase 2+3)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 功能模块

### 1. 异常检测模块

#### 1.1 统计方法

**基于阈值的检测**:
- 3-sigma规则
- 移动平均线
- 指数加权移动平均（EWMA）

**变化点检测**:
- CUSUM (累积和控制图)
- Page-Hinkley检验
- Bayesian变化点检测

**时序分析**:
- ARIMA异常检测
- 季节性分解
- 趋势分析

#### 1.2 机器学习方法

**无监督学习**:
- Isolation Forest
- One-Class SVM
- Local Outlier Factor (LOF)
- DBSCAN聚类

**半监督学习**:
- Autoencoder
- Variational Autoencoder (VAE)

#### 1.3 深度学习方法（Phase 4.2）

**时序模型**:
- LSTM Autoencoder
- GRU-based Detector
- Transformer

**混合模型**:
- CNN + LSTM
- Attention机制

#### 1.4 多传感器融合

**融合策略**:
- 投票机制
- 加权融合
- Dempster-Shafer证据理论

### 2. 故障诊断模块

#### 2.1 规则推理

**专家规则库**:
```
IF 水位异常高 AND 入流正常 AND 出流低 
   THEN 出口闸门故障

IF 水位异常低 AND 入流低 AND 出流正常
   THEN 入口闸门故障 OR 上游供水不足

IF 水位波动大 AND 流量变化快
   THEN 控制系统故障 OR 传感器故障
```

**规则置信度**:
- 每条规则有置信度评分
- 多条规则组合推理
- 不确定性处理

#### 2.2 因果分析

**因果图建模**:
```
上游供水 → 入口流量 → 水位 → 出口流量 → 下游需求
            ↑            ↑         ↑
         闸门1       传感器     闸门2
```

**因果推断**:
- Pearl因果推断
- Do-calculus
- 反事实分析

#### 2.3 专家系统

**知识表示**:
- 产生式规则
- 框架表示
- 语义网络

**推理机制**:
- 正向推理
- 反向推理
- 混合推理

#### 2.4 故障树分析（FTA）

**故障树构建**:
- 顶事件：系统失效
- 中间事件：子系统故障
- 基本事件：设备故障

**概率分析**:
- 故障概率计算
- 关键路径识别
- 薄弱环节定位

### 3. 自愈控制模块

#### 3.1 故障隔离

**隔离策略**:
- 故障设备停用
- 故障池段隔离
- 备用通道启用

**隔离决策**:
```python
def isolate_fault(fault_type, fault_location):
    if fault_type == "sensor":
        # 切换到备用传感器
        switch_to_backup_sensor()
    elif fault_type == "actuator":
        # 使用相邻闸门补偿
        use_adjacent_gate()
    elif fault_type == "controller":
        # 切换到备用控制器
        switch_to_backup_controller()
```

#### 3.2 降级运行

**降级模式**:
- **安全模式**: 最小流量维持
- **保守模式**: 降低控制精度
- **局部模式**: 故障段独立运行

**性能保障**:
- 优先保证关键池段
- 调整控制目标
- 放宽约束条件

#### 3.3 备用切换

**备用方案**:
1. **主-备切换**
   - 设备备用（传感器、闸门、控制器）
   - 路径备用（水路备用通道）
   - 控制备用（PID备用MPC）

2. **冗余设计**
   - 双传感器冗余
   - 多路径冗余
   - 分布式控制冗余

#### 3.4 自动恢复

**恢复流程**:
1. 故障确认已排除
2. 逐步恢复功能
3. 性能监测验证
4. 完全恢复正常

**渐进式恢复**:
- 先恢复基本功能
- 再恢复完整功能
- 最后恢复最优控制

### 4. 预测性维护模块

#### 4.1 设备寿命预测（RUL）

**数据驱动方法**:
- 回归模型（Linear, SVR）
- 随机森林
- XGBoost
- LSTM

**物理模型方法**:
- 磨损模型
- 退化模型
- Paris公式（疲劳裂纹）

#### 4.2 故障预警

**预警指标**:
- 健康指数（Health Index）
- 剩余寿命（RUL）
- 故障概率（PoF）

**预警级别**:
- 正常（绿色）
- 注意（黄色）
- 警告（橙色）
- 危险（红色）

#### 4.3 维护优化

**维护策略**:
- 事后维护（Reactive）
- 预防性维护（Preventive）
- 预测性维护（Predictive）
- 主动维护（Proactive）

**优化目标**:
- 最小化维护成本
- 最大化设备可用性
- 平衡成本与风险

---

## 🔬 关键技术

### 异常检测算法

**统计方法**:
```python
# 3-sigma检测
def detect_3sigma(data, window=20):
    mean = np.mean(data[-window:])
    std = np.std(data[-window:])
    threshold = 3 * std
    is_anomaly = abs(data[-1] - mean) > threshold
    return is_anomaly

# CUSUM检测
def detect_cusum(data, threshold=5, drift=1):
    cumsum = np.cumsum(data - np.mean(data))
    is_anomaly = np.max(np.abs(cumsum)) > threshold
    return is_anomaly
```

**机器学习**:
```python
from sklearn.ensemble import IsolationForest

# Isolation Forest
detector = IsolationForest(contamination=0.1)
detector.fit(normal_data)
anomaly_score = detector.predict(new_data)
```

**深度学习**:
```python
# LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim)
    
    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded
```

### 故障诊断推理

**贝叶斯网络**:
```python
# 简化示例
P(故障|症状) = P(症状|故障) * P(故障) / P(症状)

# 多个症状的联合概率
P(闸门故障 | 水位高, 流量低) = 
    P(水位高, 流量低 | 闸门故障) * P(闸门故障) / 
    P(水位高, 流量低)
```

**规则推理引擎**:
```python
class RuleEngine:
    def infer(self, symptoms):
        for rule in self.rules:
            if rule.matches(symptoms):
                return rule.diagnosis, rule.confidence
        return "unknown", 0.0
```

### 自愈决策

**决策树**:
```
故障类型？
├─ 传感器故障
│  └─ 切换到备用传感器
├─ 闸门故障
│  ├─ 有备用闸门？
│  │  ├─ 是 → 切换到备用
│  │  └─ 否 → 使用相邻闸门补偿
│  └─ ...
└─ 控制器故障
   └─ 切换到备用控制器
```

---

## 📈 实施路线

### Phase 4.1: 基础框架（当前）

**目标**: 建立异常检测和故障诊断的基本能力

**任务**:
- [x] 创建Phase 4规划文档
- [ ] 异常检测基础框架
  - [ ] 统计方法检测器（3-sigma, CUSUM, EWMA）
  - [ ] 基于阈值的检测
  - [ ] 时间窗口管理
- [ ] 故障诊断基础框架
  - [ ] 规则库定义
  - [ ] 规则推理引擎
  - [ ] 诊断报告生成
- [ ] 简单演示系统
  - [ ] 传感器故障检测
  - [ ] 闸门故障诊断
  - [ ] 自动告警

**交付物**:
- 异常检测基础类
- 规则推理引擎
- 故障诊断器
- 演示程序

### Phase 4.2: 机器学习增强（未来）

**目标**: 使用ML提升检测准确率和诊断能力

**任务**:
- [ ] Isolation Forest检测器
- [ ] One-Class SVM检测器
- [ ] Autoencoder检测器
- [ ] 集成多个检测器
- [ ] 模型训练和评估

### Phase 4.3: 自愈控制（未来）

**目标**: 实现自动故障隔离和恢复

**任务**:
- [ ] 故障隔离策略
- [ ] 降级运行模式
- [ ] 备用方案切换
- [ ] 自动恢复流程
- [ ] 完整演示

### Phase 4.4: 预测性维护（未来）

**目标**: 实现设备寿命预测和维护优化

**任务**:
- [ ] RUL预测模型
- [ ] 健康指数计算
- [ ] 故障预警系统
- [ ] 维护计划优化
- [ ] 决策支持工具

---

## 🎯 验收标准

### Phase 4.1 基础框架

| 指标 | 目标 | 备注 |
|------|------|------|
| 异常检测准确率 | >90% | 基于模拟数据 |
| 误报率 | <10% | 正常数据被误判为异常 |
| 检测延迟 | <5步 | 从异常发生到检测出 |
| 诊断准确率 | >85% | 正确识别故障类型 |
| 响应时间 | <1秒 | 从检测到诊断 |

### Phase 4 完整版

| 指标 | 目标 | 备注 |
|------|------|------|
| 异常检测准确率 | >95% | 多种方法融合 |
| 误报率 | <5% | 严格控制 |
| 故障定位准确率 | >90% | 准确定位故障源 |
| 自愈成功率 | >80% | 自动恢复成功 |
| 系统可用性 | >99% | 包含自愈能力 |

---

## 🚀 快速开始（开发完成后）

```bash
# 异常检测演示
cd phase4/examples
python3 anomaly_detection_demo.py

# 故障诊断演示
python3 fault_diagnosis_demo.py

# 自愈控制演示
python3 self_healing_demo.py

# 完整演示
python3 complete_demo.py
```

---

## 📖 技术参考

### 异常检测

- **统计方法**: Shewhart控制图、CUSUM、EWMA
- **机器学习**: Isolation Forest (Liu et al., 2008)
- **深度学习**: LSTM-based Anomaly Detection (Malhotra et al., 2015)

### 故障诊断

- **因果推断**: Pearl's Causality (Pearl, 2009)
- **贝叶斯网络**: Probabilistic Reasoning (Jensen, 2007)
- **专家系统**: Rule-based Systems (Durkin, 1994)

### 预测性维护

- **RUL预测**: Data-driven Prognostics (Si et al., 2011)
- **PHM**: Prognostics and Health Management (Vachtsevanos et al., 2006)

---

## 💡 关键挑战

1. **误报与漏报平衡**
   - 挑战：降低误报同时保持高检出率
   - 方案：多方法融合 + 自适应阈值

2. **实时性要求**
   - 挑战：复杂算法计算开销大
   - 方案：轻量化模型 + 分层检测

3. **故障类型多样**
   - 挑战：覆盖所有可能故障
   - 方案：分层诊断 + 持续学习

4. **自愈决策可靠性**
   - 挑战：避免错误的自愈动作
   - 方案：多重验证 + 人工确认

---

## 🔮 未来展望

- **智能化**: AI驱动的异常检测和诊断
- **预测性**: 从被动响应到主动预防
- **自主性**: 从人工干预到自动自愈
- **学习性**: 从固定规则到持续学习

---

**Phase 4状态**: 🚧 **开发中**  
**当前任务**: Phase 4.1 基础框架  
**预计完成**: Phase 4.1  

*让系统更加健壮和可靠* 🛡️
