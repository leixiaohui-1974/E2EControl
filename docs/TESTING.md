# 测试指南

## 测试概览

E2EControl采用多层次测试策略，包括单元测试、集成测试、性能测试和端到端测试。

### 测试文件分布

| 文件 | 类型 | 说明 |
|------|------|------|
| `tests/` | 单元测试 | pytest测试套件 |
| `test_units.py` | 单元测试 | 核心模块测试 |
| `test_api.py` | API测试 | REST API端点测试 |
| `test_enhanced.py` | 增强测试 | 扩展功能测试 |
| `comprehensive_test.py` | 集成测试 | 基础集成测试 |
| `comprehensive_test_100.py` | 集成测试 | 完整覆盖测试 |
| `comprehensive_test_ultimate.py` | 集成测试 | 终极测试套件 |
| `performance_test.py` | 性能测试 | 性能基准测试 |
| `benchmark.py` | 性能测试 | 算法基准测试 |
| `run_comprehensive_hil_test.py` | HIL测试 | 在环仿真测试 |

## 运行测试

### 基础测试

```bash
# 运行pytest测试套件
python -m pytest tests/ -v

# 运行单元测试
python test_units.py

# 运行API测试（需先启动API服务）
python test_api.py
```

### 综合测试

```bash
# 基础集成测试
python comprehensive_test.py

# 完整覆盖测试
python comprehensive_test_100.py

# 终极测试套件
python comprehensive_test_ultimate.py
```

### 性能测试

```bash
# 性能基准测试
python performance_test.py

# 算法基准测试
python benchmark.py
```

### 一键运行全部演示

```bash
# 运行所有演示程序
bash run_all_demos.sh

# 快速测试
bash quick_test.sh
```

## 测试覆盖范围

### Phase 1: 基础MPC控制

- 语义解释器（SemanticInterpreter）初始化与指令解析
- MPC求解器（UniversalMPCSolver）优化求解
- 物理仿真器（CanalPoolSimulator）状态演化
- 场景识别与策略选择
- 前馈控制

### Phase 2: 分布式DMPC

- ADMM分布式优化收敛
- Over-relaxation加速效果
- 多池协调控制
- 自适应惩罚参数

### Phase 3: 数字孪生

- 物理本体（SingleChannelFidelity）初始化与状态演化
- 智能感知层（IntelligentObserver）参数辨识
- 鲁棒ADMM求解器（SinglePoolADMM）约束处理
- 深度仿真场景完整运行

### Phase 4: 智能决策与自愈

- 统计检测器（5种）准确率
- 机器学习检测器（4种）准确率
- 深度学习检测器（3种）准确率
- 集成检测器融合效果
- 故障诊断引擎准确率
- 故障隔离策略执行
- 降级模式切换
- 恢复管理器策略执行
- 10步自愈闭环完整性

### Phase 5: 系统集成

- 端到端系统初始化
- 场景自动切换
- 故障注入与自愈验证
- API端点响应

## 性能测试指标

### 目标值

| 指标 | 目标 | 测试方法 |
|------|------|---------|
| MPC求解时间 | < 100ms | `performance_test.py` |
| 物理仿真步长 | < 1ms | `performance_test.py` |
| 异常检测延迟 | < 100ms | `benchmark.py` |
| API响应时间 | < 500ms | `test_api.py` |
| 100步完整仿真 | < 10s | `benchmark.py` |
| 内存占用 | < 200MB | `performance_test.py` |

### 运行性能测试

```bash
python performance_test.py
```

输出示例：

```
MPC求解时间:    ~33ms   (目标<100ms)
物理仿真:       <0.001ms/步
异常检测准确率:  96%
故障诊断准确率:  95%
自愈成功率:      85-90%
系统可用性:      99.7%
```

## 已知问题

以下是当前测试中已发现的问题（详见项目 issue tracker）：

1. **智能感知层导入路径错误**（高优先级）：`intelligent_observer.py` 中的 `physics` 导入路径需修正为 `digital_twin.physics`
2. **故障诊断模块缺失**（高优先级）：`diagnosis_engine.py` 文件缺失
3. **12个目录缺少 `__init__.py`**（中优先级）：影响包导入
4. **部分模块集成测试被跳过**（中优先级）：4个功能模块未完全覆盖
5. **物理模型构造函数参数不匹配**（低优先级）：测试代码需适配 `ChannelGeometry` 参数

## 编写测试

### 测试文件命名

- 单元测试：`test_*.py` 或 `*_test.py`
- 测试类：`TestClassName`
- 测试方法：`test_method_name`

### 测试示例

```python
import pytest
from hydroe2e.brain import SemanticInterpreter

class TestSemanticInterpreter:
    def setup_method(self):
        self.interpreter = SemanticInterpreter()

    def test_normal_instruction(self):
        result = self.interpreter.interpret("保持水位平稳，正常供水。")
        assert result is not None
        assert result["confidence"] > 0.5

    def test_emergency_instruction(self):
        result = self.interpreter.interpret("收到暴雨预警，立刻降低水位！")
        assert result["Z_ref"] < 3.0

    def test_empty_instruction(self):
        with pytest.raises(ValueError):
            self.interpreter.interpret("")
```

### 运行单个测试

```bash
# 运行指定测试文件
python -m pytest test_units.py -v

# 运行指定测试类
python -m pytest test_units.py::TestSemanticInterpreter -v

# 运行指定测试方法
python -m pytest test_units.py::TestSemanticInterpreter::test_normal_instruction -v
```
