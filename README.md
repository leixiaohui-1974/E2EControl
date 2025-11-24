# 🌊 智能闸门控制系统 (Smart Pool Agent)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于模型预测控制（MPC）的智能渠道水位调度系统，实现自然语言指令到优化控制策略的自动转换。

## 📋 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用指南](#使用指南)
- [测试](#测试)
- [性能](#性能)
- [开发指南](#开发指南)
- [常见问题](#常见问题)

## 🎯 项目概述

本项目实现了一个**智能化的水闸控制系统**，能够：

1. **理解自然语言指令**：如"收到暴雨预警，立刻降低水位腾出库容！"
2. **自动生成控制策略**：将语义转换为MPC优化问题
3. **实时优化控制**：考虑系统延迟、约束和未来预测
4. **全方位监控**：水位、流量、偏差等多维度监控告警

### 应用场景

- 🚰 南水北调渠道调度
- 🏞️ 水库大坝闸门控制  
- 🌧️ 洪水防御预警响应
- ❄️ 冰期安全输水管理

## ✨ 核心特性

### 🧠 增强版语义解释器
- ✅ 精确关键词匹配
- ✅ 模糊语义理解
- ✅ 相似度计算与置信度评分
- ✅ 场景库动态加载

### 🎛️ 通用MPC求解器
- ✅ 凸优化求解（CVXPY）
- ✅ 系统延迟建模
- ✅ 动态约束处理
- ✅ 多目标权重配置

### 📊 监控告警系统
- ✅ 实时状态监控
- ✅ 多级别告警（信息/警告/严重）
- ✅ 阈值自定义
- ✅ 告警回调机制

### 💾 数据持久化
- ✅ SQLite数据库存储
- ✅ 仿真历史回放
- ✅ 告警记录查询
- ✅ 性能指标统计

### 📈 可视化
- ✅ 静态结果图表
- ✅ 动态GIF动画
- ✅ 详细Markdown报告
- ✅ 中文字体支持

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────┐
│         自然语言指令输入层                    │
│   "收到暴雨预警，立刻降低水位腾出库容！"       │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │   增强语义解释器    │  [Brain Enhanced]
         │  - 模糊匹配         │
         │  - 置信度评分       │
         └─────────┬──────────┘
                   │ 控制配置
         ┌─────────▼──────────┐
         │   通用MPC求解器     │  [Control]
         │  - 凸优化          │
         │  - 约束处理        │
         └─────────┬──────────┘
                   │ 控制指令
         ┌─────────▼──────────┐
         │   渠道物理模拟器    │  [Physics]
         │  - 积分延迟模型    │
         │  - 状态演化        │
         └─────────┬──────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼───┐    ┌────▼────┐    ┌───▼────┐
│监控系统│    │日志系统 │    │数据库  │
│告警    │    │记录     │    │持久化  │
└───────┘    └─────────┘    └────────┘
```

### 模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| **配置管理** | `config_manager.py` | YAML配置加载与验证 |
| **日志系统** | `logger.py` | 结构化多级别日志 |
| **异常处理** | `exceptions.py` | 自定义异常类 |
| **语义解释** | `brain_enhanced.py` | NLP指令解析 |
| **MPC控制** | `control.py` | 优化求解器 |
| **物理仿真** | `physics.py` | 渠道模型 |
| **监控告警** | `monitor.py` | 状态监控 |
| **数据持久化** | `database.py` | SQLite存储 |
| **主程序** | `main_enhanced.py` | 系统集成 |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 安装依赖

```bash
# 克隆项目
git clone <repository_url>
cd smart-pool-agent

# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 运行仿真

```bash
# 运行增强版仿真
python main_enhanced.py

# 运行原始版本（兼容）
python main.py
```

### 运行测试

```bash
# 运行所有测试
python -m pytest test_enhanced.py -v

# 运行特定测试类
python -m pytest test_enhanced.py::TestEnhancedSemanticInterpreter -v

# 查看测试覆盖率
python -m pytest test_enhanced.py --cov=. --cov-report=html
```

## ⚙️ 配置说明

主配置文件：`config.yaml`

### 关键配置项

```yaml
# 仿真参数
simulation:
  total_hours: 50        # 仿真时长（小时）
  dt: 3600.0            # 时间步长（秒）
  area: 10000.0         # 渠池面积（m²）
  delay_steps: 1        # 系统延迟步数

# MPC求解器
mpc:
  horizon: 10           # 预测时域
  Q_cap: 20.0          # 最大流量（m³/s）
  Z_min: 0.0           # 最小水位（m）
  Z_max: 10.0          # 最大水位（m）

# 监控告警
monitoring:
  level_warning_high: 8.0    # 高水位警戒
  level_warning_low: 1.0     # 低水位警戒
  level_critical_high: 9.5   # 高水位危险
  level_critical_low: 0.5    # 低水位危险

# 日志配置
logging:
  level: "INFO"              # 日志级别
  file: "smart_pool.log"     # 日志文件
  console_output: true       # 控制台输出

# 数据库
database:
  enabled: true              # 是否启用
  path: "simulation_data.db" # 数据库路径
```

### 场景定义

在 `config.yaml` 中定义控制场景：

```yaml
scenarios:
  - name: "暴雨预警"
    keywords: ["暴雨预警", "降低水位", "腾出库容"]
    config:
      W_level: 100.0      # 提高水位跟踪权重
      Z_ref: 2.0          # 降低目标水位
      delta_Q_max: 5.0    # 允许大流量变化
```

## 📖 使用指南

### 基本使用

```python
from main_enhanced import SmartPoolSimulation

# 创建仿真实例
sim = SmartPoolSimulation("config.yaml")

# 定义场景脚本
script = [
    (0, "保持水位平稳，正常供水。"),
    (10, "收到暴雨预警，立刻降低水位！"),
    (20, "恢复正常供水。")
]

# 运行仿真
sim.run(script)

# 关闭资源
if sim.db:
    sim.db.close()
```

### 自定义场景

```python
from brain_enhanced import EnhancedSemanticInterpreter

# 创建解释器
brain = EnhancedSemanticInterpreter()

# 解释指令
config, confidence = brain.interpret("进入冰期输水模式")

print(f"置信度: {confidence:.2f}")
print(f"目标水位: {config['Z_ref']}m")
print(f"平滑权重: {config['W_smooth']}")
```

### 监控告警

```python
from monitor import MonitoringSystem, AlertLevel

# 创建监控系统
monitor = MonitoringSystem()

# 注册告警回调
def alert_handler(alert):
    if alert.level == AlertLevel.CRITICAL:
        print(f"🚨 严重告警: {alert.message}")

monitor.register_callback(alert_handler)

# 检查状态
alerts = monitor.check_state(
    time_step=10,
    level=9.5,
    q_in=15.0,
    q_out=5.0,
    config={'Z_ref': 3.0}
)
```

### 数据库查询

```python
from database import SimulationDatabase

# 打开数据库
db = SimulationDatabase("simulation_data.db")

# 获取最近的仿真
recent_sims = db.get_recent_simulations(limit=5)

# 查询特定仿真的历史
history = db.get_simulation_history(simulation_id=1)

# 获取告警记录
alerts = db.get_simulation_alerts(simulation_id=1)

db.close()
```

## 🧪 测试

### 测试结构

```
test_enhanced.py
├── TestConfigManager           # 配置管理器测试
├── TestEnhancedSemanticInterpreter  # 语义解释器测试
├── TestPhysicsEdgeCases        # 物理模拟边界测试
├── TestSolverEdgeCases         # 求解器边界测试
├── TestMonitoringSystem        # 监控系统测试
├── TestDatabaseOperations      # 数据库测试
├── TestIntegration             # 集成测试
└── TestPerformance             # 性能测试
```

### 运行测试

```bash
# 运行所有测试
python -m pytest test_enhanced.py -v

# 运行原始测试
python test_units.py

# 性能测试
python -m pytest test_enhanced.py::TestPerformance -v -s
```

## ⚡ 性能

### 基准指标

| 指标 | 典型值 | 说明 |
|------|--------|------|
| **MPC求解时间** | ~50ms | 单步优化（10步时域）|
| **语义解释时间** | <1ms | 基于规则匹配 |
| **数据库写入** | ~5ms/100条 | 批量提交 |
| **总仿真时间** | ~3-5秒 | 50小时仿真 |

### 性能优化建议

1. **减少MPC时域**：`horizon: 5` 可加速2倍
2. **调整日志级别**：`level: WARNING` 减少I/O
3. **禁用数据库**：`database.enabled: false` 加速10%
4. **批量提交**：每10步提交一次数据库

## 👨‍💻 开发指南

### 项目结构

```
smart-pool-agent/
├── config.yaml              # 主配置文件
├── config_manager.py        # 配置管理
├── logger.py                # 日志系统
├── exceptions.py            # 异常定义
├── brain_enhanced.py        # 语义解释器
├── control.py               # MPC求解器
├── physics.py               # 物理模拟
├── monitor.py               # 监控系统
├── database.py              # 数据库
├── main_enhanced.py         # 增强主程序
├── main.py                  # 原始主程序（兼容）
├── test_enhanced.py         # 增强测试
├── test_units.py            # 原始测试
├── requirements.txt         # 依赖列表
└── README.md               # 本文档
```

### 添加新场景

1. 编辑 `config.yaml`
2. 在 `scenarios` 下添加：
```yaml
- name: "你的场景名"
  keywords: ["关键词1", "关键词2"]
  config:
    W_level: 10.0
    Z_ref: 3.0
    # ... 其他参数
```

### 扩展监控指标

在 `monitor.py` 的 `check_state` 方法中添加：

```python
# 自定义检查
if custom_condition:
    alert = self._create_alert(
        AlertLevel.WARNING,
        "自定义告警类型",
        "告警信息",
        {'data': value}
    )
    alerts.append(alert)
```

### 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## ❓ 常见问题

### Q: 优化求解失败怎么办？

**A:** 检查约束是否冲突。例如：
- `Q_in_max: 0` 但 `Z_min` 很高且有持续出流
- `delta_Q_max` 太小无法响应需求变化

解决方法：
1. 放宽 `delta_Q_max`
2. 调整目标水位 `Z_ref`
3. 检查日志中的详细错误信息

### Q: 中文显示乱码？

**A:** 确保系统安装了中文字体：

```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-microhei

# 或在config.yaml中指定可用字体
visualization:
  chinese_font: ["SimHei", "Arial Unicode MS"]
```

### Q: 如何提高求解速度？

**A:** 
1. 减少MPC时域: `mpc.horizon: 5`
2. 使用更快的求解器（安装商业求解器如MOSEK）
3. 减少仿真步数或增大时间步长

### Q: 数据库文件太大？

**A:**
```python
# 定期清理旧数据
import sqlite3
conn = sqlite3.connect('simulation_data.db')
cursor = conn.cursor()
cursor.execute("DELETE FROM states WHERE simulation_id < ?", (old_id,))
conn.commit()
conn.close()
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- **CVXPY**: 凸优化建模库
- **Matplotlib**: 可视化工具
- **NumPy**: 科学计算基础

## 📧 联系方式

- 项目主页: [GitHub Repository]
- 问题反馈: [Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

---

**Happy Coding! 🚀**
