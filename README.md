# HydroE2E - 智能水网端到端控制系统

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-v1.1.0-green.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HydroMind](https://img.shields.io/badge/HydroMind-Satellite-orange.svg)](#hydromind-生态集成)

面向南水北调等大型调水工程的端到端智能控制系统。**HydroMind 生态卫星项目**，支持独立开发、测试和部署。

## 系统架构

```
+-----------------------------------------------------------+
|                     用户指令层                              |
|          自然语言 / REST API / MCP工具 / 仿真脚本            |
+-----------------------------+-----------------------------+
                              |
              +---------------v---------------+
              |      智能决策引擎 (Brain)       |
              |   场景识别 | 策略选择 | 参数自适应  |
              +---------------+---------------+
                              |
              +---------------v---------------+
              |    分布式MPC控制层 (DMPC)       |
              |   ADMM优化 | 多池协调 | 约束处理  |
              +---------------+---------------+
                              |
              +---------------v---------------+
              |     数字孪生系统 (Twin)         |
              |  高精度物理模型 | 智能感知 | 鲁棒控制 |
              +---------------+---------------+
                              |
          +-------------------+-------------------+
          |                   |                   |
   +------v------+    +------v------+    +-------v-----+
   |  异常检测    |    |  故障诊断    |    |  自愈控制    |
   |  15+算法     |    |  5大类型     |    |  10步闭环    |
   |  准确率96%   |    |  准确率95%   |    |  成功率90%   |
   +-------------+    +-------------+    +-------------+
```

## 功能特性

| Phase | 模块 | 核心能力 |
|-------|------|----------|
| **Phase 1** | 基础MPC控制 | 模型预测控制、自然语言指令解析、多场景策略自动切换 |
| **Phase 2** | 分布式DMPC | ADMM优化、Over-relaxation加速40%、多池协调 |
| **Phase 3** | 数字孪生 | 20km渠池物理模型、圣维南方程、FDIA防御、经济调度 |
| **Phase 4** | 智能决策与自愈 | 15+异常检测算法、5类故障诊断、10步自愈闭环 |
| **Phase 5** | 系统集成 | 端到端框架、REST API、Docker部署、性能监控 |

## 快速开始

```bash
# 克隆项目
git clone https://github.com/leixiaohui-1974/e2econtrol.git
cd e2econtrol

# 安装（开发模式）
pip install -e ".[dev]"

# 运行测试
pytest

# 启动API服务
hydroe2e-api
```

### 基本使用

```python
from hydroe2e.brain import SemanticInterpreter
from hydroe2e.physics import CanalPoolSimulator
from hydroe2e.control import UniversalMPCSolver

# 解析自然语言指令
interpreter = SemanticInterpreter()
config = interpreter.interpret("收到暴雨预警，立刻降低水位腾出库容！")

# 创建物理仿真器
physics = CanalPoolSimulator(area=10000.0, dt=3600.0, delay_steps=1, initial_level=3.0)

# MPC最优控制
solver = UniversalMPCSolver(horizon=10, dt=3600.0, area=10000.0, delay_steps=1)
q_opt = solver.solve(current_level=3.0, config=config)
```

## 项目结构

```
e2econtrol/
├── hydroe2e/                       # 主包
│   ├── brain.py                    #   智能决策引擎
│   ├── brain_enhanced.py           #   增强语义解析
│   ├── config_manager.py           #   YAML配置管理
│   ├── database.py                 #   SQLite持久化
│   ├── monitor.py                  #   实时监控告警
│   ├── logger.py                   #   结构化日志
│   ├── exceptions.py               #   自定义异常体系
│   ├── simulation_manager.py       #   仿真编排器
│   ├── api.py                      #   Flask REST API (13端点)
│   ├── api_client.py               #   Python客户端SDK
│   ├── role.py                     #   HydroMind角色注册
│   ├── _compat.py                  #   hydromind-contracts桥接
│   ├── mcp_servers/                #   MCP工具服务
│   ├── control/                    #   MPC/DMPC控制器
│   ├── physics/                    #   水力学仿真
│   ├── digital_twin/               #   数字孪生系统
│   ├── phase2/                     #   多池级联控制
│   ├── phase3/                     #   场景识别与决策
│   ├── phase4/                     #   异常检测+故障诊断+自愈
│   └── phase5/                     #   高级集成
├── tests/                          # pytest测试套件 (271+)
├── docs/                           # 文档体系
├── web/                            # Web前端
├── docker/                         # Docker配置
├── config.yaml                     # 系统配置
└── pyproject.toml                  # 包配置
```

## 性能指标

| 类别 | 指标 | 目标值 | 实际值 |
|------|------|--------|--------|
| 控制 | MPC求解时间 | < 100ms | ~33ms |
| 控制 | 水位控制精度 | ±10cm | ±5cm |
| 检测 | 异常检测准确率 | > 90% | 96% |
| 诊断 | 故障诊断准确率 | > 85% | 95% |
| 自愈 | 自愈成功率 | > 80% | 85-90% |
| 系统 | 可用性 | > 99% | 99.7% |

## HydroMind 生态集成

HydroE2E 作为 HydroMind 生态卫星项目，通过以下机制接入：

- **角色注册**：`hydromind.roles` entry_point → `e2e_controller`
- **MCP工具**：`hydromind.engines` entry_point → 6个MCP工具
- **协议兼容**：可选依赖 `hydromind-contracts`，未安装时独立运行

```bash
# 安装含 HydroMind 集成
pip install -e ".[hydromind]"
```

## 文档

| 文档 | 说明 |
|------|------|
| [架构设计](docs/ARCHITECTURE.md) | 系统分层架构、模块关系、数据流 |
| [API参考](docs/API_REFERENCE.md) | 完整的REST API端点文档 |
| [快速入门](docs/QUICK_START.md) | 安装、配置、运行一站式指南 |
| [部署指南](docs/DEPLOYMENT.md) | Docker与生产环境部署 |
| [开发指南](docs/DEVELOPMENT.md) | 开发路线图与贡献指南 |
| [测试指南](docs/TESTING.md) | 测试策略与运行方法 |
| [故障排查](docs/TROUBLESHOOTING.md) | 常见问题与解决方案 |
| [更新日志](docs/CHANGELOG.md) | 版本发布记录 |

## 许可证

本项目采用 [MIT 许可证](LICENSE)。
