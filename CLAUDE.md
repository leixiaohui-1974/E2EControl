# HydroE2E - 智能水网端到端控制系统

## 概述
HydroMind 生态卫星项目，可独立开发测试。提供从单池MPC控制到多池分布式协调、数字孪生、异常检测、故障诊断和自愈控制的完整端到端能力。

## 项目结构
```
hydroe2e/                    # 主包
├── brain.py                 # 语义解析（SemanticInterpreter）
├── brain_enhanced.py        # 增强语义解析（模糊匹配+置信度）
├── config_manager.py        # YAML配置管理
├── database.py              # SQLite持久化
├── monitor.py               # 实时监控与告警
├── logger.py                # 结构化日志
├── exceptions.py            # 自定义异常体系
├── simulation_manager.py    # 仿真编排器
├── api.py                   # Flask REST API (13端点)
├── api_client.py            # Python客户端SDK
├── role.py                  # HydroMind角色注册
├── _compat.py               # hydromind-contracts桥接
├── control/                 # MPC/DMPC控制器
├── physics/                 # 水力学仿真（Saint-Venant）
├── digital_twin/            # 数字孪生（20km渠道）
├── phase2/                  # 多池级联控制
├── phase3/                  # 场景识别与决策
├── phase4/                  # 异常检测+故障诊断+自愈
├── phase5/                  # 高级集成与自主学习
└── mcp_servers/             # MCP工具服务
tests/                       # pytest测试套件
docs/                        # 文档体系
```

## 导入约定
所有导入使用 `hydroe2e.` 前缀：
```python
from hydroe2e.brain import SemanticInterpreter
from hydroe2e.control import UniversalMPCSolver
from hydroe2e.physics import CanalPoolSimulator
from hydroe2e.config_manager import get_config
```

## 关键命令
```bash
pip install -e ".[dev]"      # 开发安装
pytest                       # 运行测试
hydroe2e-api                 # 启动API服务
```

## 依赖
- 必需：numpy, scipy, cvxpy, flask, networkx, scikit-learn, pandas
- 可选：hydromind-contracts（生态集成）, torch（深度学习）, mcp（MCP服务）

## HydroMind 集成
- 通过 `hydromind.roles` entry_point 注册 `e2e_controller` 角色
- 通过 `hydromind.engines` entry_point 暴露 MCP 工具
- hydromind-contracts 为可选依赖，不安装时系统独立运行
