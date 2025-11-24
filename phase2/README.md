# Phase 2: 多池级联控制系统

## 目标
实现2-3个渠池的级联协同控制，建立基础的水网控制能力。

## 目录结构

```
phase2/
├── models/
│   └── cascaded_system.py      # 级联系统物理模型
├── controllers/
│   └── distributed_mpc.py      # 分布式MPC控制器
├── topology/
│   └── network.py              # 网络拓扑管理（待实现）
├── examples/
│   └── three_pool_demo.py      # 三池演示
└── tests/
    └── test_cascaded.py        # 测试用例（待实现）
```

## 快速开始

```bash
# 1. 进入目录
cd phase2/examples

# 2. 运行三池演示
python3 three_pool_demo.py
```

## 核心功能

### 1. 级联系统模型
- ✅ 多池串联结构
- ✅ 闸门动力学
- ✅ 流量传播延迟
- ✅ 水力学耦合

### 2. 分布式MPC
- ✅ ADMM算法
- ✅ 本地优化
- ✅ 耦合协调
- ✅ 迭代收敛

### 3. 协同控制
- ⏳ 多目标优化（开发中）
- ⏳ 前馈补偿（开发中）
- ⏳ 扰动抑制（开发中）

## 下一步

- [ ] 实现网络拓扑管理
- [ ] 添加前馈补偿
- [ ] 优化ADMM收敛速度
- [ ] 增加测试用例
- [ ] 支持5池以上系统

## 参考文献

1. Negenborn et al. "Distributed MPC for canal systems" (2009)
2. Boyd et al. "Distributed optimization and statistical learning via ADMM" (2011)

