#!/bin/bash

# ============================================================================
# 运行所有演示脚本
# Run All Demonstration Scripts
# ============================================================================

echo "================================================================================"
echo "                    智能水网控制系统 - 完整演示"
echo "                Smart Water Network Control System - Full Demo"
echo "================================================================================"

# 检查Python环境
echo ""
echo "[1/5] 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3，请先安装Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python版本: $PYTHON_VERSION"

# 检查依赖
echo ""
echo "[2/5] 检查依赖库..."
python3 -c "import numpy, matplotlib, cvxpy, networkx" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  警告: 部分依赖未安装"
    echo "   请运行: pip install numpy matplotlib cvxpy networkx"
    read -p "   是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ 所有核心依赖已安装"
fi

# 创建输出目录
mkdir -p demo_outputs

# ============================================================================
# Phase 1: 基础MPC控制演示
# ============================================================================
echo ""
echo "================================================================================"
echo "[3/5] Phase 1: 基础MPC控制演示"
echo "================================================================================"
echo ""
echo "运行 hydroe2e.main ..."
python3 -m hydroe2e.main

if [ $? -eq 0 ]; then
    echo "✓ Phase 1演示完成"
    # 移动输出文件
    [ -f "simulation_result.png" ] && cp simulation_result.png demo_outputs/
    [ -f "simulation.gif" ] && cp simulation.gif demo_outputs/
else
    echo "❌ Phase 1演示失败"
fi

# ============================================================================
# Phase 3: 数字孪生系统演示
# ============================================================================
echo ""
echo "================================================================================"
echo "[4/5] Phase 3: 数字孪生系统演示"
echo "================================================================================"
echo ""
echo "运行 hydroe2e/digital_twin/scenarios/deep_dive_simulation.py ..."
python3 hydroe2e/digital_twin/scenarios/deep_dive_simulation.py

if [ $? -eq 0 ]; then
    echo "✓ Phase 3演示完成"
    [ -f "digital_twin_dashboard.png" ] && cp digital_twin_dashboard.png demo_outputs/
else
    echo "❌ Phase 3演示失败（可能缺少依赖）"
fi

# ============================================================================
# Phase 4: 自愈系统演示
# ============================================================================
echo ""
echo "================================================================================"
echo "[5/5] Phase 4: 自愈系统演示"
echo "================================================================================"
echo ""
echo "运行 hydroe2e/phase4/self_healing/self_healing_system.py ..."
python3 hydroe2e/phase4/self_healing/self_healing_system.py 2>&1 | head -n 100

if [ $? -eq 0 ]; then
    echo "✓ Phase 4演示完成"
    [ -f "hydroe2e/phase4/self_healing/self_healing_report.png" ] && cp hydroe2e/phase4/self_healing/self_healing_report.png demo_outputs/
else
    echo "❌ Phase 4演示失败（可能缺少依赖）"
fi

# ============================================================================
# Phase 5: 完整系统集成演示
# ============================================================================
echo ""
echo "================================================================================"
echo "[Bonus] Phase 5: 完整系统集成演示"
echo "================================================================================"
echo ""
echo "运行 hydroe2e/phase5/integrated_system.py ..."
python3 hydroe2e/phase5/integrated_system.py 2>&1 | tail -n 50

if [ $? -eq 0 ]; then
    echo "✓ Phase 5演示完成"
    [ -f "hydroe2e/phase5/integrated_system_results.png" ] && cp hydroe2e/phase5/integrated_system_results.png demo_outputs/
else
    echo "❌ Phase 5演示失败"
fi

# ============================================================================
# 总结
# ============================================================================
echo ""
echo "================================================================================"
echo "                           演示总结"
echo "================================================================================"
echo ""
echo "生成的文件已保存到 demo_outputs/ 目录："
ls -lh demo_outputs/ 2>/dev/null || echo "  (空目录)"
echo ""
echo "查看结果："
echo "  - Phase 1: demo_outputs/simulation_result.png"
echo "  - Phase 3: demo_outputs/digital_twin_dashboard.png"
echo "  - Phase 4: demo_outputs/self_healing_report.png"
echo "  - Phase 5: demo_outputs/integrated_system_results.png"
echo ""
echo "================================================================================"
echo "✅ 所有演示完成！"
echo "================================================================================"
echo ""
echo "下一步："
echo "  - 查看文档: cat README.md"
echo "  - 快速开始: cat QUICK_START.md"
echo "  - 性能测试: python3 performance_test.py"
echo ""
