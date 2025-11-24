#!/bin/bash

# 快速测试脚本
# 用于验证系统各模块功能

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║         智能闸门控制系统 - 快速测试脚本                         ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 测试计数
TOTAL=0
PASSED=0
FAILED=0

# 测试函数
test_module() {
    local name=$1
    local command=$2
    
    TOTAL=$((TOTAL + 1))
    echo -e "${BLUE}[TEST $TOTAL]${NC} 测试: $name"
    
    if eval "$command" > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}  ✓ 通过${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}  ✗ 失败${NC}"
        FAILED=$((FAILED + 1))
        cat /tmp/test_output.log | head -5
        return 1
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. 检查Python环境"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_module "Python版本" "python3 --version"
test_module "NumPy安装" "python3 -c 'import numpy'"
test_module "CVXPY安装" "python3 -c 'import cvxpy'"
test_module "Matplotlib安装" "python3 -c 'import matplotlib'"
test_module "PyYAML安装" "python3 -c 'import yaml'"
test_module "Flask安装" "python3 -c 'import flask'"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2. 测试核心模块"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_module "配置管理器" "python3 config_manager.py"
test_module "日志系统" "python3 logger.py"
test_module "语义解释器" "python3 -c 'from brain_enhanced import EnhancedSemanticInterpreter; EnhancedSemanticInterpreter()'"
test_module "监控系统" "python3 -c 'from monitor import MonitoringSystem; MonitoringSystem()'"
test_module "数据库系统" "python3 -c 'from database import SimulationDatabase; db = SimulationDatabase(\"test.db\"); import os; os.remove(\"test.db\")'"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3. 运行单元测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_module "原始单元测试" "python3 test_units.py 2>&1 | grep -q OK"
test_module "API测试" "python3 -m unittest test_api.TestAPI 2>&1 | grep -q OK"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4. 检查文件完整性"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_module "config.yaml存在" "test -f config.yaml"
test_module "README.md存在" "test -f README.md"
test_module "requirements.txt存在" "test -f requirements.txt"
test_module "api.py存在" "test -f api.py"
test_module "main_enhanced.py存在" "test -f main_enhanced.py"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "测试结果汇总"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "总测试数: $TOTAL"
echo -e "${GREEN}通过: $PASSED${NC}"
echo -e "${RED}失败: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                                ║${NC}"
    echo -e "${GREEN}║                  ✓ 所有测试通过！系统正常！                     ║${NC}"
    echo -e "${GREEN}║                                                                ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "可以运行以下命令:"
    echo "  • python3 main_enhanced.py    - 运行完整仿真"
    echo "  • python3 api.py              - 启动API服务器"
    echo "  • python3 demo.py             - 运行交互式演示"
    echo "  • python3 benchmark.py        - 性能测试"
    exit 0
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                                                                ║${NC}"
    echo -e "${RED}║                    ✗ 部分测试失败                              ║${NC}"
    echo -e "${RED}║                                                                ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "请检查失败的测试，确保:"
    echo "  1. 安装了所有依赖: pip3 install -r requirements.txt"
    echo "  2. Python版本 >= 3.8"
    echo "  3. 配置文件正确"
    exit 1
fi
