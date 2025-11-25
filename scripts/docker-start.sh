#!/bin/bash
# E2EControl Docker Startup Script
# 智能水网控制系统 - Docker启动脚本

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║           E2EControl - 智能水网控制系统                    ║"
    echo "║              Docker Deployment Manager                     ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "检查运行环境..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi

    log_success "环境检查通过"
}

# Setup environment
setup_env() {
    log_info "配置环境变量..."

    if [ ! -f "$PROJECT_DIR/.env" ]; then
        if [ -f "$PROJECT_DIR/docker/.env.example" ]; then
            cp "$PROJECT_DIR/docker/.env.example" "$PROJECT_DIR/.env"
            log_warn "已从模板创建 .env 文件，请检查并修改配置"
        fi
    fi
}

# Create directories
create_dirs() {
    log_info "创建必要目录..."
    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/data"
    mkdir -p "$PROJECT_DIR/reports"
    log_success "目录创建完成"
}

# Build images
build_images() {
    log_info "构建 Docker 镜像..."
    cd "$PROJECT_DIR"
    docker compose build
    log_success "镜像构建完成"
}

# Start services
start_services() {
    local profile="${1:-}"

    log_info "启动服务..."
    cd "$PROJECT_DIR"

    if [ -n "$profile" ]; then
        docker compose --profile "$profile" up -d
    else
        docker compose up -d
    fi

    log_success "服务启动完成"
}

# Stop services
stop_services() {
    log_info "停止服务..."
    cd "$PROJECT_DIR"
    docker compose down
    log_success "服务已停止"
}

# Show status
show_status() {
    log_info "服务状态:"
    cd "$PROJECT_DIR"
    docker compose ps

    echo ""
    log_info "服务地址:"
    echo "  - Web Dashboard: http://localhost:8080"
    echo "  - API Server:    http://localhost:8000"
    echo "  - Grafana:       http://localhost:3000 (如启用)"
    echo "  - Prometheus:    http://localhost:9090 (如启用)"
}

# Show logs
show_logs() {
    local service="${1:-}"
    cd "$PROJECT_DIR"

    if [ -n "$service" ]; then
        docker compose logs -f "$service"
    else
        docker compose logs -f
    fi
}

# Run certification
run_certification() {
    log_info "运行认证测试..."
    cd "$PROJECT_DIR"
    docker compose --profile certification up certification
    log_success "认证测试完成，报告保存在 reports/ 目录"
}

# Clean up
cleanup() {
    log_warn "清理所有容器和数据..."
    cd "$PROJECT_DIR"
    docker compose down -v --remove-orphans
    log_success "清理完成"
}

# Main menu
show_help() {
    echo "用法: $0 <命令> [选项]"
    echo ""
    echo "命令:"
    echo "  start [profile]  启动服务 (可选 profile: monitoring, certification)"
    echo "  stop             停止所有服务"
    echo "  restart          重启所有服务"
    echo "  status           显示服务状态"
    echo "  logs [service]   查看日志"
    echo "  build            构建镜像"
    echo "  certify          运行认证测试"
    echo "  clean            清理所有容器和数据"
    echo "  help             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 start                    # 启动基础服务"
    echo "  $0 start monitoring         # 启动含监控的完整服务"
    echo "  $0 logs controller          # 查看控制器日志"
    echo "  $0 certify                  # 运行 L0-L5 认证测试"
}

# Main
main() {
    print_banner

    case "${1:-help}" in
        start)
            check_prerequisites
            setup_env
            create_dirs
            start_services "${2:-}"
            show_status
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            start_services "${2:-}"
            show_status
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "${2:-}"
            ;;
        build)
            check_prerequisites
            build_images
            ;;
        certify)
            check_prerequisites
            run_certification
            ;;
        clean)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
