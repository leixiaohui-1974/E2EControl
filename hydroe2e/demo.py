"""
交互式演示脚本
展示系统各模块功能
"""

import sys
import time
from colorama import init, Fore, Style
import os

# 初始化colorama（如果可用）
try:
    init(autoreset=True)
    COLOR_SUPPORT = True
except Exception:
    COLOR_SUPPORT = False

from hydroe2e.config_manager import get_config
from hydroe2e.logger import get_logger, setup_logging
from hydroe2e.brain_enhanced import EnhancedSemanticInterpreter
from hydroe2e.monitor import MonitoringSystem
from hydroe2e.database import SimulationDatabase


def print_header(text):
    """打印标题"""
    if COLOR_SUPPORT:
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{text.center(60)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    else:
        print(f"\n{'='*60}")
        print(text.center(60))
        print(f"{'='*60}\n")


def print_success(text):
    """打印成功信息"""
    if COLOR_SUPPORT:
        print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")
    else:
        print(f"✓ {text}")


def print_info(text):
    """打印信息"""
    if COLOR_SUPPORT:
        print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")
    else:
        print(f"ℹ {text}")


def print_warning(text):
    """打印警告"""
    if COLOR_SUPPORT:
        print(f"{Fore.YELLOW}⚠ {text}{Style.RESET_ALL}")
    else:
        print(f"⚠ {text}")


def demo_config_system():
    """演示配置管理系统"""
    print_header("演示1: 配置管理系统")
    
    try:
        config = get_config()
        
        print_info("加载配置文件...")
        time.sleep(0.5)
        print_success("配置加载成功！")
        
        print(f"\n📊 仿真参数:")
        print(f"  - 总时长: {config.get('simulation.total_hours')} 小时")
        print(f"  - 时间步长: {config.get('simulation.dt')} 秒")
        print(f"  - 渠池面积: {config.get('simulation.area')} m²")
        
        print(f"\n🎛️ MPC配置:")
        print(f"  - 预测时域: {config.get('mpc.horizon')} 步")
        print(f"  - 最大流量: {config.get('mpc.Q_cap')} m³/s")
        print(f"  - 水位范围: {config.get('mpc.Z_min')}-{config.get('mpc.Z_max')} m")
        
        scenarios = config.get_all_scenarios()
        print(f"\n📋 场景数量: {len(scenarios)}")
        for s in scenarios:
            print(f"  • {s['name']}: {', '.join(s['keywords'][:2])}...")
        
        print_success("\n配置系统运行正常！")
        
    except Exception as e:
        print_warning(f"配置系统测试失败: {e}")
    
    input("\n按回车继续...")


def demo_semantic_interpreter():
    """演示语义解释器"""
    print_header("演示2: 增强版语义解释器")
    
    setup_logging({'level': 'ERROR', 'console_output': False})
    
    try:
        brain = EnhancedSemanticInterpreter(similarity_threshold=0.5)
        
        print_info("初始化语义解释器...")
        time.sleep(0.5)
        print_success("语义解释器就绪！")
        
        test_instructions = [
            "保持水位平稳，正常供水。",
            "保持水位平稳",
            "收到暴雨预警",
            "进入冰期模式",
            "发现污染",
            "这是一个完全未知的指令"
        ]
        
        print(f"\n🧠 测试 {len(test_instructions)} 条指令:\n")
        
        for i, instruction in enumerate(test_instructions, 1):
            print(f"{i}. 指令: \"{instruction}\"")
            
            config, confidence = brain.interpret(instruction)
            
            if confidence >= 0.9:
                status = "精确匹配"
                color = Fore.GREEN if COLOR_SUPPORT else ""
            elif confidence >= 0.6:
                status = "模糊匹配"
                color = Fore.YELLOW if COLOR_SUPPORT else ""
            else:
                status = "使用默认"
                color = Fore.RED if COLOR_SUPPORT else ""
            
            if COLOR_SUPPORT:
                print(f"   {color}置信度: {confidence:.2f} ({status}){Style.RESET_ALL}")
            else:
                print(f"   置信度: {confidence:.2f} ({status})")
            
            print(f"   目标水位: {config['Z_ref']}m, 平滑权重: {config['W_smooth']}")
            print()
        
        print_success("语义解释器工作正常！")
        
    except Exception as e:
        print_warning(f"语义解释器测试失败: {e}")
    
    input("\n按回车继续...")


def demo_monitoring_system():
    """演示监控告警系统"""
    print_header("演示3: 监控告警系统")
    
    setup_logging({'level': 'ERROR', 'console_output': False})
    
    try:
        monitor = MonitoringSystem()
        
        print_info("初始化监控系统...")
        time.sleep(0.5)
        print_success("监控系统就绪！")
        
        test_cases = [
            (0, 3.0, 5.0, 5.0, "正常状态"),
            (10, 8.5, 10.0, 5.0, "水位偏高"),
            (20, 9.8, 15.0, 5.0, "水位严重偏高"),
            (30, 0.3, 2.0, 5.0, "水位严重偏低"),
            (40, 3.0, 30.0, 5.0, "流量过大"),
        ]
        
        print(f"\n🔍 测试 {len(test_cases)} 个监控场景:\n")
        
        for t, level, q_in, q_out, desc in test_cases:
            print(f"场景: {desc}")
            print(f"  时间={t}h, 水位={level}m, 入流={q_in}m³/s")
            
            alerts = monitor.check_state(t, level, q_in, q_out, {'Z_ref': 3.0})
            
            if not alerts:
                print_success(f"  ✓ 无告警")
            else:
                for alert in alerts:
                    if alert.level.name == "WARNING":
                        print_warning(f"  ⚠ {alert.message}")
                    elif alert.level.name == "CRITICAL":
                        if COLOR_SUPPORT:
                            print(f"{Fore.RED}  🚨 {alert.message}{Style.RESET_ALL}")
                        else:
                            print(f"  🚨 {alert.message}")
            print()
        
        stats = monitor.get_statistics()
        print(f"📊 监控统计:")
        print(f"  - 总告警数: {stats['total_alerts']}")
        print(f"  - 警告数: {stats['warning_count']}")
        print(f"  - 严重告警数: {stats['critical_count']}")
        
        print_success("\n监控系统工作正常！")
        
    except Exception as e:
        print_warning(f"监控系统测试失败: {e}")
    
    input("\n按回车继续...")


def demo_database_system():
    """演示数据库系统"""
    print_header("演示4: 数据持久化系统")
    
    setup_logging({'level': 'ERROR', 'console_output': False})
    
    try:
        db_path = "demo_test.db"
        if os.path.exists(db_path):
            os.remove(db_path)
        
        db = SimulationDatabase(db_path)
        
        print_info("初始化数据库...")
        time.sleep(0.5)
        print_success("数据库连接成功！")
        
        print(f"\n💾 创建测试仿真记录...")
        sim_id = db.create_simulation(
            total_hours=50,
            dt=3600.0,
            area=10000.0,
            config={'test': True},
            notes="演示仿真"
        )
        print_success(f"仿真ID: {sim_id}")
        
        print(f"\n📝 写入状态数据...")
        for t in range(10):
            db.save_state(
                sim_id, t, 3.0 + t * 0.1, 5.0, 5.0, 3.0,
                "测试指令", {'Z_ref': 3.0}
            )
        print_success(f"写入 10 条状态记录")
        
        print(f"\n⚠️ 写入告警数据...")
        db.save_alert(sim_id, 5, "WARNING", "测试告警", "水位偏高", {})
        print_success(f"写入 1 条告警记录")
        
        db.finish_simulation(sim_id)
        
        print(f"\n📖 读取数据...")
        history = db.get_simulation_history(sim_id)
        alerts = db.get_simulation_alerts(sim_id)
        
        print(f"  - 历史记录数: {len(history)}")
        print(f"  - 告警记录数: {len(alerts)}")
        
        db.close()
        
        file_size = os.path.getsize(db_path) / 1024
        print(f"\n📦 数据库文件: {file_size:.2f} KB")
        
        print_success("\n数据库系统工作正常！")
        
        # 清理
        os.remove(db_path)
        
    except Exception as e:
        print_warning(f"数据库系统测试失败: {e}")
    
    input("\n按回车继续...")


def demo_api_endpoints():
    """演示API端点"""
    print_header("演示5: REST API端点")
    
    print_info("API服务器需要单独启动")
    print(f"\n🌐 可用的API端点:\n")
    
    endpoints = [
        ("GET", "/", "获取API信息"),
        ("GET", "/health", "健康检查"),
        ("GET", "/config", "获取配置信息"),
        ("POST", "/interpret", "解释自然语言指令"),
        ("POST", "/simulation/run", "运行仿真"),
        ("GET", "/simulation/<id>/status", "查询仿真状态"),
        ("GET", "/simulation/<id>/history", "获取历史数据"),
        ("GET", "/simulation/<id>/alerts", "获取告警记录"),
        ("GET", "/simulation/<id>/report", "获取仿真报告"),
        ("GET", "/simulation/<id>/result.png", "获取结果图表"),
        ("GET", "/simulations", "获取所有仿真"),
        ("GET", "/scenarios", "获取场景列表"),
    ]
    
    for method, path, desc in endpoints:
        method_color = Fore.GREEN if method == "GET" else Fore.YELLOW
        if COLOR_SUPPORT:
            print(f"{method_color}{method:6s}{Style.RESET_ALL} {path:35s} - {desc}")
        else:
            print(f"{method:6s} {path:35s} - {desc}")
    
    print(f"\n📝 启动API服务器:")
    print(f"  python3 api.py")
    
    print(f"\n🔍 测试API:")
    print(f"  curl http://localhost:5000/health")
    
    print(f"\n📚 详细文档:")
    print(f"  cat API_EXAMPLES.md")
    
    print_success("\nAPI端点定义完整！")
    
    input("\n按回车继续...")


def demo_full_system():
    """演示完整系统"""
    print_header("演示6: 完整系统集成")
    
    print_info("准备运行完整仿真...")
    print()
    
    print(f"此演示将:")
    print(f"  1. 加载配置")
    print(f"  2. 初始化所有模块")
    print(f"  3. 运行50小时仿真")
    print(f"  4. 生成可视化结果")
    print(f"  5. 保存数据到数据库")
    
    print(f"\n⏱️ 预计耗时: ~5-10秒")
    
    choice = input(f"\n是否运行完整仿真？(y/n): ").lower()
    
    if choice == 'y':
        print()
        print_info("启动完整仿真...")
        
        try:
            from main_enhanced import SmartPoolSimulation
            
            sim = SmartPoolSimulation()
            sim.run()
            
            if sim.db:
                sim.db.close()
            
            print()
            print_success("仿真完成！")
            
            print(f"\n📁 生成的文件:")
            files = [
                'simulation_result_enhanced.png',
                'simulation_enhanced.gif',
                'simulation_report_enhanced.md',
                'simulation_data.db',
                'smart_pool.log'
            ]
            
            for f in files:
                if os.path.exists(f):
                    size = os.path.getsize(f) / 1024
                    print(f"  ✓ {f:35s} ({size:.1f} KB)")
            
        except Exception as e:
            print_warning(f"仿真运行出错: {e}")
    else:
        print_info("跳过完整仿真")
    
    input("\n按回车继续...")


def main_menu():
    """主菜单"""
    while True:
        print_header("🌊 智能闸门控制系统 - 交互式演示")
        
        print("请选择演示项目:\n")
        print("  1. 配置管理系统")
        print("  2. 增强版语义解释器")
        print("  3. 监控告警系统")
        print("  4. 数据持久化系统")
        print("  5. REST API端点")
        print("  6. 完整系统集成")
        print("  7. 运行所有演示")
        print("  0. 退出")
        
        choice = input(f"\n请输入选项 (0-7): ").strip()
        
        if choice == '1':
            demo_config_system()
        elif choice == '2':
            demo_semantic_interpreter()
        elif choice == '3':
            demo_monitoring_system()
        elif choice == '4':
            demo_database_system()
        elif choice == '5':
            demo_api_endpoints()
        elif choice == '6':
            demo_full_system()
        elif choice == '7':
            demo_config_system()
            demo_semantic_interpreter()
            demo_monitoring_system()
            demo_database_system()
            demo_api_endpoints()
            demo_full_system()
            print_header("🎉 所有演示完成！")
            input("\n按回车返回主菜单...")
        elif choice == '0':
            print()
            print_success("感谢使用！再见！")
            break
        else:
            print_warning("无效选项，请重新选择")
            time.sleep(1)


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW if COLOR_SUPPORT else ''}演示中断{Style.RESET_ALL if COLOR_SUPPORT else ''}")
        sys.exit(0)
