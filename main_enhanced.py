"""
增强版主程序
集成所有新功能：配置管理、日志系统、监控告警、数据持久化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import time
from typing import Dict, List

# 新模块
from config_manager import get_config
from logger import get_logger, setup_logging
from brain_enhanced import EnhancedSemanticInterpreter
from monitor import MonitoringSystem
from database import SimulationDatabase

# 原有模块
from physics import CanalPoolSimulator
from control import UniversalMPCSolver


class SmartPoolSimulation:
    """智能闸门仿真系统"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化仿真系统
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = get_config(config_path)
        
        # 设置日志
        setup_logging(self.config.get_section('logging'))
        self.logger = get_logger()
        
        self.logger.info("=" * 60)
        self.logger.info("智能闸门仿真系统启动")
        self.logger.info("=" * 60)
        
        # 加载仿真参数
        sim_config = self.config.get_section('simulation')
        self.total_hours = sim_config['total_hours']
        self.dt = sim_config['dt']
        self.area = sim_config['area']
        self.delay_steps = sim_config['delay_steps']
        self.initial_level = sim_config['initial_level']
        
        # 初始化模块
        self.brain = EnhancedSemanticInterpreter()
        self.physics = CanalPoolSimulator(
            area=self.area,
            dt=self.dt,
            delay_steps=self.delay_steps,
            initial_level=self.initial_level
        )
        
        mpc_config = self.config.get_section('mpc')
        self.solver = UniversalMPCSolver(
            horizon=mpc_config['horizon'],
            dt=self.dt,
            area=self.area,
            delay_steps=self.delay_steps
        )
        
        self.monitor = MonitoringSystem()
        
        # 数据库（可选）
        self.db = None
        if self.config.get('database.enabled', False):
            db_path = self.config.get('database.path', 'simulation_data.db')
            self.db = SimulationDatabase(db_path)
            self.logger.info(f"数据库已启用: {db_path}")
        
        # 仿真数据
        self.history = {
            'time': [],
            'level': [],
            'q_in': [],
            'q_out': [],
            'target_level': [],
            'instruction': [],
            'config': [],
            'confidence': []
        }
        
        self.simulation_id = None
        
        # 配置中文字体
        chinese_fonts = self.config.get('visualization.chinese_font', [])
        plt.rcParams['font.sans-serif'] = chinese_fonts + ['sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    
    def run(self, script: List[tuple] = None):
        """
        运行仿真
        
        Args:
            script: 场景脚本 [(时间, 指令), ...]
        """
        if script is None:
            # 使用默认脚本
            script = [
                (0, "保持水位平稳，正常供水。"),
                (10, "收到暴雨预警，立刻降低水位腾出库容！安全第一！"),
                (20, "进入冰期输水模式，严禁扰动冰盖。"),
                (30, "下游检测到污染，紧急切断出流！"),
                (40, "保持水位平稳，正常供水。")
            ]
        
        # 创建数据库记录
        if self.db:
            self.simulation_id = self.db.create_simulation(
                total_hours=self.total_hours,
                dt=self.dt,
                area=self.area,
                config=self.config.config,
                notes="增强版仿真"
            )
        
        # 生成需求预测
        demand_config = self.config.get_section('demand')
        np.random.seed(self.config.get('simulation.random_seed', 42))
        base_demand = demand_config.get('base_demand', 5.0)
        noise_std = demand_config.get('noise_std', 0.5)
        demands = base_demand + np.random.normal(0, noise_std, self.total_hours + 20)
        
        # 初始状态
        current_instruction = script[0][1]
        last_control_action = base_demand
        
        self.logger.info(f"开始仿真: 总时长={self.total_hours}h, 步长={self.dt}s")
        self.logger.info(f"场景数量: {len(script)}")
        
        start_time = time.time()
        
        # 仿真循环
        for t in range(self.total_hours):
            # Interactive Mode: Slow down for visualization
            if getattr(self, 'interactive', False):
                time.sleep(1.0) # 1 second per step

            # 检查场景切换
            for start_hour, instruction in script:
                if t == start_hour:
                    current_instruction = instruction
                    self.logger.info(f"\n{'='*50}")
                    self.logger.info(f"[时间 {t}h] 新指令: {current_instruction}")
                    self.logger.info(f"{'='*50}\n")
                    break
            
            # 语义解释
            config, confidence = self.brain.interpret(current_instruction)
            
            if t % 10 == 0:  # 定期记录
                self.logger.log_scenario_change(t, current_instruction, config)
            
            # MPC求解
            current_level = self.physics.get_level()
            q_out_forecast = demands[t : t + self.solver.N]
            
            solve_start = time.time()
            q_in_cmd = self.solver.solve(
                current_level=current_level,
                q_prev=last_control_action,
                q_out_forecast=q_out_forecast,
                config=config
            )
            solve_time = time.time() - solve_start
            
            if t % 10 == 0:
                self.logger.log_optimization_result("optimal", solve_time=solve_time)
            
            # 物理仿真
            q_out_actual = demands[t]
            next_level = self.physics.step(q_in_command=q_in_cmd, q_out=q_out_actual)
            
            # 监控检查
            alerts = self.monitor.check_state(t, next_level, q_in_cmd, q_out_actual, config)
            
            # 记录数据
            self.history['time'].append(t)
            self.history['level'].append(current_level)
            self.history['q_in'].append(q_in_cmd)
            self.history['q_out'].append(q_out_actual)
            self.history['target_level'].append(config['Z_ref'])
            self.history['instruction'].append(current_instruction)
            self.history['config'].append(config)
            self.history['confidence'].append(confidence)
            
            # 保存到数据库
            if self.db and self.simulation_id:
                self.db.save_state(
                    self.simulation_id, t, current_level,
                    q_in_cmd, q_out_actual, config['Z_ref'],
                    current_instruction, config
                )
                
                # 保存告警
                for alert in alerts:
                    self.db.save_alert(
                        self.simulation_id, t,
                        alert.level.name,
                        alert.alert_type,
                        alert.message,
                        alert.data
                    )
            
            last_control_action = q_in_cmd
        
        elapsed = time.time() - start_time
        
        # 完成仿真
        if self.db and self.simulation_id:
            self.db.finish_simulation(self.simulation_id)
        
        self.logger.info(f"\n仿真完成!")
        self.logger.info(f"总耗时: {elapsed:.2f}秒")
        self.logger.info(f"平均每步: {elapsed/self.total_hours*1000:.1f}ms")
        
        # 生成报告
        self._generate_report(script)
        
        # 可视化
        self._plot_results(script)
        self._create_animation(script)
        
        self.logger.info("\n所有产物已生成完毕")
    
    def _generate_report(self, script: List[tuple]):
        """生成仿真报告"""
        self.logger.info("生成仿真报告...")
        
        with open("simulation_report_enhanced.md", "w", encoding='utf-8') as f:
            f.write("# 智能闸门仿真报告（增强版）\n\n")
            f.write(f"**生成时间:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 系统配置
            f.write("## 系统配置\n\n")
            f.write(f"- 仿真时长: {self.total_hours} 小时\n")
            f.write(f"- 时间步长: {self.dt} 秒\n")
            f.write(f"- 渠池面积: {self.area} m²\n")
            f.write(f"- 系统延迟: {self.delay_steps} 步\n\n")
            
            # 按阶段分析
            f.write("## 阶段分析\n\n")
            
            current_phase_start = 0
            current_instr = self.history['instruction'][0]
            
            for t in range(1, len(self.history['time']) + 1):
                if t >= len(self.history['time']) or self.history['instruction'][t] != current_instr:
                    # 阶段结束
                    end = t
                    phase_levels = self.history['level'][current_phase_start:end]
                    phase_targets = self.history['target_level'][current_phase_start:end]
                    phase_qin = self.history['q_in'][current_phase_start:end]
                    
                    avg_dev = np.mean(np.abs(np.array(phase_levels) - np.array(phase_targets)))
                    max_dev = np.max(np.abs(np.array(phase_levels) - np.array(phase_targets)))
                    avg_qin = np.mean(phase_qin)
                    max_qin = np.max(phase_qin)
                    
                    f.write(f"### 阶段: {current_phase_start}h - {end}h\n")
                    f.write(f"**指令:** {current_instr}\n\n")
                    f.write(f"- **平均水位偏差:** {avg_dev:.4f} m\n")
                    f.write(f"- **最大水位偏差:** {max_dev:.4f} m\n")
                    f.write(f"- **平均入流:** {avg_qin:.2f} m³/s\n")
                    f.write(f"- **最大入流:** {max_qin:.2f} m³/s\n\n")
                    
                    if t < len(self.history['time']):
                        current_phase_start = t
                        current_instr = self.history['instruction'][t]
            
            # 监控统计
            f.write("## 监控统计\n\n")
            stats = self.monitor.get_statistics()
            f.write(f"- **总告警数:** {stats['total_alerts']}\n")
            f.write(f"- **警告数:** {stats['warning_count']}\n")
            f.write(f"- **严重告警数:** {stats['critical_count']}\n\n")
            
            # 告警详情
            if stats['total_alerts'] > 0:
                f.write("### 告警详情\n\n")
                for alert in self.monitor.get_alerts(limit=20):
                    f.write(f"- [{alert.level.value}] T={alert.data.get('time_step', '?')}h: {alert.message}\n")
        
        self.logger.info("报告已保存: simulation_report_enhanced.md")
    
    def _plot_results(self, script: List[tuple]):
        """绘制静态结果图"""
        self.logger.info("生成可视化图表...")
        
        t = self.history['time']
        z = self.history['level']
        z_ref = self.history['target_level']
        qin = self.history['q_in']
        qout = self.history['q_out']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 水位图
        ax1.plot(t, z, 'b-', label='实际水位', linewidth=2)
        ax1.plot(t, z_ref, 'r--', label='目标水位', linewidth=2, alpha=0.7)
        ax1.axhline(y=self.monitor.level_warning_high, color='orange', linestyle=':', alpha=0.5, label='警戒线')
        ax1.axhline(y=self.monitor.level_warning_low, color='orange', linestyle=':', alpha=0.5)
        ax1.set_ylabel('水位 (m)', fontsize=12)
        ax1.set_title('智能闸门控制系统 - 水位控制', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # 标注场景切换
        for start, instr in script:
            ax1.axvline(x=start, color='k', linestyle=':', alpha=0.5)
            ax1.text(start + 0.5, ax1.get_ylim()[1]*0.95, f"{start}h", rotation=90, fontsize=9)
        
        # 流量图
        ax2.plot(t, qin, 'g-', label='入流（控制）', linewidth=2)
        ax2.plot(t, qout, 'k:', label='出流（需求）', alpha=0.6)
        ax2.set_ylabel('流量 (m³/s)', fontsize=12)
        ax2.set_xlabel('时间 (小时)', fontsize=12)
        ax2.set_title('闸门控制动作', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        for start, instr in script:
            ax2.axvline(x=start, color='k', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('simulation_result_enhanced.png', dpi=150)
        plt.close()

        self.logger.info("图表已保存: simulation_result_enhanced.png")
    
    def _create_animation(self, script: List[tuple]):
        """创建动画"""
        self.logger.info("生成动画...")
        
        fig = plt.figure(figsize=(12, 9))
        gs = GridSpec(3, 1, height_ratios=[1, 1, 0.2])
        
        ax_level = fig.add_subplot(gs[0])
        ax_flow = fig.add_subplot(gs[1])
        ax_text = fig.add_subplot(gs[2])
        ax_text.axis('off')
        
        line_level, = ax_level.plot([], [], 'b-', lw=2, label='水位')
        line_target, = ax_level.plot([], [], 'r--', lw=2, label='目标')
        line_qin, = ax_flow.plot([], [], 'g-', lw=2, label='入流')
        line_qout, = ax_flow.plot([], [], 'k:', lw=1, label='出流')
        
        point_level, = ax_level.plot([], [], 'bo', markersize=8)
        point_qin, = ax_flow.plot([], [], 'go', markersize=8)
        
        text_instr = ax_text.text(0.5, 0.5, "", ha='center', va='center', fontsize=11)
        
        ax_level.set_xlim(0, len(self.history['time']))
        ax_level.set_ylim(0, 10)
        ax_level.set_ylabel("水位 (m)")
        ax_level.legend(loc='upper right')
        ax_level.grid(True, alpha=0.3)
        ax_level.set_title("智能闸门仿真（增强版）")
        
        ax_flow.set_xlim(0, len(self.history['time']))
        ax_flow.set_ylim(0, 25)
        ax_flow.set_ylabel("流量 (m³/s)")
        ax_flow.set_xlabel("时间 (小时)")
        ax_flow.legend(loc='upper right')
        ax_flow.grid(True, alpha=0.3)
        
        def init():
            line_level.set_data([], [])
            line_target.set_data([], [])
            line_qin.set_data([], [])
            line_qout.set_data([], [])
            point_level.set_data([], [])
            point_qin.set_data([], [])
            text_instr.set_text("")
            return line_level, line_target, line_qin, line_qout, point_level, point_qin, text_instr
        
        def update(frame):
            times = self.history['time'][:frame+1]
            levels = self.history['level'][:frame+1]
            targets = self.history['target_level'][:frame+1]
            qins = self.history['q_in'][:frame+1]
            qouts = self.history['q_out'][:frame+1]
            
            line_level.set_data(times, levels)
            line_target.set_data(times, targets)
            line_qin.set_data(times, qins)
            line_qout.set_data(times, qouts)
            
            if frame < len(self.history['time']):
                point_level.set_data([times[-1]], [levels[-1]])
                point_qin.set_data([times[-1]], [qins[-1]])
                
                current_instr = self.history['instruction'][frame]
                confidence = self.history['confidence'][frame]
                text_instr.set_text(
                    f"时间: {frame}h | 置信度: {confidence:.2f}\n"
                    f"指令: {current_instr[:40]}..."
                )
            
            return line_level, line_target, line_qin, line_qout, point_level, point_qin, text_instr
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(self.history['time']),
            init_func=init, blit=False,
            interval=self.config.get('visualization.animation_interval', 100)
        )
        
        try:
            fps = self.config.get('visualization.animation_fps', 10)
            ani.save('simulation_enhanced.gif', writer='pillow', fps=fps)
            self.logger.info("动画已保存: simulation_enhanced.gif")
        except Exception as e:
            self.logger.error(f"动画保存失败: {e}")
        
        plt.close()


def main():
    """主函数"""
    # 创建仿真实例
    sim = SmartPoolSimulation("config.yaml")
    
    # 运行仿真
    sim.run()
    
    # 关闭数据库
    if sim.db:
        sim.db.close()


if __name__ == "__main__":
    main()
