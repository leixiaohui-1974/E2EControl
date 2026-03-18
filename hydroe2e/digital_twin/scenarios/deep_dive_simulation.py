"""
六重内卷场景 - 数字孪生深度仿真
The "Deep-Dive" Scenario: 100 steps of intensive physical-information-control coupling
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from typing import Dict, List

from hydroe2e.digital_twin.physics.single_channel_fidelity import SingleChannelFidelity, ChannelGeometry
from hydroe2e.digital_twin.perception.intelligent_observer import IntelligentObserver
from hydroe2e.digital_twin.control.single_pool_admm import SinglePoolADMM


class DeepDiveSimulation:
    """六重内卷场景仿真"""
    
    def __init__(self):
        """初始化仿真系统"""
        # 物理层
        self.geom = ChannelGeometry(length=20000.0, N=20)
        self.physical = SingleChannelFidelity(self.geom, dt=60.0)
        
        # 感知层
        self.observer = IntelligentObserver(self.physical)
        
        # 控制层
        self.controller = SinglePoolADMM(N=20, horizon=10, dt=60.0)
        
        # 仿真参数
        self.T_total = 100
        self.current_step = 0
        
        # 场景阶段定义
        self.scenarios = {
            'phase1': (0, 20, '参数漂移'),
            'phase2': (20, 40, '经济调度'),
            'phase3': (40, 60, '网络攻击'),
            'phase4': (60, 80, '突发污染'),
            'phase5': (80, 100, '边坡危机')
        }
        
        # 记录
        self.history = {
            'Z_spatial': [],  # 空间水位分布
            'Q_spatial': [],
            'C_spatial': [],
            'n_spatial': [],
            'n_estimated': [],
            'measured_Z_mid': [],  # 中游测量值
            'cleaned_Z_mid': [],   # 清洗后值
            'slope_safety': [],
            'mode': [],
            'u_in': [],
            'u_out': [],
            'energy_cost': [],
            'electricity_price': [],
            'risk_level': [],
            'debris_eta': []
        }
    
    def run(self):
        """运行完整仿真"""
        logger.info("\n" + "="*80)
        logger.info(" "*15 + "🌊 单渠池数字孪生深度仿真 🌊")
        logger.info(" "*20 + "六重内卷场景 (T=100步)")
        logger.info("="*80)
        
        for t in range(self.T_total):
            self.current_step = t
            
            # 判断当前阶段并执行相应场景
            self._execute_scenario_stage(t)
            
            # 获取测量状态
            measured_state = self.physical.get_measured_state()
            
            # 感知层分析
            cleaned_state, risk, constraints = self.observer.observe_and_analyze(
                measured_state, t
            )
            
            # 更新控制器约束
            self.controller.update_dynamic_constraints(constraints)
            
            # ADMM求解
            target_Z = np.ones(self.geom.N) * 3.0  # 目标水位
            current_hour = (8 + t // 4) % 24  # 模拟24小时循环
            
            (u_in, u_out), debug_info = self.controller.solve(
                cleaned_state.Z,
                cleaned_state.Q,
                target_Z,
                current_hour
            )
            
            # 物理演化
            pollution_source = self._get_pollution_source(t)
            self.physical.step(u_in, u_out, pollution_source)
            
            # 记录数据
            self._record_history(measured_state, cleaned_state, risk, 
                                debug_info, current_hour)
            
            # 打印关键信息
            if t % 10 == 0 or self._is_critical_moment(t):
                self._print_status(t, risk, debug_info)
        
        logger.info("\n" + "="*80)
        logger.info("✅ 仿真完成！")
        logger.info("="*80)
    
    def _execute_scenario_stage(self, t: int):
        """执行场景阶段"""
        # Phase 1: T=0-20 参数漂移
        if t == 10:
            logger.info(f"\n{'='*80}")
            logger.info(f"📍 阶段1 (T={t}): 参数漂移 - 中游水草疯长")
            logger.info(f"{'='*80}")
            self.physical.grow_vegetation(start_seg=9, end_seg=14, growth_factor=1.8)
        
        # Phase 2: T=20-40 经济调度
        if t == 20:
            logger.info(f"\n{'='*80}")
            logger.info(f"📍 阶段2 (T={t}): 经济调度 - 低谷电价期，蓄能运行")
            logger.info(f"{'='*80}")
        
        # Phase 3: T=40-60 网络攻击
        if t == 40:
            logger.info(f"\n{'='*80}")
            logger.info(f"📍 阶段3 (T={t}): 网络攻击 - 黑客篡改中游传感器")
            logger.info(f"{'='*80}")
            self.physical.inject_attack(active=True, bias=-0.5)
        
        if t == 60:
            # 攻击结束
            self.physical.inject_attack(active=False)
        
        # Phase 4: T=60-80 突发污染
        if t == 60:
            logger.info(f"\n{'='*80}")
            logger.info(f"📍 阶段4 (T={t}): 突发污染 + 漂浮物")
            logger.info(f"{'='*80}")
            # 注入漂浮物
            self.observer.inject_debris(position=0.0)
        
        # Phase 5: T=80-100 边坡危机
        if t == 80:
            logger.info(f"\n{'='*80}")
            logger.info(f"📍 阶段5 (T={t}): 边坡危机 - 暴雨后急剧退水")
            logger.info(f"{'='*80}")
            # 模拟暴雨：增大侧向入流
            lateral = np.zeros(self.geom.N)
            lateral[10:15] = 15.0  # 中游暴雨
            self.physical.set_lateral_inflow(lateral)
    
    def _get_pollution_source(self, t: int) -> tuple:
        """获取污染源"""
        # 在T=60-80期间注入污染
        if 60 <= t < 80:
            return (0, 50.0)  # 上游污染源，负荷50 mg/L·m³/s
        return None
    
    def _is_critical_moment(self, t: int) -> bool:
        """判断是否是关键时刻"""
        critical_moments = [10, 20, 40, 60, 80]
        return t in critical_moments
    
    def _record_history(self, measured_state, cleaned_state, risk, debug_info, hour):
        """记录历史数据"""
        self.history['Z_spatial'].append(cleaned_state.Z.copy())
        self.history['Q_spatial'].append(cleaned_state.Q.copy())
        self.history['C_spatial'].append(cleaned_state.C.copy())
        self.history['n_spatial'].append(self.physical.state.n_roughness.copy())
        
        # 参数辨识结果
        id_result = self.observer.get_identification_result()
        self.history['n_estimated'].append(id_result['estimated_n'].copy())
        
        # 攻击监测
        mid_seg = 9
        self.history['measured_Z_mid'].append(measured_state.Z[mid_seg])
        self.history['cleaned_Z_mid'].append(cleaned_state.Z[mid_seg])
        
        # 边坡安全
        self.history['slope_safety'].append(self.physical.compute_slope_safety_factor().copy())
        
        # 运行模式
        self.history['mode'].append(self.observer.current_mode.value)
        
        # 控制量
        self.history['u_in'].append(debug_info['u_in_safe'])
        self.history['u_out'].append(debug_info['u_out_safe'])
        
        # 能耗
        self.history['energy_cost'].append(debug_info['energy_cost'])
        self.history['electricity_price'].append(debug_info['electricity_price'])
        
        # 风险
        self.history['risk_level'].append(risk.max_risk_level)
        self.history['debris_eta'].append(risk.debris_eta if risk.debris_eta > 0 else np.nan)
    
    def _print_status(self, t, risk, debug_info):
        """打印状态"""
        stats = self.physical.get_statistics()
        mode = self.observer.current_mode.value
        
        logger.info(f"\n[T={t:3d}] 状态报告:")
        logger.info(f"  物理: 平均水位={stats['mean_level']:.2f}m, "
              f"平均流量={stats['mean_flow']:.1f}m³/s")
        logger.info(f"  控制: u_in={debug_info['u_in_safe']:.1f}, "
              f"u_out={debug_info['u_out_safe']:.1f} m³/s")
        logger.info(f"  能耗: {debug_info['energy_cost']:.4f}元 "
              f"(电价{debug_info['electricity_price']:.2f}元/kWh)")
        logger.info(f"  风险: {risk.max_risk_level}")
        logger.info(f"  模式: {mode}")
        
        if risk.cyber_risk:
            logger.info(f"  🔴 网络攻击告警！")
        
        if risk.debris_eta > 0 and risk.debris_eta < 3600:
            logger.info(f"  ⚠️  漂浮物ETA: {risk.debris_eta:.0f}s")
        
        if np.max(risk.slope_risk) >= 2:
            critical_segs = np.where(risk.slope_risk >= 2)[0]
            logger.info(f"  ⚠️  边坡风险: 切片 {critical_segs + 1}")
    
    def visualize(self, save_path: str = 'digital_twin_dashboard.png'):
        """生成3x3可视化大屏"""
        logger.info(f"\n生成可视化大屏...")
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('🌊 单渠池数字孪生深度仿真 - 六重内卷场景分析', 
                     fontsize=20, fontweight='bold', y=0.995)
        
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3,
                     left=0.05, right=0.98, top=0.96, bottom=0.04)
        
        # 1. Spatial Profile (最终水位剖面)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_spatial_profile(ax1)
        
        # 2. Heatmap (时空热力图)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_heatmap(ax2)
        
        # 3. Roughness ID (粗糙度辨识)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_roughness_id(ax3)
        
        # 4. Attack Monitor (攻击监测)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_attack_monitor(ax4)
        
        # 5. Risk Curves (风险曲线)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_risk_curves(ax5)
        
        # 6. Visual Countdown (漂浮物倒计时)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_debris_countdown(ax6)
        
        # 7. Cost (累计能耗)
        ax7 = fig.add_subplot(gs[2, 0])
        self._plot_cost(ax7)
        
        # 8. Control Actions (控制动作)
        ax8 = fig.add_subplot(gs[2, 1])
        self._plot_control_actions(ax8)
        
        # 9. Mode Flags (模式标志)
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_mode_flags(ax9)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ 可视化大屏已保存: {save_path}")

        return fig
    
    def _plot_spatial_profile(self, ax):
        """1. 空间剖面图"""
        x = self.geom.x_positions() / 1000.0  # km
        z_bed = self.physical.z_bed
        z_surface = self.physical.get_surface_elevation()
        
        ax.fill_between(x, 0, z_bed, color='#8B4513', alpha=0.6, label='河床')
        ax.fill_between(x, z_bed, z_surface, color='#4A90E2', alpha=0.5, label='水体')
        ax.plot(x, z_surface, 'b-', linewidth=2, label='水面线')
        ax.plot(x, z_bed, 'k-', linewidth=1.5)
        
        # 标注水草区域
        ax.axvspan(9*1, 15*1, color='green', alpha=0.15, label='水草区')
        
        ax.set_xlabel('距离 [km]', fontsize=11, fontweight='bold')
        ax.set_ylabel('高程 [m]', fontsize=11, fontweight='bold')
        ax.set_title('① 最终空间剖面 (T=100)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_heatmap(self, ax):
        """2. 时空热力图"""
        Z_array = np.array(self.history['Z_spatial']).T
        
        im = ax.imshow(Z_array, aspect='auto', cmap='RdYlBu_r', origin='lower',
                      extent=[0, self.T_total, 0, 20])
        
        # 标注关键时刻
        for t, label in [(10, 'P1'), (20, 'P2'), (40, 'P3'), (60, 'P4'), (80, 'P5')]:
            ax.axvline(t, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.text(t, 19, label, color='white', fontweight='bold', 
                   ha='center', fontsize=9, bbox=dict(boxstyle='round', 
                   facecolor='black', alpha=0.6))
        
        ax.set_xlabel('时间步 [step]', fontsize=11, fontweight='bold')
        ax.set_ylabel('切片索引', fontsize=11, fontweight='bold')
        ax.set_title('② 水位时空演化热力图', fontsize=12, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('水位 [m]', fontsize=10)
    
    def _plot_roughness_id(self, ax):
        """3. 粗糙度辨识"""
        x = self.geom.x_positions() / 1000.0
        
        n_true_final = self.history['n_spatial'][-1]
        n_est_final = self.history['n_estimated'][-1]
        
        ax.plot(x, n_true_final, 'r-', linewidth=2.5, marker='o', 
               markersize=6, label='真实粗糙度')
        ax.plot(x, n_est_final, 'b--', linewidth=2, marker='s', 
               markersize=5, label='辨识粗糙度')
        
        # 标注水草区
        ax.axvspan(9*1, 15*1, color='green', alpha=0.1)
        
        ax.set_xlabel('距离 [km]', fontsize=11, fontweight='bold')
        ax.set_ylabel('曼宁系数 n', fontsize=11, fontweight='bold')
        ax.set_title('③ 自适应参数辨识 (T=100)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_attack_monitor(self, ax):
        """4. 攻击监测"""
        t = np.arange(len(self.history['measured_Z_mid']))
        
        ax.plot(t, self.history['measured_Z_mid'], 'r-', linewidth=2, 
               alpha=0.7, label='传感器实测值')
        ax.plot(t, self.history['cleaned_Z_mid'], 'g-', linewidth=2, 
               label='孪生清洗值')
        
        # 标注攻击区间
        ax.axvspan(40, 60, color='red', alpha=0.15, label='攻击期')
        ax.axvline(40, color='red', linestyle='--', alpha=0.6)
        ax.axvline(60, color='green', linestyle='--', alpha=0.6)
        
        ax.set_xlabel('时间步', fontsize=11, fontweight='bold')
        ax.set_ylabel('中游水位 [m]', fontsize=11, fontweight='bold')
        ax.set_title('④ 网络攻击监测 (切片10)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_curves(self, ax):
        """5. 风险曲线"""
        t = np.arange(len(self.history['slope_safety']))
        
        # 提取第13切片的边坡安全系数
        slope_safety_13 = np.array([fs[12] for fs in self.history['slope_safety']])
        
        # 提取最大污染浓度
        max_C = np.array([np.max(C) for C in self.history['C_spatial']])
        
        ax2 = ax.twinx()
        
        l1 = ax.plot(t, slope_safety_13, 'b-', linewidth=2, label='边坡安全系数 (切片13)')
        ax.axhline(1.5, color='orange', linestyle='--', alpha=0.6, label='警戒线')
        ax.axhline(1.2, color='red', linestyle='--', alpha=0.6, label='危险线')
        
        l2 = ax2.plot(t, max_C, 'g-', linewidth=2, alpha=0.7, label='最大污染浓度')
        
        # 标注边坡危机
        ax.axvspan(80, 100, color='red', alpha=0.1)
        
        ax.set_xlabel('时间步', fontsize=11, fontweight='bold')
        ax.set_ylabel('边坡安全系数', fontsize=11, fontweight='bold', color='b')
        ax2.set_ylabel('污染浓度 [mg/L]', fontsize=11, fontweight='bold', color='g')
        ax.set_title('⑤ 多维风险曲线', fontsize=12, fontweight='bold')
        
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_debris_countdown(self, ax):
        """6. 漂浮物倒计时"""
        t = np.arange(len(self.history['debris_eta']))
        eta = np.array(self.history['debris_eta'])
        
        # 只在有效范围内绘制
        valid_mask = ~np.isnan(eta)
        
        if np.any(valid_mask):
            ax.plot(t[valid_mask], eta[valid_mask] / 60.0, 'ro-', 
                   linewidth=2.5, markersize=6, label='漂浮物ETA')
            
            ax.fill_between(t[valid_mask], 0, eta[valid_mask] / 60.0, 
                           color='red', alpha=0.2)
        
        ax.axhline(10, color='orange', linestyle='--', alpha=0.6, label='预警阈值 (10min)')
        
        ax.set_xlabel('时间步', fontsize=11, fontweight='bold')
        ax.set_ylabel('到达时间 [分钟]', fontsize=11, fontweight='bold')
        ax.set_title('⑥ 漂浮物视觉跟踪', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_cost(self, ax):
        """7. 累计能耗"""
        t = np.arange(len(self.history['energy_cost']))
        cumulative_cost = np.cumsum(self.history['energy_cost'])
        
        ax.plot(t, cumulative_cost, 'b-', linewidth=2.5)
        ax.fill_between(t, 0, cumulative_cost, color='blue', alpha=0.2)
        
        # 标注低谷电价期
        ax.axvspan(20, 40, color='green', alpha=0.1, label='低谷电价期')
        
        ax.set_xlabel('时间步', fontsize=11, fontweight='bold')
        ax.set_ylabel('累计费用 [元]', fontsize=11, fontweight='bold')
        ax.set_title('⑦ 能耗成本累计', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 显示总费用
        total_cost = cumulative_cost[-1]
        ax.text(0.95, 0.95, f'总费用: {total_cost:.2f}元', 
               transform=ax.transAxes, ha='right', va='top',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    def _plot_control_actions(self, ax):
        """8. 控制动作"""
        t = np.arange(len(self.history['u_in']))
        
        ax.plot(t, self.history['u_in'], 'b-', linewidth=2, label='上游入流 $u_{in}$')
        ax.plot(t, self.history['u_out'], 'r-', linewidth=2, label='下游出流 $u_{out}$')
        
        # 标注关键时刻
        for t_key in [10, 20, 40, 60, 80]:
            ax.axvline(t_key, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('时间步', fontsize=11, fontweight='bold')
        ax.set_ylabel('流量 [m³/s]', fontsize=11, fontweight='bold')
        ax.set_title('⑧ 控制动作序列', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_mode_flags(self, ax):
        """9. 模式标志"""
        # 统计各模式的时间占比
        from collections import Counter
        mode_counts = Counter(self.history['mode'])
        
        modes = list(mode_counts.keys())
        counts = list(mode_counts.values())
        colors = ['green', 'blue', 'red', 'orange', 'purple'][:len(modes)]
        
        bars = ax.barh(modes, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{int(width)}步', ha='left', va='center', 
                   fontsize=9, fontweight='bold')
        
        ax.set_xlabel('运行时长 [步]', fontsize=11, fontweight='bold')
        ax.set_title('⑨ 运行模式统计', fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # 添加总结文本
        total_steps = sum(counts)
        summary = f"总步数: {total_steps}\n" + \
                 f"模式切换: {len(modes)}种"
        
        ax.text(0.98, 0.02, summary, transform=ax.transAxes,
               ha='right', va='bottom', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))


# 主执行
if __name__ == "__main__":
    # 创建并运行仿真
    simulation = DeepDiveSimulation()
    simulation.run()
    
    # 生成可视化
    simulation.visualize(save_path='/workspace/digital_twin_dashboard.png')
    
    logger.info("\n" + "="*80)
    logger.info("🎊 数字孪生深度仿真完成！")
    logger.info("="*80)
