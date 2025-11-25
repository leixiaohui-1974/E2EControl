import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

class ReportGenerator:
    def __init__(self, history, script):
        self.history = history
        self.script = script
        # Configure Matplotlib for Chinese Support
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

    def generate_all_artifacts(self):
        print("Generating report and visualizations...")
        self.generate_markdown_report()
        self.plot_static_results()
        self.create_animation()
        print("Artifacts generation complete.")

    def generate_markdown_report(self):
        """Generates a detailed markdown report of the simulation."""
        with open("simulation_report.md", "w", encoding='utf-8') as f:
            f.write("# Smart Pool Agent Simulation Report\n\n")

            # Overall Performance Metrics
            levels = np.array(self.history['level'])
            targets = np.array(self.history['target_level'])
            rmse = np.sqrt(np.mean((levels - targets)**2))
            max_abs_error = np.max(np.abs(levels - targets))

            f.write("## 总体性能指标\n")
            f.write(f"- **水位跟踪均方根误差 (RMSE):** {rmse:.4f} m\n")
            f.write(f"- **最大绝对误差:** {max_abs_error:.4f} m\n\n")

            # Phase-by-phase analysis
            f.write("## 分阶段分析\n")
            current_phase_start = 0
            current_instr = self.history['instruction'][0]

            for t in range(1, len(self.history['time'])):
                if self.history['instruction'][t] != current_instr:
                    self._summarize_phase(f, current_phase_start, t, current_instr)
                    current_phase_start = t
                    current_instr = self.history['instruction'][t]

            # Summarize the last phase
            self._summarize_phase(f, current_phase_start, len(self.history['time']), current_instr)

    def _summarize_phase(self, f, start, end, instr):
        phase_levels = np.array(self.history['level'][start:end])
        phase_targets = np.array(self.history['target_level'][start:end])
        phase_qin = np.array(self.history['q_in'][start:end])

        if len(phase_levels) == 0: return

        avg_dev = np.mean(np.abs(phase_levels - phase_targets))
        max_dev = np.max(np.abs(phase_levels - phase_targets))
        avg_qin = np.mean(phase_qin)
        max_qin = np.max(phase_qin)

        f.write(f"### 阶段: {start}h - {end-1}h\n")
        f.write(f"**指令:** `{instr}`\n\n")
        f.write("| 指标 | 数值 |\n")
        f.write("|:---|:---|\n")
        f.write(f"| 平均水位偏差 | {avg_dev:.4f} m |\n")
        f.write(f"| 最大水位偏差 | {max_dev:.4f} m |\n")
        f.write(f"| 平均流入量 | {avg_qin:.2f} m³/s |\n")
        f.write(f"| 最大流入量 | {max_qin:.2f} m³/s |\n\n")

    def plot_static_results(self):
        """Generates a static summary plot."""
        t = self.history['time']
        z = self.history['level']
        z_ref = self.history['target_level']
        qin = self.history['q_in']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.plot(t, z, 'b-', label='实际水位', linewidth=2)
        ax1.plot(t, z_ref, 'r--', label='目标水位', linewidth=2)
        ax1.set_ylabel('水位 (m)')
        ax1.set_title('水位控制性能')
        ax1.grid(True)
        ax1.legend()

        for start, instr in self.script:
            ax1.axvline(x=start, color='k', linestyle=':', alpha=0.5)
            ax1.text(start + 0.5, ax1.get_ylim()[1]*0.95, f"{start}h: {instr[:10]}...", rotation=90, verticalalignment='top')

        ax2.plot(t, qin, 'g-', label='流入量 (控制动作)', linewidth=2)
        ax2.plot(t, self.history['q_out'], 'k:', label='流出量 (需求)', alpha=0.6)
        ax2.set_ylabel('流量 (m³/s)')
        ax2.set_xlabel('时间 (小时)')
        ax2.set_title('闸门控制动作')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('simulation_result.png')
        plt.close()

    def create_animation(self):
        """Generates a GIF animation of the simulation."""
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(3, 1, height_ratios=[1, 1, 0.2])
        ax_level = fig.add_subplot(gs[0])
        ax_flow = fig.add_subplot(gs[1])
        ax_text = fig.add_subplot(gs[2])
        ax_text.axis('off')

        line_level, = ax_level.plot([], [], 'b-', lw=2, label='水位')
        line_target, = ax_level.plot([], [], 'r--', lw=2, label='目标')
        point_level, = ax_level.plot([], [], 'bo', markersize=8)

        line_qin, = ax_flow.plot([], [], 'g-', lw=2, label='流入量')
        line_qout, = ax_flow.plot([], [], 'k:', lw=1, label='流出量')
        point_qin, = ax_flow.plot([], [], 'go', markersize=8)

        text_instr = ax_text.text(0.5, 0.5, "", ha='center', va='center', fontsize=12, wrap=True)

        ax_level.set_xlim(0, len(self.history['time']))
        ax_level.set_ylim(min(self.history['level']) - 0.5, max(self.history['level']) + 0.5)
        ax_level.set_ylabel("水位 (m)")
        ax_level.legend(loc='upper right')
        ax_level.grid(True)

        ax_flow.set_xlim(0, len(self.history['time']))
        ax_flow.set_ylim(min(self.history['q_in']) - 2, max(self.history['q_in']) + 2)
        ax_flow.set_ylabel("流量 (m³/s)")
        ax_flow.legend(loc='upper right')
        ax_flow.grid(True)

        def init():
            # All lines and points to animate
            animated_elements = [line_level, line_target, point_level, line_qin, line_qout, point_qin, text_instr]
            for element in animated_elements:
                if isinstance(element, plt.Line2D):
                    element.set_data([], [])
                else:
                    element.set_text("")
            return animated_elements

        def update(frame):
            times = self.history['time'][:frame+1]
            line_level.set_data(times, self.history['level'][:frame+1])
            line_target.set_data(times, self.history['target_level'][:frame+1])
            point_level.set_data([times[-1]], [self.history['level'][frame]])

            line_qin.set_data(times, self.history['q_in'][:frame+1])
            line_qout.set_data(times, self.history['q_out'][:frame+1])
            point_qin.set_data([times[-1]], [self.history['q_in'][frame]])

            text_instr.set_text(f"时间: {frame}h\n指令: {self.history['instruction'][frame]}")

            return line_level, line_target, point_level, line_qin, line_qout, point_qin, text_instr

        ani = animation.FuncAnimation(fig, update, frames=len(self.history['time']), init_func=init, blit=False, interval=100)

        try:
            ani.save('simulation.gif', writer='pillow', fps=10)
        except Exception as e:
            print(f"无法保存动画: {e}")
        plt.close()
