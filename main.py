import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from brain import SemanticInterpreter
from physics import CanalPoolSimulator
from control import UniversalMPCSolver
import os

def main():
    # --- 1. Initialization ---
    TOTAL_HOURS = 50
    DT = 3600.0  # 1 hour steps
    AREA = 10000.0

    # Configure Matplotlib for Chinese Support (Windows/Linux compatibility)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False # Fix for minus sign display

    # Initialize Modules
    brain = SemanticInterpreter()
    physics = CanalPoolSimulator(area=AREA, dt=DT, delay_steps=1, initial_level=3.0)
    solver = UniversalMPCSolver(horizon=10, dt=DT, area=AREA, delay_steps=1)

    # --- 2. Scenario Script ---
    # Define timeline: (start_hour, instruction)
    # The instruction persists until the next one.
    script = [
        (0, "保持水位平稳，正常供水。"),
        (10, "收到暴雨预警，立刻降低水位腾出库容！安全第一！"),
        (20, "进入冰期输水模式，严禁扰动冰盖。"),
        (30, "下游检测到污染，紧急切断出流！"),
        (40, "保持水位平稳，正常供水。") # Restoration
    ]

    # --- 3. Simulation Loop ---
    # Data logging
    history = {
        'time': [],
        'level': [],
        'q_in': [],
        'q_out': [],
        'target_level': [],
        'instruction': [],
        'config': []
    }

    # External Demand Profile (Disturbance)
    # Assume a base demand that fluctuates slightly
    np.random.seed(42)
    base_demand = 5.0
    demands = base_demand + np.random.normal(0, 0.5, TOTAL_HOURS + 20) # +20 for forecast horizon

    # Initial Conditions
    current_instruction = script[0][1]
    last_control_action = 5.0 # Assume steady state start

    print("Starting Simulation...")

    for t in range(TOTAL_HOURS):
        # A. Check Script for new instruction
        for start_time, instruction in script:
            if t == start_time:
                current_instruction = instruction
                print(f"[Time {t}h] New Instruction: {current_instruction}")
                break

        # B. Brain: Interpret Instruction
        config = brain.interpret(current_instruction)

        # C. Solver: Calculate Optimal Control
        # Get forecast for next N steps
        q_out_forecast = demands[t : t + solver.N]

        current_level = physics.get_level()

        q_in_cmd = solver.solve(
            current_level=current_level,
            q_prev=last_control_action,
            q_out_forecast=q_out_forecast,
            config=config
        )

        # D. Physics: Execute Step
        # Actual demand at time t
        q_out_actual = demands[t]

        # Physics step
        next_level = physics.step(q_in_command=q_in_cmd, q_out=q_out_actual)

        # E. Log Data
        history['time'].append(t)
        history['level'].append(current_level)
        history['q_in'].append(q_in_cmd)
        history['q_out'].append(q_out_actual)
        history['target_level'].append(config['Z_ref'])
        history['instruction'].append(current_instruction)
        history['config'].append(config)

        last_control_action = q_in_cmd

    # --- 4. Report Generation ---
    generate_report(history)

    # --- 5. Visualization (Static & Dynamic) ---
    plot_results(history, script)
    create_animation(history, script)

    print("Simulation Complete. Artifacts generated.")

def generate_report(history):
    """Generates a markdown report of the simulation."""
    with open("simulation_report.md", "w", encoding='utf-8') as f:
        f.write("# Smart Pool Agent Simulation Report\n\n")

        # Analyze by phase
        phases = []
        # Reconstruct phases from script logic
        # Or just iterate history
        current_phase_start = 0
        current_instr = history['instruction'][0]

        # Helper to summarize a phase
        def summarize_phase(start, end, instr):
            phase_levels = history['level'][start:end]
            phase_targets = history['target_level'][start:end]
            phase_qin = history['q_in'][start:end]

            avg_dev = np.mean(np.abs(np.array(phase_levels) - np.array(phase_targets)))
            max_qin = np.max(phase_qin) if len(phase_qin) > 0 else 0
            avg_qin = np.mean(phase_qin) if len(phase_qin) > 0 else 0

            f.write(f"## Phase: {start}h - {end}h\n")
            f.write(f"**Instruction:** {instr}\n\n")
            f.write(f"- **Avg Level Deviation:** {avg_dev:.4f} m\n")
            f.write(f"- **Avg Inflow:** {avg_qin:.2f} m3/s\n")
            f.write(f"- **Max Inflow:** {max_qin:.2f} m3/s\n")
            f.write("\n")

        for t in range(1, len(history['time'])):
            if history['instruction'][t] != current_instr:
                summarize_phase(current_phase_start, t, current_instr)
                current_phase_start = t
                current_instr = history['instruction'][t]

        # Last phase
        summarize_phase(current_phase_start, len(history['time']), current_instr)

def plot_results(history, script):
    """Generates a static summary plot."""
    t = history['time']
    z = history['level']
    z_ref = history['target_level']
    qin = history['q_in']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Subplot 1: Water Level
    ax1.plot(t, z, 'b-', label='Actual Level', linewidth=2)
    ax1.plot(t, z_ref, 'r--', label='Target Level', linewidth=2)
    ax1.set_ylabel('Level (m)')
    ax1.set_title('Water Level Control')
    ax1.grid(True)
    ax1.legend()

    # Annotate phases
    for start, instr in script:
        ax1.axvline(x=start, color='k', linestyle=':', alpha=0.5)
        # Add text label slightly offset
        # Only show short label
        short_label = instr[:10] + "..."
        ax1.text(start + 0.5, ax1.get_ylim()[1]*0.95, str(start)+"h", rotation=90)

    # Subplot 2: Flow Rates
    ax2.plot(t, qin, 'g-', label='Inflow (Control)', linewidth=2)
    ax2.plot(t, history['q_out'], 'k:', label='Outflow (Demand)', alpha=0.6)
    ax2.set_ylabel('Flow (m3/s)')
    ax2.set_xlabel('Time (hours)')
    ax2.set_title('Gate Control Actions')
    ax2.grid(True)
    ax2.legend()

    for start, instr in script:
        ax2.axvline(x=start, color='k', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('simulation_result.png')
    plt.close()

def create_animation(history, script):
    """Generates a GIF animation of the simulation."""
    # Setup Figure
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 0.2])

    ax_level = fig.add_subplot(gs[0])
    ax_flow = fig.add_subplot(gs[1])
    ax_text = fig.add_subplot(gs[2])
    ax_text.axis('off')

    # Initialize Lines
    line_level, = ax_level.plot([], [], 'b-', lw=2, label='Level')
    line_target, = ax_level.plot([], [], 'r--', lw=2, label='Target')
    line_qin, = ax_flow.plot([], [], 'g-', lw=2, label='Inflow')
    line_qout, = ax_flow.plot([], [], 'k:', lw=1, label='Outflow')

    # Markers for current time
    point_level, = ax_level.plot([], [], 'bo', markersize=8)
    point_qin, = ax_flow.plot([], [], 'go', markersize=8)

    # Text instruction
    text_instr = ax_text.text(0.5, 0.5, "", ha='center', va='center', fontsize=12, wrap=True)

    # Limits
    ax_level.set_xlim(0, len(history['time']))
    ax_level.set_ylim(0, 10) # Fixed scale for consistent view
    ax_level.set_ylabel("Level (m)")
    ax_level.legend(loc='upper right')
    ax_level.grid(True)

    ax_flow.set_xlim(0, len(history['time']))
    ax_flow.set_ylim(0, 25) # Cap at slightly above max
    ax_flow.set_ylabel("Flow (m3/s)")
    ax_flow.legend(loc='upper right')
    ax_flow.grid(True)

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
        # Update data up to current frame
        times = history['time'][:frame+1]
        levels = history['level'][:frame+1]
        targets = history['target_level'][:frame+1]
        qins = history['q_in'][:frame+1]
        qouts = history['q_out'][:frame+1]

        line_level.set_data(times, levels)
        line_target.set_data(times, targets)
        line_qin.set_data(times, qins)
        line_qout.set_data(times, qouts)

        if frame < len(history['time']):
            point_level.set_data([times[-1]], [levels[-1]])
            point_qin.set_data([times[-1]], [qins[-1]])

            # Instruction text
            current_instr = history['instruction'][frame]
            text_instr.set_text(f"Time: {frame}h\nInstruction: {current_instr}")

        return line_level, line_target, line_qin, line_qout, point_level, point_qin, text_instr

    ani = animation.FuncAnimation(fig, update, frames=len(history['time']), init_func=init, blit=False, interval=100)

    # Save as GIF
    # Try using Pillow writer
    try:
        ani.save('simulation.gif', writer='pillow', fps=10)
        print("Animation saved as simulation.gif")
    except Exception as e:
        print(f"Could not save animation: {e}")

if __name__ == "__main__":
    main()
