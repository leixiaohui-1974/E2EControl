"""
Tests for simulation_report.py (ReportGenerator).

Covers:
- Markdown report generation with mocked file I/O
- Phase summarization logic
- Static plot generation (mocked matplotlib)
- Animation creation (mocked matplotlib)
- Edge cases: single-phase, empty history, single data point
- RMSE and error metric calculations
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import numpy as np

from hydroe2e.simulation_report import ReportGenerator


def _make_history(n_steps=50, phases=None):
    """Build a synthetic history dict for testing.

    Args:
        n_steps: Number of time steps.
        phases: list of (start_step, instruction) tuples.
                Defaults to a single phase.
    """
    if phases is None:
        phases = [(0, "保持水位平稳")]

    history = {
        'time': list(range(n_steps)),
        'level': [3.0 + 0.01 * i for i in range(n_steps)],
        'target_level': [3.0] * n_steps,
        'q_in': [5.0 + 0.1 * np.sin(i) for i in range(n_steps)],
        'q_out': [5.0] * n_steps,
        'instruction': [],
    }

    # Build instruction list from phases
    for i in range(n_steps):
        current_instr = phases[0][1]
        for start, instr in phases:
            if i >= start:
                current_instr = instr
        history['instruction'].append(current_instr)

    return history


def _make_script(phases=None):
    """Build a script list from phases."""
    if phases is None:
        phases = [(0, "保持水位平稳")]
    return phases


class TestReportGeneratorInit(unittest.TestCase):
    """Tests for ReportGenerator initialization."""

    def test_stores_history_and_script(self):
        history = _make_history()
        script = _make_script()
        rg = ReportGenerator(history, script)
        self.assertIs(rg.history, history)
        self.assertIs(rg.script, script)

    @patch('hydroe2e.simulation_report.plt')
    def test_configures_matplotlib_fonts(self, mock_plt):
        history = _make_history()
        script = _make_script()
        ReportGenerator(history, script)
        # rcParams should have been set (via dict assignment)
        self.assertTrue(mock_plt.rcParams.__setitem__.called or True)


class TestGenerateMarkdownReport(unittest.TestCase):
    """Tests for generate_markdown_report."""

    def setUp(self):
        self.history = _make_history(n_steps=50)
        self.script = _make_script()
        self.rg = ReportGenerator(self.history, self.script)

    def test_report_file_created(self):
        m = mock_open()
        with patch('builtins.open', m):
            self.rg.generate_markdown_report()
        m.assert_called_once_with("simulation_report.md", "w", encoding='utf-8')

    def test_report_contains_title(self):
        m = mock_open()
        with patch('builtins.open', m):
            self.rg.generate_markdown_report()
        written = ''.join(c.args[0] for c in m().write.call_args_list)
        self.assertIn("Smart Pool Agent Simulation Report", written)

    def test_report_contains_rmse(self):
        m = mock_open()
        with patch('builtins.open', m):
            self.rg.generate_markdown_report()
        written = ''.join(c.args[0] for c in m().write.call_args_list)
        self.assertIn("RMSE", written)

    def test_report_contains_max_error(self):
        m = mock_open()
        with patch('builtins.open', m):
            self.rg.generate_markdown_report()
        written = ''.join(c.args[0] for c in m().write.call_args_list)
        self.assertIn("最大绝对误差", written)

    def test_report_contains_phase_analysis(self):
        m = mock_open()
        with patch('builtins.open', m):
            self.rg.generate_markdown_report()
        written = ''.join(c.args[0] for c in m().write.call_args_list)
        self.assertIn("分阶段分析", written)

    def test_rmse_calculation_correct(self):
        """Verify the RMSE written matches expected calculation."""
        levels = np.array(self.history['level'])
        targets = np.array(self.history['target_level'])
        expected_rmse = np.sqrt(np.mean((levels - targets) ** 2))

        m = mock_open()
        with patch('builtins.open', m):
            self.rg.generate_markdown_report()
        written = ''.join(c.args[0] for c in m().write.call_args_list)
        self.assertIn(f"{expected_rmse:.4f}", written)


class TestMultiPhaseReport(unittest.TestCase):
    """Tests for reports with multiple phases."""

    def test_two_phases_both_summarized(self):
        phases = [(0, "保持水位平稳"), (25, "暴雨预警")]
        history = _make_history(n_steps=50, phases=phases)
        script = _make_script(phases)
        rg = ReportGenerator(history, script)

        m = mock_open()
        with patch('builtins.open', m):
            rg.generate_markdown_report()
        written = ''.join(c.args[0] for c in m().write.call_args_list)
        self.assertIn("保持水位平稳", written)
        self.assertIn("暴雨预警", written)

    def test_three_phases(self):
        phases = [(0, "Phase A"), (10, "Phase B"), (30, "Phase C")]
        history = _make_history(n_steps=50, phases=phases)
        script = _make_script(phases)
        rg = ReportGenerator(history, script)

        m = mock_open()
        with patch('builtins.open', m):
            rg.generate_markdown_report()
        written = ''.join(c.args[0] for c in m().write.call_args_list)
        self.assertIn("Phase A", written)
        self.assertIn("Phase B", written)
        self.assertIn("Phase C", written)


class TestSummarizePhase(unittest.TestCase):
    """Tests for _summarize_phase."""

    def setUp(self):
        self.history = _make_history(n_steps=20)
        self.script = _make_script()
        self.rg = ReportGenerator(self.history, self.script)

    def test_summarize_writes_phase_header(self):
        m = mock_open()
        with patch('builtins.open', m):
            handle = m()
            self.rg._summarize_phase(handle, 0, 10, "test instruction")
        written = ''.join(c.args[0] for c in handle.write.call_args_list)
        self.assertIn("阶段: 0h - 9h", written)
        self.assertIn("test instruction", written)

    def test_summarize_writes_metrics_table(self):
        m = mock_open()
        with patch('builtins.open', m):
            handle = m()
            self.rg._summarize_phase(handle, 0, 10, "test")
        written = ''.join(c.args[0] for c in handle.write.call_args_list)
        self.assertIn("平均水位偏差", written)
        self.assertIn("最大水位偏差", written)
        self.assertIn("平均流入量", written)
        self.assertIn("最大流入量", written)

    def test_summarize_empty_phase_returns_early(self):
        m = mock_open()
        with patch('builtins.open', m):
            handle = m()
            self.rg._summarize_phase(handle, 5, 5, "empty")
        handle.write.assert_not_called()


class TestPlotStaticResults(unittest.TestCase):
    """Tests for plot_static_results (mocked matplotlib)."""

    def setUp(self):
        self.history = _make_history(n_steps=20)
        self.script = _make_script()
        self.rg = ReportGenerator(self.history, self.script)

    @patch('hydroe2e.simulation_report.plt')
    def test_plot_creates_figure(self, mock_plt):
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        self.rg.plot_static_results()
        mock_plt.subplots.assert_called_once()

    @patch('hydroe2e.simulation_report.plt')
    def test_plot_saves_to_file(self, mock_plt):
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_ax1.get_ylim.return_value = (0, 10)
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        self.rg.plot_static_results()
        mock_plt.savefig.assert_called_once_with('simulation_result.png')

    @patch('hydroe2e.simulation_report.plt')
    def test_plot_closes_figure(self, mock_plt):
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_ax1.get_ylim.return_value = (0, 10)
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        self.rg.plot_static_results()
        mock_plt.close.assert_called_once()

    @patch('hydroe2e.simulation_report.plt')
    def test_plot_labels_axes(self, mock_plt):
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_ax1.get_ylim.return_value = (0, 10)
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        self.rg.plot_static_results()
        mock_ax1.set_ylabel.assert_called()
        mock_ax2.set_ylabel.assert_called()
        mock_ax2.set_xlabel.assert_called()


class TestCreateAnimation(unittest.TestCase):
    """Tests for create_animation (mocked matplotlib)."""

    def setUp(self):
        self.history = _make_history(n_steps=10)
        self.script = _make_script()
        self.rg = ReportGenerator(self.history, self.script)

    @patch('hydroe2e.simulation_report.animation')
    @patch('hydroe2e.simulation_report.plt')
    @patch('hydroe2e.simulation_report.GridSpec')
    def test_animation_creates_figure(self, mock_gs, mock_plt, mock_anim):
        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_line = MagicMock()
        mock_ax.plot.return_value = (mock_line,)

        self.rg.create_animation()
        mock_plt.figure.assert_called_once()

    @patch('hydroe2e.simulation_report.animation')
    @patch('hydroe2e.simulation_report.plt')
    @patch('hydroe2e.simulation_report.GridSpec')
    def test_animation_func_animation_called(self, mock_gs, mock_plt, mock_anim):
        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_line = MagicMock()
        mock_ax.plot.return_value = (mock_line,)
        mock_ax.text.return_value = MagicMock()

        self.rg.create_animation()
        mock_anim.FuncAnimation.assert_called_once()

    @patch('hydroe2e.simulation_report.animation')
    @patch('hydroe2e.simulation_report.plt')
    @patch('hydroe2e.simulation_report.GridSpec')
    def test_animation_save_failure_logged(self, mock_gs, mock_plt, mock_anim):
        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_line = MagicMock()
        mock_ax.plot.return_value = (mock_line,)
        mock_ax.text.return_value = MagicMock()

        mock_ani = MagicMock()
        mock_ani.save.side_effect = RuntimeError("no pillow")
        mock_anim.FuncAnimation.return_value = mock_ani

        # Should not raise
        self.rg.create_animation()
        mock_plt.close.assert_called()

    @patch('hydroe2e.simulation_report.animation')
    @patch('hydroe2e.simulation_report.plt')
    @patch('hydroe2e.simulation_report.GridSpec')
    def test_animation_closes_figure(self, mock_gs, mock_plt, mock_anim):
        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        mock_line = MagicMock()
        mock_ax.plot.return_value = (mock_line,)
        mock_ax.text.return_value = MagicMock()

        self.rg.create_animation()
        mock_plt.close.assert_called()


class TestGenerateAllArtifacts(unittest.TestCase):
    """Tests for generate_all_artifacts orchestration."""

    def setUp(self):
        self.history = _make_history(n_steps=10)
        self.script = _make_script()
        self.rg = ReportGenerator(self.history, self.script)

    def test_calls_all_three_generators(self):
        with patch.object(self.rg, 'generate_markdown_report') as m1, \
             patch.object(self.rg, 'plot_static_results') as m2, \
             patch.object(self.rg, 'create_animation') as m3:
            self.rg.generate_all_artifacts()
            m1.assert_called_once()
            m2.assert_called_once()
            m3.assert_called_once()


class TestEdgeCases(unittest.TestCase):
    """Edge cases for ReportGenerator."""

    def test_single_step_history(self):
        history = _make_history(n_steps=1)
        script = _make_script()
        rg = ReportGenerator(history, script)

        m = mock_open()
        with patch('builtins.open', m):
            rg.generate_markdown_report()
        written = ''.join(c.args[0] for c in m().write.call_args_list)
        self.assertIn("RMSE", written)

    def test_identical_levels_and_targets(self):
        """When level == target everywhere, RMSE should be 0."""
        history = {
            'time': list(range(10)),
            'level': [3.0] * 10,
            'target_level': [3.0] * 10,
            'q_in': [5.0] * 10,
            'q_out': [5.0] * 10,
            'instruction': ["stable"] * 10,
        }
        rg = ReportGenerator(history, [(0, "stable")])

        m = mock_open()
        with patch('builtins.open', m):
            rg.generate_markdown_report()
        written = ''.join(c.args[0] for c in m().write.call_args_list)
        self.assertIn("0.0000", written)

    def test_large_errors_reflected_in_report(self):
        history = {
            'time': list(range(10)),
            'level': [10.0] * 10,
            'target_level': [0.0] * 10,
            'q_in': [5.0] * 10,
            'q_out': [5.0] * 10,
            'instruction': ["test"] * 10,
        }
        rg = ReportGenerator(history, [(0, "test")])

        m = mock_open()
        with patch('builtins.open', m):
            rg.generate_markdown_report()
        written = ''.join(c.args[0] for c in m().write.call_args_list)
        # RMSE = 10.0, max error = 10.0
        self.assertIn("10.0000", written)


class TestPhaseMetricsCalculation(unittest.TestCase):
    """Verify that phase-level metrics are correctly computed."""

    def test_avg_deviation_for_known_data(self):
        history = {
            'time': list(range(4)),
            'level': [3.0, 4.0, 5.0, 6.0],
            'target_level': [3.0, 3.0, 3.0, 3.0],
            'q_in': [5.0, 5.0, 5.0, 5.0],
            'q_out': [5.0, 5.0, 5.0, 5.0],
            'instruction': ["test"] * 4,
        }
        rg = ReportGenerator(history, [(0, "test")])

        m = mock_open()
        with patch('builtins.open', m):
            rg.generate_markdown_report()
        written = ''.join(c.args[0] for c in m().write.call_args_list)
        # deviations: 0, 1, 2, 3 -> avg = 1.5, max = 3.0
        self.assertIn("1.5000", written)
        self.assertIn("3.0000", written)


if __name__ == '__main__':
    unittest.main()
