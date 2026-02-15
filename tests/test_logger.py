"""Tests for the logger module."""

import logging
import os
import tempfile

import pytest

from logger import (
    SmartPoolLogger,
    StructuredFormatter,
    get_logger,
    setup_logging,
)


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_format_info_includes_module(self):
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "hello" in output
        assert "test" in output  # module name

    def test_format_warning_omits_module(self):
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="test.py",
            lineno=1, msg="danger", args=(), exc_info=None,
        )
        output = formatter.format(record)
        assert "WARNING" in output
        assert "danger" in output

    def test_format_includes_timestamp(self):
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="ts check", args=(), exc_info=None,
        )
        output = formatter.format(record)
        # ISO timestamp starts with year
        assert "20" in output  # year like 2024, 2025, 2026...


class TestSmartPoolLogger:
    """Tests for SmartPoolLogger singleton."""

    def setup_method(self):
        # Reset singleton between tests
        SmartPoolLogger._instance = None

    def test_singleton(self):
        a = SmartPoolLogger()
        b = SmartPoolLogger()
        assert a is b

    def test_initial_level_is_debug(self):
        spl = SmartPoolLogger()
        assert spl.logger.level == logging.DEBUG

    def test_setup_changes_level(self):
        spl = SmartPoolLogger()
        spl.setup(level="WARNING", console_output=False)
        assert spl.logger.level == logging.WARNING

    def test_setup_console_handler(self):
        spl = SmartPoolLogger()
        spl.setup(level="INFO", console_output=True)
        handler_types = [type(h) for h in spl.logger.handlers]
        assert logging.StreamHandler in handler_types

    def test_setup_no_console(self):
        spl = SmartPoolLogger()
        spl.setup(level="INFO", console_output=False)
        stream_handlers = [
            h for h in spl.logger.handlers
            if type(h) is logging.StreamHandler
        ]
        assert len(stream_handlers) == 0

    def test_setup_file_handler(self):
        spl = SmartPoolLogger()
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = f.name
        try:
            spl.setup(level="DEBUG", console_output=False, log_file=log_path)
            from logging.handlers import RotatingFileHandler
            file_handlers = [
                h for h in spl.logger.handlers
                if isinstance(h, RotatingFileHandler)
            ]
            assert len(file_handlers) == 1
            # Write a log entry and verify it appears in the file
            spl.info("file test message")
            file_handlers[0].flush()
            with open(log_path) as fh:
                content = fh.read()
            assert "file test message" in content
        finally:
            # Close handlers before cleanup
            for h in spl.logger.handlers[:]:
                h.close()
                spl.logger.removeHandler(h)
            os.unlink(log_path)

    def test_log_levels(self):
        spl = SmartPoolLogger()
        spl.setup(level="DEBUG", console_output=False)
        # These should not raise
        spl.debug("d")
        spl.info("i")
        spl.warning("w")
        spl.error("e")
        spl.critical("c")

    def test_log_control_action(self):
        spl = SmartPoolLogger()
        spl.setup(level="DEBUG", console_output=False)
        # Should not raise
        spl.log_control_action(
            time_step=5, level=3.5, q_in=5.0,
            q_out=4.8, config={"Z_ref": 3.0},
        )

    def test_log_scenario_change(self):
        spl = SmartPoolLogger()
        spl.setup(level="DEBUG", console_output=False)
        spl.log_scenario_change(
            time_step=10, instruction="暴雨预警",
            config={"W_level": 100, "Z_ref": 2.0, "delta_Q_max": 1.0},
        )

    def test_log_optimization_result_optimal(self):
        spl = SmartPoolLogger()
        spl.setup(level="DEBUG", console_output=False)
        spl.log_optimization_result("optimal", cost=1.23, solve_time=0.05)

    def test_log_optimization_result_no_cost(self):
        spl = SmartPoolLogger()
        spl.setup(level="DEBUG", console_output=False)
        spl.log_optimization_result("optimal", solve_time=0.05)

    def test_log_optimization_result_minimal(self):
        spl = SmartPoolLogger()
        spl.setup(level="DEBUG", console_output=False)
        spl.log_optimization_result("optimal")

    def test_log_optimization_result_abnormal(self):
        spl = SmartPoolLogger()
        spl.setup(level="DEBUG", console_output=False)
        spl.log_optimization_result("infeasible")

    def test_log_alert(self):
        spl = SmartPoolLogger()
        spl.setup(level="DEBUG", console_output=False)
        spl.log_alert("水位偏高", "当前水位超过阈值", level=8.5)


class TestModuleFunctions:
    """Tests for module-level helper functions."""

    def setup_method(self):
        SmartPoolLogger._instance = None
        import logger as _mod
        _mod._logger_instance = None

    def test_get_logger_returns_singleton(self):
        a = get_logger()
        b = get_logger()
        assert a is b
        assert isinstance(a, SmartPoolLogger)

    def test_setup_logging(self):
        setup_logging({"level": "WARNING", "console_output": False})
        lg = get_logger()
        assert lg.logger.level == logging.WARNING

    def test_setup_logging_defaults(self):
        setup_logging({})
        lg = get_logger()
        assert lg.logger.level == logging.INFO
