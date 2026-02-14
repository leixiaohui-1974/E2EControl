"""
Cognitive Layer: Semantic Interpreter.

Translates natural language instructions (Chinese) into mathematical
weights, constraints, and targets for the MPC controller using a
keyword-based approach with numerical extraction and modifier logic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


# Default MPC configuration values
DEFAULT_W_LEVEL: float = 10.0     # Level-tracking weight
DEFAULT_W_SMOOTH: float = 5.0     # Smoothness weight
DEFAULT_Z_REF: float = 3.0        # Reference water level (m)
DEFAULT_DELTA_Q_MAX: float = 2.0  # Max flow change per step (m^3/s)


class SemanticInterpreter:
    """Translates natural language instructions into MPC configuration.

    Uses a keyword-matching approach to modify default control
    parameters based on Chinese language instructions about water
    management scenarios.
    """

    def __init__(self) -> None:
        self.default_config: Dict[str, Any] = {
            'W_level': DEFAULT_W_LEVEL,
            'W_smooth': DEFAULT_W_SMOOTH,
            'Z_ref': DEFAULT_Z_REF,
            'delta_Q_max': DEFAULT_DELTA_Q_MAX,
            'constraints': {},
        }

        # Keyword -> effects mapping.
        # Each keyword can trigger multiple parameter modifications.
        self.keyword_map: Dict[str, List[Dict[str, Any]]] = {
            # Keywords affecting level tracking (W_level)
            '紧急': [{'param': 'W_level', 'action': 'multiply', 'value': 10.0}],
            '暴雨': [{'param': 'W_level', 'action': 'multiply', 'value': 10.0}],
            '安全第一': [{'param': 'W_level', 'action': 'multiply', 'value': 5.0}],
            '重要': [{'param': 'W_level', 'action': 'multiply', 'value': 2.0}],

            # Keywords affecting smoothness (W_smooth)
            '平稳': [
                {'param': 'W_smooth', 'action': 'multiply', 'value': 8.0},
                {'param': 'W_level', 'action': 'multiply', 'value': 1.5},
            ],
            '稳定': [
                {'param': 'W_smooth', 'action': 'multiply', 'value': 8.0},
                {'param': 'W_level', 'action': 'multiply', 'value': 1.5},
            ],
            '小心': [{'param': 'W_smooth', 'action': 'multiply', 'value': 5.0}],

            # Keywords affecting control action limits (delta_Q_max)
            '立刻': [{'param': 'delta_Q_max', 'action': 'multiply', 'value': 2.5}],
            '快速': [{'param': 'delta_Q_max', 'action': 'multiply', 'value': 2.0}],
            '缓慢': [{'param': 'delta_Q_max', 'action': 'multiply', 'value': 0.5}],

            # Keywords setting specific states or constraints
            '降低水位': [{'param': 'Z_ref', 'action': 'set', 'value': 2.0}],
            '提升水位': [{'param': 'Z_ref', 'action': 'set', 'value': 4.0}],
            '切断': [{'param': 'constraints', 'action': 'set', 'value': {'Q_in_max': 0.0, 'Z_min': -10.0}}],
            '污染': [{'param': 'constraints', 'action': 'set', 'value': {'Q_in_max': 0.0, 'Z_min': -10.0}}],
        }

        self.negations: List[str] = ['不要', '避免', '禁止']

    def interpret(self, instruction: str) -> Dict[str, Any]:
        """Interpret a natural language instruction into MPC config.

        Args:
            instruction: Chinese text describing the desired scenario.

        Returns:
            Dictionary of MPC parameters (W_level, W_smooth, Z_ref,
            delta_Q_max, constraints).
        """
        print(f"[Brain] Receiving instruction: {instruction}")

        config: Dict[str, Any] = self.default_config.copy()
        config['constraints'] = self.default_config['constraints'].copy()

        config = self._extract_numerical_values(instruction, config)
        config = self._apply_keyword_modifiers(instruction, config)

        print(f"[Brain] Interpreted as: {config}")
        return config

    def _extract_numerical_values(
        self, instruction: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract explicit numerical values from the instruction."""
        level_match = re.search(
            r'(?:level|水位)\s*([+-]?\d+\.?\d*)', instruction, re.IGNORECASE
        )
        if level_match:
            try:
                val = float(level_match.group(1))
                config['Z_ref'] = val
                print(f"[Brain] Extracted numerical value for Z_ref: {val}")
            except ValueError:
                pass
        return config

    def _apply_keyword_modifiers(
        self, instruction: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply keyword-based modifiers to the configuration."""
        if "严禁扰动" in instruction:
            config['W_smooth'] *= 100.0
            instruction = instruction.replace("严禁扰动", "")

        is_negated = any(neg in instruction for neg in self.negations)

        for keyword, effects in self.keyword_map.items():
            if keyword in instruction:
                for effect in effects:
                    param = effect['param']
                    action = effect['action']
                    value = effect['value']

                    if is_negated and action == 'multiply' and value != 0:
                        value = 1 / value
                    elif is_negated and action == 'set':
                        continue

                    if action == 'multiply' and param in config:
                        config[param] *= value
                    elif action == 'set':
                        if param == 'constraints':
                            config[param].update(value)
                        else:
                            config[param] = value
        return config
