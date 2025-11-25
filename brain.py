import re

class SemanticInterpreter:
    """
    Cognitive Layer: Semantic Interpreter (Enhanced).
    Translates natural language instructions into mathematical weights, constraints, and targets
    using a keyword-based approach with numerical extraction and modifier logic.
    """
    def __init__(self):
        self.default_config = {
            'W_level': 10.0,
            'W_smooth': 5.0,
            'Z_ref': 3.0,
            'delta_Q_max': 2.0,
            'constraints': {}
        }

        # Structure: keyword -> list of effects. Allows one keyword to have multiple effects.
        self.keyword_map = {
            # Keywords affecting level tracking (W_level)
            '紧急': [{'param': 'W_level', 'action': 'multiply', 'value': 10.0}],
            '暴雨': [{'param': 'W_level', 'action': 'multiply', 'value': 10.0}],
            '安全第一': [{'param': 'W_level', 'action': 'multiply', 'value': 5.0}],
            '重要': [{'param': 'W_level', 'action': 'multiply', 'value': 2.0}],

            # Keywords affecting smoothness (W_smooth)
            '平稳': [
                {'param': 'W_smooth', 'action': 'multiply', 'value': 8.0},
                {'param': 'W_level', 'action': 'multiply', 'value': 1.5}
            ],
            '稳定': [
                {'param': 'W_smooth', 'action': 'multiply', 'value': 8.0},
                {'param': 'W_level', 'action': 'multiply', 'value': 1.5}
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
            '污染': [{'param': 'constraints', 'action': 'set', 'value': {'Q_in_max': 0.0, 'Z_min': -10.0}}]
        }

        self.negations = ['不要', '避免', '禁止']

    def interpret(self, instruction: str) -> dict:
        print(f"[Brain] Receiving instruction: {instruction}")

        config = self.default_config.copy()
        config['constraints'] = self.default_config['constraints'].copy()

        config = self._extract_numerical_values(instruction, config)
        config = self._apply_keyword_modifiers(instruction, config)

        print(f"[Brain] Interpreted as: {config}")
        return config

    def _extract_numerical_values(self, instruction: str, config: dict) -> dict:
        level_match = re.search(r'(?:level|水位)\s*([+-]?\d+\.?\d*)', instruction, re.IGNORECASE)
        if level_match:
            try:
                val = float(level_match.group(1))
                config['Z_ref'] = val
                print(f"[Brain] Extracted numerical value for Z_ref: {val}")
            except ValueError:
                pass
        return config

    def _apply_keyword_modifiers(self, instruction: str, config: dict) -> dict:
        if "严禁扰动" in instruction:
            config['W_smooth'] *= 100.0
            instruction = instruction.replace("严禁扰动", "")

        is_negated = any(neg in instruction for neg in self.negations)

        for keyword, effects in self.keyword_map.items():
            if keyword in instruction:
                for effect in effects:
                    param, action, value = effect['param'], effect['action'], effect['value']

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
