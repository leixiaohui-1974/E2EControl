class SemanticInterpreter:
    """
    Cognitive Layer: Semantic Interpreter.

    Simulates the 'Intent Recognition' and 'Parameter Mapping' process of a Large Language Model.
    Translates natural language instructions into mathematical weights, constraints, and targets.
    """

    def __init__(self):
        # Default configuration (Baseline)
        self.default_config = {
            'W_level': 10.0,
            'W_smooth': 5.0,
            'Z_ref': 3.0,
            'delta_Q_max': 2.0,
            'constraints': {}
        }

        # Scenario Library: Mapping instructions to parameter updates
        self.scenario_map = {
            "保持水位平稳，正常供水。": {
                'W_level': 10.0,
                'W_smooth': 5.0,
                'Z_ref': 3.0,
                'delta_Q_max': 2.0,
                'constraints': {} # No special extra constraints
            },
            "收到暴雨预警，立刻降低水位腾出库容！安全第一！": {
                'W_level': 100.0,
                # W_smooth not specified, keep default or low?
                # Prompt says "allow drastic action", so maybe smooth is low?
                # Actually prompt only lists changes usually.
                # But Scenario A lists all.
                # I will assume W_smooth stays default (5.0) or I should infer.
                # "Allow drastic action" is covered by delta_Q_max = 5.0
                'W_smooth': 5.0,
                'Z_ref': 2.0,
                'delta_Q_max': 5.0,
                'constraints': {}
            },
            "进入冰期输水模式，严禁扰动冰盖。": {
                'W_level': 1.0,
                'W_smooth': 500.0,
                'delta_Q_max': 0.1,
                # Z_ref not specified, assume normal 3.0
                'Z_ref': 3.0,
                'constraints': {}
            },
            "下游检测到污染，紧急切断出流！": {
                # "Special handling"
                # Prompt says: Add constraint Q_in_max = 0
                # Other params? Maybe keep defaults?
                'W_level': 10.0,
                'W_smooth': 5.0,
                'Z_ref': 3.0,
                'delta_Q_max': 20.0, # Updated to allow immediate closure
                'constraints': {
                    'Q_in_max': 0.0,
                    'Z_min': -10.0 # Relax level constraint to allow draining
                }
            },
            "水位计读数异常，切换到开环保持模式。": {
                'W_level': 0.0,
                'W_smooth': 100.0,
                # Assume keeping current flow means high smooth weight.
                'Z_ref': 3.0, # Irrelevant since W_level is 0
                'delta_Q_max': 2.0,
                'constraints': {}
            }
        }

    def interpret(self, instruction: str) -> dict:
        """
        Translates a natural language instruction into a control configuration.

        Args:
            instruction (str): The natural language command.

        Returns:
            dict: Configuration dictionary containing weights, constraints, and targets.
        """
        print(f"[Brain] Receiving instruction: {instruction}")

        # Exact match lookup
        if instruction in self.scenario_map:
            config = self.scenario_map[instruction]
            # Ensure all keys exist by merging with default if necessary
            # (In this implementation, I tried to ensure all keys are present in the map,
            # but merging is safer)
            final_config = self.default_config.copy()
            final_config.update(config)

            # Print interpretation for debugging/demo
            print(f"[Brain] Interpreted as: {final_config}")
            return final_config
        else:
            print(f"[Brain] Unknown instruction. Using default configuration.")
            return self.default_config
