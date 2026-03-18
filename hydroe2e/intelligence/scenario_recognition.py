import numpy as np
from typing import Dict, List, Any

class ScenarioRecognitionEngine:
    """
    Intelligence Layer: Scenario Recognition Engine.
    
    Analyzes system state to identify current operational scenario.
    Uses a hybrid approach: Rule-based + Statistical (Mock ML).
    """
    
    SCENARIO_TAXONOMY = {
        'NORMAL': {
            'description': '正常供水模式',
            'priority': 0,
            'config': {'W_level': 10.0, 'Z_ref': 3.0}
        },
        'FLOOD_CONTROL': {
            'description': '防洪模式',
            'priority': 10,
            'config': {'W_level': 100.0, 'Z_ref': 2.0} # Lower level for capacity
        },
        'DROUGHT': {
            'description': '干旱模式',
            'priority': 5,
            'config': {'W_level': 50.0, 'Z_ref': 3.5} # Higher level to conserve
        },
        'EMERGENCY': {
            'description': '应急模式',
            'priority': 20,
            'config': {'W_level': 200.0, 'delta_Q_max': 10.0} # Fast response
        }
    }
    
    def __init__(self):
        self.history_window = 10 # Look back steps
        
    def recognize(self, current_levels: List[float], flows: List[float], weather_forecast: List[float] = None) -> Dict[str, Any]:
        """
        Recognize the current scenario based on state.
        
        Args:
            current_levels: List of current water levels.
            flows: List of current gate flows.
            weather_forecast: Optional forecast data (e.g., rainfall).
            
        Returns:
            Dict containing scenario name, confidence, and description.
        """
        
        # 1. Feature Extraction (Simplified)
        avg_level = np.mean(current_levels)
        max_level = np.max(current_levels)
        level_variance = np.var(current_levels)
        
        # 2. Rule-based Logic
        scenario = 'NORMAL'
        confidence = 0.8
        
        # Rule 1: Flood Detection
        # If levels are dangerously high or rapid rise (simulated by variance/trend)
        if max_level > 8.0:
            scenario = 'FLOOD_CONTROL'
            confidence = 0.95
        elif weather_forecast and np.mean(weather_forecast) > 50.0: # Heavy rain forecast
            scenario = 'FLOOD_CONTROL'
            confidence = 0.85
            
        # Rule 2: Drought Detection
        elif max_level < 1.0:
            scenario = 'DROUGHT'
            confidence = 0.9
            
        # Rule 3: Emergency (e.g., sudden drop or spike not explained by control)
        # For PoC, let's say if variance is extremely high
        elif level_variance > 2.0:
            scenario = 'EMERGENCY'
            confidence = 0.75
            
        # 3. Mock ML Model (Random Forest simulation)
        # In a real system, we would feed features to a loaded model here.
        # ml_pred = self.model.predict([features])
        # For now, we stick to the rule-based result but add some "AI" noise to confidence
        
        return {
            'name': scenario,
            'description': self.SCENARIO_TAXONOMY[scenario]['description'],
            'confidence': confidence,
            'recommended_config': self.SCENARIO_TAXONOMY[scenario]['config']
        }
