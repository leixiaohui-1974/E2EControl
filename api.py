"""
REST API 接口
提供HTTP服务访问智能闸门控制系统
"""

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

from config_manager import get_config
from logger import get_logger, setup_logging
from brain_enhanced import EnhancedSemanticInterpreter
from simulation_manager import SimulationManager
import simulation_manager
print(f"LOADING simulation_manager from: {simulation_manager.__file__}")
import sys
print(f"SYS.PATH: {sys.path}")

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)
# Initialize System
config = get_config()
setup_logging(config.get_section('logging'))
logger = get_logger()

# Initialize Simulation Manager
sim_manager = SimulationManager()

@app.route('/')
def index():
    """Serve the frontend application."""
    try:
        full_path = os.path.join(os.getcwd(), 'web', 'index.html')
        if os.path.exists(full_path):
            return send_file(full_path)
        else:
            return "Index file not found", 404
    except Exception as e:
        return str(e), 500

@app.route('/api')
def api_info():
    """API Info"""
    return jsonify({
        'name': 'Smart Pool Agent API',
        'version': '2.1',
        'description': '智能闸门控制系统REST API',
        'endpoints': {
            'GET /': 'Frontend UI',
            'GET /api': 'API Info',
            'GET /health': 'Health Check',
            'GET /config': 'Get Configuration',
            'POST /interpret': 'Interpret Instruction',
            'POST /simulation/run': 'Run Simulation',
            'GET /simulation/<id>/status': 'Get Simulation Status',
            'GET /simulation/<id>/history': 'Get Simulation History',
            'GET /simulations': 'List Simulations',
            'GET /scenarios': 'List Scenarios'
        }
    })

@app.route('/health')
def health():
    """Health Check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.1'
    })

@app.route('/config')
def get_config_info():
    """Get Configuration"""
    try:
        return jsonify({
            'success': True,
            'config': {
                'simulation': config.get_section('simulation'),
                'mpc': config.get_section('mpc'),
                'monitoring': config.get_section('monitoring')
            }
        })
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/interpret', methods=['POST'])
def interpret_instruction():
    """Interpret Natural Language Instruction"""
    try:
        data = request.get_json()
        if not data or 'instruction' not in data:
            return jsonify({'success': False, 'error': 'Missing instruction parameter'}), 400
        
        instruction = data['instruction']
        brain = EnhancedSemanticInterpreter()
        config_result, confidence = brain.interpret(instruction)
        
        return jsonify({
            'success': True,
            'instruction': instruction,
            'confidence': confidence,
            'config': config_result,
            'scenario': _get_scenario_name(confidence)
        })
    except Exception as e:
        logger.error(f"Interpretation failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def _get_scenario_name(confidence: float) -> str:
    if confidence >= 0.9: return "Exact Match"
    elif confidence >= 0.6: return "Fuzzy Match"
    else: return "Default Config"

@app.route('/simulation/run', methods=['POST'])
def run_simulation():
    """Run Simulation"""
    try:
        data = request.get_json() or {}
        script = data.get('script')
        is_async = data.get('async', False)
        system_type = data.get('system_type', 'single') # 'single' or 'cascaded'
        
        result = sim_manager.run_simulation(script=script, is_async=is_async, system_type=system_type)
        
        status_code = 202 if is_async else 200
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Simulation run failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/simulation/<int:sim_id>/event', methods=['POST'])
def trigger_event(sim_id: int):
    """Trigger Event in Simulation"""
    try:
        data = request.get_json()
        event_type = data.get('type')
        event_data = data.get('data', {})
        
        if not event_type:
            return jsonify({'success': False, 'error': 'Missing event type'}), 400
            
        success = sim_manager.inject_event(sim_id, event_type, event_data)
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Simulation not running or found'}), 404
    except Exception as e:
        logger.error(f"Trigger event failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/simulation/<int:sim_id>/control/reset', methods=['POST'])
def reset_simulation_overrides(sim_id):
    success = sim_manager.clear_overrides(sim_id)
    if success:
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Simulation not found'}), 404

@app.route('/simulation/<int:sim_id>/stop', methods=['POST'])
def stop_simulation(sim_id):
    success = sim_manager.stop_simulation(sim_id)
    if success:
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Simulation not found'}), 404

@app.route('/simulation/<id>/status', methods=['GET'])
def get_simulation_status(id):
    """Get the status of a simulation."""
    try:
        try:
            sim_id = int(id)
        except ValueError:
            sim_id = id
        
        status = sim_manager.get_status(sim_id)
        if status:
            return jsonify(status)
        else:
            return jsonify({'error': 'Simulation not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulation/<id>/history', methods=['GET'])
def get_simulation_history(id):
    """Get the history of a simulation."""
    try:
        try:
            sim_id = int(id)
        except ValueError:
            sim_id = id
            
        history = sim_manager.get_history(sim_id)
        if history:
            return jsonify(history)
        else:
            return jsonify({'error': 'Simulation not found or history empty'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulation/<id>/health', methods=['GET'])
def get_simulation_health(id):
    """Get the health status of a simulation (Phase 5)."""
    try:
        try:
            sim_id = int(id)
        except ValueError:
            sim_id = id
            
        with sim_manager.lock:
            if sim_id in sim_manager.running_simulations:
                sim_data = sim_manager.running_simulations[sim_id]
                if 'system_instance' in sim_data:
                    status = sim_data['system_instance'].get_system_status()
                    return jsonify(status)
                else:
                    return jsonify({'error': 'System instance not found (not a Phase 5 simulation?)'}), 404
            else:
                return jsonify({'error': 'Simulation not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/simulations', methods=['GET'])
def list_simulations():
    """List All Simulations"""
    try:
        sims = sim_manager.list_simulations()
        return jsonify({
            'success': True,
            'count': len(sims),
            'simulations': sims
        })
    except Exception as e:
        logger.error(f"List simulations failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/scenarios')
def list_scenarios():
    """List Scenarios"""
    try:
        scenarios = config.get_all_scenarios()
        return jsonify({
            'success': True,
            'count': len(scenarios),
            'scenarios': [
                {
                    'name': s['name'],
                    'keywords': s['keywords'],
                    'config': s['config']
                }
                for s in scenarios
            ]
        })
    except Exception as e:
        logger.error(f"List scenarios failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Not Found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal Server Error', 'message': str(error)}), 500

def run_api_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    logger.info(f"Starting API Server: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == '__main__':
    api_config = config.get_section('api')
    if api_config.get('enabled', False):
        run_api_server(
            host=api_config.get('host', '0.0.0.0'),
            port=api_config.get('port', 5000),
            debug=api_config.get('debug', False)
        )
    else:
        print("API service disabled in config.yaml")
        print("Manual start: python3 api.py")
