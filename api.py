from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import uuid
import numpy as np
from datetime import datetime
import logging

from config_manager import ConfigManager
from simulation_manager import SimulationManager
from brain import SemanticInterpreter

# --- Initialization ---
app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

config_manager = ConfigManager()
brain = SemanticInterpreter()
logger = logging.getLogger(__name__)

simulations = {}
sim_lock = threading.Lock()

# --- API Endpoints ---

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api')
def api_info():
    return jsonify({
        'name': 'Smart Pool Agent API',
        'version': '3.0',
        'description': '智能水池代理系统的REST API'
    })

@app.route('/config')
def get_config_endpoint():
    return jsonify({'success': True, 'config': config_manager.config})

@app.route('/interpret', methods=['POST'])
def interpret_instruction_endpoint():
    data = request.get_json()
    if not data or 'instruction' not in data:
        return jsonify({'success': False, 'error': 'Missing instruction'}), 400

    instruction = data['instruction']
    config_result = brain.interpret(instruction)

    return jsonify({
        'success': True,
        'instruction': instruction,
        'confidence': 0.95,
        'scenario': 'Keyword-based Interpretation',
        'config': config_result
    })

@app.route('/simulation/run', methods=['POST'])
def run_simulation_endpoint():
    data = request.get_json() or {}
    instruction = data.get('instruction', '保持水位平稳，正常供水。')
    system_type = data.get('system_type', 'single')

    sim_id = str(uuid.uuid4())

    with sim_lock:
        simulations[sim_id] = {'status': 'starting', 'start_time': datetime.now().isoformat()}

    thread = threading.Thread(target=run_simulation_worker, args=(sim_id, instruction, system_type))
    simulations[sim_id]['thread'] = thread
    thread.start()

    return jsonify({'success': True, 'simulation_id': sim_id, 'status': 'running'}), 202

@app.route('/simulation/<sim_id>/history')
def get_simulation_history_endpoint(sim_id):
    with sim_lock:
        sim = simulations.get(sim_id)
    if not sim:
        return jsonify({'success': False, 'error': 'Simulation not found'}), 404

    response_data = {k: v for k, v in sim.items() if k != 'thread'}
    return jsonify(response_data)

@app.route('/simulations')
def list_simulations_endpoint():
    with sim_lock:
        sim_list = [
            {
                'id': sid,
                'start_time': s.get('start_time'),
                'status': s.get('status'),
                'total_hours': len(s.get('history', {}).get('time', []))
            }
            for sid, s in simulations.items()
        ]
    return jsonify({'success': True, 'simulations': sim_list})

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '3.0'
    })


@app.route('/scenarios')
def list_scenarios():
    """List available predefined scenarios."""
    scenarios = [
        {'id': 'normal', 'name': '正常供水', 'instruction': '保持水位平稳，正常供水。'},
        {'id': 'flood', 'name': '暴雨预警', 'instruction': '收到暴雨预警，立刻降低水位腾出库容！安全第一！'},
        {'id': 'pollution', 'name': '污染应急', 'instruction': '下游检测到污染，紧急切断出流！'},
        {'id': 'drought', 'name': '干旱调度', 'instruction': '上游来水减少，提升水位储备水源。'},
        {'id': 'maintenance', 'name': '检修维护', 'instruction': '下游渠道需要检修，缓慢降低流量。'},
    ]
    return jsonify({'success': True, 'scenarios': scenarios, 'count': len(scenarios)})


@app.route('/simulation/<sim_id>/event', methods=['POST'])
def trigger_event_endpoint(sim_id):
    # This is a placeholder for re-implementing event injection
    return jsonify({'success': True, 'message': 'Event injection placeholder'})

@app.route('/simulation/<sim_id>/control', methods=['POST'])
def set_control_override_endpoint(sim_id):
    # This is a placeholder for re-implementing control overrides
    return jsonify({'success': True, 'message': 'Control override placeholder'})


@app.errorhandler(404)
def not_found(error):
    """Custom 404 handler returning JSON."""
    return jsonify({'success': False, 'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Custom 500 handler returning JSON."""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# --- Simulation Worker ---

def format_history_for_frontend(history, num_pools):
    if num_pools == 1:
        return {
            'time': history['time'],
            'levels': [history['level']],
            'flows': [history['q_in']]
        }
    else:
        levels = [history[f'level_{i}'] for i in range(num_pools)]
        flows = [history[f'q_in_{i}'] for i in range(num_pools)]
        return {'time': history['time'], 'levels': levels, 'flows': flows}

def run_simulation_worker(sim_id, instruction, system_type):
    try:
        with sim_lock:
            simulations[sim_id]['status'] = 'running'

        sim_params = config_manager.get_simulation_params()
        physical_params = config_manager.get_physical_params()
        demand_params = config_manager.get_demand_profile_params()

        total_hours = sim_params['total_hours']
        np.random.seed(sim_params['seed'])
        demands = (demand_params['base_demand'] +
                   np.random.normal(0, demand_params['noise_std_dev'], total_hours + 20))

        script = [(0, instruction)]
        num_pools = 3 if system_type == 'cascaded' else 1

        sim_manager = SimulationManager(
            system_type=system_type,
            num_pools=num_pools,
            dt=sim_params['time_step'],
            area=physical_params['area'],
            initial_level=physical_params['initial_level'],
            initial_flow=demand_params['base_demand']
        )

        raw_history = sim_manager.run_simulation(total_hours, script, demands)

        with sim_lock:
            simulations[sim_id]['history'] = format_history_for_frontend(raw_history, num_pools)
            simulations[sim_id]['status'] = 'completed'

    except Exception as e:
        logger.error(f"Simulation {sim_id} failed: {e}")
        with sim_lock:
            simulations[sim_id]['status'] = 'failed'
            simulations[sim_id]['error'] = str(e)

# --- Main Execution ---

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=5000, debug=False)
