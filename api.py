"""
REST API for the Smart Pool Agent system.

Provides endpoints for instruction interpretation, simulation
management, health monitoring, and scenario listing.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from brain import SemanticInterpreter
from config_manager import ConfigManager
from simulation_manager import SimulationManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_INSTRUCTION_LENGTH: int = 500     # max characters for an instruction
API_VERSION: str = "3.0"

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

config_manager = ConfigManager()
brain = SemanticInterpreter()
logger = logging.getLogger(__name__)

simulations: Dict[str, Dict[str, Any]] = {}
sim_lock = threading.Lock()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route('/')
def index() -> Response:
    return app.send_static_file('index.html')


@app.route('/api')
def api_info() -> Tuple[Response, int]:
    return jsonify({
        'name': 'Smart Pool Agent API',
        'version': API_VERSION,
        'description': '智能水池代理系统的REST API',
    }), 200


@app.route('/config')
def get_config_endpoint() -> Tuple[Response, int]:
    try:
        return jsonify({'success': True, 'config': config_manager.config}), 200
    except Exception as exc:
        logger.error("Failed to get config: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': 'Failed to load config'}), 500


@app.route('/interpret', methods=['POST'])
def interpret_instruction_endpoint() -> Tuple[Response, int]:
    data = request.get_json(silent=True)
    if not data or 'instruction' not in data:
        return jsonify({'success': False, 'error': 'Missing instruction'}), 400

    instruction = data['instruction']

    if not isinstance(instruction, str) or not instruction.strip():
        return jsonify({
            'success': False,
            'error': 'Instruction must be a non-empty string',
        }), 400

    if len(instruction) > MAX_INSTRUCTION_LENGTH:
        return jsonify({
            'success': False,
            'error': f'Instruction too long (max {MAX_INSTRUCTION_LENGTH} chars)',
        }), 400

    try:
        config_result = brain.interpret(instruction)
    except Exception as exc:
        logger.error("Interpret failed: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': 'Interpretation failed'}), 500

    return jsonify({
        'success': True,
        'instruction': instruction,
        'confidence': 0.95,
        'scenario': 'Keyword-based Interpretation',
        'config': config_result,
    }), 200


@app.route('/simulation/run', methods=['POST'])
def run_simulation_endpoint() -> Tuple[Response, int]:
    data = request.get_json(silent=True) or {}
    instruction = data.get('instruction', '保持水位平稳，正常供水。')

    if not isinstance(instruction, str) or not instruction.strip():
        return jsonify({
            'success': False,
            'error': 'Instruction must be a non-empty string',
        }), 400

    if len(instruction) > MAX_INSTRUCTION_LENGTH:
        return jsonify({
            'success': False,
            'error': f'Instruction too long (max {MAX_INSTRUCTION_LENGTH} chars)',
        }), 400

    try:
        sim_id = str(uuid.uuid4())

        with sim_lock:
            simulations[sim_id] = {
                'status': 'starting',
                'start_time': datetime.now().isoformat(),
            }

        thread = threading.Thread(
            target=_run_simulation_worker,
            args=(sim_id, instruction),
            daemon=True,
        )
        with sim_lock:
            simulations[sim_id]['thread'] = thread
        thread.start()

        return jsonify({
            'success': True,
            'simulation_id': sim_id,
            'status': 'running',
        }), 202
    except Exception as exc:
        logger.error("Failed to start simulation: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': 'Failed to start simulation'}), 500


@app.route('/simulation/<sim_id>/history')
def get_simulation_history_endpoint(sim_id: str) -> Tuple[Response, int]:
    try:
        with sim_lock:
            sim = simulations.get(sim_id)
        if not sim:
            return jsonify({'success': False, 'error': 'Simulation not found'}), 404

        response_data = {k: v for k, v in sim.items() if k != 'thread'}
        return jsonify(response_data), 200
    except Exception as exc:
        logger.error("Failed to get history for %s: %s", sim_id, exc, exc_info=True)
        return jsonify({'success': False, 'error': 'Failed to retrieve history'}), 500


@app.route('/simulations')
def list_simulations_endpoint() -> Tuple[Response, int]:
    try:
        with sim_lock:
            sim_list = [
                {
                    'id': sid,
                    'start_time': s.get('start_time'),
                    'status': s.get('status'),
                    'total_hours': len(s.get('history', {}).get('time', [])),
                }
                for sid, s in simulations.items()
            ]
        return jsonify({'success': True, 'simulations': sim_list}), 200
    except Exception as exc:
        logger.error("Failed to list simulations: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': 'Failed to list simulations'}), 500


@app.route('/health')
def health_check() -> Tuple[Response, int]:
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': API_VERSION,
    }), 200


@app.route('/scenarios')
def list_scenarios() -> Tuple[Response, int]:
    """List available predefined scenarios."""
    scenarios = [
        {'id': 'normal', 'name': '正常供水',
         'instruction': '保持水位平稳，正常供水。'},
        {'id': 'flood', 'name': '暴雨预警',
         'instruction': '收到暴雨预警，立刻降低水位腾出库容！安全第一！'},
        {'id': 'pollution', 'name': '污染应急',
         'instruction': '下游检测到污染，紧急切断出流！'},
        {'id': 'drought', 'name': '干旱调度',
         'instruction': '上游来水减少，提升水位储备水源。'},
        {'id': 'maintenance', 'name': '检修维护',
         'instruction': '下游渠道需要检修，缓慢降低流量。'},
    ]
    return jsonify({
        'success': True,
        'scenarios': scenarios,
        'count': len(scenarios),
    }), 200


@app.route('/simulation/<sim_id>/event', methods=['POST'])
def trigger_event_endpoint(sim_id: str) -> Tuple[Response, int]:
    return jsonify({
        'success': True,
        'message': 'Event injection placeholder',
    }), 200


@app.route('/simulation/<sim_id>/control', methods=['POST'])
def set_control_override_endpoint(sim_id: str) -> Tuple[Response, int]:
    return jsonify({
        'success': True,
        'message': 'Control override placeholder',
    }), 200


@app.errorhandler(404)
def not_found(error: Exception) -> Tuple[Response, int]:
    """Custom 404 handler returning JSON."""
    return jsonify({'success': False, 'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error: Exception) -> Tuple[Response, int]:
    """Custom 500 handler returning JSON."""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


# ---------------------------------------------------------------------------
# Simulation Worker
# ---------------------------------------------------------------------------

def _format_history_for_frontend(
    history: Dict[str, List[Any]], num_pools: int,
) -> Dict[str, Any]:
    """Convert raw simulation history to frontend-friendly format."""
    if num_pools == 1:
        return {
            'time': history['time'],
            'levels': [history['level']],
            'flows': [history['q_in']],
        }
    else:
        levels = [history[f'level_{i}'] for i in range(num_pools)]
        flows = [history[f'q_in_{i}'] for i in range(num_pools)]
        return {'time': history['time'], 'levels': levels, 'flows': flows}


def _run_simulation_worker(sim_id: str, instruction: str) -> None:
    """Run a simulation in a background thread."""
    try:
        with sim_lock:
            simulations[sim_id]['status'] = 'running'

        sim_params = config_manager.get_simulation_params()
        physical_params = config_manager.get_physical_params()
        demand_params = config_manager.get_demand_profile_params()

        total_hours = sim_params['total_hours']
        np.random.seed(sim_params['seed'])
        demands = (
            demand_params['base_demand']
            + np.random.normal(0, demand_params['noise_std_dev'], total_hours + 20)
        )

        script: List[Tuple[int, str]] = [(0, instruction)]

        sim_manager = SimulationManager(
            total_hours=total_hours,
            dt=sim_params['time_step'],
            area=physical_params['area'],
            initial_level=physical_params['initial_level'],
            script=script,
            demands=demands,
        )

        raw_history = sim_manager.run_simulation()

        with sim_lock:
            simulations[sim_id]['history'] = _format_history_for_frontend(
                raw_history, num_pools=1,
            )
            simulations[sim_id]['status'] = 'completed'

    except Exception as e:
        logger.error("Simulation %s failed: %s", sim_id, e, exc_info=True)
        with sim_lock:
            simulations[sim_id]['status'] = 'failed'
            simulations[sim_id]['error'] = str(e)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    port = int(os.environ.get('E2E_API_PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
