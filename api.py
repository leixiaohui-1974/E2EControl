"""
REST API 接口
提供HTTP服务访问智能闸门控制系统
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import threading

from config_manager import get_config
from logger import get_logger, setup_logging
from brain_enhanced import EnhancedSemanticInterpreter
from database import SimulationDatabase
from main_enhanced import SmartPoolSimulation


app = Flask(__name__)
CORS(app)  # 启用跨域访问

# 初始化
config = get_config()
setup_logging(config.get_section('logging'))
logger = get_logger()

# 全局变量
running_simulations = {}  # 存储运行中的仿真
simulation_lock = threading.Lock()


@app.route('/')
def index():
    """API首页"""
    return jsonify({
        'name': 'Smart Pool Agent API',
        'version': '2.0',
        'description': '智能闸门控制系统REST API',
        'endpoints': {
            'GET /': '获取API信息',
            'GET /health': '健康检查',
            'GET /config': '获取配置信息',
            'POST /interpret': '解释自然语言指令',
            'POST /simulation/run': '运行仿真',
            'GET /simulation/<id>': '获取仿真详情',
            'GET /simulation/<id>/status': '获取仿真状态',
            'GET /simulation/<id>/history': '获取仿真历史数据',
            'GET /simulation/<id>/alerts': '获取仿真告警',
            'GET /simulation/<id>/report': '获取仿真报告',
            'GET /simulation/<id>/result.png': '获取仿真图表',
            'GET /simulations': '获取所有仿真列表',
            'GET /scenarios': '获取所有场景定义'
        }
    })


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0'
    })


@app.route('/config')
def get_config_info():
    """获取配置信息"""
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
        logger.error(f"获取配置失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/interpret', methods=['POST'])
def interpret_instruction():
    """
    解释自然语言指令
    
    请求体:
    {
        "instruction": "收到暴雨预警，立刻降低水位！"
    }
    """
    try:
        data = request.get_json()
        if not data or 'instruction' not in data:
            return jsonify({
                'success': False,
                'error': '缺少instruction参数'
            }), 400
        
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
        logger.error(f"指令解释失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def _get_scenario_name(confidence: float) -> str:
    """根据置信度获取场景名称"""
    if confidence >= 0.9:
        return "精确匹配"
    elif confidence >= 0.6:
        return "模糊匹配"
    else:
        return "使用默认配置"


@app.route('/simulation/run', methods=['POST'])
def run_simulation():
    """
    运行仿真
    
    请求体:
    {
        "script": [
            [0, "保持水位平稳，正常供水。"],
            [10, "收到暴雨预警，立刻降低水位！"]
        ],
        "async": true  # 可选，是否异步运行
    }
    """
    try:
        data = request.get_json() or {}
        script = data.get('script')
        is_async = data.get('async', False)
        
        # 创建仿真实例
        sim = SmartPoolSimulation()
        
        if is_async:
            # 异步运行
            sim_id = sim.simulation_id or len(running_simulations) + 1
            
            def run_async():
                with simulation_lock:
                    running_simulations[sim_id] = {
                        'status': 'running',
                        'start_time': datetime.now().isoformat(),
                        'simulation': sim
                    }
                
                try:
                    sim.run(script=script)
                    with simulation_lock:
                        running_simulations[sim_id]['status'] = 'completed'
                        running_simulations[sim_id]['end_time'] = datetime.now().isoformat()
                except Exception as e:
                    with simulation_lock:
                        running_simulations[sim_id]['status'] = 'failed'
                        running_simulations[sim_id]['error'] = str(e)
                finally:
                    if sim.db:
                        sim.db.close()
            
            thread = threading.Thread(target=run_async)
            thread.start()
            
            return jsonify({
                'success': True,
                'simulation_id': sim_id,
                'status': 'running',
                'message': '仿真已启动，使用 /simulation/{id}/status 查询状态'
            }), 202
        
        else:
            # 同步运行
            sim.run(script=script)
            sim_id = sim.simulation_id
            
            if sim.db:
                sim.db.close()
            
            return jsonify({
                'success': True,
                'simulation_id': sim_id,
                'status': 'completed',
                'results': {
                    'total_hours': sim.total_hours,
                    'final_level': sim.history['level'][-1] if sim.history['level'] else None,
                    'alerts_count': len(sim.monitor.alerts)
                }
            })
        
    except Exception as e:
        logger.error(f"仿真运行失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/simulation/<int:sim_id>/status')
def get_simulation_status(sim_id: int):
    """获取仿真状态"""
    try:
        with simulation_lock:
            if sim_id in running_simulations:
                sim_info = running_simulations[sim_id]
                return jsonify({
                    'success': True,
                    'simulation_id': sim_id,
                    'status': sim_info['status'],
                    'start_time': sim_info.get('start_time'),
                    'end_time': sim_info.get('end_time'),
                    'error': sim_info.get('error')
                })
        
        # 查询数据库
        if config.get('database.enabled', False):
            db = SimulationDatabase(config.get('database.path'))
            sims = db.get_recent_simulations(limit=100)
            db.close()
            
            for sim in sims:
                if sim['id'] == sim_id:
                    return jsonify({
                        'success': True,
                        'simulation_id': sim_id,
                        'status': 'completed',
                        'start_time': sim['start_time'],
                        'end_time': sim['end_time'],
                        'total_hours': sim['total_hours']
                    })
        
        return jsonify({
            'success': False,
            'error': f'仿真ID {sim_id} 不存在'
        }), 404
        
    except Exception as e:
        logger.error(f"获取仿真状态失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/simulation/<int:sim_id>/history')
def get_simulation_history(sim_id: int):
    """获取仿真历史数据"""
    try:
        if not config.get('database.enabled', False):
            return jsonify({
                'success': False,
                'error': '数据库未启用'
            }), 503
        
        db = SimulationDatabase(config.get('database.path'))
        history = db.get_simulation_history(sim_id)
        db.close()
        
        if not history:
            return jsonify({
                'success': False,
                'error': f'仿真ID {sim_id} 没有历史数据'
            }), 404
        
        return jsonify({
            'success': True,
            'simulation_id': sim_id,
            'count': len(history),
            'history': history
        })
        
    except Exception as e:
        logger.error(f"获取仿真历史失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/simulation/<int:sim_id>/alerts')
def get_simulation_alerts(sim_id: int):
    """获取仿真告警"""
    try:
        if not config.get('database.enabled', False):
            return jsonify({
                'success': False,
                'error': '数据库未启用'
            }), 503
        
        db = SimulationDatabase(config.get('database.path'))
        alerts = db.get_simulation_alerts(sim_id)
        db.close()
        
        return jsonify({
            'success': True,
            'simulation_id': sim_id,
            'count': len(alerts),
            'alerts': alerts
        })
        
    except Exception as e:
        logger.error(f"获取仿真告警失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/simulation/<int:sim_id>/report')
def get_simulation_report(sim_id: int):
    """获取仿真报告（文本）"""
    try:
        report_file = f'simulation_report_{sim_id}.md'
        
        if os.path.exists(report_file):
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({
                'success': True,
                'simulation_id': sim_id,
                'report': content
            })
        elif os.path.exists('simulation_report_enhanced.md'):
            # 返回最新的报告
            with open('simulation_report_enhanced.md', 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({
                'success': True,
                'simulation_id': sim_id,
                'report': content,
                'note': '返回最新仿真报告'
            })
        else:
            return jsonify({
                'success': False,
                'error': '报告文件不存在'
            }), 404
        
    except Exception as e:
        logger.error(f"获取仿真报告失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/simulation/<int:sim_id>/result.png')
def get_simulation_image(sim_id: int):
    """获取仿真结果图表"""
    try:
        image_file = f'simulation_result_{sim_id}.png'
        
        if os.path.exists(image_file):
            return send_file(image_file, mimetype='image/png')
        elif os.path.exists('simulation_result_enhanced.png'):
            return send_file('simulation_result_enhanced.png', mimetype='image/png')
        else:
            return jsonify({
                'success': False,
                'error': '图表文件不存在'
            }), 404
        
    except Exception as e:
        logger.error(f"获取仿真图表失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/simulations')
def list_simulations():
    """获取所有仿真列表"""
    try:
        result = []
        
        # 运行中的仿真
        with simulation_lock:
            for sim_id, info in running_simulations.items():
                result.append({
                    'id': sim_id,
                    'status': info['status'],
                    'start_time': info.get('start_time'),
                    'end_time': info.get('end_time'),
                    'source': 'memory'
                })
        
        # 数据库中的仿真
        if config.get('database.enabled', False):
            db = SimulationDatabase(config.get('database.path'))
            sims = db.get_recent_simulations(limit=20)
            db.close()
            
            for sim in sims:
                result.append({
                    'id': sim['id'],
                    'status': 'completed',
                    'start_time': sim['start_time'],
                    'end_time': sim['end_time'],
                    'total_hours': sim['total_hours'],
                    'source': 'database'
                })
        
        return jsonify({
            'success': True,
            'count': len(result),
            'simulations': result
        })
        
    except Exception as e:
        logger.error(f"获取仿真列表失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/scenarios')
def list_scenarios():
    """获取所有场景定义"""
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
        logger.error(f"获取场景列表失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        'success': False,
        'error': '接口不存在',
        'message': 'API endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({
        'success': False,
        'error': '服务器内部错误',
        'message': str(error)
    }), 500


def run_api_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """
    启动API服务器
    
    Args:
        host: 监听地址
        port: 监听端口
        debug: 调试模式
    """
    logger.info(f"启动API服务器: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    # 从配置读取API设置
    api_config = config.get_section('api')
    
    if api_config.get('enabled', False):
        run_api_server(
            host=api_config.get('host', '0.0.0.0'),
            port=api_config.get('port', 5000),
            debug=api_config.get('debug', False)
        )
    else:
        print("API服务未启用，请在config.yaml中设置 api.enabled: true")
        print("\n手动启动: python3 api.py")
