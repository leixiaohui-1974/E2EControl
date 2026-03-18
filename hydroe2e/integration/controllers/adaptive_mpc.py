"""
自适应MPC控制器
集成Phase 2的分布式MPC + Phase 3的智能决策
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Phase 2 导入
from hydroe2e.phase2.controllers.improved_admm import ImprovedDistributedMPC, ADMMParameters
from hydroe2e.phase2.controllers.multi_objective_mpc import ObjectiveWeights
from hydroe2e.phase2.controllers.feedforward_control import (
    IntegratedFeedforwardMPC,
    DemandPredictor,
    WeatherForecast,
    DisturbanceForecast
)

# Phase 3 导入
from hydroe2e.phase3.scenario_recognition.rule_engine import SystemState
from hydroe2e.phase3.decision.decision_engine import DecisionEngine


@dataclass
class AdaptiveControlConfig:
    """自适应控制配置"""
    enable_scenario_recognition: bool = True
    enable_adaptive_weights: bool = True
    enable_feedforward: bool = True
    enable_risk_adjustment: bool = True
    
    # 更新频率
    scenario_update_interval: int = 1  # 每步更新
    decision_update_interval: int = 1


class AdaptiveMPCController:
    """
    自适应MPC控制器
    
    核心功能:
    1. 场景自动识别
    2. 策略自动选择
    3. 参数动态调整
    4. 风险自适应
    """
    
    def __init__(self, 
                 num_pools: int = 3,
                 horizon: int = 10,
                 dt: float = 3600.0,
                 config: AdaptiveControlConfig = None):
        """
        初始化自适应MPC控制器
        
        Args:
            num_pools: 池数量
            horizon: 预测时域
            dt: 时间步长
            config: 自适应配置
        """
        self.num_pools = num_pools
        self.horizon = horizon
        self.dt = dt
        self.config = config or AdaptiveControlConfig()
        
        # Phase 3: 决策引擎
        self.decision_engine = DecisionEngine()
        
        # Phase 2: 分布式MPC
        self.distributed_mpc = ImprovedDistributedMPC(
            num_pools=num_pools,
            horizon=horizon,
            dt=dt
        )
        
        # 前馈控制
        if self.config.enable_feedforward:
            self.demand_predictor = DemandPredictor()
            self.weather_forecast = WeatherForecast()
        
        # 状态记录
        self.current_scenario = None
        self.current_strategy = None
        self.current_risk_level = 'low'
        
        # 性能统计
        self.stats = {
            'scenario_switches': 0,
            'strategy_switches': 0,
            'risk_escalations': 0,
            'total_steps': 0
        }
        
        # 历史记录
        self.history = {
            'scenarios': [],
            'strategies': [],
            'risk_levels': [],
            'control_actions': [],
            'performance': []
        }
    
    def compute_control(self, 
                       current_levels: List[float],
                       current_flows: List[float],
                       current_demands: List[float],
                       time: int,
                       weather: Dict = None,
                       alerts: List[str] = None) -> Tuple[List[Tuple[float, float]], Dict]:
        """
        计算控制动作（自适应）
        
        Args:
            current_levels: 当前水位
            current_flows: 当前流量
            current_demands: 当前需求
            time: 当前时间
            weather: 天气信息
            alerts: 告警信息
            
        Returns:
            (控制动作列表, 调试信息)
        """
        debug_info = {'time': time}
        
        # 1. 场景识别
        if self.config.enable_scenario_recognition:
            scenario_result = self._recognize_scenario(
                current_levels, current_flows, current_demands,
                time, weather, alerts
            )
            debug_info['scenario'] = scenario_result
            
            # 检查场景切换
            if (self.current_scenario is None or 
                scenario_result['scenario_id'] != self.current_scenario):
                self.stats['scenario_switches'] += 1
                self.current_scenario = scenario_result['scenario_id']
                debug_info['scenario_switched'] = True
            
            # 2. 策略选择和参数配置
            mpc_config = self._configure_mpc(scenario_result)
            debug_info['mpc_config'] = mpc_config
            
            # 检查策略切换
            if (self.current_strategy is None or 
                mpc_config['strategy_id'] != self.current_strategy):
                self.stats['strategy_switches'] += 1
                self.current_strategy = mpc_config['strategy_id']
                debug_info['strategy_switched'] = True
            
            # 3. 风险调整
            if self.config.enable_risk_adjustment:
                mpc_config = self._adjust_for_risk(
                    mpc_config, 
                    scenario_result['risk_level']
                )
                debug_info['risk_adjusted'] = True
                
                # 检查风险升级
                if (self.current_risk_level != scenario_result['risk_level'] and
                    self._risk_level_to_int(scenario_result['risk_level']) > 
                    self._risk_level_to_int(self.current_risk_level)):
                    self.stats['risk_escalations'] += 1
                
                self.current_risk_level = scenario_result['risk_level']
        else:
            # 使用默认配置
            mpc_config = self._default_config()
            scenario_result = None
        
        # 4. 更新MPC参数
        self._update_mpc_parameters(mpc_config)
        
        # 5. 生成需求预测
        demand_forecasts = self._generate_demand_forecasts(
            current_demands, time, mpc_config
        )
        debug_info['demand_forecasts'] = demand_forecasts
        
        # 6. 前馈补偿（如果启用）
        feedforward_signals = None
        if self.config.enable_feedforward and mpc_config.get('use_feedforward', False):
            feedforward_signals = self._compute_feedforward(
                current_levels, current_flows, current_demands,
                demand_forecasts, weather, time
            )
            debug_info['feedforward'] = feedforward_signals
        
        # 7. 求解分布式MPC
        control_actions, admm_info = self.distributed_mpc.solve(
            current_levels=current_levels,
            q_in_prevs=current_flows[:-1],  # 前n个流量作为输入
            q_out_forecasts=demand_forecasts
        )
        debug_info['admm'] = admm_info
        
        # 8. 应用前馈补偿
        if feedforward_signals is not None:
            control_actions = self._apply_feedforward(
                control_actions, feedforward_signals
            )
        
        # 9. 更新统计和历史
        self.stats['total_steps'] += 1
        self._update_history(scenario_result, mpc_config, control_actions, admm_info)
        
        return control_actions, debug_info
    
    def _recognize_scenario(self, levels, flows, demands, time, weather, alerts):
        """场景识别"""
        # 构建系统状态
        state = SystemState(
            time=time,
            levels=levels,
            flows=flows,
            demands=demands,
            weather=weather or {},
            alerts=alerts or []
        )
        
        # 做出决策
        decision = self.decision_engine.make_decision(state)
        
        return {
            'scenario_id': decision.scenario.scenario_id,
            'scenario_name': decision.scenario.scenario_name,
            'confidence': decision.scenario.confidence,
            'risk_level': decision.risk_level,
            'risk_factors': decision.risk_factors,
            'strategy': decision.strategy,
            'mpc_config': decision.mpc_config,
            'actions': decision.immediate_actions
        }
    
    def _configure_mpc(self, scenario_result):
        """配置MPC参数"""
        if scenario_result is None:
            return self._default_config()
        
        mpc_config = scenario_result['mpc_config'].copy()
        mpc_config['strategy_id'] = scenario_result['strategy'].strategy_id
        mpc_config['scenario_id'] = scenario_result['scenario_id']
        
        return mpc_config
    
    def _adjust_for_risk(self, mpc_config, risk_level):
        """根据风险等级调整参数"""
        if risk_level == 'critical':
            # 紧急情况：缩短时域，增加安全裕度
            mpc_config['horizon'] = min(mpc_config['horizon'], 5)
            mpc_config['constraints']['safety_margin'] *= 1.5
            mpc_config['weights']['level_tracking'] *= 1.5
            
        elif risk_level == 'high':
            # 高风险：增加安全裕度
            mpc_config['constraints']['safety_margin'] *= 1.2
            mpc_config['weights']['level_tracking'] *= 1.2
            
        elif risk_level == 'low':
            # 低风险：可以更激进地优化
            mpc_config['weights']['energy_cost'] *= 1.2
        
        return mpc_config
    
    def _update_mpc_parameters(self, mpc_config):
        """更新MPC参数"""
        # 更新时域
        if mpc_config.get('horizon') != self.distributed_mpc.horizon:
            self.distributed_mpc.horizon = mpc_config['horizon']
            # 重新构建问题
            self.distributed_mpc._rebuild_problems()
        
        # 更新权重
        if 'weights' in mpc_config:
            for controller in self.distributed_mpc.local_controllers:
                controller.weights = ObjectiveWeights(**mpc_config['weights'])
        
        # 更新约束
        if 'constraints' in mpc_config:
            for controller in self.distributed_mpc.local_controllers:
                controller.level_min = 1.0 - mpc_config['constraints']['safety_margin']
                controller.level_max = 5.0 - mpc_config['constraints']['safety_margin']
                controller.flow_change_max = mpc_config['constraints']['max_flow_change_rate']
    
    def _generate_demand_forecasts(self, current_demands, time, mpc_config):
        """生成需求预测"""
        forecasts = []
        horizon = mpc_config.get('horizon', self.horizon)
        
        for i in range(self.num_pools):
            if hasattr(self, 'demand_predictor'):
                # 使用预测器
                forecast = self.demand_predictor.predict(
                    time, horizon=horizon
                )
            else:
                # 简单假设需求不变
                forecast = [current_demands[i]] * horizon
            
            forecasts.append(forecast)
        
        return forecasts
    
    def _compute_feedforward(self, levels, flows, demands, 
                            demand_forecasts, weather, time):
        """计算前馈补偿"""
        feedforward_signals = []
        
        for i in range(self.num_pools):
            # 简化的前馈计算
            demand_change = demand_forecasts[i][0] - demands[i]
            signal = demand_change * 0.8  # 80%补偿
            feedforward_signals.append(signal)
        
        return feedforward_signals
    
    def _apply_feedforward(self, control_actions, feedforward_signals):
        """应用前馈补偿"""
        adjusted_actions = []
        
        for i, (q_in, q_out) in enumerate(control_actions):
            if i < len(feedforward_signals):
                # 在输出流量上应用补偿
                q_out_adjusted = q_out + feedforward_signals[i]
                q_out_adjusted = np.clip(q_out_adjusted, 0.0, 20.0)
                adjusted_actions.append((q_in, q_out_adjusted))
            else:
                adjusted_actions.append((q_in, q_out))
        
        return adjusted_actions
    
    def _update_history(self, scenario_result, mpc_config, 
                       control_actions, admm_info):
        """更新历史记录"""
        if scenario_result:
            self.history['scenarios'].append(scenario_result['scenario_id'])
            self.history['strategies'].append(mpc_config.get('strategy_id'))
            self.history['risk_levels'].append(scenario_result['risk_level'])
        
        self.history['control_actions'].append(control_actions)
        self.history['performance'].append({
            'iterations': admm_info.get('iterations', 0),
            'converged': admm_info.get('converged', False),
            'solve_time': admm_info.get('solve_time', 0)
        })
    
    def _default_config(self):
        """默认配置"""
        return {
            'horizon': self.horizon,
            'weights': {
                'level_tracking': 10.0,
                'flow_smoothness': 5.0,
                'energy_cost': 0.3,
                'water_delivery': 2.0
            },
            'constraints': {
                'safety_margin': 0.5,
                'max_flow_change_rate': 2.0
            },
            'use_feedforward': False,
            'strategy_id': 'default',
            'scenario_id': 'unknown'
        }
    
    def _risk_level_to_int(self, risk_level):
        """风险等级转整数"""
        mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        return mapping.get(risk_level, 0)
    
    def get_statistics(self):
        """获取统计信息"""
        return {
            **self.stats,
            'scenario_switch_rate': self.stats['scenario_switches'] / max(1, self.stats['total_steps']),
            'strategy_switch_rate': self.stats['strategy_switches'] / max(1, self.stats['total_steps']),
            'risk_escalation_rate': self.stats['risk_escalations'] / max(1, self.stats['total_steps'])
        }
    
    def get_history_summary(self):
        """获取历史摘要"""
        from collections import Counter
        
        return {
            'total_steps': len(self.history['scenarios']),
            'scenario_distribution': dict(Counter(self.history['scenarios'])),
            'strategy_distribution': dict(Counter(self.history['strategies'])),
            'risk_distribution': dict(Counter(self.history['risk_levels'])),
            'avg_iterations': np.mean([p['iterations'] for p in self.history['performance']]),
            'convergence_rate': np.mean([p['converged'] for p in self.history['performance']]),
            'avg_solve_time': np.mean([p['solve_time'] for p in self.history['performance']])
        }


# 示例使用
if __name__ == "__main__":
    logger.info("="*70)
    logger.info(" "*20 + "自适应MPC控制器演示")
    logger.info("="*70)
    
    # 创建控制器
    controller = AdaptiveMPCController(
        num_pools=3,
        horizon=10,
        dt=3600.0
    )
    
    logger.info(f"\n✓ 自适应MPC控制器已初始化")
    logger.info(f"  池数量: {controller.num_pools}")
    logger.info(f"  预测时域: {controller.horizon}")
    logger.info(f"  场景识别: {'启用' if controller.config.enable_scenario_recognition else '禁用'}")
    logger.info(f"  自适应权重: {'启用' if controller.config.enable_adaptive_weights else '禁用'}")
    logger.info(f"  前馈控制: {'启用' if controller.config.enable_feedforward else '禁用'}")
    logger.info(f"  风险调整: {'启用' if controller.config.enable_risk_adjustment else '禁用'}")
    
    # 模拟几个场景
    test_cases = [
        {
            'name': '正常运行',
            'levels': [3.0, 3.0, 3.0],
            'flows': [5.0, 5.0, 5.0, 5.0],
            'demands': [5.0, 5.0, 5.0],
            'time': 10
        },
        {
            'name': '高峰需求',
            'levels': [2.8, 2.9, 3.0],
            'flows': [10.0, 9.5, 9.0, 8.5],
            'demands': [11.0, 10.5, 10.0],
            'time': 18
        },
        {
            'name': '洪峰预警',
            'levels': [4.5, 4.2, 4.0],
            'flows': [15.0, 14.0, 13.0, 12.0],
            'demands': [6.0, 6.0, 6.0],
            'time': 50,
            'weather': {'rainfall': 20.0},
            'alerts': ['水位快速上升']
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"测试 {i}: {test['name']}")
        logger.info('='*70)
        
        control_actions, debug_info = controller.compute_control(
            current_levels=test['levels'],
            current_flows=test['flows'],
            current_demands=test['demands'],
            time=test['time'],
            weather=test.get('weather'),
            alerts=test.get('alerts')
        )
        
        # 显示结果
        if 'scenario' in debug_info:
            scenario = debug_info['scenario']
            logger.info(f"\n🎯 场景识别:")
            logger.info(f"  场景: {scenario['scenario_name']}")
            logger.info(f"  置信度: {scenario['confidence']:.1%}")
            logger.info(f"  风险: {scenario['risk_level']}")
            
            if debug_info.get('scenario_switched'):
                logger.info(f"  ⚡ 场景切换!")
        
        if 'mpc_config' in debug_info:
            config = debug_info['mpc_config']
            logger.info(f"\n⚙️ MPC配置:")
            logger.info(f"  时域: {config['horizon']}步")
            logger.info(f"  策略: {config.get('strategy_id', 'default')}")
            
            if debug_info.get('strategy_switched'):
                logger.info(f"  ⚡ 策略切换!")
        
        if 'admm' in debug_info:
            admm = debug_info['admm']
            logger.info(f"\n📊 ADMM求解:")
            logger.info(f"  迭代次数: {admm.get('iterations', 0)}")
            logger.info(f"  收敛: {'是' if admm.get('converged', False) else '否'}")
            logger.info(f"  求解时间: {admm.get('solve_time', 0)*1000:.1f}ms")
        
        logger.info(f"\n🎮 控制动作:")
        for j, (q_in, q_out) in enumerate(control_actions):
            logger.info(f"  池{j}: q_in={q_in:.2f}, q_out={q_out:.2f} m³/s")
    
    # 统计信息
    logger.info(f"\n{'='*70}")
    logger.info("统计信息")
    logger.info('='*70)
    
    stats = controller.get_statistics()
    logger.info(f"  总步数: {stats['total_steps']}")
    logger.info(f"  场景切换: {stats['scenario_switches']}次")
    logger.info(f"  策略切换: {stats['strategy_switches']}次")
    logger.info(f"  风险升级: {stats['risk_escalations']}次")
    
    logger.info("\n" + "="*70)
    logger.info("演示完成！")
    logger.info("="*70)
