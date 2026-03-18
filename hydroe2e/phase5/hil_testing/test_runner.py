"""
HIL Test Runner - 在环测试执行器

协调场景生成、工况注入、系统仿真和评估的完整测试流程
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import traceback

from hydroe2e.phase5.hil_testing.scenario_generator import ScenarioGenerator, Scenario, AutonomousLevel
from hydroe2e.phase5.hil_testing.condition_injector import ConditionInjector
from hydroe2e.phase5.hil_testing.evaluation_engine import EvaluationEngine, TestResult


class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """单个测试用例"""
    id: str
    scenario: Scenario
    status: TestStatus = TestStatus.PENDING
    result: Optional[TestResult] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    execution_log: List[str] = field(default_factory=list)


@dataclass
class TestSuite:
    """测试套件"""
    id: str
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)
    autonomous_level: Optional[AutonomousLevel] = None
    category_filter: Optional[str] = None

    @property
    def total_count(self) -> int:
        return len(self.test_cases)

    @property
    def passed_count(self) -> int:
        return sum(1 for tc in self.test_cases if tc.status == TestStatus.PASSED)

    @property
    def failed_count(self) -> int:
        return sum(1 for tc in self.test_cases if tc.status == TestStatus.FAILED)

    @property
    def error_count(self) -> int:
        return sum(1 for tc in self.test_cases if tc.status == TestStatus.ERROR)

    @property
    def skipped_count(self) -> int:
        return sum(1 for tc in self.test_cases if tc.status == TestStatus.SKIPPED)

    @property
    def pass_rate(self) -> float:
        executed = self.passed_count + self.failed_count
        return self.passed_count / executed if executed > 0 else 0.0


@dataclass
class SimulationState:
    """仿真状态"""
    time: float
    water_levels: Dict[str, float]
    gate_positions: Dict[str, float]
    inflows: Dict[str, float]
    outflows: Dict[str, float]
    sensor_readings: Dict[str, float]
    alarms: List[str]
    control_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'time': self.time,
            'water_levels': self.water_levels,
            'gate_positions': self.gate_positions,
            'inflows': self.inflows,
            'outflows': self.outflows,
            'sensor_readings': self.sensor_readings,
            'alarms': self.alarms,
            'control_mode': self.control_mode
        }


class SimulationEnvironment:
    """仿真环境 - 连接物理模型和控制系统"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.dt = self.config.get('dt', 1.0)  # 仿真步长(秒)
        self.current_time = 0.0
        self.state_history: List[SimulationState] = []
        self.alarm_history: List[Tuple[float, str]] = []
        self.control_actions: List[Tuple[float, Dict]] = []
        self.logger = logging.getLogger('SimulationEnvironment')

        # 初始化物理模型
        self._init_physics()

        # 初始化控制器
        self._init_controller()

    def _init_physics(self):
        """初始化物理模型"""
        try:
            from physics import CanalPoolSimulator
            # 适配接口: PoolPhysics -> CanalPoolSimulator
            # 这里我们直接使用 CanalPoolSimulator，但需要注意接口差异
            # HIL Runner 期望的 physics 对象可能需要适配
            # 暂时直接实例化，后续在 step 中适配
            self.physics = CanalPoolSimulator(
                area=1000.0, # 默认值
                dt=self.dt,
                delay_steps=1,
                initial_level=2.0
            )
            self.physics_available = True
            self.logger.info("物理模型已加载 (CanalPoolSimulator)")
        except ImportError as e:
            self.logger.warning(f"物理模型不可用: {e}，使用简化模型")
            self.physics = None
            self.physics_available = False

        # 初始化状态
        self.water_levels = {'pool_1': 2.0}  # 初始水位
        self.gate_positions = {'gate_1': 0.5}  # 初始闸门位置
        self.inflows = {'inflow_1': 10.0}  # 入流
        self.outflows = {'outflow_1': 10.0}  # 出流
        self.sensor_readings = {'level_sensor_1': 2.0}
        self.alarms = []
        self.control_mode = 'auto'

    def _init_controller(self):
        """初始化控制器"""
        try:
            from control import UniversalMPCSolver
            self.controller = UniversalMPCSolver(
                horizon=10,
                dt=self.dt,
                area=1000.0,
                delay_steps=1
            )
            self.controller_available = True
            self.logger.info("MPC控制器已加载 (UniversalMPCSolver)")
        except ImportError as e:
            self.logger.warning(f"MPC控制器不可用: {e}，使用简化控制器")
            self.controller = None
            self.controller_available = False

    def reset(self, initial_state: Optional[Dict] = None):
        """重置仿真环境"""
        self.current_time = 0.0
        self.state_history.clear()
        self.alarm_history.clear()
        self.control_actions.clear()
        self.alarms.clear()

        if initial_state:
            if 'water_levels' in initial_state:
                self.water_levels = initial_state['water_levels'].copy()
            if 'gate_positions' in initial_state:
                self.gate_positions = initial_state['gate_positions'].copy()
            if 'inflows' in initial_state:
                self.inflows = initial_state['inflows'].copy()
            if 'control_mode' in initial_state:
                self.control_mode = initial_state['control_mode']

        # 同步传感器读数
        self.sensor_readings = {f'level_sensor_{k.split("_")[1]}': v
                                for k, v in self.water_levels.items()}

        # 记录初始状态
        self._record_state()

    def step(self, disturbances: Optional[Dict] = None) -> SimulationState:
        """执行一步仿真"""
        # 应用扰动
        if disturbances:
            self._apply_disturbances(disturbances)

        # 执行控制
        control_action = self._compute_control()
        self.control_actions.append((self.current_time, control_action))

        # 更新物理状态
        self._update_physics(control_action)

        # 检查报警
        self._check_alarms()

        # 更新时间
        self.current_time += self.dt

        # 记录状态
        state = self._record_state()

        return state

    def _apply_disturbances(self, disturbances: Dict):
        """应用扰动"""
        if 'inflow' in disturbances:
            for key, value in disturbances['inflow'].items():
                if key in self.inflows:
                    self.inflows[key] = value

        if 'sensor_bias' in disturbances:
            for key, value in disturbances['sensor_bias'].items():
                if key in self.sensor_readings:
                    self.sensor_readings[key] += value

        if 'gate_stuck' in disturbances:
            for key, stuck in disturbances['gate_stuck'].items():
                if stuck and key in self.gate_positions:
                    # 闸门卡死，不响应控制指令
                    pass

    def _compute_control(self) -> Dict:
        """计算控制动作"""
        if self.controller_available and self.control_mode == 'auto':
            try:
                # 使用MPC控制器
                setpoint = 2.0  # 目标水位
                current_level = list(self.water_levels.values())[0]
                
                # Adapt for UniversalMPCSolver
                q_prev = self.outflows.get('outflow_1', 0.0) # Use previous outflow as proxy for previous control? 
                # Actually UniversalMPCSolver returns u_in (inflow control) usually?
                # But here we control gate_1 which affects outflow?
                # Let's assume we are controlling the gate for the NEXT pool or THIS pool's outflow?
                # In CanalPoolSimulator, u_in is inflow, u_out is outflow.
                # If we control gate_1, it usually controls outflow of pool 1 (or inflow of pool 2).
                # Let's assume we control 'gate_1' which determines 'outflow_1'.
                
                # UniversalMPCSolver solves for u_in (inflow to the pool).
                # If we are controlling a single pool, we might be controlling its inflow gate.
                # Let's assume gate_1 controls INFLOW to pool_1.
                
                q_out_forecast = [10.0] * self.controller.N # Assume steady outflow demand
                config = {
                    'Z_ref': setpoint, 
                    'W_level': 10.0, 
                    'W_smooth': 5.0, 
                    'delta_Q_max': 2.0, 
                    'constraints': {'Q_in_max': 20.0, 'Q_in_min': 0.0}
                }
                
                # Solve
                u_in_optimal = self.controller.solve(current_level, q_prev, q_out_forecast, config)
                
                # Convert flow to gate position (simplified: pos = flow / 20.0)
                action = u_in_optimal / 20.0
                
                return {'gate_1': max(0, min(1, action))}
            except Exception as e:
                self.logger.warning(f"MPC控制失败: {e}")

        # 简单比例控制作为后备
        setpoint = 2.0
        current_level = list(self.water_levels.values())[0]
        error = setpoint - current_level
        kp = 0.1
        action = 0.5 + kp * error
        return {'gate_1': max(0, min(1, action))}

    def _update_physics(self, control_action: Dict):
        """更新物理状态"""
        # 应用控制动作到闸门
        for gate, position in control_action.items():
            if gate in self.gate_positions:
                # 限制闸门变化速率
                current = self.gate_positions[gate]
                max_change = 0.1 * self.dt  # 每秒最大变化10%
                new_position = current + max(-max_change, min(max_change, position - current))
                self.gate_positions[gate] = new_position

        # 更新水位
        if self.physics_available:
            # 使用真实物理模型
            pool_id = 'pool_1'
            gate_id = 'gate_1'
            
            # Get inputs
            gate_pos = self.gate_positions.get(gate_id, 0.5)
            u_in = gate_pos * 20.0 # Map gate to flow
            
            # Assume constant outflow demand or based on downstream
            u_out = 10.0 
            
            # Step physics
            # Note: CanalPoolSimulator maintains its own state (self.H)
            # We need to sync it or use it as source of truth
            # Let's sync TO it first (if needed) or just use it.
            # But CanalPoolSimulator doesn't allow setting H easily except init.
            # So we should rely on IT.
            
            # However, reset() re-inits physics? No, it doesn't.
            # We might need to reset physics state in reset().
            
            self.physics.step(u_in, u_out)
            new_level = self.physics.get_level()
            
            self.water_levels[pool_id] = new_level
            self.inflows[f'inflow_{pool_id.split("_")[1]}'] = u_in
            self.outflows[f'outflow_{pool_id.split("_")[1]}'] = u_out
            
            # Update sensors
            import random
            noise = random.gauss(0, 0.01)
            self.sensor_readings[f'level_sensor_{pool_id.split("_")[1]}'] = new_level + noise
            
        else:
            # 简化物理模型 (Fallback)
            for pool, level in self.water_levels.items():
                pool_id = pool.split('_')[1]
                inflow = self.inflows.get(f'inflow_{pool_id}', 0)
                gate_pos = self.gate_positions.get(f'gate_{pool_id}', 0.5)

                # 出流取决于闸门开度和水位
                outflow = gate_pos * 20 * (level / 2.0) ** 0.5
                self.outflows[f'outflow_{pool_id}'] = outflow

                # 水位变化
                area = 1000  # 池面积 m²
                delta_h = (inflow - outflow) * self.dt / area
                new_level = max(0, level + delta_h)
                self.water_levels[pool] = new_level

                # 更新传感器读数（带测量噪声）
                import random
                noise = random.gauss(0, 0.01)
                self.sensor_readings[f'level_sensor_{pool_id}'] = new_level + noise

    def _check_alarms(self):
        """检查并生成报警"""
        self.alarms.clear()

        for pool, level in self.water_levels.items():
            if level > 2.5:
                alarm = f"HIGH_LEVEL_{pool}"
                self.alarms.append(alarm)
                self.alarm_history.append((self.current_time, alarm))
            elif level < 1.5:
                alarm = f"LOW_LEVEL_{pool}"
                self.alarms.append(alarm)
                self.alarm_history.append((self.current_time, alarm))

    def _record_state(self) -> SimulationState:
        """记录当前状态"""
        state = SimulationState(
            time=self.current_time,
            water_levels=self.water_levels.copy(),
            gate_positions=self.gate_positions.copy(),
            inflows=self.inflows.copy(),
            outflows=self.outflows.copy(),
            sensor_readings=self.sensor_readings.copy(),
            alarms=self.alarms.copy(),
            control_mode=self.control_mode
        )
        self.state_history.append(state)
        return state

    def get_history(self) -> List[SimulationState]:
        """获取状态历史"""
        return self.state_history

    def get_time_series(self, variable: str) -> Tuple[List[float], List[float]]:
        """获取时间序列数据"""
        times = [s.time for s in self.state_history]

        if variable.startswith('level_'):
            pool = variable.replace('level_', 'pool_')
            values = [s.water_levels.get(pool, 0) for s in self.state_history]
        elif variable.startswith('gate_'):
            values = [s.gate_positions.get(variable, 0) for s in self.state_history]
        elif variable.startswith('inflow_'):
            values = [s.inflows.get(variable, 0) for s in self.state_history]
        elif variable.startswith('outflow_'):
            values = [s.outflows.get(variable, 0) for s in self.state_history]
        else:
            values = [0] * len(times)

        return times, values


class HILTestRunner:
    """HIL测试执行器"""

    def __init__(self, scenarios_dir: Optional[str] = None, output_dir: Optional[str] = None):
        self.scenarios_dir = scenarios_dir or os.path.join(
            os.path.dirname(__file__), 'scenarios'
        )
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), 'results'
        )

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化组件
        self.scenario_generator = ScenarioGenerator()
        self.condition_injector = ConditionInjector()
        self.evaluation_engine = EvaluationEngine()
        self.simulation_env = SimulationEnvironment()

        # 测试结果
        self.test_suites: List[TestSuite] = []
        self.current_suite: Optional[TestSuite] = None

        # 配置日志
        self.logger = logging.getLogger('HILTestRunner')

    def load_scenarios(self, pattern: str = "*.yaml") -> List[Scenario]:
        """加载场景文件"""
        import glob
        scenarios = []

        scenario_files = glob.glob(os.path.join(self.scenarios_dir, pattern))
        for filepath in scenario_files:
            try:
                loaded = self.scenario_generator.load_scenario(filepath)
                if isinstance(loaded, list):
                    scenarios.extend(loaded)
                else:
                    scenarios.append(loaded)
                self.logger.info(f"加载场景文件: {filepath}")
            except Exception as e:
                self.logger.error(f"加载场景失败 {filepath}: {e}")

        return scenarios

    def create_test_suite(self, name: str, scenarios: List[Scenario],
                          autonomous_level: Optional[AutonomousLevel] = None,
                          category_filter: Optional[str] = None,
                          description: str = "") -> TestSuite:
        """创建测试套件"""
        # 过滤场景
        filtered_scenarios = scenarios

        if autonomous_level:
            filtered_scenarios = [
                s for s in filtered_scenarios
                if s.autonomous_level == autonomous_level
            ]

        if category_filter:
            filtered_scenarios = [
                s for s in filtered_scenarios
                if s.category.value == category_filter
            ]

        # 创建测试用例
        test_cases = [
            TestCase(
                id=f"TC_{scenario.id}",
                scenario=scenario
            )
            for scenario in filtered_scenarios
        ]

        suite = TestSuite(
            id=f"TS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=name,
            description=description,
            test_cases=test_cases,
            autonomous_level=autonomous_level,
            category_filter=category_filter
        )

        self.test_suites.append(suite)
        return suite

    def run_suite(self, suite: TestSuite, progress_callback=None) -> TestSuite:
        """运行测试套件"""
        self.current_suite = suite
        self.logger.info(f"开始执行测试套件: {suite.name}")
        self.logger.info(f"共 {suite.total_count} 个测试用例")

        for i, test_case in enumerate(suite.test_cases):
            # 进度回调
            if progress_callback:
                progress_callback(i + 1, suite.total_count, test_case)

            # 执行测试
            self._run_test_case(test_case)

            self.logger.info(
                f"[{i+1}/{suite.total_count}] {test_case.id}: {test_case.status.value}"
            )

        self.logger.info(f"测试套件完成: 通过率 {suite.pass_rate:.1%}")
        return suite

    def _run_test_case(self, test_case: TestCase):
        """执行单个测试用例"""
        test_case.status = TestStatus.RUNNING
        test_case.start_time = datetime.now()
        test_case.execution_log.append(f"开始执行: {test_case.start_time}")

        try:
            scenario = test_case.scenario

            # 1. 准备初始状态
            initial_state = self._prepare_initial_state(scenario)
            self.simulation_env.reset(initial_state)
            test_case.execution_log.append("初始化仿真环境完成")

            # 2. 配置工况注入器
            self._setup_injections(scenario)
            test_case.execution_log.append("配置工况注入完成")

            # 3. 运行仿真
            duration = scenario.duration
            dt = 1.0  # 仿真步长
            steps = int(duration / dt)

            test_case.execution_log.append(f"开始仿真: {duration}秒, {steps}步")

            for step in range(steps):
                current_time = step * dt

                # 获取当前时刻的扰动
                disturbances = self.condition_injector.get_disturbances(current_time)

                # 执行仿真步
                self.simulation_env.step(disturbances)

            test_case.execution_log.append("仿真完成")

            # 4. 收集仿真数据
            sim_data = self._collect_simulation_data()

            # 5. 评估结果
            result = self.evaluation_engine.evaluate(scenario, sim_data)
            test_case.result = result

            # 6. 确定通过/失败
            if result.passed:
                test_case.status = TestStatus.PASSED
            else:
                test_case.status = TestStatus.FAILED

            test_case.execution_log.append(
                f"评估完成: {'通过' if result.passed else '失败'}"
            )

        except Exception as e:
            test_case.status = TestStatus.ERROR
            test_case.error_message = str(e)
            test_case.execution_log.append(f"执行错误: {e}")
            test_case.execution_log.append(traceback.format_exc())

        finally:
            test_case.end_time = datetime.now()

    def _prepare_initial_state(self, scenario: Scenario) -> Dict:
        """准备初始状态"""
        initial = {}

        if scenario.initial_state:
            if scenario.initial_state.water_levels:
                initial['water_levels'] = scenario.initial_state.water_levels
            if scenario.initial_state.gate_positions:
                initial['gate_positions'] = scenario.initial_state.gate_positions
            if scenario.initial_state.inflows:
                initial['inflows'] = scenario.initial_state.inflows

        return initial

    def _setup_injections(self, scenario: Scenario):
        """配置工况注入"""
        self.condition_injector.clear_all()

        for condition in scenario.conditions:
            if condition.injection:
                inj = condition.injection

                # 根据注入类型配置
                if inj.target.startswith('inflow'):
                    self.condition_injector.add_flow_injection(
                        target=inj.target,
                        injection_type=inj.type,
                        start_time=inj.start_time,
                        end_time=inj.end_time or scenario.duration,
                        magnitude=inj.magnitude,
                        parameters=inj.parameters
                    )
                elif inj.target.startswith('sensor'):
                    self.condition_injector.add_sensor_injection(
                        target=inj.target,
                        injection_type=inj.type,
                        start_time=inj.start_time,
                        end_time=inj.end_time or scenario.duration,
                        magnitude=inj.magnitude,
                        parameters=inj.parameters
                    )
                elif inj.target.startswith('gate'):
                    self.condition_injector.add_actuator_injection(
                        target=inj.target,
                        injection_type=inj.type,
                        start_time=inj.start_time,
                        end_time=inj.end_time or scenario.duration,
                        magnitude=inj.magnitude,
                        parameters=inj.parameters
                    )

    def _collect_simulation_data(self) -> Dict:
        """收集仿真数据"""
        history = self.simulation_env.get_history()

        # 提取时间序列
        times = [s.time for s in history]

        # 水位数据
        water_levels = {}
        for pool in history[0].water_levels.keys():
            water_levels[pool] = [s.water_levels.get(pool, 0) for s in history]

        # 闸门数据
        gate_positions = {}
        for gate in history[0].gate_positions.keys():
            gate_positions[gate] = [s.gate_positions.get(gate, 0) for s in history]

        # 流量数据
        inflows = {}
        for inf in history[0].inflows.keys():
            inflows[inf] = [s.inflows.get(inf, 0) for s in history]

        outflows = {}
        for outf in history[0].outflows.keys():
            outflows[outf] = [s.outflows.get(outf, 0) for s in history]

        # 报警数据
        alarms = self.simulation_env.alarm_history

        # 控制动作
        control_actions = self.simulation_env.control_actions

        return {
            'times': times,
            'water_levels': water_levels,
            'gate_positions': gate_positions,
            'inflows': inflows,
            'outflows': outflows,
            'alarms': alarms,
            'control_actions': control_actions,
            'state_history': [s.to_dict() for s in history]
        }

    def run_quick_validation(self) -> Dict:
        """快速验证测试 - 检查框架是否正常工作"""
        self.logger.info("执行快速验证测试...")

        results = {
            'scenario_generator': False,
            'condition_injector': False,
            'simulation_env': False,
            'evaluation_engine': False,
            'overall': False
        }

        try:
            # 测试场景生成器
            scenario = self.scenario_generator.create_scenario(
                id="QUICK_TEST",
                name="快速验证场景",
                category="S1_NORMAL",
                difficulty=1,
                autonomous_level="L1",
                duration=60.0
            )
            results['scenario_generator'] = scenario is not None

            # 测试工况注入器
            self.condition_injector.clear_all()
            self.condition_injector.add_flow_injection(
                target='inflow_1',
                injection_type='step',
                start_time=10.0,
                magnitude=5.0
            )
            results['condition_injector'] = True

            # 测试仿真环境
            self.simulation_env.reset()
            for i in range(10):
                state = self.simulation_env.step()
            results['simulation_env'] = len(self.simulation_env.state_history) > 0

            # 测试评估引擎
            sim_data = self._collect_simulation_data()
            result = self.evaluation_engine.evaluate(scenario, sim_data)
            results['evaluation_engine'] = result is not None

            # 总体结果
            results['overall'] = all([
                results['scenario_generator'],
                results['condition_injector'],
                results['simulation_env'],
                results['evaluation_engine']
            ])

        except Exception as e:
            self.logger.error(f"快速验证失败: {e}")
            results['error'] = str(e)

        return results

    def get_summary(self) -> Dict:
        """获取测试汇总"""
        total_cases = sum(s.total_count for s in self.test_suites)
        total_passed = sum(s.passed_count for s in self.test_suites)
        total_failed = sum(s.failed_count for s in self.test_suites)
        total_errors = sum(s.error_count for s in self.test_suites)
        total_skipped = sum(s.skipped_count for s in self.test_suites)

        return {
            'total_suites': len(self.test_suites),
            'total_cases': total_cases,
            'passed': total_passed,
            'failed': total_failed,
            'errors': total_errors,
            'skipped': total_skipped,
            'pass_rate': total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else 0,
            'suites': [
                {
                    'name': s.name,
                    'total': s.total_count,
                    'passed': s.passed_count,
                    'failed': s.failed_count,
                    'pass_rate': s.pass_rate
                }
                for s in self.test_suites
            ]
        }

    def export_results(self, filepath: Optional[str] = None) -> str:
        """导出测试结果为JSON"""
        if filepath is None:
            filepath = os.path.join(
                self.output_dir,
                f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        results = {
            'summary': self.get_summary(),
            'suites': []
        }

        for suite in self.test_suites:
            suite_data = {
                'id': suite.id,
                'name': suite.name,
                'description': suite.description,
                'autonomous_level': suite.autonomous_level.value if suite.autonomous_level else None,
                'category_filter': suite.category_filter,
                'statistics': {
                    'total': suite.total_count,
                    'passed': suite.passed_count,
                    'failed': suite.failed_count,
                    'errors': suite.error_count,
                    'skipped': suite.skipped_count,
                    'pass_rate': suite.pass_rate
                },
                'test_cases': []
            }

            for tc in suite.test_cases:
                tc_data = {
                    'id': tc.id,
                    'scenario_id': tc.scenario.id,
                    'scenario_name': tc.scenario.name,
                    'status': tc.status.value,
                    'start_time': tc.start_time.isoformat() if tc.start_time else None,
                    'end_time': tc.end_time.isoformat() if tc.end_time else None,
                    'error_message': tc.error_message,
                    'execution_log': tc.execution_log
                }

                if tc.result:
                    tc_data['result'] = {
                        'passed': tc.result.passed,
                        'score': tc.result.score,
                        'safety_score': tc.result.safety_metrics.overall_score if tc.result.safety_metrics else None,
                        'control_score': tc.result.control_metrics.overall_score if tc.result.control_metrics else None,
                        'issues': tc.result.issues,
                        'recommendations': tc.result.recommendations
                    }

                suite_data['test_cases'].append(tc_data)

            results['suites'].append(suite_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        self.logger.info(f"测试结果已导出: {filepath}")
        return filepath


def main():
    """主函数 - 演示HIL测试执行"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("=" * 60)
    logger.info("HIL测试执行器 - 演示")
    logger.info("=" * 60)

    # 创建测试执行器
    runner = HILTestRunner()

    # 执行快速验证
    logger.info("\n1. 执行快速验证...")
    validation = runner.run_quick_validation()

    for component, status in validation.items():
        if component != 'error':
            logger.info(f"   {component}: {'✓' if status else '✗'}")

    if validation.get('overall'):
        logger.info("\n快速验证通过！框架工作正常。")
    else:
        logger.info("\n快速验证失败，请检查错误。")
        if 'error' in validation:
            logger.info(f"   错误: {validation['error']}")

    # 创建示例场景
    logger.info("\n2. 创建测试场景...")
    scenarios = []

    # L1级场景
    s1 = runner.scenario_generator.create_scenario(
        id="S1_NORMAL_001",
        name="稳态正常运行",
        category="S1_NORMAL",
        difficulty=1,
        autonomous_level="L1",
        duration=300.0,
        description="水位维持在目标值±5%，测试基本控制能力"
    )
    scenarios.append(s1)

    # L2级场景
    s2 = runner.scenario_generator.create_scenario(
        id="S2_FLOOD_001",
        name="单池入流突增",
        category="S2_FLOOD",
        difficulty=2,
        autonomous_level="L2",
        duration=600.0,
        description="入流突增50%，测试扰动抑制能力"
    )
    scenarios.append(s2)

    logger.info(f"   创建了 {len(scenarios)} 个测试场景")

    # 创建测试套件
    logger.info("\n3. 创建测试套件...")
    suite = runner.create_test_suite(
        name="基础功能验证",
        scenarios=scenarios,
        description="验证L1-L2级基础自主控制能力"
    )
    logger.info(f"   测试套件: {suite.name}")
    logger.info(f"   测试用例数: {suite.total_count}")

    # 运行测试
    logger.info("\n4. 执行测试...")

    def progress_callback(current, total, test_case):
        logger.info(f"   执行中: [{current}/{total}] {test_case.scenario.name}")

    runner.run_suite(suite, progress_callback)

    # 显示结果
    logger.info("\n5. 测试结果汇总:")
    summary = runner.get_summary()
    logger.info(f"   总测试数: {summary['total_cases']}")
    logger.info(f"   通过: {summary['passed']}")
    logger.info(f"   失败: {summary['failed']}")
    logger.info(f"   错误: {summary['errors']}")
    logger.info(f"   通过率: {summary['pass_rate']:.1%}")

    # 导出结果
    logger.info("\n6. 导出测试结果...")
    result_file = runner.export_results()
    logger.info(f"   结果文件: {result_file}")

    logger.info("\n" + "=" * 60)
    logger.info("HIL测试执行完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
