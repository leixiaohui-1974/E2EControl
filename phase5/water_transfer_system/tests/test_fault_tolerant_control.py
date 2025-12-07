"""
故障诊断与容错控制系统测试
Test suite for Fault Diagnosis and Fault-Tolerant Control System
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from water_transfer_system.fault_tolerant_control import (
    FaultType, FaultSeverity, FaultStatus, ControlMode, EmergencyType,
    FaultEvent, DiagnosisResult, ControlReconfiguration,
    EmergencyEvent, EmergencyResponse, OperatingRule,
    ResidualGenerator, SensorFaultDetector, ActuatorFaultDetector,
    FaultDetectionEngine, FaultDiagnosisEngine, FaultTolerantController,
    EmergencyResponseSystem, RuleCondition, RuleAction, OperatingRuleEngine,
    FaultTolerantSystem,
)


def test_fault_types():
    """测试故障类型定义"""
    print("测试1: 故障类型定义...")

    # 传感器故障
    assert FaultType.SENSOR_DRIFT.value > 0
    assert FaultType.SENSOR_STUCK.value > 0
    assert FaultType.SENSOR_LOSS.value > 0

    # 执行器故障
    assert FaultType.ACTUATOR_STUCK.value > 0
    assert FaultType.ACTUATOR_SATURATION.value > 0

    # 严重程度
    assert FaultSeverity.INFO.value < FaultSeverity.WARNING.value
    assert FaultSeverity.WARNING.value < FaultSeverity.CRITICAL.value
    assert FaultSeverity.CRITICAL.value < FaultSeverity.EMERGENCY.value

    # 控制模式
    assert ControlMode.NORMAL.value > 0
    assert ControlMode.DEGRADED.value > 0
    assert ControlMode.EMERGENCY.value > 0

    print("  ✓ 故障类型定义正确")
    return True


def test_sensor_fault_detector():
    """测试传感器故障检测器"""
    print("测试2: 传感器故障检测器...")

    detector = SensorFaultDetector()

    # 测试正常数据 - 使用正弦波动确保有变化
    import math
    for i in range(20):
        value = 2.5 + 0.1 * math.sin(i * 0.5)  # 正弦波动
        detector.add_reading("sensor_1", value, float(i))

    faults = detector.detect_faults("sensor_1")
    # 正常波动数据不应检测到卡死
    assert FaultType.SENSOR_STUCK not in faults, "正常波动不应检测到卡死"

    # 测试卡死检测 - 完全相同的值
    detector2 = SensorFaultDetector()
    for i in range(20):
        detector2.add_reading("sensor_2", 2.5, float(i))  # 恒定值

    faults = detector2.detect_faults("sensor_2")
    assert FaultType.SENSOR_STUCK in faults, "应检测到卡死故障"

    # 测试漂移检测 - 明显持续上升
    detector3 = SensorFaultDetector()
    for i in range(20):
        detector3.add_reading("sensor_3", 2.5 + i * 0.1, float(i))  # 明显上升

    faults = detector3.detect_faults("sensor_3")
    assert FaultType.SENSOR_DRIFT in faults, "应检测到漂移故障"

    print("  ✓ 传感器故障检测正确")
    return True


def test_actuator_fault_detector():
    """测试执行器故障检测器"""
    print("测试3: 执行器故障检测器...")

    detector = ActuatorFaultDetector()

    # 测试正常响应
    for i in range(30):
        command = 0.5 + 0.1 * (i % 3 - 1)
        position = command - 0.02  # 略有滞后
        detector.add_command("actuator_1", command, float(i))
        detector.add_position("actuator_1", position, float(i))

    faults = detector.detect_faults("actuator_1", 30.0)
    stuck_detected = any(f[0] == FaultType.ACTUATOR_STUCK for f in faults)
    # 正常情况不应该检测到卡死
    assert not stuck_detected or faults[0][1] < 0.5, "正常数据不应有高置信度卡死检测"

    # 测试卡死
    detector2 = ActuatorFaultDetector()
    for i in range(30):
        detector2.add_command("actuator_2", 0.3 + i * 0.01, float(i))  # 变化的命令
        detector2.add_position("actuator_2", 0.5, float(i))  # 固定位置

    faults = detector2.detect_faults("actuator_2", 30.0)
    stuck_faults = [(f, c) for f, c in faults if f == FaultType.ACTUATOR_STUCK]
    assert len(stuck_faults) > 0 and stuck_faults[0][1] > 0.5, "应检测到卡死故障"

    print("  ✓ 执行器故障检测正确")
    return True


def test_residual_generator():
    """测试残差生成器"""
    print("测试4: 残差生成器...")

    generator = ResidualGenerator(pool_id=1)

    # 正常情况
    for i in range(50):
        residual = generator.compute_residual(
            measured_level=2.5 + 0.01 * (i % 3 - 1),
            measured_flow=10.0 + 0.1 * (i % 5 - 2),
            predicted_level=2.5,
            predicted_flow=10.0,
        )

    assert residual['level_alarm'] == False, "正常残差不应触发告警"
    assert residual['flow_alarm'] == False, "正常残差不应触发告警"

    # 异常情况
    generator2 = ResidualGenerator(pool_id=2)
    for i in range(50):
        residual = generator2.compute_residual(
            measured_level=2.5 + 0.5,  # 持续偏差
            measured_flow=10.0,
            predicted_level=2.5,
            predicted_flow=10.0,
        )

    assert residual['level_alarm'] == True, "持续偏差应触发告警"

    print("  ✓ 残差生成器正确")
    return True


def test_fault_detection_engine():
    """测试综合故障检测引擎"""
    print("测试5: 综合故障检测引擎...")

    engine = FaultDetectionEngine(num_pools=3)

    # 模拟传感器读数
    for t in range(20):
        # 正常传感器
        engine.process_sensor_reading(
            "level_0", 2.5 + 0.01 * t, "level", "pool_0", float(t)
        )

        # 卡死传感器
        engine.process_sensor_reading(
            "level_1", 2.5, "level", "pool_1", float(t)
        )

    # 检查检测到的故障
    faults = engine.get_active_faults()
    stuck_faults = [f for f in faults if f.fault_type == FaultType.SENSOR_STUCK]
    assert len(stuck_faults) > 0, "应检测到传感器卡死"

    # 按位置筛选
    pool1_faults = engine.get_faults_by_location("pool_1")
    assert len(pool1_faults) > 0, "pool_1 应有故障"

    print("  ✓ 综合故障检测正确")
    return True


def test_fault_diagnosis():
    """测试故障诊断"""
    print("测试6: 故障诊断引擎...")

    engine = FaultDiagnosisEngine()

    # 创建模拟故障
    faults = [
        FaultEvent(
            fault_id="F001",
            fault_type=FaultType.ACTUATOR_STUCK,
            severity=FaultSeverity.CRITICAL,
            component_id="gate_2",
            component_type="actuator",
            location="pool_2",
            detected_time=100.0,
            description="Gate 2 stuck",
        )
    ]

    diagnosis = engine.diagnose(faults)
    assert len(diagnosis) > 0, "应生成诊断结果"
    assert diagnosis[0].root_cause is not None, "应有根因分析"
    assert len(diagnosis[0].recommended_actions) > 0, "应有推荐动作"

    print("  ✓ 故障诊断正确")
    return True


def test_fault_tolerant_controller():
    """测试容错控制器"""
    print("测试7: 容错控制器...")

    controller = FaultTolerantController(num_pools=5)

    # 注册备份
    controller.register_backup("sensor_1", "sensor_1_backup", "sensor")
    controller.register_backup("gate_1", "gate_1_backup", "actuator")

    # 创建故障
    faults = [
        FaultEvent(
            fault_id="F001",
            fault_type=FaultType.SENSOR_LOSS,
            severity=FaultSeverity.MAJOR,
            component_id="sensor_1",
            component_type="sensor",
            location="pool_1",
            detected_time=100.0,
            description="Sensor 1 loss",
        )
    ]

    # 重构控制
    reconfig = controller.reconfigure(faults, [], 100.0)

    assert reconfig.mode in [ControlMode.DEGRADED, ControlMode.BACKUP], \
        f"应切换到降级或备用模式, 实际: {reconfig.mode}"
    assert reconfig.reason is not None, "应有切换原因"

    # 测试严重故障
    critical_faults = [
        FaultEvent(
            fault_id="F002",
            fault_type=FaultType.ACTUATOR_LOSS,
            severity=FaultSeverity.EMERGENCY,
            component_id="gate_0",
            component_type="actuator",
            location="gate_0",
            detected_time=200.0,
            description="Gate 0 complete failure",
        )
    ]

    reconfig = controller.reconfigure(critical_faults, [], 200.0)
    assert reconfig.mode == ControlMode.SAFE_SHUTDOWN, "紧急故障应触发安全停机"

    print("  ✓ 容错控制器正确")
    return True


def test_emergency_response():
    """测试应急响应系统"""
    print("测试8: 应急响应系统...")

    system = EmergencyResponseSystem()

    # 报告应急事件
    event = system.report_emergency(
        event_type=EmergencyType.POLLUTION,
        location="pool_5",
        description="Water quality contamination detected",
        severity=FaultSeverity.CRITICAL,
        timestamp=100.0,
    )

    assert event.event_id is not None
    assert event.event_type == EmergencyType.POLLUTION
    assert len(event.affected_area) > 0, "应估计影响范围"

    # 生成响应
    response = system.generate_response(event)

    assert response.response_level in [1, 2, 3, 4]
    assert len(response.actions) > 0, "应有应急措施"
    assert len(response.communication_plan) > 0, "应有通信计划"

    # 检查活动应急事件
    active = system.get_active_emergencies()
    assert len(active) == 1

    # 解除应急
    system.resolve_emergency(event.event_id, 500.0)
    active = system.get_active_emergencies()
    assert len(active) == 0

    print("  ✓ 应急响应系统正确")
    return True


def test_rule_conditions():
    """测试规则条件"""
    print("测试9: 规则条件...")

    # 水位条件
    level_above = RuleCondition.level_above(1, 3.5)
    assert level_above({'pool_1_level': 4.0}) == True
    assert level_above({'pool_1_level': 3.0}) == False

    level_below = RuleCondition.level_below(2, 1.5)
    assert level_below({'pool_2_level': 1.0}) == True
    assert level_below({'pool_2_level': 2.0}) == False

    # 时间条件
    time_check = RuleCondition.time_in_range(22, 6)
    assert time_check({'current_hour': 23}) == True
    assert time_check({'current_hour': 3}) == True
    assert time_check({'current_hour': 12}) == False

    print("  ✓ 规则条件正确")
    return True


def test_rule_engine():
    """测试运行规则引擎"""
    print("测试10: 运行规则引擎...")

    engine = OperatingRuleEngine()

    # 添加自定义规则
    engine.add_rule(OperatingRule(
        rule_id="TEST_001",
        name="测试规则",
        priority=5,
        condition="水位超过3.5m",
        action="发送告警",
        condition_func=RuleCondition.level_above(0, 3.5),
        action_func=RuleAction.send_alert("Test alert", "warning"),
        category="test",
    ))

    # 规则统计
    stats = engine.get_rule_statistics()
    assert stats['total_rules'] > 0

    # 评估规则 - 不触发
    state = {'pool_0_level': 3.0}
    actions = engine.evaluate(state, None, 100.0)
    test_actions = [a for a in actions if a.get('rule_id') == 'TEST_001']
    assert len(test_actions) == 0, "不应触发规则"

    # 评估规则 - 触发
    state = {'pool_0_level': 4.0}
    actions = engine.evaluate(state, None, 200.0)
    test_actions = [a for a in actions if a.get('rule_id') == 'TEST_001']
    assert len(test_actions) > 0, "应触发规则"
    assert test_actions[0]['action'] == 'alert'

    print("  ✓ 运行规则引擎正确")
    return True


def test_integrated_fault_tolerant_system():
    """测试综合故障诊断与容错系统"""
    print("测试11: 综合故障诊断与容错系统...")

    system = FaultTolerantSystem(num_pools=3)

    # 注册备份
    system.register_backup_sensor("level_sensor_0", "level_sensor_0_backup")
    system.register_backup_actuator("gate_0", "gate_0_backup")

    # 测试系统能够处理状态更新
    import math
    for t in range(10):
        state = {
            'pool_0_level': 2.5 + 0.1 * math.sin(t * 0.3),
            'pool_0_flow': 10.0 + 0.5 * math.sin(t * 0.2),
            'pool_1_level': 2.5 + 0.1 * math.sin(t * 0.25 + 1),
            'pool_1_flow': 10.0 + 0.5 * math.sin(t * 0.22 + 1),
            'pool_2_level': 2.5 + 0.1 * math.sin(t * 0.28 + 2),
            'pool_2_flow': 10.0 + 0.5 * math.sin(t * 0.18 + 2),
            'gate_0_command': 0.5 + 0.02 * math.sin(t * 0.1),
            'gate_0_position': 0.5 + 0.02 * math.sin(t * 0.1 - 0.5),
            'gate_1_command': 0.5 + 0.02 * math.sin(t * 0.12),
            'gate_1_position': 0.5 + 0.02 * math.sin(t * 0.12 - 0.5),
            'gate_2_command': 0.5 + 0.02 * math.sin(t * 0.15),
            'gate_2_position': 0.5 + 0.02 * math.sin(t * 0.15 - 0.5),
            'gate_3_command': 0.5 + 0.02 * math.sin(t * 0.11),
            'gate_3_position': 0.5 + 0.02 * math.sin(t * 0.11 - 0.5),
        }

        result = system.process_step(state, float(t))
        # 确保结果包含必要字段
        assert 'timestamp' in result
        assert 'faults_detected' in result
        assert 'system_health' in result

    # 获取系统状态
    status = system.get_system_status()
    assert 'health' in status
    assert 'control_mode' in status
    assert 'active_faults' in status
    assert 'rules_enabled' in status

    # 模拟故障 - 传感器卡死（所有值恒定）
    for t in range(30, 60):
        state = {
            'pool_0_level': 2.5,  # 完全卡死
            'pool_0_flow': 10.0,
            'pool_1_level': 2.5,
            'pool_1_flow': 10.0,
            'pool_2_level': 2.5,
            'pool_2_flow': 10.0,
            'gate_0_command': 0.5,
            'gate_0_position': 0.5,
            'gate_1_command': 0.5,
            'gate_1_position': 0.5,
            'gate_2_command': 0.5,
            'gate_2_position': 0.5,
            'gate_3_command': 0.5,
            'gate_3_position': 0.5,
        }

        result = system.process_step(state, float(t))

    # 应检测到故障（所有传感器都卡死了）
    status = system.get_system_status()
    assert status['active_faults'] > 0, "卡死情况应检测到故障"

    print("  ✓ 综合系统正确")
    return True


def test_control_mode_transitions():
    """测试控制模式切换"""
    print("测试12: 控制模式切换...")

    controller = FaultTolerantController(num_pools=3)

    # 轻微故障 -> 降级模式
    minor_fault = [FaultEvent(
        fault_id="F1", fault_type=FaultType.SENSOR_NOISE,
        severity=FaultSeverity.MINOR, component_id="s1",
        component_type="sensor", location="pool_0",
        detected_time=0, description="Minor noise"
    )]
    reconfig = controller.reconfigure(minor_fault, [], 0)
    assert reconfig.mode == ControlMode.DEGRADED

    # 主要故障 -> 可能降级或备用
    major_fault = [FaultEvent(
        fault_id="F2", fault_type=FaultType.SENSOR_STUCK,
        severity=FaultSeverity.MAJOR, component_id="s2",
        component_type="sensor", location="pool_1",
        detected_time=1, description="Major stuck"
    )]
    reconfig = controller.reconfigure(major_fault, [], 1)
    assert reconfig.mode in [ControlMode.DEGRADED, ControlMode.BACKUP]

    # 严重故障 -> 应急模式
    critical_fault = [FaultEvent(
        fault_id="F3", fault_type=FaultType.ACTUATOR_STUCK,
        severity=FaultSeverity.CRITICAL, component_id="a1",
        component_type="actuator", location="gate_0",
        detected_time=2, description="Critical stuck"
    )]
    reconfig = controller.reconfigure(critical_fault, [], 2)
    assert reconfig.mode == ControlMode.EMERGENCY

    # 紧急故障 -> 安全停机
    emergency_fault = [FaultEvent(
        fault_id="F4", fault_type=FaultType.ACTUATOR_LOSS,
        severity=FaultSeverity.EMERGENCY, component_id="a2",
        component_type="actuator", location="gate_1",
        detected_time=3, description="Emergency loss"
    )]
    reconfig = controller.reconfigure(emergency_fault, [], 3)
    assert reconfig.mode == ControlMode.SAFE_SHUTDOWN

    print("  ✓ 控制模式切换正确")
    return True


def test_rule_execution():
    """测试规则执行"""
    print("测试13: 规则执行与历史...")

    engine = OperatingRuleEngine()

    # 高水位触发测试
    state = {
        'pool_0_level': 4.0,
        'pool_1_level': 4.0,
        'pool_2_level': 4.0,
    }

    actions = engine.evaluate(state, None, 100.0)

    # 应触发高水位告警规则
    alert_actions = [a for a in actions if a.get('action') == 'alert']
    assert len(alert_actions) > 0, "应触发告警规则"

    # 检查规则统计
    stats = engine.get_rule_statistics()
    assert stats['total_executions'] > 0

    print("  ✓ 规则执行正确")
    return True


def test_fault_propagation_analysis():
    """测试故障传播分析"""
    print("测试14: 故障传播分析...")

    engine = FaultDiagnosisEngine()

    # 设置传播图
    engine.update_propagation_graph({
        'pool_0': ['pool_1', 'pool_2'],
        'pool_1': ['pool_2', 'pool_3'],
        'pool_2': ['pool_3', 'pool_4'],
    })

    # 创建故障
    fault = FaultEvent(
        fault_id="F1",
        fault_type=FaultType.OVERFLOW,
        severity=FaultSeverity.CRITICAL,
        component_id="level_0",
        component_type="sensor",
        location="pool_0",
        detected_time=100.0,
        description="Overflow at pool 0"
    )

    # 分析传播
    affected = engine.analyze_propagation(fault)
    assert 'pool_1' in affected
    assert 'pool_2' in affected

    print("  ✓ 故障传播分析正确")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("故障诊断与容错控制系统测试")
    print("=" * 60)
    print()

    tests = [
        test_fault_types,
        test_sensor_fault_detector,
        test_actuator_fault_detector,
        test_residual_generator,
        test_fault_detection_engine,
        test_fault_diagnosis,
        test_fault_tolerant_controller,
        test_emergency_response,
        test_rule_conditions,
        test_rule_engine,
        test_integrated_fault_tolerant_system,
        test_control_mode_transitions,
        test_rule_execution,
        test_fault_propagation_analysis,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test.__name__} 失败")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__} 异常: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
