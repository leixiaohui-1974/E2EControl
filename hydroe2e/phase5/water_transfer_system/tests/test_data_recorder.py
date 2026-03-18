"""
历史数据记录与仿真回放系统测试
Test suite for Data Recording and Replay System
"""

import math
import tempfile

from hydroe2e.phase5.water_transfer_system.data_recorder import (
    DataChannel, RecordingMode, PlaybackState,
    DataPoint, TimeSeriesSegment, EventRecord, SimulationSnapshot,
    RecordingSession, PlaybackFrame,
    TimeSeriesStorage, SimulationRecorderV2, SimulationReplayer,
    DataAnalyzer, DataExporter, DataImporter, DataRecordingSystem,
)


def test_data_types():
    """测试数据类型定义"""
    print("测试1: 数据类型定义...")

    # 数据通道
    assert DataChannel.POOL_LEVEL.value > 0
    assert DataChannel.GATE_POSITION.value > 0
    assert DataChannel.FAULT_EVENT.value > 0

    # 记录模式
    assert RecordingMode.FULL.value > 0
    assert RecordingMode.SAMPLED.value > 0
    assert RecordingMode.COMPRESSED.value > 0

    # 回放状态
    assert PlaybackState.STOPPED.value > 0
    assert PlaybackState.PLAYING.value > 0
    assert PlaybackState.PAUSED.value > 0

    print("  ✓ 数据类型定义正确")
    return True


def test_time_series_segment():
    """测试时间序列片段"""
    print("测试2: 时间序列片段...")

    segment = TimeSeriesSegment(
        channel=DataChannel.POOL_LEVEL,
        source_id="pool_1",
        start_time=0.0,
        end_time=0.0,
        sample_rate=1.0,
    )

    # 添加数据
    for i in range(100):
        segment.add_point(float(i), 2.5 + 0.1 * math.sin(i * 0.1))

    assert len(segment.values) == 100
    assert segment.end_time == 99.0

    # 获取插值
    value = segment.get_value_at(50.5)
    assert value is not None
    assert abs(value - 2.5) < 0.2  # 合理范围内

    # 获取切片
    slice_seg = segment.get_slice(20.0, 40.0)
    assert len(slice_seg.values) > 0
    assert all(20.0 <= t <= 40.0 for t in slice_seg.timestamps)

    print("  ✓ 时间序列片段正确")
    return True


def test_time_series_storage():
    """测试时序存储引擎"""
    print("测试3: 时序存储引擎...")

    storage = TimeSeriesStorage()

    # 添加数据点
    for i in range(100):
        point = DataPoint(
            timestamp=float(i),
            channel=DataChannel.POOL_LEVEL,
            source_id="pool_0",
            value=2.5 + 0.1 * i,
        )
        storage.add_data_point(point)

    assert storage.total_points == 100

    # 获取数据
    segment = storage.get_data(DataChannel.POOL_LEVEL, "pool_0", 0, 100)
    assert segment is not None
    assert len(segment.values) > 0

    # 获取指定时间值
    value = storage.get_value_at(DataChannel.POOL_LEVEL, "pool_0", 50.0)
    assert value is not None
    assert abs(value - 7.5) < 0.1  # 2.5 + 0.1 * 50

    # 添加事件
    event = EventRecord(
        event_id="EVT_001",
        timestamp=50.0,
        event_type="fault",
        source="sensor_1",
        severity="warning",
        description="Test event",
    )
    storage.add_event(event)

    events = storage.get_events(event_type="fault")
    assert len(events) == 1
    assert events[0].event_id == "EVT_001"

    # 统计
    stats = storage.get_statistics()
    assert stats['total_points'] == 100
    assert stats['total_events'] == 1

    print("  ✓ 时序存储引擎正确")
    return True


def test_simulation_recorder():
    """测试仿真记录器"""
    print("测试4: 仿真记录器...")

    recorder = SimulationRecorderV2(mode=RecordingMode.FULL)
    recorder.start_recording(0.0, "Test simulation")

    # 记录数据
    for t in range(60):
        # 渠池状态
        recorder.record_pool_state(0, float(t), 2.5 + 0.1 * t, 10.0)
        recorder.record_pool_state(1, float(t), 2.6 + 0.08 * t, 9.5)

        # 闸门状态
        recorder.record_gate_state(0, float(t), 0.5, 0.5)
        recorder.record_gate_state(1, float(t), 0.6, 0.6)

        # 性能指标
        recorder.record_performance_metric("efficiency", float(t), 0.95)

    # 记录事件
    recorder.record_event("scenario", "system", 30.0, "info",
                          "Scenario started", {"type": "flood"})

    # 记录快照
    recorder.record_snapshot(
        30.0,
        pool_states={0: {'level': 5.5, 'flow': 10.0}},
        gate_states={0: {'position': 0.5, 'command': 0.5}},
        system_metrics={'health': 0.95},
    )

    recorder.stop_recording(60.0)

    # 检查会话信息
    info = recorder.get_session_info()
    assert info['total_points'] > 0
    assert info['total_events'] == 1
    assert info['total_snapshots'] == 1

    print("  ✓ 仿真记录器正确")
    return True


def test_simulation_replayer():
    """测试仿真回放器"""
    print("测试5: 仿真回放器...")

    # 准备数据
    storage = TimeSeriesStorage()

    for i in range(100):
        storage.add_data_point(DataPoint(
            timestamp=float(i),
            channel=DataChannel.POOL_LEVEL,
            source_id="pool_0",
            value=2.5 + 0.1 * math.sin(i * 0.1),
        ))
        storage.add_data_point(DataPoint(
            timestamp=float(i),
            channel=DataChannel.GATE_POSITION,
            source_id="gate_0",
            value=0.5,
        ))

    storage.add_event(EventRecord(
        event_id="EVT_001",
        timestamp=50.0,
        event_type="test",
        source="test",
        severity="info",
        description="Test event",
    ))

    # 创建回放器
    replayer = SimulationReplayer(storage)

    assert replayer.state == PlaybackState.STOPPED
    assert replayer.start_time == 0.0
    assert replayer.end_time == 99.0

    # 播放控制
    replayer.play()
    assert replayer.state == PlaybackState.PLAYING

    # 步进
    frame = replayer.step(10.0)
    assert frame is not None
    assert frame.timestamp > 0

    # 暂停
    replayer.pause()
    assert replayer.state == PlaybackState.PAUSED

    # 跳转
    replayer.seek(50.0)
    assert abs(replayer.current_time - 50.0) < 0.1

    # 获取帧
    frame = replayer.get_frame_at(50.0)
    assert frame.timestamp == 50.0
    assert 0 in frame.pool_states

    # 进度
    progress = replayer.get_progress()
    assert 0 <= progress <= 1

    # 回放信息
    info = replayer.get_playback_info()
    assert 'state' in info
    assert 'current_time' in info

    print("  ✓ 仿真回放器正确")
    return True


def test_data_analyzer():
    """测试数据分析器"""
    print("测试6: 数据分析器...")

    storage = TimeSeriesStorage()

    # 添加正弦数据
    for i in range(200):
        storage.add_data_point(DataPoint(
            timestamp=float(i),
            channel=DataChannel.POOL_LEVEL,
            source_id="pool_0",
            value=2.5 + 0.5 * math.sin(i * 0.05),
        ))

    # 添加线性上升数据
    for i in range(200):
        storage.add_data_point(DataPoint(
            timestamp=float(i),
            channel=DataChannel.POOL_FLOW,
            source_id="pool_0",
            value=10.0 + 0.1 * i,
        ))

    analyzer = DataAnalyzer(storage)

    # 统计量
    stats = analyzer.compute_statistics(DataChannel.POOL_LEVEL, "pool_0")
    assert 'mean' in stats
    assert 'std' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert abs(stats['mean'] - 2.5) < 0.1

    # 趋势检测
    trends = analyzer.detect_trends(DataChannel.POOL_FLOW, "pool_0")
    assert len(trends) > 0
    assert trends[0]['type'] == 'rising'

    # 异常检测
    # 添加一个异常点
    storage.add_data_point(DataPoint(
        timestamp=250.0,
        channel=DataChannel.POOL_LEVEL,
        source_id="pool_0",
        value=10.0,  # 远超正常范围
    ))

    anomalies = analyzer.detect_anomalies(DataChannel.POOL_LEVEL, "pool_0")
    assert len(anomalies) > 0

    # 汇总报告
    report = analyzer.generate_summary_report()
    assert 'total_data_points' in report
    assert 'channel_statistics' in report

    print("  ✓ 数据分析器正确")
    return True


def test_correlation_analysis():
    """测试相关性分析"""
    print("测试7: 相关性分析...")

    storage = TimeSeriesStorage()

    # 添加相关数据（同相位）
    for i in range(100):
        storage.add_data_point(DataPoint(
            timestamp=float(i),
            channel=DataChannel.POOL_LEVEL,
            source_id="pool_0",
            value=2.5 + 0.5 * math.sin(i * 0.1),
        ))
        storage.add_data_point(DataPoint(
            timestamp=float(i),
            channel=DataChannel.POOL_LEVEL,
            source_id="pool_1",
            value=2.5 + 0.5 * math.sin(i * 0.1),
        ))

    analyzer = DataAnalyzer(storage)

    # 相关性应该接近1
    corr = analyzer.compute_correlation(
        DataChannel.POOL_LEVEL, "pool_0",
        DataChannel.POOL_LEVEL, "pool_1"
    )
    assert abs(corr - 1.0) < 0.01

    print("  ✓ 相关性分析正确")
    return True


def test_data_export_import():
    """测试数据导入导出"""
    print("测试8: 数据导入导出...")

    storage = TimeSeriesStorage()

    # 添加数据
    for i in range(50):
        storage.add_data_point(DataPoint(
            timestamp=float(i),
            channel=DataChannel.POOL_LEVEL,
            source_id="pool_0",
            value=2.5 + 0.1 * i,
        ))

    storage.add_event(EventRecord(
        event_id="EVT_001",
        timestamp=25.0,
        event_type="test",
        source="test",
        severity="info",
        description="Test event",
    ))

    storage.add_snapshot(SimulationSnapshot(
        snapshot_id="SNAP_001",
        timestamp=25.0,
        pool_states={0: {'level': 5.0}},
        gate_states={0: {'position': 0.5}},
        control_states={},
        active_events=[],
        system_metrics={'health': 0.95},
    ))

    exporter = DataExporter(storage)

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        # 导出
        exporter.export_to_json(temp_path, compress=False)

        # 导入到新存储
        new_storage = TimeSeriesStorage()
        importer = DataImporter(new_storage)
        importer.import_from_json(temp_path)

        # 验证
        assert new_storage.total_points == storage.total_points
        assert len(new_storage.events) == len(storage.events)
        assert len(new_storage.snapshots) == len(storage.snapshots)

    finally:
        # 清理
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("  ✓ 数据导入导出正确")
    return True


def test_compressed_export():
    """测试压缩导出"""
    print("测试9: 压缩导出...")

    storage = TimeSeriesStorage()

    for i in range(100):
        storage.add_data_point(DataPoint(
            timestamp=float(i),
            channel=DataChannel.POOL_LEVEL,
            source_id="pool_0",
            value=2.5 + 0.1 * i,
        ))

    exporter = DataExporter(storage)

    with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as f:
        temp_path = f.name

    try:
        # 压缩导出
        exporter.export_to_json(temp_path, compress=True)

        # 验证文件存在
        assert os.path.exists(temp_path)

        # 导入
        new_storage = TimeSeriesStorage()
        importer = DataImporter(new_storage)
        importer.import_from_json(temp_path)

        assert new_storage.total_points == 100

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("  ✓ 压缩导出正确")
    return True


def test_recording_system():
    """测试综合记录系统"""
    print("测试10: 综合记录系统...")

    system = DataRecordingSystem(mode=RecordingMode.FULL)
    system.start("Test simulation")

    # 模拟仿真过程
    for t in range(100):
        pool_states = {
            0: {'level': 2.5 + 0.1 * math.sin(t * 0.1), 'flow': 10.0},
            1: {'level': 2.6 + 0.1 * math.sin(t * 0.1 + 0.5), 'flow': 9.5},
        }
        gate_states = {
            0: {'position': 0.5, 'command': 0.5},
            1: {'position': 0.6, 'command': 0.6},
        }
        events = []
        if t == 50:
            events = [{'type': 'test', 'source': 'system',
                       'severity': 'info', 'description': 'Mid-point'}]

        metrics = {'efficiency': 0.95, 'stability': 0.9}

        system.record_simulation_step(float(t), pool_states, gate_states,
                                      events, metrics)

    system.stop(100.0)

    # 创建回放器
    replayer = system.create_replayer()
    assert replayer is not None

    # 获取分析报告
    report = system.get_analysis_report()
    assert 'total_data_points' in report
    assert report['total_data_points'] > 0

    print("  ✓ 综合记录系统正确")
    return True


def test_sampled_recording():
    """测试采样记录模式"""
    print("测试11: 采样记录模式...")

    recorder = SimulationRecorderV2(
        mode=RecordingMode.SAMPLED,
        sample_interval=5.0  # 每5秒采样一次
    )
    recorder.start_recording(0.0)

    # 快速记录（应该只保留每5秒的采样）
    for t in range(100):
        recorder.record_pool_state(0, float(t), 2.5, 10.0)

    recorder.stop_recording(100.0)

    # 采样记录应该少于100个点
    info = recorder.get_session_info()
    assert info['total_points'] < 100
    assert info['total_points'] >= 20  # 至少20个采样点

    print("  ✓ 采样记录模式正确")
    return True


def test_playback_events():
    """测试回放事件处理"""
    print("测试12: 回放事件处理...")

    storage = TimeSeriesStorage()

    for i in range(100):
        storage.add_data_point(DataPoint(
            timestamp=float(i),
            channel=DataChannel.POOL_LEVEL,
            source_id="pool_0",
            value=2.5,
        ))

    # 添加多个事件
    for i in range(5):
        storage.add_event(EventRecord(
            event_id=f"EVT_{i:03d}",
            timestamp=float(i * 20),
            event_type="test",
            source="test",
            severity="info",
            description=f"Event at {i * 20}",
        ))

    replayer = SimulationReplayer(storage)

    # 记录触发的事件
    triggered_events = []
    replayer.on_event = lambda e: triggered_events.append(e)

    # 播放整个过程
    replayer.play()
    while replayer.state == PlaybackState.PLAYING:
        replayer.step(10.0)

    # 应该触发所有事件
    assert len(triggered_events) == 5

    print("  ✓ 回放事件处理正确")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("历史数据记录与仿真回放系统测试")
    print("=" * 60)
    print()

    tests = [
        test_data_types,
        test_time_series_segment,
        test_time_series_storage,
        test_simulation_recorder,
        test_simulation_replayer,
        test_data_analyzer,
        test_correlation_analysis,
        test_data_export_import,
        test_compressed_export,
        test_recording_system,
        test_sampled_recording,
        test_playback_events,
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
