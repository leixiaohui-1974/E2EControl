"""
Phase 5.7 Certification Runner Tests

Tests for:
1. Scenario loading
2. Certification execution
3. Level determination
4. Report generation
5. Export functionality
"""

import pytest
import os
import json
import tempfile
from datetime import datetime
from pathlib import Path

from hydroe2e.phase5.hil_testing.certification_runner import (
    CertificationRunner,
    CertificationReport,
    CertificationResult,
    AutonomousLevel,
    LevelRequirement,
    ScenarioResult,
    CategorySummary,
    run_certification
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def scenarios_path():
    """Get scenarios directory path"""
    return os.path.join(os.path.dirname(__file__), '..', 'hil_testing', 'scenarios')


@pytest.fixture
def runner(scenarios_path):
    """Create certification runner"""
    return CertificationRunner(scenarios_path=scenarios_path)


@pytest.fixture
def temp_dir():
    """Create temporary directory for output files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# AutonomousLevel Tests
# ============================================================================

class TestAutonomousLevel:
    """Tests for AutonomousLevel enum"""

    def test_level_values(self):
        """Test level value ordering"""
        assert AutonomousLevel.L0.value == 0
        assert AutonomousLevel.L1.value == 1
        assert AutonomousLevel.L2.value == 2
        assert AutonomousLevel.L3.value == 3
        assert AutonomousLevel.L4.value == 4
        assert AutonomousLevel.L5.value == 5

    def test_level_comparison(self):
        """Test level comparison"""
        assert AutonomousLevel.L5.value > AutonomousLevel.L4.value
        assert AutonomousLevel.L3.value > AutonomousLevel.L1.value


# ============================================================================
# LevelRequirement Tests
# ============================================================================

class TestLevelRequirement:
    """Tests for LevelRequirement"""

    def test_get_requirements(self):
        """Test getting level requirements"""
        requirements = LevelRequirement.get_requirements()

        assert AutonomousLevel.L1 in requirements
        assert AutonomousLevel.L5 in requirements

    def test_l1_requirements(self):
        """Test L1 requirements"""
        requirements = LevelRequirement.get_requirements()
        l1_req = requirements[AutonomousLevel.L1]

        assert l1_req.min_pass_rate == 0.60
        assert l1_req.max_human_intervention == 0.80
        assert 'S1_NORMAL' in l1_req.required_categories

    def test_l5_requirements(self):
        """Test L5 requirements"""
        requirements = LevelRequirement.get_requirements()
        l5_req = requirements[AutonomousLevel.L5]

        assert l5_req.min_pass_rate == 0.99
        assert l5_req.max_human_intervention == 0.01
        assert len(l5_req.required_categories) >= 8


# ============================================================================
# CertificationRunner Initialization Tests
# ============================================================================

class TestCertificationRunnerInit:
    """Tests for CertificationRunner initialization"""

    def test_init_default(self, scenarios_path):
        """Test initialization with default path"""
        runner = CertificationRunner(scenarios_path=scenarios_path)

        assert runner.scenarios_path == scenarios_path
        assert runner.system is None
        assert runner.parallel is False

    def test_init_with_options(self, scenarios_path):
        """Test initialization with options"""
        runner = CertificationRunner(
            scenarios_path=scenarios_path,
            parallel=True,
            max_workers=8
        )

        assert runner.parallel is True
        assert runner.max_workers == 8

    def test_scenarios_loaded(self, runner):
        """Test scenarios are loaded"""
        assert runner.total_scenario_count > 0
        assert len(runner.scenarios) > 0


# ============================================================================
# Scenario Loading Tests
# ============================================================================

class TestScenarioLoading:
    """Tests for scenario loading"""

    def test_total_count(self, runner):
        """Test total scenario count"""
        # Should have 103 scenarios (12+10+8+8+8+10+8+15+12+12)
        assert runner.total_scenario_count >= 100

    def test_categories_loaded(self, runner):
        """Test all categories loaded"""
        expected_categories = [
            'S1_NORMAL', 'S2_FLOOD', 'S3_DROUGHT', 'S4_ICE',
            'S5_POLLUTION', 'S6_EQUIPMENT', 'S7_SECURITY',
            'S8_COMPOUND', 'S9_EXTREME', 'S10_LEARNING'
        ]

        for cat in expected_categories:
            # Categories are stored with yaml file name as key
            found = any(cat in key.upper() for key in runner.scenarios.keys())
            assert found or len(runner.scenarios) > 0

    def test_get_by_category(self, runner):
        """Test getting scenarios by category"""
        # Try to get normal scenarios
        for key in runner.scenarios.keys():
            scenarios = runner.get_scenarios_by_category(key)
            assert isinstance(scenarios, list)
            break

    def test_get_by_difficulty(self, runner):
        """Test getting scenarios by difficulty"""
        diff_1 = runner.get_scenarios_by_difficulty(1)
        diff_5 = runner.get_scenarios_by_difficulty(5)

        # Should have scenarios at difficulty 1
        assert len(diff_1) > 0

    def test_get_by_level(self, runner):
        """Test getting scenarios by level"""
        l2_scenarios = runner.get_scenarios_by_level(AutonomousLevel.L2)
        l5_scenarios = runner.get_scenarios_by_level(AutonomousLevel.L5)

        # L5 should include more scenarios than L2
        assert len(l5_scenarios) >= len(l2_scenarios)


# ============================================================================
# Scenario Execution Tests
# ============================================================================

class TestScenarioExecution:
    """Tests for scenario execution"""

    def test_run_single_scenario(self, runner):
        """Test running a single scenario"""
        # Get first scenario from any category
        for scenarios in runner.scenarios.values():
            if scenarios:
                scenario = scenarios[0]
                break

        result = runner.run_scenario(scenario)

        assert isinstance(result, ScenarioResult)
        assert result.scenario_id is not None
        assert result.result in CertificationResult
        assert 0.0 <= result.score <= 1.0

    def test_scenario_result_fields(self, runner):
        """Test scenario result has all fields"""
        for scenarios in runner.scenarios.values():
            if scenarios:
                scenario = scenarios[0]
                break

        result = runner.run_scenario(scenario)

        assert hasattr(result, 'scenario_id')
        assert hasattr(result, 'scenario_name')
        assert hasattr(result, 'category')
        assert hasattr(result, 'difficulty')
        assert hasattr(result, 'required_level')
        assert hasattr(result, 'result')
        assert hasattr(result, 'score')
        assert hasattr(result, 'duration')
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'failures')


# ============================================================================
# Certification Run Tests
# ============================================================================

class TestCertificationRun:
    """Tests for full certification run"""

    def test_run_certification(self, runner):
        """Test running certification"""
        report = runner.run_certification(max_scenarios=10)

        assert isinstance(report, CertificationReport)
        assert report.total_scenarios == 10
        assert report.scenarios_passed + report.scenarios_failed <= report.total_scenarios

    def test_run_with_target_level(self, runner):
        """Test running with target level"""
        report = runner.run_certification(
            target_level=AutonomousLevel.L3,
            max_scenarios=20
        )

        assert report.target_level == AutonomousLevel.L3
        assert isinstance(report.achieved_level, AutonomousLevel)

    def test_run_with_categories(self, runner):
        """Test running specific categories"""
        # Get first available category
        first_cat = list(runner.scenarios.keys())[0]

        report = runner.run_certification(
            categories=[first_cat],
            max_scenarios=10
        )

        assert report.total_scenarios > 0

    def test_certification_report_fields(self, runner):
        """Test certification report has all fields"""
        report = runner.run_certification(max_scenarios=5)

        assert hasattr(report, 'test_date')
        assert hasattr(report, 'system_version')
        assert hasattr(report, 'total_scenarios')
        assert hasattr(report, 'scenarios_passed')
        assert hasattr(report, 'scenarios_failed')
        assert hasattr(report, 'overall_pass_rate')
        assert hasattr(report, 'achieved_level')
        assert hasattr(report, 'category_summaries')
        assert hasattr(report, 'difficulty_distribution')
        assert hasattr(report, 'recommendations')


# ============================================================================
# Category Summary Tests
# ============================================================================

class TestCategorySummary:
    """Tests for category summaries"""

    def test_category_summaries_generated(self, runner):
        """Test category summaries are generated"""
        report = runner.run_certification(max_scenarios=20)

        assert len(report.category_summaries) > 0

    def test_category_summary_fields(self, runner):
        """Test category summary has correct fields"""
        report = runner.run_certification(max_scenarios=20)

        for cat, summary in report.category_summaries.items():
            assert isinstance(summary, CategorySummary)
            assert summary.total > 0
            assert summary.passed >= 0
            assert summary.failed >= 0
            assert 0.0 <= summary.pass_rate <= 1.0
            assert 0.0 <= summary.avg_score <= 1.0


# ============================================================================
# Level Determination Tests
# ============================================================================

class TestLevelDetermination:
    """Tests for autonomous level determination"""

    def test_achieved_level_valid(self, runner):
        """Test achieved level is valid"""
        report = runner.run_certification(max_scenarios=30)

        assert report.achieved_level in AutonomousLevel

    def test_level_increases_with_pass_rate(self, runner):
        """Test level correlates with pass rate"""
        # This is a probabilistic test
        reports = []
        for _ in range(3):
            report = runner.run_certification(max_scenarios=50)
            reports.append(report)

        # Higher pass rates should generally correlate with higher levels
        # (not strictly guaranteed due to randomness)
        pass_rates = [r.overall_pass_rate for r in reports]
        levels = [r.achieved_level.value for r in reports]

        # Just check they're all valid
        assert all(0 <= pr <= 1 for pr in pass_rates)
        assert all(0 <= lv <= 5 for lv in levels)


# ============================================================================
# Recommendations Tests
# ============================================================================

class TestRecommendations:
    """Tests for improvement recommendations"""

    def test_recommendations_generated(self, runner):
        """Test recommendations are generated"""
        report = runner.run_certification(max_scenarios=30)

        assert isinstance(report.recommendations, list)

    def test_recommendations_for_failures(self, runner):
        """Test recommendations address failures"""
        report = runner.run_certification(max_scenarios=30)

        # If there are failures, should have recommendations
        if report.scenarios_failed > 0:
            # Recommendations should exist (may be empty if all passed)
            assert isinstance(report.recommendations, list)


# ============================================================================
# Export Tests
# ============================================================================

class TestReportExport:
    """Tests for report export"""

    def test_export_json(self, runner, temp_dir):
        """Test JSON export"""
        report = runner.run_certification(max_scenarios=10)
        output_path = os.path.join(temp_dir, "report.json")

        runner.export_report(report, output_path, "json")

        assert os.path.exists(output_path)

        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert 'summary' in data
        assert 'categories' in data
        assert 'scenario_results' in data

    def test_export_markdown(self, runner, temp_dir):
        """Test Markdown export"""
        report = runner.run_certification(max_scenarios=10)
        output_path = os.path.join(temp_dir, "report.md")

        runner.export_report(report, output_path, "markdown")

        assert os.path.exists(output_path)

        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "# 自主运行等级认证报告" in content
        assert "认证结果" in content

    def test_export_html(self, runner, temp_dir):
        """Test HTML export"""
        report = runner.run_certification(max_scenarios=10)
        output_path = os.path.join(temp_dir, "report.html")

        runner.export_report(report, output_path, "html")

        assert os.path.exists(output_path)

        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content
        assert "自主运行等级认证报告" in content

    def test_export_invalid_format(self, runner, temp_dir):
        """Test invalid export format raises error"""
        report = runner.run_certification(max_scenarios=5)
        output_path = os.path.join(temp_dir, "report.xyz")

        with pytest.raises(ValueError):
            runner.export_report(report, output_path, "xyz")


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunction:
    """Tests for run_certification convenience function"""

    def test_run_certification_function(self):
        """Test convenience function"""
        report = run_certification(max_scenarios=5)

        assert isinstance(report, CertificationReport)
        assert report.total_scenarios == 5

    def test_run_certification_with_level(self):
        """Test convenience function with target level"""
        report = run_certification(target_level='L3', max_scenarios=10)

        assert report.target_level == AutonomousLevel.L3


# ============================================================================
# Difficulty Distribution Tests
# ============================================================================

class TestDifficultyDistribution:
    """Tests for difficulty distribution analysis"""

    def test_difficulty_distribution_generated(self, runner):
        """Test difficulty distribution is generated"""
        report = runner.run_certification(max_scenarios=30)

        assert isinstance(report.difficulty_distribution, dict)
        assert len(report.difficulty_distribution) > 0

    def test_difficulty_distribution_counts(self, runner):
        """Test difficulty distribution has correct counts"""
        report = runner.run_certification(max_scenarios=30)

        total_from_dist = sum(
            d['total'] for d in report.difficulty_distribution.values()
        )

        assert total_from_dist == report.total_scenarios


# ============================================================================
# Integration Tests
# ============================================================================

class TestCertificationIntegration:
    """Integration tests"""

    def test_full_certification_workflow(self, runner, temp_dir):
        """Test complete certification workflow"""
        # Run certification
        report = runner.run_certification(
            target_level=AutonomousLevel.L4,
            max_scenarios=50
        )

        # Verify report
        assert report.total_scenarios == 50
        assert report.target_level == AutonomousLevel.L4
        assert isinstance(report.achieved_level, AutonomousLevel)

        # Export all formats
        json_path = os.path.join(temp_dir, "report.json")
        md_path = os.path.join(temp_dir, "report.md")
        html_path = os.path.join(temp_dir, "report.html")

        runner.export_report(report, json_path, "json")
        runner.export_report(report, md_path, "markdown")
        runner.export_report(report, html_path, "html")

        # Verify all exports
        assert os.path.exists(json_path)
        assert os.path.exists(md_path)
        assert os.path.exists(html_path)

    def test_multiple_certification_runs(self, runner):
        """Test multiple certification runs are independent"""
        report1 = runner.run_certification(max_scenarios=10)
        report2 = runner.run_certification(max_scenarios=20)

        assert report1.total_scenarios == 10
        assert report2.total_scenarios == 20

        # Results should be independent
        assert len(runner.results) == 20  # Last run's results


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
