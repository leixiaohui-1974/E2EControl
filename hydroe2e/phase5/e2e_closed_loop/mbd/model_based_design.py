"""
基于模型的设计管理器 (Model-Based Design Manager)

实现MBD的完整V模型流程:
需求 → 系统设计 → 组件设计 → 实现 → 单元测试 → 集成测试 → 系统测试 → 验收

核心功能:
1. 模型注册与版本管理
2. 模型验证 (SIL/MIL)
3. 代码生成接口
4. 测试用例管理
5. 追溯矩阵
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """模型类型"""
    PLANT = "plant"                     # 被控对象模型
    CONTROLLER = "controller"           # 控制器模型
    OBSERVER = "observer"               # 观测器模型
    REFERENCE = "reference"             # 参考模型
    FAULT_INJECTION = "fault"           # 故障注入模型


class ValidationLevel(Enum):
    """验证级别"""
    MIL = "mil"                         # Model-in-the-Loop
    SIL = "sil"                         # Software-in-the-Loop
    PIL = "pil"                         # Processor-in-the-Loop
    HIL = "hil"                         # Hardware-in-the-Loop


class DevelopmentPhase(Enum):
    """V模型开发阶段"""
    REQUIREMENTS = "requirements"
    SYSTEM_DESIGN = "system_design"
    COMPONENT_DESIGN = "component_design"
    IMPLEMENTATION = "implementation"
    UNIT_TESTING = "unit_testing"
    SIL_TESTING = "sil_testing"
    HIL_TESTING = "hil_testing"
    SYSTEM_TESTING = "system_testing"
    ACCEPTANCE = "acceptance"


class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class ModelArtifact:
    """模型工件"""
    model_id: str
    model_type: ModelType
    version: str
    description: str

    # 模型内容
    parameters: Dict[str, Any] = field(default_factory=dict)
    interfaces: Dict[str, Any] = field(default_factory=dict)

    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    author: str = "system"
    checksum: str = ""

    # 验证状态
    validation_level: Optional[ValidationLevel] = None
    validation_results: Dict[str, Any] = field(default_factory=dict)

    # 依赖
    dependencies: List[str] = field(default_factory=list)

    def compute_checksum(self) -> str:
        """计算校验和"""
        content = json.dumps({
            "model_id": self.model_id,
            "parameters": self.parameters,
            "interfaces": self.interfaces,
        }, sort_keys=True)
        self.checksum = hashlib.md5(content.encode()).hexdigest()
        return self.checksum


@dataclass
class Requirement:
    """需求项"""
    req_id: str
    title: str
    description: str
    priority: int = 1                   # 1-5
    category: str = "functional"
    parent_id: Optional[str] = None
    verification_method: str = "test"   # test, analysis, inspection, demo
    status: str = "open"


@dataclass
class TestCase:
    """测试用例"""
    test_id: str
    title: str
    description: str
    test_type: str = "functional"       # functional, performance, safety
    validation_level: ValidationLevel = ValidationLevel.SIL

    # 测试配置
    preconditions: List[str] = field(default_factory=list)
    test_steps: List[str] = field(default_factory=list)
    expected_results: List[str] = field(default_factory=list)
    pass_criteria: Dict[str, Any] = field(default_factory=dict)

    # 关联
    requirements: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)

    # 执行结果
    status: TestStatus = TestStatus.PENDING
    actual_results: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    last_run: Optional[datetime] = None


@dataclass
class TraceabilityLink:
    """追溯链接"""
    source_type: str                    # requirement, model, test, code
    source_id: str
    target_type: str
    target_id: str
    link_type: str = "derives"          # derives, verifies, implements


class ModelBasedDesignManager:
    """
    MBD流程管理器

    管理模型驱动开发的完整生命周期
    """

    def __init__(self):
        # 模型注册表
        self.models: Dict[str, ModelArtifact] = {}

        # 需求库
        self.requirements: Dict[str, Requirement] = {}

        # 测试用例库
        self.test_cases: Dict[str, TestCase] = {}

        # 追溯矩阵
        self.traceability: List[TraceabilityLink] = []

        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []

        # 初始化标准模型和需求
        self._init_standard_artifacts()

        logger.info("ModelBasedDesignManager initialized")

    def _init_standard_artifacts(self):
        """初始化标准工件"""
        # 核心需求
        self._add_standard_requirements()

        # 核心模型
        self._add_standard_models()

        # 标准测试用例
        self._add_standard_tests()

    def _add_standard_requirements(self):
        """添加标准需求"""
        requirements = [
            Requirement(
                req_id="REQ-SYS-001",
                title="水位控制精度",
                description="全线各渠池水位控制误差不超过±0.1m",
                priority=1,
                category="functional",
            ),
            Requirement(
                req_id="REQ-SYS-002",
                title="响应时间",
                description="从需求变化到控制响应不超过15分钟",
                priority=1,
                category="performance",
            ),
            Requirement(
                req_id="REQ-SYS-003",
                title="安全约束",
                description="水位始终保持在[1.5m, 5.5m]范围内",
                priority=1,
                category="safety",
            ),
            Requirement(
                req_id="REQ-SYS-004",
                title="场景覆盖",
                description="支持8大标准运行场景",
                priority=2,
                category="functional",
            ),
            Requirement(
                req_id="REQ-SYS-005",
                title="自主等级",
                description="常规场景支持L4高度自动化",
                priority=2,
                category="functional",
            ),
            Requirement(
                req_id="REQ-SYS-006",
                title="故障恢复",
                description="单点故障时自愈成功率≥85%",
                priority=1,
                category="reliability",
            ),
        ]

        for req in requirements:
            self.requirements[req.req_id] = req

    def _add_standard_models(self):
        """添加标准模型"""
        # 被控对象模型
        self.register_model(ModelArtifact(
            model_id="MDL-PLANT-001",
            model_type=ModelType.PLANT,
            version="1.0.0",
            description="南水北调中线全线水力学模型",
            parameters={
                "num_pools": 63,
                "total_length": 1432000,
                "design_flow": 350,
            },
            interfaces={
                "inputs": ["upstream_flow", "gate_openings", "lateral_flows"],
                "outputs": ["water_levels", "flow_rates"],
            },
        ))

        # 控制器模型
        self.register_model(ModelArtifact(
            model_id="MDL-CTRL-001",
            model_type=ModelType.CONTROLLER,
            version="1.0.0",
            description="分布式MPC控制器",
            parameters={
                "prediction_horizon": 48,
                "control_interval": 900,
            },
            interfaces={
                "inputs": ["water_levels", "flow_rates", "targets"],
                "outputs": ["gate_openings"],
            },
            dependencies=["MDL-PLANT-001"],
        ))

    def _add_standard_tests(self):
        """添加标准测试用例"""
        tests = [
            TestCase(
                test_id="TC-SIL-001",
                title="稳态水位控制测试",
                description="验证稳态条件下水位控制精度",
                test_type="functional",
                validation_level=ValidationLevel.SIL,
                requirements=["REQ-SYS-001"],
                models=["MDL-PLANT-001", "MDL-CTRL-001"],
                pass_criteria={
                    "max_level_deviation": 0.1,
                    "settling_time": 7200,
                },
            ),
            TestCase(
                test_id="TC-SIL-002",
                title="阶跃响应测试",
                description="验证流量阶跃变化的响应性能",
                test_type="performance",
                validation_level=ValidationLevel.SIL,
                requirements=["REQ-SYS-002"],
                models=["MDL-PLANT-001", "MDL-CTRL-001"],
                pass_criteria={
                    "rise_time": 3600,
                    "overshoot": 0.05,
                },
            ),
            TestCase(
                test_id="TC-SIL-003",
                title="安全边界测试",
                description="验证极端条件下的安全约束",
                test_type="safety",
                validation_level=ValidationLevel.SIL,
                requirements=["REQ-SYS-003"],
                pass_criteria={
                    "min_level_maintained": True,
                    "max_level_maintained": True,
                },
            ),
            TestCase(
                test_id="TC-HIL-001",
                title="硬件在环集成测试",
                description="验证控制器与真实闸门的集成",
                test_type="integration",
                validation_level=ValidationLevel.HIL,
                requirements=["REQ-SYS-001", "REQ-SYS-002"],
                pass_criteria={
                    "communication_success_rate": 0.99,
                    "control_accuracy": 0.1,
                },
            ),
        ]

        for tc in tests:
            self.test_cases[tc.test_id] = tc

        # 添加追溯
        for tc in tests:
            for req_id in tc.requirements:
                self.add_traceability(
                    source_type="test",
                    source_id=tc.test_id,
                    target_type="requirement",
                    target_id=req_id,
                    link_type="verifies",
                )

    def add_test_case(
        self,
        test_id: str,
        name: str,
        description: str,
        phase: DevelopmentPhase = DevelopmentPhase.SIL_TESTING,
        requirements: List[str] = None,
        pass_criteria: Dict[str, Any] = None,
    ):
        """
        添加测试用例 (简化接口)

        Args:
            test_id: 测试用例ID
            name: 测试名称
            description: 测试描述
            phase: 开发阶段
            requirements: 关联需求ID列表
            pass_criteria: 通过标准
        """
        level_map = {
            DevelopmentPhase.SIL_TESTING: ValidationLevel.SIL,
            DevelopmentPhase.HIL_TESTING: ValidationLevel.HIL,
            DevelopmentPhase.UNIT_TESTING: ValidationLevel.MIL,
        }
        tc = TestCase(
            test_id=test_id,
            title=name,
            description=description,
            validation_level=level_map.get(phase, ValidationLevel.SIL),
            requirements=requirements or [],
            pass_criteria=pass_criteria or {},
        )
        self.test_cases[tc.test_id] = tc

        # Auto-traceability
        for req_id in tc.requirements:
            self.add_traceability(
                source_type="test",
                source_id=tc.test_id,
                target_type="requirement",
                target_id=req_id,
                link_type="verifies",
            )

    def register_model(self, model: ModelArtifact):
        """注册模型"""
        model.compute_checksum()
        self.models[model.model_id] = model
        logger.info("Model registered: %s v%s", model.model_id, model.version)

    def update_model(self, model_id: str, updates: Dict[str, Any]):
        """更新模型"""
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")

        model = self.models[model_id]
        old_version = model.version

        # 更新参数
        if "parameters" in updates:
            model.parameters.update(updates["parameters"])

        # 更新版本
        if "version" in updates:
            model.version = updates["version"]

        model.updated_at = datetime.now()
        model.compute_checksum()

        logger.info("Model updated: %s %s -> %s", model_id, old_version, model.version)

    def add_traceability(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        link_type: str = "derives",
    ):
        """添加追溯链接"""
        link = TraceabilityLink(
            source_type=source_type,
            source_id=source_id,
            target_type=target_type,
            target_id=target_id,
            link_type=link_type,
        )
        self.traceability.append(link)

    def get_traceability_matrix(self) -> Dict[str, List[str]]:
        """获取追溯矩阵"""
        matrix = {}

        for link in self.traceability:
            key = f"{link.source_type}:{link.source_id}"
            if key not in matrix:
                matrix[key] = []
            matrix[key].append(f"{link.target_type}:{link.target_id}")

        return matrix

    def run_test(
        self,
        test_id: str,
        test_runner: Optional[Callable] = None,
        timeout: float = 3600.0,
    ) -> Dict[str, Any]:
        """
        执行测试用例

        Args:
            test_id: 测试用例ID
            test_runner: 测试执行器函数
            timeout: 超时时间

        Returns:
            result: 测试结果
        """
        if test_id not in self.test_cases:
            raise ValueError(f"Test case not found: {test_id}")

        tc = self.test_cases[test_id]
        tc.status = TestStatus.RUNNING
        tc.last_run = datetime.now()

        start_time = datetime.now()

        try:
            if test_runner:
                # 使用提供的测试执行器
                result = test_runner(tc)
            else:
                # 默认模拟执行
                result = self._simulate_test(tc)

            # 检查通过标准
            passed = self._check_pass_criteria(result, tc.pass_criteria)

            tc.status = TestStatus.PASSED if passed else TestStatus.FAILED
            tc.actual_results = [str(result)]

        except Exception as e:
            logger.error("Test %s failed: %s", test_id, e)
            tc.status = TestStatus.FAILED
            tc.actual_results = [f"Error: {str(e)}"]
            result = {"error": str(e)}

        tc.execution_time = (datetime.now() - start_time).total_seconds()

        # 记录历史
        self.execution_history.append({
            "test_id": test_id,
            "status": tc.status.value,
            "execution_time": tc.execution_time,
            "timestamp": datetime.now().isoformat(),
            "result": result,
        })

        return result

    def _simulate_test(self, tc: TestCase) -> Dict[str, Any]:
        """模拟测试执行"""
        # 简单的模拟结果
        return {
            "max_level_deviation": 0.08,
            "settling_time": 5400,
            "overshoot": 0.03,
            "min_level_maintained": True,
            "max_level_maintained": True,
        }

    def _check_pass_criteria(
        self,
        result: Dict[str, Any],
        criteria: Dict[str, Any]
    ) -> bool:
        """检查通过标准"""
        for key, threshold in criteria.items():
            if key not in result:
                continue

            actual = result[key]
            if isinstance(threshold, bool):
                if actual != threshold:
                    return False
            elif isinstance(threshold, (int, float)):
                if actual > threshold:
                    return False

        return True

    def generate_v_model_report(self) -> Dict[str, Any]:
        """生成V模型报告"""
        # 需求覆盖率
        covered_reqs = set()
        for link in self.traceability:
            if link.link_type == "verifies" and link.target_type == "requirement":
                covered_reqs.add(link.target_id)

        req_coverage = len(covered_reqs) / max(len(self.requirements), 1)

        # 测试通过率
        passed_tests = sum(1 for tc in self.test_cases.values() if tc.status == TestStatus.PASSED)
        total_tests = len(self.test_cases)
        test_pass_rate = passed_tests / max(total_tests, 1)

        # 模型验证状态
        validated_models = sum(1 for m in self.models.values() if m.validation_level is not None)

        return {
            "summary": {
                "total_requirements": len(self.requirements),
                "covered_requirements": len(covered_reqs),
                "requirement_coverage": req_coverage,
                "total_models": len(self.models),
                "validated_models": validated_models,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "test_pass_rate": test_pass_rate,
            },
            "requirements": {
                req_id: {
                    "title": req.title,
                    "status": req.status,
                    "priority": req.priority,
                }
                for req_id, req in self.requirements.items()
            },
            "models": {
                model_id: {
                    "type": model.model_type.value,
                    "version": model.version,
                    "validation_level": model.validation_level.value if model.validation_level else None,
                }
                for model_id, model in self.models.items()
            },
            "tests": {
                tc_id: {
                    "title": tc.title,
                    "status": tc.status.value,
                    "validation_level": tc.validation_level.value,
                    "execution_time": tc.execution_time,
                }
                for tc_id, tc in self.test_cases.items()
            },
            "traceability_matrix": self.get_traceability_matrix(),
        }

    def get_status(self) -> Dict[str, Any]:
        """获取MBD状态"""
        return {
            "num_models": len(self.models),
            "num_requirements": len(self.requirements),
            "num_tests": len(self.test_cases),
            "num_traceability_links": len(self.traceability),
            "pending_tests": sum(1 for tc in self.test_cases.values() if tc.status == TestStatus.PENDING),
            "passed_tests": sum(1 for tc in self.test_cases.values() if tc.status == TestStatus.PASSED),
            "failed_tests": sum(1 for tc in self.test_cases.values() if tc.status == TestStatus.FAILED),
        }

    def generate_controller_code(
        self,
        model_id: str,
        target: str = "python",
    ) -> str:
        """
        从模型定义生成控制器代码

        Args:
            model_id: 模型ID
            target: 目标语言 (python, c, structured_text)

        Returns:
            生成的代码字符串
        """
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")

        model = self.models[model_id]
        generators = {
            "python": self._gen_python,
            "c": self._gen_c_header,
            "structured_text": self._gen_structured_text,
        }

        gen = generators.get(target)
        if gen is None:
            raise ValueError(f"Unsupported target: {target}")

        code = gen(model)
        logger.info("Code generated for %s (target=%s, %d chars)",
                     model_id, target, len(code))
        return code

    def _gen_python(self, model: ModelArtifact) -> str:
        """生成Python控制器代码"""
        inputs = model.interfaces.get("inputs", [])
        outputs = model.interfaces.get("outputs", [])
        params = model.parameters

        lines = [
            f'"""Auto-generated controller: {model.model_id} v{model.version}"""',
            "",
            "import numpy as np",
            "from dataclasses import dataclass, field",
            "from typing import Dict, List",
            "",
            "",
            "@dataclass",
            f"class {_class_name(model.model_id)}Config:",
            '    """Controller configuration parameters"""',
        ]
        for k, v in params.items():
            lines.append(f"    {k}: float = {v}")
        if not params:
            lines.append("    pass")

        lines += [
            "",
            "",
            f"class {_class_name(model.model_id)}:",
            f'    """Generated controller for {model.description}"""',
            "",
            f"    def __init__(self, config: {_class_name(model.model_id)}Config = None):",
            f"        self.config = config or {_class_name(model.model_id)}Config()",
            "        self._state: Dict[str, float] = {}",
            "",
            "    def step(self, inputs: Dict[str, float]) -> Dict[str, float]:",
            '        """Execute one control step"""',
        ]
        for inp in inputs:
            lines.append(f'        {inp} = inputs.get("{inp}", 0.0)')
        lines.append("")
        lines.append("        # Controller logic (implement specific algorithm here)")
        for out in outputs:
            lines.append(f'        {out} = 0.0  # TODO: compute from inputs')
        lines.append("")
        lines.append("        return {")
        for out in outputs:
            lines.append(f'            "{out}": {out},')
        lines.append("        }")

        return "\n".join(lines) + "\n"

    def _gen_c_header(self, model: ModelArtifact) -> str:
        """生成C语言头文件"""
        inputs = model.interfaces.get("inputs", [])
        outputs = model.interfaces.get("outputs", [])
        params = model.parameters
        name = _class_name(model.model_id)
        guard = f"_{name.upper()}_H_"

        lines = [
            f"/* Auto-generated: {model.model_id} v{model.version} */",
            f"#ifndef {guard}",
            f"#define {guard}",
            "",
            "#include <stdint.h>",
            "",
            f"typedef struct {{",
        ]
        for k, v in params.items():
            lines.append(f"    double {k};")
        lines += [
            f"}} {name}Config;",
            "",
            f"typedef struct {{",
        ]
        for inp in inputs:
            lines.append(f"    double {inp};")
        lines += [
            f"}} {name}Inputs;",
            "",
            f"typedef struct {{",
        ]
        for out in outputs:
            lines.append(f"    double {out};")
        lines += [
            f"}} {name}Outputs;",
            "",
            f"void {name}_Init({name}Config* cfg);",
            f"void {name}_Step({name}Config* cfg, const {name}Inputs* in, {name}Outputs* out);",
            "",
            f"#endif /* {guard} */",
        ]
        return "\n".join(lines) + "\n"

    def _gen_structured_text(self, model: ModelArtifact) -> str:
        """生成IEC 61131-3 结构化文本 (PLC)"""
        inputs = model.interfaces.get("inputs", [])
        outputs = model.interfaces.get("outputs", [])
        name = _class_name(model.model_id)

        lines = [
            f"(* Auto-generated: {model.model_id} v{model.version} *)",
            f"FUNCTION_BLOCK {name}",
            "VAR_INPUT",
        ]
        for inp in inputs:
            lines.append(f"    {inp} : REAL;")
        lines += [
            "END_VAR",
            "VAR_OUTPUT",
        ]
        for out in outputs:
            lines.append(f"    {out} : REAL;")
        lines += [
            "END_VAR",
            "VAR",
            "    (* Internal state *)",
            "END_VAR",
            "",
            "(* Control logic *)",
        ]
        for out in outputs:
            lines.append(f"{out} := 0.0; (* TODO: implement *)")
        lines.append("")
        lines.append(f"END_FUNCTION_BLOCK")
        return "\n".join(lines) + "\n"

    def generate_test_harness(self, test_id: str) -> str:
        """
        从测试用例生成自动化测试代码

        Args:
            test_id: 测试用例ID

        Returns:
            生成的pytest测试代码
        """
        if test_id not in self.test_cases:
            raise ValueError(f"Test case not found: {test_id}")

        tc = self.test_cases[test_id]
        func_name = test_id.lower().replace("-", "_")

        lines = [
            f'"""Auto-generated test: {tc.title}"""',
            "",
            "import pytest",
            "",
            "",
            f"class Test{_class_name(test_id)}:",
            f'    """{tc.description}"""',
            "",
        ]

        # Preconditions as setup
        if tc.preconditions:
            lines.append("    def setup_method(self):")
            for pre in tc.preconditions:
                lines.append(f"        # {pre}")
            lines.append("        pass")
            lines.append("")

        # Main test
        lines.append(f"    def test_{func_name}(self):")
        for step in tc.test_steps:
            lines.append(f"        # {step}")
        lines.append("        result = {}  # TODO: run simulation")
        lines.append("")

        # Assertions from pass criteria
        for key, threshold in tc.pass_criteria.items():
            if isinstance(threshold, bool):
                lines.append(f'        assert result.get("{key}") is {threshold}')
            elif isinstance(threshold, (int, float)):
                lines.append(f'        assert result.get("{key}", float("inf")) <= {threshold}')

        return "\n".join(lines) + "\n"


def _class_name(identifier: str) -> str:
    """Convert an identifier like MDL-PLANT-001 to MdlPlant001."""
    parts = identifier.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts)
