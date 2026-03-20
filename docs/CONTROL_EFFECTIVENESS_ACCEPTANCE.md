# 控制效果专项验收说明（ACC-CTRL-001）

## 1. 目的

本文件定义控制效果专项验收的最小闭环要求，用于把“控制能运行”与“控制效果达标”明确区分。

本专项覆盖：

- 单池 MPC 控制效果
- 多池参数化 DMPC / 双层控制效果
- 扰动、约束、退化模式下的控制表现

## 2. 范围

当前以严格重验证脚本为准，优先覆盖：

- `base_mpc`
- `parameterized_dmpc`

物理后端优先级：

- `single_channel`
- `tank`
- `segmented_hf`（CI 环境稳定后纳入）

## 3. 执行入口

基础执行命令：

```bash
python scripts/run_strict_revalidation.py \
  --modules control \
  --scenarios 100 \
  --physics-backend single_channel \
  --controller-backend base_mpc \
  --output reports/acceptance/acc_ctrl001_base_single_channel.json

python scripts/run_strict_revalidation.py \
  --modules control \
  --scenarios 100 \
  --physics-backend single_channel \
  --controller-backend parameterized_dmpc \
  --output reports/acceptance/acc_ctrl001_param_single_channel.json
```

可选对比命令：

```bash
python scripts/compare_revalidation_backends.py \
  --modules control \
  --scenarios 100 \
  --controller-backend base_mpc
```

## 4. 最小门槛

在以下条件全部满足前，`ACC-CTRL-001` 不能标记为“已通过”：

- `control.pass_rate >= 0.90`
- `control.average_score >= 0.85`
- `skipped_tests == 0`
- `setpoint_tracking` 通过率 `>= 0.90`
- `constraint_handling` 通过率 `>= 0.95`
- `stability` 通过率 `>= 0.95`

如果本次抽样包含以下类型，则追加要求：

- `overshoot` 通过率 `>= 0.85`
- `response_time` 通过率 `>= 0.85`
- `degraded_control` 通过率 `>= 0.90`

## 5. 必要产物

每次执行必须至少归档以下文件：

- 严格重验证 JSON 汇总
- 使用的命令行与参数
- 代码版本与依赖版本
- 验收结论记录

建议路径：

- `reports/acceptance/acc_ctrl001_*.json`
- `reports/acceptance/ACC-CTRL-001_<YYYYMMDD>.md`
- `reports/acceptance/acc_ctrl001_gate_status.json`

## 6. 当前状态（截至 2026-03-19）

- 当前状态：`待验收`
- 当前问题：
  - 严格控制 tester 已存在，但默认 CI 未将其作为发布门槛
  - `skipped_tests` 尚未作为发布失败条件处理
  - 现有部分 Phase 5 集成测试偏 smoke check，未直接约束关键 KPI

## 7. 与总体验收计划的关系

本专项属于 `docs/ACCEPTANCE_PLAN.md` 中“控制性能（单池/多池）”条目的正式落地说明。

在 `ACC-CTRL-001` 形成正式“已通过”结论前：

- 控制模块相关 HIL 分数只能视为候选证据
- `pass_rate` 和 `certification_level` 不得直接等同于发布结论
- 发布说明中不得宣称控制效果已正式验收通过
