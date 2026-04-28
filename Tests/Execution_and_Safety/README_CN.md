# Execution_and_Safety

`Execution_and_Safety` 用于评估 `VLA_Agent_Core` 执行层的安全性与正确性，重点验证执行路径中的关键安全约束与命令分发逻辑。

当前包含三类评测：

- FSM 状态转移拒绝/放行正确率
- 紧急停机下的队列抢占正确率
- Dry-run 硬件分发与技能 Schema 一致性

## 目录结构

```text
Execution_and_Safety/
├── FSM_correct_rejection_rate/
│   ├── evaluate_fsm_rejection.py
│   ├── fsm_rejection_results.jsonl
│   └── fsm_rejection_summary.txt
├── Queue_preemption_correctness/
│   ├── evaluate_queue_preemption.py
│   ├── evaluate_queue_preemption.md
│   ├── queue_preemption_results.jsonl
│   └── queue_preemption_summary.txt
└── execution_guardian_dryrun/
    ├── dryrun_hardware_wrapper.py
    ├── evaluate_dryrun_dispatch.py
    ├── dryrun_dispatch_results.jsonl
    └── dryrun_dispatch_summary.txt
```

## 评测内容

### 1) FSM 正确拒绝率

脚本：`FSM_correct_rejection_rate/evaluate_fsm_rejection.py`

验证点：

- 未知 `action_id` 必须拒绝
- `stop (action_id=1)` 始终允许
- `BLOCKING` 状态下仅允许 stop
- `EMERGENCY_STOP` 后仅允许 `stand_up`（以及 stop）
- 连续运动/阻塞技能只能在合法 FSM 状态下执行

输出：

- `fsm_rejection_results.jsonl`（逐 case 结果）
- `fsm_rejection_summary.txt`（汇总指标）

### 2) 队列抢占正确率

脚本：`Queue_preemption_correctness/evaluate_queue_preemption.py`

验证点（含重复 emergency 场景）：

- 队列容量压力下初始入队行为正确
- 触发 emergency 后队列被完全清空
- emergency stop 任务（`action_id=1`）被正确注入
- stop 执行后速度归零，并进入 `EMERGENCY_STOP` 状态

输出：

- `queue_preemption_results.jsonl`
- `queue_preemption_summary.txt`

### 3) Dry-run 执行分发

脚本：

- `execution_guardian_dryrun/dryrun_hardware_wrapper.py`
- `execution_guardian_dryrun/evaluate_dryrun_dispatch.py`

验证点：

- `action_id` 集合与 `VLA_Agent_Core/schemas/robot_skill_schema.json` 保持一致
- 先做 mode-duration 规则校验再分发
- 每个合法动作映射到预期 mock 高层方法
- 记录每次分发成功率与延迟

输出：

- `dryrun_dispatch_results.jsonl`
- `dryrun_dispatch_summary.txt`

## 运行前提

- Python 3.10+
- 本机存在 `VLA_Agent_Core`：
  - `/home/ubuntu/New_Architecture/VLA_Agent_Core`
- 必要文件：
  - `execution/fsm_guardian.py`
  - `execution/task_queue.py`
  - `schemas/robot_skill_schema.json`

## 运行方式

进入目录：

```bash
cd /home/ubuntu/New_Architecture/Tests/Execution_and_Safety
```

执行三个评测：

```bash
python3 FSM_correct_rejection_rate/evaluate_fsm_rejection.py
python3 Queue_preemption_correctness/evaluate_queue_preemption.py
python3 execution_guardian_dryrun/evaluate_dryrun_dispatch.py
```

## 当前基线结果

基于目录中现有 summary 文件：

- FSM 拒绝评测：
  - 总 case：`102`
  - FSM 判定准确率：`100.00%`
  - 正确拒绝率：`100.00%`
  - 正确放行率：`100.00%`

- 队列抢占评测：
  - 场景数：`10`
  - emergency cycle 数：`12`
  - 总体抢占正确率：`100.00%`
  - 平均清队列延迟：`0.000019 s`

- Dry-run 分发评测：
  - 任务总数：`65`（13 个动作 x 5 次）
  - Schema 合法率：`100.00%`
  - 分发成功率：`100.00%`
  - 平均分发延迟：`0.000002 s`

## 说明

- 该目录只关注执行与安全层，不包含 ASR/规划质量评测。
- Dry-run 方案用于验证“映射与约束”，不触发真实机器人运动。
- 可将这些基线指标作为回归测试阈值，在调整 FSM 规则、队列逻辑或动作映射后复测对比。
