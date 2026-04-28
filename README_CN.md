# New_Architecture

`New_Architecture` 是一个面向四足机器人（机器狗）的模块化架构，核心是“语言驱动控制 + 安全执行”。

整体分工：

- `VLA_Agent_Core`：规划与执行安全核心
- `VLA_Pipeline`：端到端编排层（文本 + 视觉反射）
- `Tests`：语音链路与执行安全评测

## 仓库结构总览

```text
New_Architecture/
├── VLA_Agent_Core/
├── VLA_Pipeline/
└── Tests/
    ├── voice_command_dataset/
    └── Execution_and_Safety/
```

## 1) VLA_Agent_Core

`VLA_Agent_Core` 是控制中枢与安全执行核心。

主要职责：

- 通过 LLM 工具调用把自然语言编译为动作序列
- 用线程安全队列承接动作
- 通过 FSM 做状态转移安全校验
- 本地优先处理紧急停机

关键模块：

- `core/agent_planner.py`
- `execution/task_queue.py`
- `execution/fsm_guardian.py`
- `schemas/robot_skill_schema.json`
- `main_agent.py`

详细文档：

- `VLA_Agent_Core/README.md`（英文）
- `VLA_Agent_Core/README_CN.md`（中文）

## 2) VLA_Pipeline

`VLA_Pipeline` 是编排层，负责把认知、感知和执行链路串联起来。

主要职责：

- 将 `SpeechEvent`、`PerceptionEvent`、系统/错误事件统一到事件总线
- 支持 `planner_only` / `reflex_only` / `hybrid` 三种模式
- 启动前自动检查资源并可选自动下载
- 检测 vLLM 是否就绪并可选自动拉起
- 复用 `VLA_Agent_Core` 的规划、队列和 FSM 执行

关键模块：

- `src/pipeline/orchestrator.py`
- `src/cognition/planner_client.py`
- `src/perception/*`
- `src/execution/fsm_adapter.py`
- `config/pipeline.yaml`

详细文档：

- `VLA_Pipeline/README.md`（英文）
- `VLA_Pipeline/README_CN.md`（中文）

## 3) Tests

`Tests` 目前有两套评测：

### A) `voice_command_dataset`

端到端语音链路评测：

- TTS 生成
- ASR 评测
- 命令语义准确率
- 紧急路由准确率
- 规划有效率
- 意图-计划匹配率
- 规划时延
- TTFT 对比

详细文档：

- `Tests/voice_command_dataset/README.md`（英文）
- `Tests/voice_command_dataset/README_CN.md`（中文）

### B) `Execution_and_Safety`

执行安全链路评测：

- FSM 拒绝/放行正确率
- emergency 下队列抢占正确率
- dry-run 命令分发正确率

详细文档：

- `Tests/Execution_and_Safety/README.md`（英文）
- `Tests/Execution_and_Safety/README_CN.md`（中文）

## 模块关系

按依赖关系看执行路径：

1. `VLA_Pipeline` 接收文本与感知事件
2. 规划链路调用 `VLA_Agent_Core` 的 planner
3. 动作进入队列，由 `VLA_Agent_Core` 的 FSM 守护执行
4. `Tests` 分别验证端到端质量（`voice_command_dataset`）和执行安全（`Execution_and_Safety`）

## 快速开始

### 仅启动核心 Agent

```bash
cd /home/ubuntu/New_Architecture/VLA_Agent_Core
python3 main_agent.py
```

### 启动完整 Pipeline

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
python3 -m src.pipeline.orchestrator
```

### 执行安全评测

```bash
cd /home/ubuntu/New_Architecture/Tests/Execution_and_Safety
python3 FSM_correct_rejection_rate/evaluate_fsm_rejection.py
python3 Queue_preemption_correctness/evaluate_queue_preemption.py
python3 execution_guardian_dryrun/evaluate_dryrun_dispatch.py
```

## 说明

- 由于 `protobuf` 版本冲突，建议将 vLLM 与 MediaPipe 分到不同 Python 环境。
- 各模块更详细的配置、安装、排障请查看对应子目录 README。
