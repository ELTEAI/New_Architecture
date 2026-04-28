# New_Architecture

`New_Architecture` is a modular robotics stack centered on language-driven control for a quadruped platform.  
At a high level:

- `VLA_Agent_Core` provides planning + execution safety primitives
- `VLA_Pipeline` provides end-to-end orchestration (text + vision reflex)
- `Tests` provides benchmark/evaluation suites for voice and execution safety

## Repository Overview

```text
New_Architecture/
├── VLA_Agent_Core/
├── VLA_Pipeline/
└── Tests/
    ├── voice_command_dataset/
    └── Execution_and_Safety/
```

## 1) VLA_Agent_Core

`VLA_Agent_Core` is the control brain + safety execution core.

Core responsibilities:

- Compile natural-language commands into structured action sequences via LLM tool-calling
- Buffer actions in a thread-safe queue
- Enforce safety via FSM transition checks
- Prioritize local emergency stop

Key modules:

- `core/agent_planner.py`
- `execution/task_queue.py`
- `execution/fsm_guardian.py`
- `schemas/robot_skill_schema.json`
- `main_agent.py`

Main doc:

- `VLA_Agent_Core/README.md` (EN)
- `VLA_Agent_Core/README_CN.md` (ZH)

## 2) VLA_Pipeline

`VLA_Pipeline` is the orchestration layer that integrates planning, perception, and execution.

Core responsibilities:

- Unify `SpeechEvent`, `PerceptionEvent`, and system/error events through one event bus
- Run planner path, reflex path, or hybrid path
- Auto-check and optionally auto-pull required model assets
- Auto-check vLLM readiness and optionally auto-start vLLM
- Reuse `VLA_Agent_Core` for planner + queue + FSM execution

Key modules:

- `src/pipeline/orchestrator.py`
- `src/cognition/planner_client.py`
- `src/perception/*`
- `src/execution/fsm_adapter.py`
- `config/pipeline.yaml`

Main doc:

- `VLA_Pipeline/README.md` (EN)
- `VLA_Pipeline/README_CN.md` (ZH)

## 3) Tests

`Tests` currently contains two test suites:

### A) `voice_command_dataset`

End-to-end voice benchmark chain:

- TTS generation
- ASR evaluation
- command semantic accuracy
- emergency routing accuracy
- plan validity
- intent-plan matching
- planning latency
- TTFT comparison

Main doc:

- `Tests/voice_command_dataset/README.md` (EN)
- `Tests/voice_command_dataset/README_CN.md` (ZH)

### B) `Execution_and_Safety`

Execution-layer safety/correctness benchmark:

- FSM rejection correctness
- queue emergency preemption correctness
- dry-run command dispatch correctness

Main doc:

- `Tests/Execution_and_Safety/README.md` (EN)
- `Tests/Execution_and_Safety/README_CN.md` (ZH)

## Component Relationship

Execution flow by dependency:

1. `VLA_Pipeline` receives text/perception events
2. Planner path calls into `VLA_Agent_Core` planning logic
3. Actions are pushed into queue and guarded by FSM from `VLA_Agent_Core`
4. Tests validate both end-to-end quality (`voice_command_dataset`) and execution safety (`Execution_and_Safety`)

## Quick Start

### Run core agent only

```bash
cd /home/ubuntu/New_Architecture/VLA_Agent_Core
python3 main_agent.py
```

### Run full pipeline

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
python3 -m src.pipeline.orchestrator
```

### Run execution safety tests

```bash
cd /home/ubuntu/New_Architecture/Tests/Execution_and_Safety
python3 FSM_correct_rejection_rate/evaluate_fsm_rejection.py
python3 Queue_preemption_correctness/evaluate_queue_preemption.py
python3 execution_guardian_dryrun/evaluate_dryrun_dispatch.py
```

## Notes

- Keep vLLM and MediaPipe in separate Python environments due to `protobuf` version constraints.
- Use the per-module README files for detailed setup, configs, and troubleshooting.
