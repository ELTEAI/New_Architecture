# VLA Agent Core

`VLA_Agent_Core` is the control core for a robot-dog pipeline. It compiles natural-language instructions into structured action sequences, then executes them through a thread-safe queue and an FSM safety guardian.

## Highlights

- Natural language -> tactical action sequence via LLM function calling
- Thread-safe action buffering (`TaskQueue`)
- FSM transition validation and guarded execution
- Local emergency stop with highest priority (bypasses LLM)
- Runtime speed monitoring (`vx` / `vy` / `omega`)

## Project Layout

```text
VLA_Agent_Core/
├── schemas/
│   └── robot_skill_schema.json   # Tool schema with 13 actions
├── core/
│   └── agent_planner.py          # Planner: text -> tool arguments
├── execution/
│   ├── task_queue.py             # Thread-safe queue
│   └── fsm_guardian.py           # FSM checks + execution guard
├── config.yaml                   # LLM / Audio / Hardware config
├── prompts.py                    # System prompt templates
├── architecture.md               # Architecture note
└── main_agent.py                 # Main entrypoint
```

## How It Works

1. User types a command into `main_agent.py`
2. `VLABrainPlanner` calls an OpenAI-compatible endpoint (default: local vLLM)
3. The model returns arguments for `execute_robot_tactical_sequence` based on `schemas/robot_skill_schema.json`
4. Actions are pushed into `TaskQueue`
5. `FSMGuardian` consumes actions one by one, validates transitions, and executes them (hardware-independent simulation in current version)

## Requirements

- Python 3.10+
- Reachable OpenAI-compatible Chat Completions endpoint (local vLLM recommended)
- Python packages:
  - `openai`
  - `PyYAML`

Install dependencies:

```bash
pip install openai PyYAML
```

## Configuration

Edit `config.yaml`:

- `llm.base_url`: OpenAI-compatible endpoint (default `http://127.0.0.1:8000/v1`)
- `llm.model_name`: must match server-side `--served-model-name`
- `llm.temperature`: recommended range `0.0 ~ 0.3` for stable tool calling
- `audio.*`: reserved for audio layer configuration (main loop is text-first right now)

> Do not commit real API keys to the repository; prefer environment variables at runtime.

## Run

From the `VLA_Agent_Core` directory:

```bash
python main_agent.py
```

Example commands:

- `walk forward two steps then bow`
- `sit down and then stand up`
- `turn left and then stop`

Exit commands:

- `exit` / `quit`

Local emergency commands (without LLM):

- `停下` / `停止` / `急停` / `stop` / `halt` / `emergency`

## Action IDs (Schema-Aligned)

- Mode 0 (emergency): `1` full stop, `duration=0`
- Mode 1 (continuous motion, `duration=0`):
  - `0` forward
  - `2` backward
  - `3` strafe left
  - `4` strafe right
  - `5` turn left
  - `6` turn right
- Mode 2 (blocking actions, recommended `duration=2~5`):
  - `7` sit
  - `8` stand up
  - `9` stretch
  - `10` roll over
  - `11` pose
  - `12` greeting/bow

## Safety Rules

- Emergency stop is always allowed
- During a blocking action, only stop is accepted
- After emergency stop, `stand_up (action_id=8)` is required before normal actions
- Rejections can be inspected via `FSMGuardian.get_rejection_log()`

## Troubleshooting

- **No tool call from model**  
  Verify server-side tool-calling support and `tool-call-parser` configuration.

- **Model name mismatch**  
  Ensure `llm.model_name` exactly matches server `--served-model-name`.

- **Action rejected by FSM**  
  Check transition constraints (for example, some actions require standing state first).

## Version Note

The current `FSMGuardian` execution path is hardware-independent simulation, focused on validating:

- tactical planning output correctness
- queue/thread coordination
- FSM safety constraints
- runtime motion-state observability

You can later replace the simulation logic in `_execute_single_action` with real robot SDK calls.
