# VLA_Pipeline

`VLA_Pipeline` is the end-to-end orchestration layer. It unifies the text-planning path and the vision-reflex path into one event bus, then reuses `VLA_Agent_Core` (`TaskQueue + FSMGuardian`) for action execution.

## Core Capabilities

- Text instruction -> LLM planning -> action sequence execution
- Visual perception (Go2 camera first, auto-fallback to local camera)
- MediaPipe Tasks keypoint extraction (Pose + Hand)
- 1D-CNN+GRU gesture recognition -> ReflexBridge -> action trigger
- Event-bus dispatching (`SpeechEvent` / `PerceptionEvent` / `SystemEvent` / `ErrorEvent`)
- Startup resource gate (auto-download when assets are missing)
- vLLM health check and optional auto-start

## Directory Layout (Key Parts)

```text
VLA_Pipeline/
├── config/
│   └── pipeline.yaml
├── scripts/
│   ├── run_vllm_server.sh
│   └── download_qwen35_4b.py
├── src/
│   ├── pipeline/
│   │   ├── orchestrator.py
│   │   ├── contracts.py
│   │   ├── event_bus.py
│   │   └── health.py
│   ├── cognition/
│   │   ├── planner_client.py
│   │   └── prompt_router.py
│   ├── perception/
│   │   ├── mediapipe_stream.py
│   │   ├── gesture_classifier.py
│   │   └── reflex_bridge.py
│   ├── execution/
│   │   ├── task_queue_adapter.py
│   │   └── fsm_adapter.py
│   └── runtime/
│       ├── logger.py
│       └── metrics.py
├── requirements.txt
├── requirements-vllm.txt
└── README.md
```

## Running Modes

Set `pipeline.mode` in `config/pipeline.yaml`:

- `planner_only`: text planning path only
- `reflex_only`: vision reflex path only
- `hybrid`: both text planning and vision reflex (default)

## Important: Two Python Environments (Strongly Recommended)

`mediapipe` and `vllm` require conflicting `protobuf` versions, so use two separate environments:

- Pipeline environment: install `requirements.txt` (`protobuf<4`)
- vLLM environment: install `requirements-vllm.txt` (`protobuf>=5`)

Do not install both dependency sets in the same environment.

## Quick Start

### 1) Install pipeline dependencies

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
python3 -m pip install -r requirements.txt
```

### 2) Prepare a dedicated vLLM environment

```bash
/path/to/vllm-env/bin/python -m pip install -r /home/ubuntu/New_Architecture/VLA_Pipeline/requirements-vllm.txt
```

### 3) Start the pipeline

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
export VLLM_PYTHON=/path/to/vllm-env/bin/python
python3 -m src.pipeline.orchestrator
```

If `pipeline.use_text_cli: true`, you can type commands directly:

- Normal command: `walk forward two steps and bow`
- Emergency stop: `停下` / `停止` / `急停` / `stop` / `halt` / `emergency`
- Exit: `exit` / `quit`

## What Happens During Startup

`orchestrator` runs a resource gate at initialization (auto-pull when `resources.auto_pull=true`) and only continues after all required assets are ready:

- `models/best_mp_gesture_model.pth`
- `models/Qwen3.5-4B/config.json`
- `models/MediaPipe_Models/pose_landmarker_lite.task`
- `models/MediaPipe_Models/hand_landmarker.task`

If the vLLM endpoint at `planner.base_url_for_healthcheck` is not ready:

- `vllm.autostart=true`: automatically runs `scripts/run_vllm_server.sh`
- `vllm.autostart=false`: waits for readiness only, then exits on timeout

## Configuration (`config/pipeline.yaml`)

- `pipeline.mode`: `planner_only` / `reflex_only` / `hybrid`
- `pipeline.use_text_cli`: enable/disable terminal text input
- `pipeline.event_bus_size`: event queue capacity
- `pipeline.reflex_conf_threshold`: confidence threshold for reflex actions
- `pipeline.reflex_cooldown_sec`: cooldown for repeated reflex actions
- `planner.config_path`: points to `VLA_Agent_Core/config.yaml`
- `planner.base_url_for_healthcheck`: vLLM health-check URL
- `vllm.autostart` / `vllm.script_path` / `vllm.startup_timeout_sec`
- `perception.*`: camera, MediaPipe, and gesture-model settings
- `resources.*`: auto-download settings (including Hugging Face fields)

## Relationship with VLA_Agent_Core

- Planning: `src/cognition/planner_client.py` directly reuses `VLA_Agent_Core/core/agent_planner.py`
- Execution: `src/execution/fsm_adapter.py` reuses `VLA_Agent_Core/execution/fsm_guardian.py`
- Queue: `src/execution/task_queue_adapter.py` reuses `VLA_Agent_Core/execution/task_queue.py`

So `VLA_Pipeline` acts as the orchestration layer, while action semantics and execution safety mainly come from `VLA_Agent_Core`.

## Troubleshooting

- **`MessageFactory` / `GetPrototype` related errors**  
  Usually a `mediapipe` and `protobuf` version conflict. Make sure pipeline uses `requirements.txt` and vLLM stays in a separate environment.

- **vLLM stays "not ready"**  
  Check `planner.base_url_for_healthcheck` and `vllm.script_path` in `pipeline.yaml`, and ensure `VLLM_PYTHON` points to an environment with `vllm` installed.

- **Camera cannot be opened**  
  Check Go2 network connectivity; for local camera, verify `perception.local_camera_index`, `/dev/video*`, and permissions.

- **Gesture weights not found**  
  Put the file at `models/best_mp_gesture_model.pth`, or set `perception.gesture_weights_path` / `VLA_GESTURE_WEIGHTS`.

## Run Tests

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
python3 -m pytest -q
```
