# VLA_Pipeline

`VLA_Pipeline` 是端到端编排层：把文本规划链路与视觉反射链路统一到同一个事件总线中，最终复用 `VLA_Agent_Core` 的 `TaskQueue + FSMGuardian` 执行动作。

## 核心能力

- 文本指令 -> LLM 规划 -> 动作序列执行
- 视觉感知（Go2 相机优先，失败自动回退本机摄像头）
- MediaPipe Tasks 关键点提取（Pose + Hand）
- 1D-CNN+GRU 手势识别 -> ReflexBridge -> 动作触发
- 事件总线分发（`SpeechEvent` / `PerceptionEvent` / `SystemEvent` / `ErrorEvent`）
- 启动前资源闸门（缺失资源自动下载）
- vLLM 服务健康检查与可选自动拉起

## 目录结构（关键部分）

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

## 运行模式

在 `config/pipeline.yaml` 中通过 `pipeline.mode` 选择：

- `planner_only`：仅文本规划链路
- `reflex_only`：仅视觉反射链路
- `hybrid`：文本规划 + 视觉反射同时启用（默认）

## 重要说明：双环境（强烈建议）

`mediapipe` 与 `vllm` 对 `protobuf` 版本要求冲突，建议拆分两个 Python 环境：

- 管道环境：安装 `requirements.txt`（`protobuf<4`）
- vLLM 环境：安装 `requirements-vllm.txt`（`protobuf>=5`）

不要把两份依赖安装在同一个环境里。

## 快速开始

### 1) 安装管道环境依赖

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
python3 -m pip install -r requirements.txt
```

### 2) 准备 vLLM 独立环境

```bash
/path/to/vllm-env/bin/python -m pip install -r /home/ubuntu/New_Architecture/VLA_Pipeline/requirements-vllm.txt
```

### 3) 启动 Pipeline

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
export VLLM_PYTHON=/path/to/vllm-env/bin/python
python3 -m src.pipeline.orchestrator
```

启动后若 `pipeline.use_text_cli: true`，可在终端直接输入指令：

- 普通指令：`向前走两步后作揖`
- 紧急停机：`停下` / `停止` / `急停` / `stop` / `halt` / `emergency`
- 退出：`exit` / `quit`

## 启动阶段自动执行内容

`orchestrator` 会在初始化阶段执行资源闸门检查（`resources.auto_pull=true` 时自动拉取），确保以下资源完备后才继续：

- `models/best_mp_gesture_model.pth`
- `models/Qwen3.5-4B/config.json`
- `models/MediaPipe_Models/pose_landmarker_lite.task`
- `models/MediaPipe_Models/hand_landmarker.task`

若 `planner.base_url_for_healthcheck` 对应的 vLLM 服务未就绪：

- `vllm.autostart=true`：自动执行 `scripts/run_vllm_server.sh`
- `vllm.autostart=false`：仅等待服务就绪，超时则退出

## 配置说明（`config/pipeline.yaml`）

- `pipeline.mode`：`planner_only` / `reflex_only` / `hybrid`
- `pipeline.use_text_cli`：是否开启终端文本输入
- `pipeline.event_bus_size`：事件队列大小
- `pipeline.reflex_conf_threshold`：视觉反射置信度阈值
- `pipeline.reflex_cooldown_sec`：同类反射动作冷却时间
- `planner.config_path`：指向 `VLA_Agent_Core/config.yaml`
- `planner.base_url_for_healthcheck`：vLLM 健康检查地址
- `vllm.autostart` / `vllm.script_path` / `vllm.startup_timeout_sec`
- `perception.*`：相机、MediaPipe、手势模型参数
- `resources.*`：自动拉取资源参数（含 Hugging Face 配置）

## 与 VLA_Agent_Core 的关系

- 规划：`src/cognition/planner_client.py` 直接复用 `VLA_Agent_Core/core/agent_planner.py`
- 执行：`src/execution/fsm_adapter.py` 复用 `VLA_Agent_Core/execution/fsm_guardian.py`
- 队列：`src/execution/task_queue_adapter.py` 复用 `VLA_Agent_Core/execution/task_queue.py`

这意味着 `VLA_Pipeline` 是“编排层”，而动作语义与安全执行能力主要由 `VLA_Agent_Core` 提供。

## 常见问题

- **出现 `MessageFactory` / `GetPrototype` 相关错误**  
  通常是 `mediapipe` 与 `protobuf` 冲突。请确认管道环境使用 `requirements.txt`，并与 vLLM 环境隔离。

- **vLLM 一直 not ready**  
  检查 `pipeline.yaml` 中 `planner.base_url_for_healthcheck` 与 `vllm.script_path`，并确认 `VLLM_PYTHON` 指向已安装 vLLM 的环境。

- **摄像头无法打开**  
  检查 Go2 网络连通性；如使用本机相机，确认 `perception.local_camera_index`、`/dev/video*` 和权限。

- **手势权重找不到**  
  放置到 `models/best_mp_gesture_model.pth`，或通过 `perception.gesture_weights_path` / `VLA_GESTURE_WEIGHTS` 显式指定。

## 运行测试

```bash
cd /home/ubuntu/New_Architecture/VLA_Pipeline
python3 -m pytest -q
```
