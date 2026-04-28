# VLA Agent Core

`VLA_Agent_Core` 是一个面向机器狗控制链路的智能体核心模块：把自然语言指令编译为结构化动作序列，并通过线程安全队列与 FSM 安全守护执行。

## 功能概览

- 自然语言 -> 战术动作序列（LLM Function Calling）
- 线程安全动作缓冲（Task Queue）
- FSM 状态校验与执行守护（含阻塞动作保护）
- 紧急停机本地优先（不依赖 LLM）
- 实时速度状态监视（`vx` / `vy` / `omega`）

## 项目结构

```text
VLA_Agent_Core/
├── schemas/
│   └── robot_skill_schema.json   # LLM 工具调用契约（13个动作）
├── core/
│   └── agent_planner.py          # 大脑：自然语言 -> tool call 参数
├── execution/
│   ├── task_queue.py             # 线程安全动作队列
│   └── fsm_guardian.py           # FSM 验证与动作执行守护
├── config.yaml                   # LLM / Audio / Hardware 配置
├── prompts.py                    # System Prompt 模板
├── architecture.md               # 架构说明
└── main_agent.py                 # 主入口
```

## 运行原理

1. 用户输入文本指令到 `main_agent.py`
2. `VLABrainPlanner` 调用本地 OpenAI 兼容接口（默认指向 vLLM）
3. 模型按 `schemas/robot_skill_schema.json` 返回 `execute_robot_tactical_sequence` 参数
4. 动作序列写入 `TaskQueue`
5. `FSMGuardian` 逐条取出动作，进行状态合法性校验后执行（当前版本为硬件无关模拟）

## 环境要求

- Python 3.10+
- 可访问的 OpenAI 兼容 Chat Completions 接口（推荐本地 vLLM）
- 依赖：
  - `openai`
  - `PyYAML`

可使用以下命令安装：

```bash
pip install openai PyYAML
```

## 配置说明

编辑 `config.yaml`：

- `llm.base_url`：OpenAI 兼容服务地址（默认 `http://127.0.0.1:8000/v1`）
- `llm.model_name`：需与服务端 `--served-model-name` 一致
- `llm.temperature`：建议 0.0~0.3（工具调用更稳定）
- `audio.*`：预留语音层配置（当前主流程以文本输入为主）

> 建议不要把真实密钥写入仓库；密钥优先通过环境变量注入。

## 启动方式

在 `VLA_Agent_Core` 目录下执行：

```bash
python main_agent.py
```

启动后可持续输入自然语言指令，例如：

- `向前走两步后作揖`
- `先坐下再起立`
- `左转一下再停下`

退出命令：

- `exit` / `quit`

本地紧急停机命令（不经过 LLM）：

- `停下` / `停止` / `急停` / `stop` / `halt` / `emergency`

## 动作 ID 说明（Schema 对齐）

- Mode 0（紧急）：`1` 完全静止，`duration=0`
- Mode 1（连续运动，`duration=0`）：
  - `0` 前进
  - `2` 后退
  - `3` 左横移
  - `4` 右横移
  - `5` 左转
  - `6` 右转
- Mode 2（阻塞动作，建议 `duration=2~5`）：
  - `7` 坐下
  - `8` 起立
  - `9` 伸懒腰
  - `10` 打滚
  - `11` 摆姿势
  - `12` 拜年/作揖

## 安全机制

- 紧急停机始终可执行
- 阻塞动作进行中仅允许停机
- 紧急停机后要求先 `起立(action_id=8)` 再执行其他动作
- FSM 拒绝记录可通过 `FSMGuardian.get_rejection_log()` 获取

## 常见问题

- **模型没有触发工具调用**  
  检查服务端工具调用能力与 `tool-call-parser` 参数配置。

- **模型名不匹配**  
  确认 `config.yaml` 中 `llm.model_name` 与服务端 `--served-model-name` 完全一致。

- **动作不执行或被拒绝**  
  查看 FSM 状态流转约束（如未起立不能执行部分动作）。

## 当前版本说明

当前 `FSMGuardian` 执行层是硬件无关模拟实现，重点验证：

- 上层战术编排正确性
- 队列与线程协同
- 状态机安全约束
- 速度状态变化可观测性

后续可将 `_execute_single_action` 中的模拟逻辑替换为真实机器人 SDK 调用。
