# voice_command_dataset

`voice_command_dataset` 是用于评估 VLA 语音指令链路的测试数据与评测脚本集合，覆盖从 `TTS/音频 -> ASR -> 语义保持 -> 路由 -> 规划有效性 -> 意图匹配 -> 时延/TTFT` 的完整指标链。

## 目录内容概览

```text
voice_command_dataset/
├── prompts.csv
├── generate_tts_dataset.py
├── build_manifest.py
├── manifest.jsonl
├── results/
│   ├── run_asr_test.py
│   ├── summarize_asr_results.py
│   ├── asr_results.csv
│   ├── asr_results.jsonl
│   └── asr_summary.txt
├── command_level_semantic_accuracy/
│   ├── evaluate_command_semantics.py
│   ├── semantic_accuracy_results.jsonl
│   └── semantic_accuracy_summary.txt
├── emergency_routing_accuracy/
│   ├── evaluate_emergency_routing.py
│   ├── emergency_routing_results.jsonl
│   └── emergency_routing_summary.txt
├── plan_validity_rate/
│   ├── evaluate_plan_validity.py
│   ├── plan_validity_results.jsonl
│   └── plan_validity_summary.txt
├── intent_plan_matching_accuracy/
│   ├── evaluate_intent_plan_matching.py
│   ├── intent_plan_matching_results.jsonl
│   └── intent_plan_matching_summary.txt
├── planning_latency/
│   ├── summarize_planning_latency.py
│   └── planning_latency_summary.txt
└── TTFT/
    ├── benchmark_vllm_ttft.py
    ├── benchmark_transformers_ttft.py
    ├── compare_ttft_results.py
    ├── ttft_vllm_results.jsonl
    ├── ttft_transformers_results.jsonl
    └── ttft_comparison_summary.txt
```

## 数据集设计

`prompts.csv` 使用如下字段：

- `id`
- `category`（`normal_action` / `combo_action` / `emergency` / `invalid`）
- `text`
- `language`
- `expected_route`
- `expected_valid`
- `expected_intent`
- `expected_steps`

当前已内置 50 条文本样本（20 normal + 10 combo + 10 emergency + 10 invalid）。  
默认会用 3 种声线（`alloy` / `nova` / `echo`）生成音频，因此全量规模是 `50 × 3 = 150` 条语音样本。

## 指标与脚本映射

| 指标 | 含义 | 对应脚本 | 默认输出 |
| --- | --- | --- | --- |
| ASR transcription success rate | Whisper 转写是否可用（含 WER 近匹配） | `results/run_asr_test.py` + `results/summarize_asr_results.py` | `results/asr_results.*`, `results/asr_summary.txt` |
| command-level semantic accuracy | ASR 是否保留命令语义 | `command_level_semantic_accuracy/evaluate_command_semantics.py` | `semantic_accuracy_results.jsonl`, `semantic_accuracy_summary.txt` |
| emergency routing accuracy | emergency 是否正确路由到紧急通道 | `emergency_routing_accuracy/evaluate_emergency_routing.py` | `emergency_routing_results.jsonl`, `emergency_routing_summary.txt` |
| plan validity rate | Planner 输出动作序列是否结构合法 | `plan_validity_rate/evaluate_plan_validity.py` | `plan_validity_results.jsonl`, `plan_validity_summary.txt` |
| intent-plan matching accuracy | 生成动作序列是否匹配期望步骤 | `intent_plan_matching_accuracy/evaluate_intent_plan_matching.py` | `intent_plan_matching_results.jsonl`, `intent_plan_matching_summary.txt` |
| planning latency | transcript -> plan 的耗时统计 | `planning_latency/summarize_planning_latency.py` | `planning_latency_summary.txt` |
| TTFT | planner 请求到首 token 时间 | `TTFT/benchmark_*.py` + `TTFT/compare_ttft_results.py` | `ttft_*_results.jsonl`, `ttft_comparison_summary.txt` |

## 前置依赖

### 1) Python 依赖

至少需要：

- `openai`
- `python-dotenv`
- `pyyaml`

（如果项目总环境已安装可跳过）

### 2) 密钥与环境

`generate_tts_dataset.py`、`results/run_asr_test.py`、`command_level_semantic_accuracy/evaluate_command_semantics.py` 都会读取 `Tests/.env`（即 `voice_command_dataset` 上级目录）中的 `OPENAI_API_KEY`。

### 3) Planner / vLLM

`plan_validity_rate/evaluate_plan_validity.py` 通过 `VLA_Pipeline` 的 `PlannerClient` 调用 `VLA_Agent_Core`，需保证：

- `VLA_Agent_Core/config.yaml` 可用
- 对应 LLM 服务（通常是 vLLM OpenAI 兼容接口）已就绪

## 一次性全流程命令

在仓库根目录执行：

```bash
cd /home/ubuntu/New_Architecture/Tests/voice_command_dataset
```

### Step 1: 生成 TTS 音频（可跳过已存在文件）

```bash
python3 generate_tts_dataset.py
```

输出到 `audio/`，文件名格式：`<id>_<voice>.wav`。

### Step 2: 构建 manifest

```bash
python3 build_manifest.py
```

默认生成 `manifest.jsonl`。若允许缺失音频可使用 `--allow-partial`。

### Step 3: ASR 评测

```bash
python3 results/run_asr_test.py
python3 results/summarize_asr_results.py
```

### Step 4: 命令级语义准确率

```bash
python3 command_level_semantic_accuracy/evaluate_command_semantics.py
```

### Step 5: 紧急路由准确率

```bash
python3 emergency_routing_accuracy/evaluate_emergency_routing.py
```

### Step 6: 规划合法率

```bash
python3 plan_validity_rate/evaluate_plan_validity.py
```

### Step 7: 意图-计划匹配率

```bash
python3 intent_plan_matching_accuracy/evaluate_intent_plan_matching.py
```

### Step 8: 规划延迟统计

```bash
python3 planning_latency/summarize_planning_latency.py
```

### Step 9: TTFT 对比（vLLM vs Transformers）

```bash
python3 TTFT/benchmark_vllm_ttft.py
python3 TTFT/benchmark_transformers_ttft.py
python3 TTFT/compare_ttft_results.py
```

## 脚本依赖关系（数据流）

1. `prompts.csv` -> `generate_tts_dataset.py` -> `audio/*.wav`
2. `prompts.csv + audio/*.wav` -> `build_manifest.py` -> `manifest.jsonl`
3. `manifest.jsonl` -> `results/run_asr_test.py` -> `asr_results.jsonl`
4. `asr_results.jsonl` -> `evaluate_command_semantics.py` -> `semantic_accuracy_results.jsonl`
5. `semantic_accuracy_results.jsonl` -> `evaluate_emergency_routing.py` -> `emergency_routing_results.jsonl`
6. `emergency_routing_results.jsonl` -> `evaluate_plan_validity.py` -> `plan_validity_results.jsonl`
7. `plan_validity_results.jsonl` -> `evaluate_intent_plan_matching.py` -> `intent_plan_matching_results.jsonl`
8. `plan_validity_results.jsonl` -> `summarize_planning_latency.py` -> `planning_latency_summary.txt`
9. `plan_validity_results.jsonl` -> `benchmark_*_ttft.py` -> `ttft_*_results.jsonl` -> `compare_ttft_results.py`

## 当前目录已有结果（示例）

以下文件已存在，可直接查看：

- `results/asr_summary.txt`
- `command_level_semantic_accuracy/semantic_accuracy_summary.txt`
- `emergency_routing_accuracy/emergency_routing_summary.txt`
- `plan_validity_rate/plan_validity_summary.txt`
- `intent_plan_matching_accuracy/intent_plan_matching_summary.txt`
- `planning_latency/planning_latency_summary.txt`
- `TTFT/ttft_comparison_summary.txt`

这些是一次完整跑通后的基线输出，后续可用于回归对比。

## 常见问题

- **`OPENAI_API_KEY` 未设置**  
  在 `Tests/.env` 设置后重试。

- **`build_manifest.py` 报缺失音频**  
  先运行 `generate_tts_dataset.py`，或临时使用 `--allow-partial`。

- **`evaluate_plan_validity.py` 报 planner/vLLM 不可用**  
  检查 `VLA_Agent_Core/config.yaml` 与 vLLM 服务状态。

- **TTFT 脚本报模型连接错误**  
  确认 `VLA_Agent_Core/config.yaml` 中 `llm.base_url/model_name` 与服务端一致，或通过命令行参数覆盖。
