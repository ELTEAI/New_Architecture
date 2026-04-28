# voice_command_dataset

`voice_command_dataset` is a test data and evaluation suite for the VLA voice-command pipeline. It covers the full metric chain from `TTS/audio -> ASR -> semantic preservation -> routing -> plan validity -> intent matching -> latency/TTFT`.

## Directory Overview

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

## Dataset Design

`prompts.csv` uses the following fields:

- `id`
- `category` (`normal_action` / `combo_action` / `emergency` / `invalid`)
- `text`
- `language`
- `expected_route`
- `expected_valid`
- `expected_intent`
- `expected_steps`

The current dataset contains 50 text samples (20 normal + 10 combo + 10 emergency + 10 invalid).  
By default, audio is generated with 3 voices (`alloy` / `nova` / `echo`), so the full set size is `50 × 3 = 150` utterances.

## Metrics to Script Mapping

| Metric | Meaning | Script(s) | Default Output |
| --- | --- | --- | --- |
| ASR transcription success rate | Whether Whisper transcription is usable (with WER near-match) | `results/run_asr_test.py` + `results/summarize_asr_results.py` | `results/asr_results.*`, `results/asr_summary.txt` |
| command-level semantic accuracy | Whether ASR preserves command semantics | `command_level_semantic_accuracy/evaluate_command_semantics.py` | `semantic_accuracy_results.jsonl`, `semantic_accuracy_summary.txt` |
| emergency routing accuracy | Whether emergency commands route to emergency path | `emergency_routing_accuracy/evaluate_emergency_routing.py` | `emergency_routing_results.jsonl`, `emergency_routing_summary.txt` |
| plan validity rate | Whether planner output action sequence is structurally valid | `plan_validity_rate/evaluate_plan_validity.py` | `plan_validity_results.jsonl`, `plan_validity_summary.txt` |
| intent-plan matching accuracy | Whether generated action sequence matches expected steps | `intent_plan_matching_accuracy/evaluate_intent_plan_matching.py` | `intent_plan_matching_results.jsonl`, `intent_plan_matching_summary.txt` |
| planning latency | Latency from transcript to plan completion | `planning_latency/summarize_planning_latency.py` | `planning_latency_summary.txt` |
| TTFT | Time from planner request to first token | `TTFT/benchmark_*.py` + `TTFT/compare_ttft_results.py` | `ttft_*_results.jsonl`, `ttft_comparison_summary.txt` |

## Prerequisites

### 1) Python packages

At minimum:

- `openai`
- `python-dotenv`
- `pyyaml`

(Skip if already installed in your project environment.)

### 2) API key and environment

`generate_tts_dataset.py`, `results/run_asr_test.py`, and `command_level_semantic_accuracy/evaluate_command_semantics.py` read `OPENAI_API_KEY` from `Tests/.env` (the parent directory of `voice_command_dataset`).

### 3) Planner / vLLM

`plan_validity_rate/evaluate_plan_validity.py` calls `VLA_Agent_Core` via `PlannerClient` in `VLA_Pipeline`. Ensure:

- `VLA_Agent_Core/config.yaml` is available
- The corresponding LLM service (typically vLLM OpenAI-compatible endpoint) is running

## End-to-End Run Commands

From repository root:

```bash
cd /home/ubuntu/New_Architecture/Tests/voice_command_dataset
```

### Step 1: Generate TTS audio (skips existing files)

```bash
python3 generate_tts_dataset.py
```

Outputs to `audio/`, file naming format: `<id>_<voice>.wav`.

### Step 2: Build manifest

```bash
python3 build_manifest.py
```

Generates `manifest.jsonl` by default. Use `--allow-partial` if missing audio should be tolerated.

### Step 3: Run ASR evaluation

```bash
python3 results/run_asr_test.py
python3 results/summarize_asr_results.py
```

### Step 4: Evaluate command-level semantic accuracy

```bash
python3 command_level_semantic_accuracy/evaluate_command_semantics.py
```

### Step 5: Evaluate emergency routing accuracy

```bash
python3 emergency_routing_accuracy/evaluate_emergency_routing.py
```

### Step 6: Evaluate plan validity rate

```bash
python3 plan_validity_rate/evaluate_plan_validity.py
```

### Step 7: Evaluate intent-plan matching accuracy

```bash
python3 intent_plan_matching_accuracy/evaluate_intent_plan_matching.py
```

### Step 8: Summarize planning latency

```bash
python3 planning_latency/summarize_planning_latency.py
```

### Step 9: TTFT comparison (vLLM vs Transformers)

```bash
python3 TTFT/benchmark_vllm_ttft.py
python3 TTFT/benchmark_transformers_ttft.py
python3 TTFT/compare_ttft_results.py
```

## Script Dependency Flow

1. `prompts.csv` -> `generate_tts_dataset.py` -> `audio/*.wav`
2. `prompts.csv + audio/*.wav` -> `build_manifest.py` -> `manifest.jsonl`
3. `manifest.jsonl` -> `results/run_asr_test.py` -> `asr_results.jsonl`
4. `asr_results.jsonl` -> `evaluate_command_semantics.py` -> `semantic_accuracy_results.jsonl`
5. `semantic_accuracy_results.jsonl` -> `evaluate_emergency_routing.py` -> `emergency_routing_results.jsonl`
6. `emergency_routing_results.jsonl` -> `evaluate_plan_validity.py` -> `plan_validity_results.jsonl`
7. `plan_validity_results.jsonl` -> `evaluate_intent_plan_matching.py` -> `intent_plan_matching_results.jsonl`
8. `plan_validity_results.jsonl` -> `summarize_planning_latency.py` -> `planning_latency_summary.txt`
9. `plan_validity_results.jsonl` -> `benchmark_*_ttft.py` -> `ttft_*_results.jsonl` -> `compare_ttft_results.py`

## Existing Result Files (Examples)

These files already exist and can be inspected directly:

- `results/asr_summary.txt`
- `command_level_semantic_accuracy/semantic_accuracy_summary.txt`
- `emergency_routing_accuracy/emergency_routing_summary.txt`
- `plan_validity_rate/plan_validity_summary.txt`
- `intent_plan_matching_accuracy/intent_plan_matching_summary.txt`
- `planning_latency/planning_latency_summary.txt`
- `TTFT/ttft_comparison_summary.txt`

These provide a baseline from a fully completed run and can be used for regression comparison.

## Common Issues

- **`OPENAI_API_KEY` not set**  
  Add it to `Tests/.env` and rerun.

- **`build_manifest.py` reports missing audio**  
  Run `generate_tts_dataset.py` first, or use `--allow-partial` temporarily.

- **`evaluate_plan_validity.py` reports planner/vLLM unavailable**  
  Check `VLA_Agent_Core/config.yaml` and verify vLLM service status.

- **TTFT scripts fail to connect model endpoint**  
  Ensure `llm.base_url` and `llm.model_name` in `VLA_Agent_Core/config.yaml` match the running server, or override via CLI arguments.
