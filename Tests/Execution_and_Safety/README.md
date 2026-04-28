# Execution_and_Safety

`Execution_and_Safety` contains evaluation scripts for the execution layer in `VLA_Agent_Core`, focused on safety invariants and command-dispatch correctness.

It currently covers three areas:

- FSM transition rejection/acceptance correctness
- Queue preemption correctness under emergency stop
- Dry-run hardware dispatch correctness against action schema

## Directory Structure

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

## Test Scope

### 1) FSM Correct Rejection Rate

Script: `FSM_correct_rejection_rate/evaluate_fsm_rejection.py`

What it verifies:

- Unknown `action_id` values are always rejected
- `stop (action_id=1)` is always allowed
- During `BLOCKING`, only stop is allowed
- After `EMERGENCY_STOP`, only `stand_up` (and stop) are allowed
- Locomotion / blocking skills are accepted only in allowed FSM states

Output:

- `fsm_rejection_results.jsonl` (per-case results)
- `fsm_rejection_summary.txt` (aggregated metrics)

### 2) Queue Preemption Correctness

Script: `Queue_preemption_correctness/evaluate_queue_preemption.py`

What it verifies (including repeated emergency cycles):

- Initial queue admission behaves as expected under queue capacity
- Queue is fully cleared on emergency preemption
- Emergency stop task (`action_id=1`) is injected correctly
- Stop execution forces runtime velocity to zero and transitions to `EMERGENCY_STOP`

Output:

- `queue_preemption_results.jsonl`
- `queue_preemption_summary.txt`

### 3) Execution Guardian Dry-run Dispatch

Scripts:

- `execution_guardian_dryrun/dryrun_hardware_wrapper.py`
- `execution_guardian_dryrun/evaluate_dryrun_dispatch.py`

What it verifies:

- `action_id` domain is aligned with `VLA_Agent_Core/schemas/robot_skill_schema.json`
- Mode-based duration rules are enforced before dispatch
- Each valid action maps to the expected high-level mock method
- Dispatch success and dispatch latency are recorded per trial

Output:

- `dryrun_dispatch_results.jsonl`
- `dryrun_dispatch_summary.txt`

## Prerequisites

- Python 3.10+
- `VLA_Agent_Core` must exist at:
  - `/home/ubuntu/New_Architecture/VLA_Agent_Core`
- In particular, these files are required:
  - `execution/fsm_guardian.py`
  - `execution/task_queue.py`
  - `schemas/robot_skill_schema.json`

## How To Run

From this directory:

```bash
cd /home/ubuntu/New_Architecture/Tests/Execution_and_Safety
```

Run each evaluation:

```bash
python3 FSM_correct_rejection_rate/evaluate_fsm_rejection.py
python3 Queue_preemption_correctness/evaluate_queue_preemption.py
python3 execution_guardian_dryrun/evaluate_dryrun_dispatch.py
```

## Current Baseline Results

From existing summary files in this directory:

- FSM rejection:
  - Total cases: `102`
  - FSM decision accuracy: `100.00%`
  - Correct rejection rate: `100.00%`
  - Correct acceptance rate: `100.00%`

- Queue preemption:
  - Total scenarios: `10`
  - Emergency cycles: `12`
  - Overall queue preemption correctness: `100.00%`
  - Mean queue clear latency: `0.000019 s`

- Dry-run dispatch:
  - Total tasks: `65` (13 action IDs x 5 trials)
  - Schema-valid tasks: `100.00%`
  - Dispatch success: `100.00%`
  - Mean dispatch latency: `0.000002 s`

## Notes

- This test suite is execution-focused and independent from ASR/planning benchmarks.
- The dry-run dispatch layer validates command mapping and schema conformance without moving real hardware.
- Baseline metrics can be used for regression checks after changing FSM policies, queue behavior, or action mappings.
