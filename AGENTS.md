# Repository contract

This repository supports two auditable workflows:

```text
Baseline → SFT → GRPO → Evaluation
Frozen Actor → Offline Experience Distillation → Experience-Augmented Agent Loop
             → Offline Experience Evolution → Evaluation
```

The runtime contract is ShopSimulator Environment v2.1, Reward v3, observation
v2 and tool schema v2. Do not add compatibility launchers, historical datasets,
old benchmarks, machine-specific paths or experiment journals.

Training data must never overlap `data/evaluation/tasks.jsonl`. Strict success
requires a complete `gold_purchase` terminal result with `reward_valid=true`.

The experience workflow never updates actor weights. Its active store is frozen
during a run, Final-200 trajectories never flow back into the store, and every
injection and semantic-compaction result must be versioned and auditable. API
credentials are supplied only through named environment variables.

Do not start training, merge models or run the 200-task evaluation unless the
user explicitly requests execution.
