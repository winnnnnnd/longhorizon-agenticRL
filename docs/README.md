# Documentation

Follow the guides in workflow order:

1. [Data collection](data-collection.md) explains how the checked-in SFT data
   was produced and audited.
2. [SFT](sft.md) trains the first useful shopping agent.
3. [GRPO](grpo.md) improves that model with online environment reward.
4. [Evaluation](evaluation.md) compares baseline, SFT and GRPO fairly.
5. [Final-200 Clean evaluation dataset](evaluation-dataset.md) defines the current
   curated benchmark and its update record.
6. The root [README](../README.md) documents segment extraction, six-key
   clustering, Experience Card retrieval, injection and store governance.

[Reward v3](reward-v3.md) is the detailed specification shared by collection,
GRPO and evaluation.

The frozen-actor Agent Loop reuses the same environment, Reward and evaluation
contracts without updating actor parameters.
