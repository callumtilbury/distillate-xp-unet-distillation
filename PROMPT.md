# unet-distillation

**Goal:** Train a small PicoSAM-style model (tiny U-NET) running on torch MPS (efficiently) to distill the same performance on bubble segmentation as the fine-tuned Cellpose-SAM model in ./models/bubble_finetuned.

**Project name:** unet-distillation

**Primary metric:** `absolute difference in bubble counts between student and teacher` (minimize)

**Time budget:** 5 minutes

<!-- PROMPT.md will be expanded by the agent on first run -->
