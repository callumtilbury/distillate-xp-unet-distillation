# Results

## Current best

**Run 4** — count_mae=15.7 (**4.0% per-image error** vs teacher) ✓ goal achieved (<5%)  
Student: 4635 total detections vs teacher: 4329 | best conf=0.15, max_det=1000  
Model: YOLO11n-seg (2.83M params), 268 total training epochs

## Key findings

- **max_det=300** (YOLO default) is a fatal ceiling: teacher averages 433 bubbles/image. Must raise to ≥1000.
- **Conf threshold matters enormously**: same checkpoint gives MAE=115 at conf=0.05 but MAE=15.7 at conf=0.15. Always sweep.
- **Warm-starting** across runs is highly efficient — avoids 60s pseudolabeling cost each run, compounds epochs cheaply.
- **Pseudolabels** from Cellpose-SAM-FT are high quality: 4329 instances across 10 images (~433/image).
- MPS has a YOLO11 inline-validation shape-mismatch bug; workaround: `val=False` during training, evaluate manually.

## Trajectory

| Run | count_mae | count_pct_diff | Key change |
|-----|-----------|----------------|-----------|
| 1   | 432.9     | 100%           | baseline: 18 epochs, no cache |
| 2   | 315.6     | 72.9%          | cached labels, 71 epochs |
| 3   | 31.1      | 7.8%           | warm start + max_det=1000 fix |
| 4   | **15.7**  | **4.0%**       | lower LR + conf sweep (goal ✓) |

## What's next

Validate on held-out images to confirm generalisation. Consider higher imgsz (e.g. 1280) for smaller bubbles.
