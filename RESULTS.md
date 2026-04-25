# Results

## Current best

**Run 3** — count_mae=31.1 (7.8% per-image error vs teacher)  
Student: 4640 total detections vs teacher: 4329 (slight overcounting)

## Key findings

- **max_det=300** (YOLO default) is a fatal ceiling: teacher averages 433 bubbles/image, so student could never match. Raising to 1000 was essential.
- Pseudolabeling (Cellpose-SAM-FT on 10 images) takes ~60-90s; must be cached across runs.
- 18 epochs → mAP≈0; 71 epochs → 27% recall; 169 epochs → 7.8% error. Training is the bottleneck.
- Warm-starting from a prior checkpoint was highly effective: run 3 went from 7.8% to within reach of 5% goal.
- MPS has a YOLO11 inline-validation shape-mismatch bug; workaround: `val=False` + manual eval.

## What's next

Tune conf threshold sweep and NMS to get per-image error below 5% (MAE≤22). Also explore more training epochs — the model is still improving. Consider raising `imgsz` if bubbles are small relative to 640px.
