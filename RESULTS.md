# Results

## Current best

**Run 1** — count_mae=432.9 (100% error; student predicts 0 bubbles)

## Key findings

- Teacher pseudolabeling (Cellpose-SAM-FT) works well: 4329 instances across 10 images, ~430 per image
- Pseudolabeling takes ~90s of the 300s budget, leaving only ~18 YOLO training epochs
- 18 epochs is nowhere near enough for dense instance segmentation: mAP50_mask=0.034
- MPS has a known shape-mismatch bug in YOLO11 inline validation (workaround: `val=False`)

## What's next

Cache pseudolabels to disk so subsequent runs skip the 90s pseudolabeling cost and can use the full budget for training. With 300s fully available for YOLO training we should get ~100+ epochs, which should be sufficient to learn basic bubble detection.
