# Results

## Current best

**Run 5** — diameter_mae=0.189um | count_mae=16.3 (4.0% error)  
Student: mean_diam=1.27um vs teacher: 1.17um | conf=0.25, max_det=1000, retina_masks=True  
Model: YOLO11n-seg (2.83M params), ~367 total training epochs

## Key findings

- **`retina_masks=True`** is essential: eliminates rectangular mask artefacts, raises mask mAP50 from 0.168 to 0.252
- **Count metric was misleading**: count MAE of 15.7 looked good but masks were squares. Diameter MAE is the correct metric.
- **max_det=300** (YOLO default) silently caps detections below teacher counts — must raise to 1000
- **Conf threshold=0.25** optimal with retina_masks (not 0.15 from run 4): higher conf filters noisy small-bubble detections
- **Systematic +0.1um diameter overestimate**: student=1.27um vs teacher=1.17um — likely inflated polygon labels from contour extraction
- Warm-starting across runs is highly efficient; pseudolabels cached in `.distillate/pseudolabels/`

## Trajectory

| Run | Primary metric | count_pct | Key change |
|-----|---------------|-----------|-----------|
| 1   | count_mae=432.9 | 100% | baseline: 18 epochs, 0 detections |
| 2   | count_mae=315.6 | 73%  | cached labels, 71 epochs |
| 3   | count_mae=31.1  | 7.8% | warm start + max_det=1000 |
| 4   | count_mae=15.7  | 4.0% | conf sweep, goal ✓ |
| 5   | **diameter_mae=0.189um** | 4.0% | retina_masks=True, circular masks ✓ |

## What's next

The student overestimates bubble diameter by ~8% systematically. Fix: tighten pseudolabel polygons (reduce max_contour_points or use tighter contour tolerance) so labels match the actual mask boundary more precisely.
