# Results

## Current best

**Run 9** — diameter_mape=2.16% | diameter_mae=0.5402um | count_mae=16.4 (7.4%)  
Pipeline: YOLO11n detection → bbox → circle (centre + r=(min(w,h)+√(w·h))/2)  
Teacher baseline: Cellpose-SAM-FT → regionprops → equivalent circle  
Model: YOLO11n (2.58M params), conf=0.40, max_det=1000, 9 runs total, 164 epochs this run

## Key findings

- **Blended radius formula is better than either extreme**: `(min(w,h)+sqrt(w*h))/2` beats both `min(w,h)/2` (+1.72pp) and `sqrt(w*h)/2` (overestimates large bubbles) alone
- **Detection + circle post-processing beats segmentation** for diameter accuracy
- **Systematic overestimate eliminated**: seg model overshot by +0.1um; det model with blended formula is +0.012um (1.2% mean bias)
- **`max_det=300`** YOLO default silently caps detections — must be 1000+ for 400+ bubble images
- **Conf threshold sweep is critical**: optimal shifted from 0.15 (run 6) → 0.30 (run 8) → 0.40 (run 9) as bbox regression improved
- **No-catheter data** essential: covers large bubbles (2-2.5um) absent in original 10-frame dataset

## Trajectory

| Run | diameter_mape | count_pct | Key change |
|-----|-------------|-----------|-----------|
| 1–4 | — (count only) | 100%→4% | baseline → max_det fix → conf sweep |
| 5   | ~9% | 4.0% | retina_masks=True, circular seg masks |
| 6   | ~9% | 5.3% | det + sqrt(w*h)/2 circle post-proc |
| 7   | 9.04% | — | expanded to 30 imgs (no-catheter data) |
| 8   | 3.88% | 5.9% | switch to min(w,h)/2; 52 warm-start epochs |
| **9** | **2.16%** | 7.4% | **blended formula (min+sqrt)/2; 164 epochs** |

## Pipeline (current)

```
Teacher: image → Cellpose-SAM-FT → instance masks → regionprops → (cx, cy, r_px=√(area/π))
Student: image → YOLO11n-det     → bboxes        → post-proc  → (cx, cy, r_px=(min(w,h)+√(w·h))/2/2)
Metric:  diameter_mape = mean|teacher_mean_diam - student_mean_diam| / teacher_mean_diam * 100%
```

## What's next

Closing the final ~0.16% gap to <2% target. Options:
1. More warm-start epochs with lower LR (finer bbox regression)
2. Wider conf sweep including 0.45/0.50 — optimal may have shifted further
3. Mosaic augmentation tuning to emphasize large bubble scenarios
