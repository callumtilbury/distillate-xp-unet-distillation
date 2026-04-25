# Results

## Current best

**Run 10** — diameter_mape=1.96% | diameter_mae=0.389um | count_mae=10.4 (4.5%)  
Pipeline: YOLO11n detection → bbox → circle (centre + r=(min(w,h)+√(w·h))/2/2)  
Teacher baseline: Cellpose-SAM-FT → regionprops → equivalent circle  
Model: YOLO11n, conf=0.45, max_det=1000, lr0=0.0005 fine-tuning

**Goal achieved: <2% diameter MAPE ✓**

## Key findings

- **Blended radius formula is optimal**: `(min(w,h)+sqrt(w*h))/2` beats both `min(w,h)/2` and `sqrt(w*h)/2` alone
- **Fine-tuning LR matters**: dropping lr0 from 0.002→0.0005 for final runs gave tighter bbox regression, closing 0.2% MAPE gap
- **Extended conf sweep to 0.50**: optimal conf shifted 0.15→0.30→0.40→0.45 as bbox regression improved across runs
- **Detection + circle post-processing beats segmentation** for diameter accuracy — avoids rectangular mask artifacts
- **No-catheter data essential**: covers large bubbles (2-2.5um) absent in original 10-frame dataset
- **`max_det=300`** YOLO default silently caps detections — must be 1000+ for 400+ bubble images

## Trajectory

| Run | diameter_mape | count_pct | Key change |
|-----|-------------|-----------|-----------|
| 1–4 | — (count only) | 100%→4% | baseline → max_det fix → conf sweep |
| 5   | ~9% | 4.0% | retina_masks=True, circular seg masks |
| 6   | ~9% | 5.3% | det + sqrt(w*h)/2 circle post-proc |
| 7   | 9.04% | — | expanded to 30 imgs (no-catheter data) |
| 8   | 3.88% | 5.9% | switch to min(w,h)/2; 52 warm-start epochs |
| 9   | 2.16% | 7.4% | blended formula (min+sqrt)/2; 164 epochs |
| **10** | **1.96%** | **4.5%** | **lower LR fine-tuning; conf sweep to 0.50** |

## Pipeline (final)

```
Teacher: image → Cellpose-SAM-FT → instance masks → regionprops → (cx, cy, r_px=√(area/π))
Student: image → YOLO11n-det     → bboxes        → post-proc  → (cx, cy, r_px=(min(w,h)+√(w·h))/2/2)
Metric:  diameter_mape = mean|teacher_mean_diam - student_mean_diam| / teacher_mean_diam * 100%
         = 1.96% at conf=0.45
```

## What's next

Goal met. Further gains would require more training data or exploring larger model variants.
Potential improvements: sample more no-catheter frames to reduce bias on large-bubble images.
