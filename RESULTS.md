# Results

## Current best

**Run 6** — diameter_mae=0.159um | count_mae=24.6 (5.3%)  
Pipeline: YOLO11n detection → bbox → circle (centre + r=√(w·h)/2)  
Teacher baseline: Cellpose-SAM-FT → regionprops → equivalent circle  
Model: YOLO11n (2.58M params, <1M active), conf=0.15, max_det=1000

## Key findings

- **Detection + circle post-processing beats segmentation** for diameter accuracy (0.159 vs 0.189um)
- **Systematic overestimate eliminated**: seg model overshot by +0.1um; det model is -0.04um (slight undercount)  
- **`retina_masks=True`** was essential when using segmentation; detection sidesteps this entirely
- **`max_det=300`** YOLO default silently caps detections — must be 1000+ for 400+ bubble images
- **Conf threshold sweep is critical**: optimal varies by run (0.15 for det, 0.25 for seg)
- Pseudolabels cached in `.distillate/pseudolabels/`; det labels regenerated in <1s from cached masks

## Trajectory

| Run | diameter_mae | count_pct | Key change |
|-----|-------------|-----------|-----------|
| 1–4 | — (count only) | 100%→4% | baseline → max_det fix → conf sweep |
| 5   | 0.189um | 4.0% | retina_masks=True, circular seg masks |
| **6**| **0.159um** | 5.3% | **det + circle post-proc (no mask head)** |

## Pipeline (current)

```
Teacher: image → Cellpose-SAM-FT → instance masks → regionprops → (cx, cy, r_px=√(area/π))
Student: image → YOLO11n-det     → bboxes        → post-proc  → (cx, cy, r_px=√(w·h)/2)
Metric:  diameter_mae_um = mean|teacher_diam - student_diam| across matched distributions
```

## What's next

Warm-start det model + reduce count error below 5% with conf tuning or more epochs.
