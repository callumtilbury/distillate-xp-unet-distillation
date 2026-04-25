"""
Run 5: retina_masks=True to fix rectangular mask artefacts.
Primary metric: diameter_mae_um (from mask area -> equivalent circle diameter).
Warm-start from run 4 checkpoint.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_DIR = Path("/Users/crt25/code/bubcount-distill")
SAMPLES_DIR = REPO_DIR / "samples"
DATASET_DIR = Path(".distillate/pseudolabels")
OUT_DIR = Path("/tmp/bubcount_distill_run5")
CKPT_DIR = Path(".distillate/checkpoints")
PREV_WEIGHTS = Path("/tmp/bubcount_distill_run4/runs/student/weights/last.pt")

SCALE_UM_PX = 0.0825  # from AnalysisParameters

sys.path.insert(0, str(REPO_DIR))


def read_train_budget():
    budget_file = Path(".distillate/budget.json")
    if budget_file.exists():
        data = json.loads(budget_file.read_text())
        return max(60, data.get("train_budget_seconds", 600) - 60)
    return 540


MAX_SECONDS = read_train_budget()
_start = time.time()
print(f"Budget: {MAX_SECONDS}s", flush=True)

# ── teacher masks -> diameter distribution ────────────────────────────────────
print("=== Loading teacher masks for diameter comparison ===", flush=True)

from skimage import io
from skimage.measure import regionprops

teacher_diameters = {}   # img_name -> sorted array of diameters in um
teacher_counts = {}

for npy_file in sorted((DATASET_DIR / "masks").glob("*.npy")):
    mask = np.load(str(npy_file))
    props = regionprops(mask)
    diams = []
    for r in props:
        d_px = 2 * np.sqrt(r.area / np.pi)
        diams.append(d_px * SCALE_UM_PX)
    img_name = npy_file.stem + ".tif"
    teacher_diameters[img_name] = np.array(sorted(diams))
    teacher_counts[img_name] = len(diams)

all_teacher_diams = np.concatenate(list(teacher_diameters.values()))
print(f"Teacher: {len(teacher_counts)} images, mean_diam={all_teacher_diams.mean():.2f}um, "
      f"std={all_teacher_diams.std():.2f}um", flush=True)

# ── train ────────────────────────────────────────────────────────────────────
print("=== Training (warm start, retina_masks=True) ===", flush=True)

remaining = MAX_SECONDS - (time.time() - _start) - 45
epochs = min(300, max(10, int(remaining / 3)))
print(f"Training up to {epochs} epochs ({remaining:.0f}s remaining)", flush=True)

from ultralytics import YOLO

start_model = str(PREV_WEIGHTS) if PREV_WEIGHTS.exists() else "yolo11n-seg.pt"
print(f"Starting from: {start_model}", flush=True)

yolo = YOLO(start_model)
results = yolo.train(
    data=str(DATASET_DIR / "data.yaml"),
    epochs=epochs,
    imgsz=640,
    batch=4,
    device="mps",
    project=str(OUT_DIR / "runs"),
    name="student",
    patience=100,
    val=False,
    plots=False,
    time=remaining / 3600,
    lr0=0.0005,
    lrf=0.01,
    retina_masks=True,   # full-resolution instance masks — fixes rectangular artefacts
)
print("Training done", flush=True)

# ── evaluate: diameter MAE + count MAE ───────────────────────────────────────
print("=== Evaluating (retina_masks=True, conf sweep) ===", flush=True)

weights_path = OUT_DIR / "runs" / "student" / "weights" / "best.pt"
if not weights_path.exists():
    weights_path = OUT_DIR / "runs" / "student" / "weights" / "last.pt"

yolo_eval = YOLO(str(weights_path))

best_conf = 0.15
best_diam_mae = float("inf")
best_count_mae = float("inf")
best_student_diameters = {}
best_student_counts = {}

for conf in (0.08, 0.10, 0.15, 0.20, 0.25):
    preds = yolo_eval.predict(
        source=str(SAMPLES_DIR),
        imgsz=640,
        conf=conf,
        device="mps",
        max_det=1000,
        retina_masks=True,
        save=False,
        verbose=False,
    )

    student_diameters = {}
    student_counts = {}
    for r in preds:
        img_name = Path(r.path).name
        diams = []
        if r.masks is not None:
            masks_np = r.masks.data.cpu().numpy()
            orig_h, orig_w = r.orig_shape
            from skimage.transform import resize as sk_resize
            for m in masks_np:
                if m.shape != (orig_h, orig_w):
                    m = sk_resize(m, (orig_h, orig_w), order=0, preserve_range=True)
                area_px = (m > 0.5).sum()
                if area_px > 0:
                    d_px = 2 * np.sqrt(area_px / np.pi)
                    diams.append(d_px * SCALE_UM_PX)
        student_diameters[img_name] = np.array(sorted(diams))
        student_counts[img_name] = len(diams)

    # diameter MAE: compare sorted diameter arrays, padded to same length
    diam_diffs = []
    count_diffs = []
    for img_name in sorted(teacher_counts):
        t_d = teacher_diameters.get(img_name, np.array([]))
        s_d = student_diameters.get(img_name, np.array([]))
        n = max(len(t_d), len(s_d), 1)
        t_pad = np.pad(t_d, (0, n - len(t_d)), constant_values=0)
        s_pad = np.pad(s_d, (0, n - len(s_d)), constant_values=0)
        diam_diffs.append(np.mean(np.abs(t_pad - s_pad)))
        count_diffs.append(abs(len(t_d) - len(s_d)))

    diam_mae = float(np.mean(diam_diffs))
    count_mae = float(np.mean(count_diffs))
    total_student = sum(student_counts.values())
    print(f"  conf={conf:.2f}: diameter_mae={diam_mae:.3f}um  count_mae={count_mae:.1f}  student_total={total_student}", flush=True)

    if diam_mae < best_diam_mae:
        best_diam_mae = diam_mae
        best_count_mae = count_mae
        best_conf = conf
        best_student_diameters = student_diameters
        best_student_counts = student_counts

print(f"\nBest conf={best_conf}: diameter_mae={best_diam_mae:.3f}um  count_mae={best_count_mae:.1f}", flush=True)

# per-image breakdown
for img_name in sorted(teacher_counts):
    t_n = teacher_counts[img_name]
    s_n = best_student_counts.get(img_name, 0)
    t_d = teacher_diameters.get(img_name, np.array([]))
    s_d = best_student_diameters.get(img_name, np.array([]))
    t_mean = t_d.mean() if len(t_d) else 0
    s_mean = s_d.mean() if len(s_d) else 0
    print(f"  {img_name}: count teacher={t_n} student={s_n}  mean_diam teacher={t_mean:.2f}um student={s_mean:.2f}um", flush=True)

all_student_diams = np.concatenate(list(best_student_diameters.values())) if best_student_diameters else np.array([])
count_pct = float(np.mean([abs(teacher_counts[k] - best_student_counts.get(k,0)) / max(teacher_counts[k],1)
                            for k in teacher_counts])) * 100
epochs_run = results.epoch if hasattr(results, "epoch") else epochs

print(f"\ndiameter_mae_um={best_diam_mae:.3f}", flush=True)
print(f"count_mae={best_count_mae:.2f}  count_pct={count_pct:.1f}%", flush=True)
print(f"student_mean_diam={all_student_diams.mean():.2f}um  teacher_mean_diam={all_teacher_diams.mean():.2f}um", flush=True)
print(f"epochs_run={epochs_run}", flush=True)
print(f"Total elapsed: {time.time() - _start:.1f}s", flush=True)

metrics = {
    "diameter_mae_um": round(best_diam_mae, 4),
    "count_mae": round(best_count_mae, 2),
    "count_pct_diff": round(count_pct, 1),
    "student_mean_diam_um": round(float(all_student_diams.mean()), 3) if len(all_student_diams) else 0,
    "teacher_mean_diam_um": round(float(all_teacher_diams.mean()), 3),
    "best_conf_threshold": best_conf,
    "max_det": 1000,
    "retina_masks": True,
    "epochs_run": epochs_run,
}
Path("metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Saved metrics.json: {metrics}", flush=True)

CKPT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(weights_path, CKPT_DIR / "best_model.pt")
print("Checkpoint saved", flush=True)
