"""
Run 3: Warm-start from run 2 checkpoint; max_det=1000; more epochs on cached pseudolabels.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_DIR = Path("/Users/crt25/code/bubcount-distill")
SAMPLES_DIR = REPO_DIR / "samples"
DATASET_DIR = Path(".distillate/pseudolabels")   # cached from run 2
OUT_DIR = Path("/tmp/bubcount_distill_run3")
CKPT_DIR = Path(".distillate/checkpoints")
PREV_WEIGHTS = Path("/tmp/bubcount_distill_run2/runs/student/weights/last.pt")

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

# ── teacher counts from cached labels ────────────────────────────────────────
teacher_counts = {}
for split in ("train", "val"):
    label_dir = DATASET_DIR / "labels" / split
    for lbl_file in sorted(label_dir.glob("*.txt")):
        count = sum(1 for line in lbl_file.read_text().splitlines() if line.strip())
        teacher_counts[lbl_file.stem + ".tif"] = count
print(f"Loaded {len(teacher_counts)} cached label files, "
      f"{sum(teacher_counts.values())} total instances", flush=True)

# ── train from warm start ────────────────────────────────────────────────────
print("=== Training (warm start from run 2) ===", flush=True)

remaining = MAX_SECONDS - (time.time() - _start) - 30
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
    patience=80,
    val=False,       # MPS shape-mismatch bug
    plots=False,
    time=remaining / 3600,
)
print("Training done", flush=True)

# ── evaluate ─────────────────────────────────────────────────────────────────
print("=== Evaluating ===", flush=True)

weights_path = OUT_DIR / "runs" / "student" / "weights" / "best.pt"
if not weights_path.exists():
    weights_path = OUT_DIR / "runs" / "student" / "weights" / "last.pt"

from bubcount.distill.train import predict_yolo_seg

_, preds, _ = predict_yolo_seg(
    weights=weights_path,
    source=SAMPLES_DIR,
    imgsz=640,
    conf=0.05,
    device="mps",
)

student_counts = {}
for r in preds:
    img_name = Path(r.path).name
    # Patch: use max_det=1000 via nms override
    n_masks = len(r.masks.data) if r.masks is not None else 0
    student_counts[img_name] = n_masks

# re-predict with high max_det
from ultralytics import YOLO as YOLOeval
yolo_eval = YOLOeval(str(weights_path))
preds2 = yolo_eval.predict(
    source=str(SAMPLES_DIR),
    imgsz=640,
    conf=0.05,
    device="mps",
    max_det=1000,
    save=False,
    verbose=False,
)
student_counts = {}
for r in preds2:
    img_name = Path(r.path).name
    n_masks = len(r.masks.data) if r.masks is not None else 0
    student_counts[img_name] = n_masks

abs_diffs = []
teacher_vals = []
for img_name in sorted(teacher_counts):
    teacher_n = teacher_counts[img_name]
    student_n = student_counts.get(img_name, 0)
    abs_diffs.append(abs(teacher_n - student_n))
    teacher_vals.append(teacher_n)
    print(f"  {img_name}: teacher={teacher_n}, student={student_n}, diff={abs(teacher_n-student_n)}", flush=True)

count_mae = float(np.mean(abs_diffs)) if abs_diffs else float("nan")
count_pct_diff = float(np.mean([d / max(t, 1) for d, t in zip(abs_diffs, teacher_vals)])) * 100
epochs_run = results.epoch if hasattr(results, "epoch") else epochs

print(f"\ncount_mae={count_mae:.2f}", flush=True)
print(f"count_pct_diff={count_pct_diff:.1f}%", flush=True)
print(f"epochs_run={epochs_run}", flush=True)
print(f"student_total={sum(student_counts.values())}  teacher_total={sum(teacher_counts.values())}", flush=True)
print(f"Total elapsed: {time.time() - _start:.1f}s", flush=True)

metrics = {
    "count_mae": round(count_mae, 3),
    "count_pct_diff": round(count_pct_diff, 1),
    "n_images": len(abs_diffs),
    "epochs_run": epochs_run,
    "conf_threshold": 0.05,
    "max_det": 1000,
    "student_total": sum(student_counts.values()),
    "teacher_total": sum(teacher_counts.values()),
}
Path("metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Saved metrics.json", flush=True)

CKPT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(weights_path, CKPT_DIR / "best_model.pt")
print("Checkpoint saved", flush=True)
