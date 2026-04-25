"""
Run 4: Warm-start from run 3 checkpoint; more epochs; sweep conf threshold to minimise MAE.
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
OUT_DIR = Path("/tmp/bubcount_distill_run4")
CKPT_DIR = Path(".distillate/checkpoints")
PREV_WEIGHTS = Path("/tmp/bubcount_distill_run3/runs/student/weights/last.pt")

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

# ── teacher counts ────────────────────────────────────────────────────────────
teacher_counts = {}
for split in ("train", "val"):
    label_dir = DATASET_DIR / "labels" / split
    for lbl_file in sorted(label_dir.glob("*.txt")):
        count = sum(1 for line in lbl_file.read_text().splitlines() if line.strip())
        teacher_counts[lbl_file.stem + ".tif"] = count
print(f"Teacher: {len(teacher_counts)} images, {sum(teacher_counts.values())} instances", flush=True)

# ── train ────────────────────────────────────────────────────────────────────
print("=== Training (warm start from run 3) ===", flush=True)

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
    lr0=0.001,    # lower LR for fine-tuning from warm start
    lrf=0.01,
)
print("Training done", flush=True)

# ── evaluate with conf sweep ─────────────────────────────────────────────────
print("=== Evaluating (conf sweep, max_det=1000) ===", flush=True)

weights_path = OUT_DIR / "runs" / "student" / "weights" / "best.pt"
if not weights_path.exists():
    weights_path = OUT_DIR / "runs" / "student" / "weights" / "last.pt"

yolo_eval = YOLO(str(weights_path))

best_conf = 0.05
best_mae = float("inf")
best_student_counts = {}

for conf in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
    preds = yolo_eval.predict(
        source=str(SAMPLES_DIR),
        imgsz=640,
        conf=conf,
        device="mps",
        max_det=1000,
        save=False,
        verbose=False,
    )
    sc = {Path(r.path).name: (len(r.masks.data) if r.masks is not None else 0) for r in preds}
    diffs = [abs(teacher_counts.get(k, 0) - sc.get(k, 0)) for k in teacher_counts]
    mae = float(np.mean(diffs))
    total = sum(sc.values())
    print(f"  conf={conf:.2f}: mae={mae:.1f}  student_total={total}", flush=True)
    if mae < best_mae:
        best_mae = mae
        best_conf = conf
        best_student_counts = sc

print(f"\nBest conf={best_conf}: count_mae={best_mae:.2f}", flush=True)

abs_diffs = []
teacher_vals = []
for img_name in sorted(teacher_counts):
    teacher_n = teacher_counts[img_name]
    student_n = best_student_counts.get(img_name, 0)
    abs_diffs.append(abs(teacher_n - student_n))
    teacher_vals.append(teacher_n)
    print(f"  {img_name}: teacher={teacher_n}, student={student_n}, diff={abs(teacher_n-student_n)}", flush=True)

count_pct_diff = float(np.mean([d / max(t, 1) for d, t in zip(abs_diffs, teacher_vals)])) * 100
epochs_run = results.epoch if hasattr(results, "epoch") else epochs

print(f"\ncount_mae={best_mae:.2f}", flush=True)
print(f"count_pct_diff={count_pct_diff:.1f}%", flush=True)
print(f"epochs_run={epochs_run}", flush=True)
print(f"student_total={sum(best_student_counts.values())}  teacher_total={sum(teacher_counts.values())}", flush=True)
print(f"Total elapsed: {time.time() - _start:.1f}s", flush=True)

metrics = {
    "count_mae": round(best_mae, 3),
    "count_pct_diff": round(count_pct_diff, 1),
    "n_images": len(abs_diffs),
    "epochs_run": epochs_run,
    "best_conf_threshold": best_conf,
    "max_det": 1000,
    "student_total": sum(best_student_counts.values()),
    "teacher_total": sum(teacher_counts.values()),
}
Path("metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Saved metrics.json: {metrics}", flush=True)

CKPT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(weights_path, CKPT_DIR / "best_model.pt")
print("Checkpoint saved", flush=True)
