"""
Run 1 baseline: pseudolabel with Cellpose-SAM-FT teacher -> train YOLO11n-seg student.
Metric: mean absolute bubble count difference (student vs teacher).
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
REPO_DIR = Path("/Users/crt25/code/bubcount-distill")
SAMPLES_DIR = REPO_DIR / "samples"
TEACHER_MODEL = str(REPO_DIR / "models" / "bubble_finetuned.pth")
OUT_DIR = Path("/tmp/bubcount_distill_run1")
CKPT_DIR = Path(".distillate/checkpoints")

sys.path.insert(0, str(REPO_DIR))

# ── budget ───────────────────────────────────────────────────────────────────
def read_train_budget():
    import json
    budget_file = Path(".distillate/budget.json")
    if budget_file.exists():
        data = json.loads(budget_file.read_text())
        return max(60, data.get("train_budget_seconds", 600) - 300)
    return 300

MAX_SECONDS = read_train_budget()
_start = time.time()

print(f"Budget: {MAX_SECONDS}s", flush=True)

# ── step 1: teacher pseudolabels ─────────────────────────────────────────────
print("=== Generating teacher pseudolabels ===", flush=True)

from bubcount.params import AnalysisParameters
from bubcount.analyzer import BubbleAnalyzer
from bubcount.distill.pseudolabel import PseudolabelConfig, generate_pseudolabels

params = AnalysisParameters(
    pretrained_model=TEACHER_MODEL,
    gpu=True,
)
analyzer = BubbleAnalyzer(params=params)

cfg = PseudolabelConfig(
    source_dir=SAMPLES_DIR,
    output_dir=OUT_DIR / "dataset",
    val_fraction=0.2,
    seed=42,
)

summary = generate_pseudolabels(cfg, analyzer=analyzer)
print(f"Pseudolabels: {summary['train_images']} train, {summary['val_images']} val, "
      f"{summary['total_instances']} total instances", flush=True)

# record teacher counts per image
teacher_counts = {}
for img_name, split, n_instances in summary["instances_per_image"]:
    teacher_counts[img_name] = n_instances

data_yaml = summary["data_yaml"]

elapsed = time.time() - _start
print(f"Pseudolabeling done in {elapsed:.1f}s", flush=True)

if elapsed > MAX_SECONDS * 0.6:
    print("WARNING: pseudolabeling used >60% of budget, training may be brief", flush=True)

# ── step 2: train YOLO11n-seg student ────────────────────────────────────────
print("=== Training YOLO11n-seg student ===", flush=True)

remaining = MAX_SECONDS - (time.time() - _start) - 60  # 60s reserve for eval
epochs = max(1, int(remaining / 8))  # rough: ~8s per epoch on MPS
epochs = min(epochs, 50)
print(f"Will train for up to {epochs} epochs ({remaining:.0f}s remaining)", flush=True)

from bubcount.distill.train import train_yolo_seg

try:
    from ultralytics import YOLO
    yolo = YOLO("yolo11n-seg.pt")
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=4,
        device="mps",
        project=str(OUT_DIR / "runs"),
        name="student",
        patience=20,
        val=False,   # MPS shape-mismatch bug in validator; eval manually
        plots=False,
    )
    device = "mps"
    print(f"Training done on {device}", flush=True)
except Exception as e:
    print(f"ERROR during training: {e}", flush=True)
    raise

# ── step 3: evaluate student vs teacher ──────────────────────────────────────
print("=== Evaluating student vs teacher ===", flush=True)

from skimage import io
from bubcount.distill.train import predict_yolo_seg

# find best weights
weights_path = OUT_DIR / "runs" / "student" / "weights" / "best.pt"
if not weights_path.exists():
    weights_path = OUT_DIR / "runs" / "student" / "weights" / "last.pt"

student_counts = {}
all_images = list(SAMPLES_DIR.glob("*.tif")) + list(SAMPLES_DIR.glob("*.png"))

_, preds, _ = predict_yolo_seg(
    weights=weights_path,
    source=SAMPLES_DIR,
    imgsz=640,
    conf=0.25,
    device="mps",
)

for r in preds:
    img_name = Path(r.path).name
    n_masks = len(r.masks.data) if r.masks is not None else 0
    student_counts[img_name] = n_masks

# compute MAE
abs_diffs = []
for img_name, teacher_n in teacher_counts.items():
    student_n = student_counts.get(img_name, 0)
    abs_diffs.append(abs(teacher_n - student_n))
    print(f"  {img_name}: teacher={teacher_n}, student={student_n}, diff={abs(teacher_n-student_n)}", flush=True)

count_mae = float(np.mean(abs_diffs)) if abs_diffs else float("nan")
count_pct_diff = float(np.mean([d / max(t, 1) for d, t in zip(abs_diffs, teacher_counts.values())])) * 100

print(f"\ncount_mae={count_mae:.2f}", flush=True)
print(f"count_pct_diff={count_pct_diff:.1f}%", flush=True)
print(f"Total elapsed: {time.time() - _start:.1f}s", flush=True)

# ── step 4: save metrics and checkpoint ──────────────────────────────────────
metrics = {
    "count_mae": round(count_mae, 3),
    "count_pct_diff": round(count_pct_diff, 1),
    "n_images": len(abs_diffs),
    "train_images": summary["train_images"],
    "val_images": summary["val_images"],
    "epochs_run": results.epoch if hasattr(results, "epoch") else epochs,
}
Path("metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Saved metrics.json: {metrics}", flush=True)

# save checkpoint
CKPT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(weights_path, CKPT_DIR / "best_model.pt")
print(f"Checkpoint saved to {CKPT_DIR}/best_model.pt", flush=True)
