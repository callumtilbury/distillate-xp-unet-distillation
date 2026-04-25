"""
Run 6: YOLO detection (bbox only) + circle post-processing.
Pipeline:
  Teacher masks (cached .npy) -> regionprops -> equivalent circle diameter
  YOLO bbox prediction         -> sqrt(w*h)/2  -> equivalent circle diameter
Metric: diameter_mae_um between both distributions.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_DIR = Path("/Users/crt25/code/bubcount-distill")
SAMPLES_DIR = REPO_DIR / "samples"
SEG_DATASET_DIR = Path(".distillate/pseudolabels")       # existing seg labels + cached masks
DET_DATASET_DIR = Path(".distillate/pseudolabels_det")   # new detection labels
OUT_DIR = Path("/tmp/bubcount_distill_run6")
CKPT_DIR = Path(".distillate/checkpoints")

SCALE_UM_PX = 0.0825

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

# ── step 1: generate YOLO detection labels from cached Cellpose masks ─────────
print("=== Building detection dataset from cached masks ===", flush=True)

from skimage import io
from skimage.measure import regionprops

def masks_to_det_labels(mask_path: Path, img_path: Path) -> tuple[np.ndarray, list[float]]:
    """Return YOLO det label lines + list of equivalent diameters in um."""
    mask = np.load(str(mask_path))
    img = io.imread(str(img_path))
    h, w = img.shape[:2]
    lines = []
    diams = []
    for r in regionprops(mask):
        min_row, min_col, max_row, max_col = r.bbox
        bw = (max_col - min_col) / w
        bh = (max_row - min_row) / h
        cx = (min_col + max_col) / 2 / w
        cy = (min_row + max_row) / 2 / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        d_px = 2 * np.sqrt(r.area / np.pi)
        diams.append(d_px * SCALE_UM_PX)
    return lines, diams

# read the existing train/val split from the seg dataset
import yaml
seg_yaml = yaml.safe_load((SEG_DATASET_DIR / "data.yaml").read_text())

# figure out which images are train vs val
train_imgs = set(p.stem for p in (SEG_DATASET_DIR / "images" / "train").glob("*.png"))
val_imgs   = set(p.stem for p in (SEG_DATASET_DIR / "images" / "val").glob("*.png"))

for split in ("train", "val"):
    (DET_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (DET_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

teacher_diameters = {}   # img_name.tif -> sorted array of diameters

for npy_file in sorted((SEG_DATASET_DIR / "masks").glob("*.npy")):
    stem = npy_file.stem          # e.g. "frame-1"
    tif_name = stem + ".tif"

    # original image in seg dataset
    seg_img = SEG_DATASET_DIR / "images" / (
        "train" if stem in train_imgs else "val"
    ) / f"{stem}.png"

    lines, diams = masks_to_det_labels(npy_file, seg_img)
    teacher_diameters[tif_name] = np.array(sorted(diams))

    split = "train" if stem in train_imgs else "val"

    # symlink / copy image
    dst_img = DET_DATASET_DIR / "images" / split / f"{stem}.png"
    if not dst_img.exists():
        shutil.copy2(seg_img, dst_img)

    # write label
    lbl = DET_DATASET_DIR / "labels" / split / f"{stem}.txt"
    lbl.write_text("\n".join(lines))

# write data.yaml
(DET_DATASET_DIR / "data.yaml").write_text(
    f"path: {DET_DATASET_DIR.resolve()}\n"
    f"train: images/train\n"
    f"val: images/val\n"
    f"names:\n  0: bubble\n"
)

all_teacher_d = np.concatenate(list(teacher_diameters.values()))
print(f"Teacher: {len(teacher_diameters)} images, "
      f"{sum(len(v) for v in teacher_diameters.values())} bubbles, "
      f"mean_diam={all_teacher_d.mean():.3f}um std={all_teacher_d.std():.3f}um", flush=True)
print(f"Det labels ready in {time.time()-_start:.1f}s", flush=True)

# ── step 2: train YOLO detection model ───────────────────────────────────────
print("=== Training YOLO11n detection ===", flush=True)

remaining = MAX_SECONDS - (time.time() - _start) - 45
epochs = min(300, max(10, int(remaining / 2)))
print(f"Training up to {epochs} epochs, {remaining:.0f}s remaining", flush=True)

from ultralytics import YOLO

yolo = YOLO("yolo11n.pt")   # detection model, ~2.6M params
results = yolo.train(
    data=str(DET_DATASET_DIR / "data.yaml"),
    epochs=epochs,
    imgsz=640,
    batch=4,
    device="mps",
    project=str(OUT_DIR / "runs"),
    name="student_det",
    patience=80,
    val=False,
    plots=False,
    time=remaining / 3600,
    lr0=0.01,
    lrf=0.01,
)
print("Training done", flush=True)

# ── step 3: evaluate ─────────────────────────────────────────────────────────
print("=== Evaluating (bbox -> circle) ===", flush=True)

weights_path = OUT_DIR / "runs" / "student_det" / "weights" / "best.pt"
if not weights_path.exists():
    weights_path = OUT_DIR / "runs" / "student_det" / "weights" / "last.pt"

yolo_eval = YOLO(str(weights_path))

def bbox_to_diameter(box_xyxy, orig_shape) -> float:
    """Equivalent circle diameter from bounding box in um."""
    x1, y1, x2, y2 = box_xyxy
    w_px = float(x2 - x1)
    h_px = float(y2 - y1)
    # geometric mean of bbox sides -> equivalent circle diameter
    d_px = np.sqrt(w_px * h_px)
    return d_px * SCALE_UM_PX

best_conf, best_diam_mae, best_count_mae = 0.25, float("inf"), float("inf")
best_student_diameters = {}

for conf in (0.10, 0.15, 0.20, 0.25, 0.35, 0.50):
    preds = yolo_eval.predict(
        source=str(SAMPLES_DIR),
        imgsz=640,
        conf=conf,
        device="mps",
        max_det=1000,
        save=False,
        verbose=False,
    )
    student_diameters = {}
    for r in preds:
        img_name = Path(r.path).name
        diams = []
        if r.boxes is not None and len(r.boxes):
            for box in r.boxes.xyxy.cpu().numpy():
                diams.append(bbox_to_diameter(box, r.orig_shape))
        student_diameters[img_name] = np.array(sorted(diams))

    diam_diffs, count_diffs = [], []
    for img_name in sorted(teacher_diameters):
        t_d = teacher_diameters[img_name]
        s_d = student_diameters.get(img_name, np.array([]))
        n = max(len(t_d), len(s_d), 1)
        t_pad = np.pad(t_d, (0, n - len(t_d)))
        s_pad = np.pad(s_d, (0, n - len(s_d)))
        diam_diffs.append(np.mean(np.abs(t_pad - s_pad)))
        count_diffs.append(abs(len(t_d) - len(s_d)))

    diam_mae = float(np.mean(diam_diffs))
    count_mae = float(np.mean(count_diffs))
    print(f"  conf={conf:.2f}: diameter_mae={diam_mae:.4f}um  count_mae={count_mae:.1f}  "
          f"student_total={sum(len(v) for v in student_diameters.values())}", flush=True)

    if diam_mae < best_diam_mae:
        best_diam_mae = diam_mae
        best_count_mae = count_mae
        best_conf = conf
        best_student_diameters = student_diameters

print(f"\nBest conf={best_conf}: diameter_mae={best_diam_mae:.4f}um  count_mae={best_count_mae:.1f}", flush=True)

all_student_d = np.concatenate(list(best_student_diameters.values())) if best_student_diameters else np.array([])
count_pct = float(np.mean([
    abs(len(teacher_diameters[k]) - len(best_student_diameters.get(k, np.array([])))) / max(len(teacher_diameters[k]), 1)
    for k in teacher_diameters
])) * 100
epochs_run = results.epoch if hasattr(results, "epoch") else epochs

print("\nPer-image breakdown:", flush=True)
for img_name in sorted(teacher_diameters):
    t_d = teacher_diameters[img_name]
    s_d = best_student_diameters.get(img_name, np.array([]))
    print(f"  {img_name}: count teacher={len(t_d)} student={len(s_d)}  "
          f"mean_diam teacher={t_d.mean():.3f}um student={s_d.mean():.3f}um", flush=True)

print(f"\ndiameter_mae_um={best_diam_mae:.4f}", flush=True)
print(f"count_mae={best_count_mae:.2f}  count_pct={count_pct:.1f}%", flush=True)
print(f"student_mean_diam={all_student_d.mean():.3f}um  teacher_mean_diam={all_teacher_d.mean():.3f}um", flush=True)
print(f"epochs_run={epochs_run}  total_elapsed={time.time()-_start:.1f}s", flush=True)

metrics = {
    "diameter_mae_um": round(best_diam_mae, 4),
    "count_mae": round(best_count_mae, 2),
    "count_pct_diff": round(count_pct, 1),
    "student_mean_diam_um": round(float(all_student_d.mean()), 4) if len(all_student_d) else 0,
    "teacher_mean_diam_um": round(float(all_teacher_d.mean()), 4),
    "best_conf_threshold": best_conf,
    "max_det": 1000,
    "model": "yolo11n-det",
    "epochs_run": epochs_run,
}
Path("metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Saved metrics.json: {metrics}", flush=True)

CKPT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(weights_path, CKPT_DIR / "best_model.pt")
print("Checkpoint saved", flush=True)
