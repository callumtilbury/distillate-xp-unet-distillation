"""
Run 10: Warm-start from run 9; lower LR for fine-tuning; wider conf sweep to 0.50.
Targeting final <2% diameter MAPE gap.
"""
import json
import shutil
import sys
import time
from pathlib import Path
import tempfile

import numpy as np

REPO_DIR = Path("/Users/crt25/code/bubcount-distill")
DET_DIR = Path(".distillate/pseudolabels_det_v2")
OUT_DIR = Path("/tmp/bubcount_distill_run10")
CKPT_DIR = Path(".distillate/checkpoints")
PREV_WEIGHTS = Path("/tmp/bubcount_distill_run9/runs/student_det/weights/last.pt")

SCALE_UM_PX = 0.0825

sys.path.insert(0, str(REPO_DIR))


def read_train_budget():
    d = json.loads(Path(".distillate/budget.json").read_text())
    return max(60, d.get("train_budget_seconds", 600) - 60)


MAX_SECONDS = read_train_budget()
_start = time.time()
print(f"Budget: {MAX_SECONDS}s", flush=True)

from skimage.measure import regionprops

teacher_diameters: dict[str, np.ndarray] = {}

for npy in sorted(Path(".distillate/pseudolabels/masks").glob("*.npy")):
    mask = np.load(str(npy))
    diams = [2 * np.sqrt(r.area / np.pi) * SCALE_UM_PX for r in regionprops(mask)]
    teacher_diameters[npy.stem + ".tif"] = np.array(sorted(diams))

for npy in sorted(Path(".distillate/pseudolabels_nocath/masks").glob("*.npy")):
    mask = np.load(str(npy))
    diams = [2 * np.sqrt(r.area / np.pi) * SCALE_UM_PX for r in regionprops(mask)]
    teacher_diameters[npy.stem] = np.array(sorted(diams))

all_t = np.concatenate(list(teacher_diameters.values()))
stem_to_teacher = {}
for k in teacher_diameters:
    stem = k[:-4] if k.endswith(".tif") else k
    stem_to_teacher[stem] = k

print(f"Teacher: {len(teacher_diameters)} imgs, {len(all_t)} bubbles, "
      f"mean_diam={all_t.mean():.3f}um", flush=True)

print("=== Training (warm start from run 9, lower LR) ===", flush=True)

remaining = MAX_SECONDS - (time.time() - _start) - 45
epochs = min(400, max(10, int(remaining / 3)))
print(f"Up to {epochs} epochs, {remaining:.0f}s remaining", flush=True)

from ultralytics import YOLO

start_model = str(PREV_WEIGHTS) if PREV_WEIGHTS.exists() else "yolo11n.pt"
print(f"Starting from: {start_model}", flush=True)

yolo = YOLO(start_model)
results = yolo.train(
    data=str(DET_DIR / "data.yaml"),
    epochs=epochs,
    imgsz=640,
    batch=4,
    device="mps",
    project=str(OUT_DIR / "runs"),
    name="student_det",
    patience=100,
    val=False,
    plots=False,
    time=remaining / 3600,
    lr0=0.0005,   # lower LR for fine-tuning
    lrf=0.01,
)
print("Training done", flush=True)

print("=== Evaluating (blended formula, wider conf sweep) ===", flush=True)

weights_path = OUT_DIR / "runs" / "student_det" / "weights" / "best.pt"
if not weights_path.exists():
    weights_path = OUT_DIR / "runs" / "student_det" / "weights" / "last.pt"

yolo_eval = YOLO(str(weights_path))

tmp_dir = Path(tempfile.mkdtemp())
for split in ("train", "val"):
    for png in (DET_DIR / "images" / split).glob("*.png"):
        shutil.copy2(png, tmp_dir / png.name)

best_conf, best_mae, best_mape = 0.20, float("inf"), float("inf")
best_sc: dict[str, np.ndarray] = {}

for conf in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
    preds = yolo_eval.predict(
        source=str(tmp_dir), imgsz=640, conf=conf,
        max_det=1000, save=False, verbose=False,
    )
    sc: dict[str, np.ndarray] = {}
    for r in preds:
        stem = Path(r.path).stem
        teacher_key = stem_to_teacher.get(stem)
        if teacher_key is None:
            continue
        diams = []
        if r.boxes is not None and len(r.boxes):
            for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy():
                w = x2 - x1
                h = y2 - y1
                d_px = (min(w, h) + np.sqrt(w * h)) / 2
                diams.append(d_px * SCALE_UM_PX)
        sc[teacher_key] = np.array(sorted(diams))

    maes, mapes = [], []
    for k, t_d in teacher_diameters.items():
        if k not in sc:
            continue
        s_d = sc[k]
        n = max(len(t_d), len(s_d), 1)
        t_p = np.pad(t_d, (0, n - len(t_d)))
        s_p = np.pad(s_d, (0, n - len(s_d)))
        maes.append(np.mean(np.abs(t_p - s_p)))
        t_mean = t_d.mean()
        s_mean = s_d.mean() if len(s_d) else 0.0
        mapes.append(abs(t_mean - s_mean) / t_mean * 100)

    diam_mae = float(np.mean(maes)) if maes else 999.0
    mape = float(np.mean(mapes)) if mapes else 999.0
    total = sum(len(v) for v in sc.values())
    print(f"  conf={conf:.2f}: mae={diam_mae:.4f}um  mape={mape:.2f}%  total={total}", flush=True)

    if diam_mae < best_mae:
        best_mae, best_mape, best_conf, best_sc = diam_mae, mape, conf, sc

shutil.rmtree(tmp_dir)

print(f"\nBest conf={best_conf}: diameter_mae={best_mae:.4f}um  diameter_mape={best_mape:.2f}%", flush=True)

count_diffs = []
for k, t_d in sorted(teacher_diameters.items()):
    if k not in best_sc:
        continue
    s_d = best_sc[k]
    t_mean = t_d.mean() if len(t_d) else 0
    s_mean = s_d.mean() if len(s_d) else 0
    count_diffs.append(abs(len(t_d) - len(s_d)))
    print(f"  {k}: t={len(t_d)} {t_mean:.2f}um  s={len(s_d)} {s_mean:.2f}um", flush=True)

all_s = np.concatenate([v for v in best_sc.values() if len(v)]) if best_sc else np.array([0.0])
count_mae = float(np.mean(count_diffs)) if count_diffs else 999.0
count_pct = float(np.mean([
    cd / max(len(teacher_diameters[k]), 1)
    for cd, k in zip(count_diffs, sorted(teacher_diameters))
    if k in best_sc
])) * 100
epochs_run = results.epoch if hasattr(results, "epoch") else epochs

print(f"\ndiameter_mae_um={best_mae:.4f}  diameter_mape={best_mape:.2f}%", flush=True)
print(f"count_mae={count_mae:.1f}  count_pct={count_pct:.1f}%", flush=True)
print(f"student_mean_diam={all_s.mean():.3f}um  teacher_mean_diam={all_t.mean():.3f}um", flush=True)
print(f"epochs_run={epochs_run}  elapsed={time.time() - _start:.1f}s", flush=True)

metrics = {
    "diameter_mae_um": round(best_mae, 4),
    "diameter_mape": round(best_mape, 2),
    "count_mae": round(count_mae, 1),
    "count_pct_diff": round(count_pct, 1),
    "student_mean_diam_um": round(float(all_s.mean()), 4) if len(all_s) else 0,
    "teacher_mean_diam_um": round(float(all_t.mean()), 4),
    "best_conf_threshold": best_conf,
    "radius_formula": "(min(w,h) + sqrt(w*h)) / 2",
    "epochs_run": epochs_run,
    "n_train_images": 24,
}
Path("metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Saved metrics.json: {metrics}", flush=True)

CKPT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(weights_path, CKPT_DIR / "best_model.pt")
print("Checkpoint saved", flush=True)
