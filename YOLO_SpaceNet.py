"""
YOLO26 on satellite tiles (chapter 4 of the learning path): train, read the curves, measure, find buildings
and their areas, and export the model for the web page.

It uses the data made by YOLO_SpaceNet_data.py (data/yolo/seg and data/yolo/sem) and Ultralytics YOLO.

Run:  python YOLO_SpaceNet.py train seg                 buildings and sports fields (instance segmentation)
      python YOLO_SpaceNet.py train sem                 land cover (semantic segmentation)
      python YOLO_SpaceNet.py train seg --scratch       the same from random weights, to see what pretraining is worth
      python YOLO_SpaceNet.py curves seg                the curves of results.csv and signs of overfitting
      python YOLO_SpaceNet.py val seg                   AP per class, confusion matrix
      python YOLO_SpaceNet.py predict data/yolo/seg/images/val/Paris_img637.jpg    buildings, m², built-up ratio
      python YOLO_SpaceNet.py export seg                ONNX for a web page (float16 weights)
Options: --epochs 100, --batch 4, --mpp 0.3 (metres per pixel of an image that is not a SpaceNet tile).

Needs: pip install ultralytics onnx onnxslim   (on Google Colab, choose a GPU runtime first: a T4 trains a
model in about an hour; on a CPU it takes days)
Licences: Ultralytics YOLO, AGPL-3.0 (models trained with it are shared under AGPL-3.0 too);
SpaceNet CC BY-SA 4.0; OpenStreetMap ODbL, (c) OpenStreetMap contributors.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


#--------------- Hyperparameters ---------------#
EPOCHS = 100
IMGSZ = 640            # the network's input: 640 × 640 (tiles of 650 × 650 are letterboxed)
BATCH = 4              # fits a 6 GB card; a Colab T4 (16 GB) takes 8
SEED = 0
CONF = 0.25            # confidence threshold for predictions
#-----------------------------------------------#

DATA = Path("data/yolo")
RUNS = Path("runs/yolo")
NAMES = {"seg": ["building", "sports_field"],
         "sem": ["background", "building", "road", "green", "water", "parking"]}


def run_dir(task: str, scratch: bool = False) -> Path:
    return RUNS / f"{task}-{'scratch' if scratch else 'pretrained'}"


def best_weights(task: str) -> Path:
    weights = run_dir(task) / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"{weights} is missing: first run  python YOLO_SpaceNet.py train {task}")
    return weights


# ===========================================================================================
# Training
# ===========================================================================================

def train(task: str, epochs: int, batch: int, scratch: bool) -> None:
    """Fine-tunes YOLO26n (or trains it from random weights) on the tiles."""
    data = DATA / task / "data.yaml"
    if not data.exists():
        raise SystemExit(f"{data} is missing: first run  python YOLO_SpaceNet_data.py")
    # yolo26n-seg.pt: weights trained on COCO, downloaded once by Ultralytics; the .yaml is the same network, random
    model = YOLO(f"yolo26n-{task}.yaml" if scratch else f"yolo26n-{task}.pt")
    start = time.time()
    model.train(
        data=str(data.resolve()),
        epochs=epochs,
        imgsz=IMGSZ,
        batch=batch,
        seed=SEED,
        deterministic=True,
        project=str(RUNS.resolve()),
        name=run_dir(task, scratch).name,
        exist_ok=True,
        flipud=0.5,        # a view from above has no "up": flip it upside down as often as left-right (fliplr=0.5)
        degrees=90,        # and turn it by up to ±90°
        save_period=10,    # keep the weights of every 10th epoch, to compare later
        workers=2,
    )
    print(f"training took {(time.time() - start) / 60:.0f} minutes; weights in {run_dir(task, scratch) / 'weights'}")


# ===========================================================================================
# Reading the curves
# ===========================================================================================

def curves(task: str, scratch: bool = False) -> None:
    """The losses and scores of results.csv at a few epochs, and where the validation loss was lowest."""
    rows = list(csv.DictReader((run_dir(task, scratch) / "results.csv").open()))
    rows = [{k.strip(): float(v) for k, v in r.items()} for r in rows]
    if task == "seg":
        columns = ["train/box_loss", "val/box_loss", "train/seg_loss", "val/seg_loss", "metrics/mAP50(M)", "metrics/mAP50-95(M)"]
        val_loss = "val/seg_loss"
    else:
        columns = ["train/ce_loss", "val/ce_loss", "metrics/mIoU", "metrics/pixel_acc"]
        val_loss = "val/ce_loss"
    print("epoch  " + "  ".join(f"{c.replace('metrics/', '').replace('/', ' '):>14s}" for c in columns))
    n = len(rows)
    for e in sorted({1, 5, 10, 25, 50, 75, n} & set(range(1, n + 1))):
        print(f"{e:5d}  " + "  ".join(f"{rows[e - 1][c]:14.4f}" for c in columns))
    best = min(range(n), key=lambda i: rows[i][val_loss]) + 1
    print(f"lowest {val_loss}: epoch {best}" + (
        f"; it rose again later (from {rows[best - 1][val_loss]:.4f} to {rows[-1][val_loss]:.4f}): a sign of overfitting"
        if best < n and rows[-1][val_loss] > 1.05 * rows[best - 1][val_loss] else "; no clear sign of overfitting"))
    print(f"plots by Ultralytics: {run_dir(task, scratch) / 'results.png'}")


# ===========================================================================================
# Measuring
# ===========================================================================================

def val(task: str) -> None:
    """The validation scores of the best weights, class by class."""
    model = YOLO(str(best_weights(task)))
    metrics = model.val(data=str((DATA / task / "data.yaml").resolve()), imgsz=IMGSZ, batch=BATCH, plots=True,
                        project=str(RUNS.resolve()), name=f"{task}-val", exist_ok=True)
    if task == "seg":
        for k, name in enumerate(NAMES[task]):
            print(f"{name:14s} mask AP50 {metrics.seg.ap50[k]:.3f}   AP50-95 {metrics.seg.ap[k]:.3f}   box AP50 {metrics.box.ap50[k]:.3f}")
        print(f"{'mean':14s} mask mAP50 {metrics.seg.map50:.3f}   mAP50-95 {metrics.seg.map:.3f}")
    else:
        print(f"mIoU {metrics.results_dict['metrics/mIoU']:.3f}, pixel accuracy {metrics.results_dict['metrics/pixel_acc']:.3f}")
    print(f"confusion matrix and curves: {metrics.save_dir}")


# ===========================================================================================
# Buildings and their areas on a new image
# ===========================================================================================

def metres_per_pixel(image: Path, default: float) -> tuple[float, float]:
    """A SpaceNet tile knows its ground size (data/yolo/tiles/<name>.json); any other image gets --mpp."""
    meta = DATA / "tiles" / f"{image.stem}.json"
    if not meta.exists():
        return default, default
    m = json.loads(meta.read_text())
    lat = (m["bounds"][1] + m["bounds"][3]) / 2
    return abs(m["transform"][0]) * 111_320 * np.cos(np.radians(lat)), abs(m["transform"][4]) * 111_320


def predict(image: Path, mpp: float) -> None:
    """Finds buildings and sports fields, and turns their masks into square metres."""
    model = YOLO(str(best_weights("seg")))
    result = model.predict(str(image), imgsz=IMGSZ, conf=CONF, retina_masks=True, verbose=False)[0]
    gx, gy = metres_per_pixel(image, mpp)
    h, w = result.orig_shape
    covered = np.zeros((h, w), bool)
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy().astype(bool)   # retina_masks: one mask per object at the image's size
        for k, (cls, score, mask) in enumerate(zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), masks)):
            print(f"{k + 1:3d}  {NAMES['seg'][int(cls)]:12s} score {score:.2f}  area {mask.sum() * gx * gy:8.1f} m²")
            if int(cls) == 0:
                covered |= mask
    built = covered.sum() * gx * gy
    print(f"one pixel = {gx:.3f} m × {gy:.3f} m; building footprint {built:.0f} m² of {w * h * gx * gy:.0f} m²: built-up ratio {covered.mean():.1%}")
    out = RUNS / "predict" / f"{image.stem}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), result.plot())
    print(f"drawn into {out}")


# ===========================================================================================
# Exporting for the web page
# ===========================================================================================

def float32_io(src: Path, dst: Path) -> None:
    """Wraps a float16 model with Cast nodes so it takes and returns float32: half the size, simpler JavaScript."""
    import onnx
    from onnx import TensorProto, helper

    m = onnx.load(str(src))
    g = m.graph
    for inp in g.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT16:
            inner = inp.name + "_fp16"
            for node in g.node:
                node.input[:] = [inner if x == inp.name else x for x in node.input]
            g.node.insert(0, helper.make_node("Cast", [inp.name], [inner], to=TensorProto.FLOAT16))
            inp.type.tensor_type.elem_type = TensorProto.FLOAT
    for out in g.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT16:
            inner = out.name + "_fp16"
            for node in g.node:
                node.output[:] = [inner if x == out.name else x for x in node.output]
            g.node.append(helper.make_node("Cast", [inner], [out.name], to=TensorProto.FLOAT))
            out.type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def export(task: str) -> None:
    """ONNX with the raw head (the page runs NMS itself), float16 weights on a GPU, float32 inputs and outputs."""
    import torch

    model = YOLO(str(best_weights(task)))
    half = torch.cuda.is_available()                     # float16 export needs a GPU; on a CPU the file stays float32
    path = Path(model.export(format="onnx", imgsz=IMGSZ, simplify=True, opset=17, half=half, **({"device": 0} if half else {})))
    web = path.with_name(f"{task}_web.onnx")
    if half:
        float32_io(path, web)
    else:
        path.replace(web)
    print(f"{web}: {web.stat().st_size / 1e6:.1f} MB; the chapter's page loads it with ONNX Runtime Web")


# ===========================================================================================

if __name__ == "__main__":   # on Windows the data loader's workers import this file again: keep the work in here
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "curves", "val", "predict", "export"])
    parser.add_argument("target", help="seg or sem; for predict, an image file")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--scratch", action="store_true", help="start from random weights instead of COCO's")
    parser.add_argument("--mpp", type=float, default=0.3, help="metres per pixel of an image that is not a SpaceNet tile")
    args = parser.parse_args()
    if args.command == "predict":
        predict(Path(args.target), args.mpp)
    elif args.target not in NAMES:
        raise SystemExit("the target is seg or sem")
    elif args.command == "train":
        train(args.target, args.epochs, args.batch, args.scratch)
    elif args.command == "curves":
        curves(args.target, args.scratch)
    elif args.command == "val":
        val(args.target)
    else:
        export(args.target)
