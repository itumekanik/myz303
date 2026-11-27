"""
Exports what the YOLO chapter of the static site needs, as JavaScript data files a page loads with a
<script> tag (which also works when the page is opened from a file, where fetch() is not allowed).

    python web/tools/export_yolo_assets.py <source> [--fixture-only | --outputs-only]      (from the repository root)

<source> is the folder the chapter's Python files worked in (the repository root if you ran them there):
data/yolo/{seg,sem,tiles} from YOLO_SpaceNet_data.py, and runs/yolo/{seg-pretrained,seg-scratch,sem-pretrained}
from `YOLO_SpaceNet.py train`, with seg_web.onnx and sem_web.onnx from `YOLO_SpaceNet.py export` (float16
weights, float32 inputs and outputs). The training folder of the first version (data/seg, runs/seg-pretrained)
works too. Needs ultralytics, onnxruntime, opencv-python and numpy.

Writes web/public/data/:

    yolo-seg.js       the instance segmentation model (building, sports_field), gzip
    yolo-sem.js       the semantic segmentation model (background, building, road, green, water, parking), gzip
    yolo-samples.js   validation tiles (JPEG as they are), their true polygons and land-cover masks, metres per pixel
    yolo-run.js       the three training runs (results.csv curves, final scores per class) and one val tile
                      predicted after 1, 5, 10, 25, 50, 75 and 100 epochs (runs/*/weights/epoch*.pt)
    yolo-outputs.js   what the Outputs tab reads about runs/…/seg-pretrained: its files, results.csv, label statistics,
                      the first and last training batches, a val batch, the confusion matrix at thresholds 0.001 and
                      0.01 … 0.95 (rebuilt with Ultralytics' ConfusionMatrix) with example crops per cell, the P, R, F1
                      and PR curves, the land-cover model's confusion matrix, and the console lines of training if the
                      console was saved to <source>/train_seg.log (paths in it are made relative)

and web/tests/fixtures/yolo-parity.json: what Ultralytics and OpenCV give on small inputs (resize, letterbox,
NMS, masks, AP), which the unit tests compare with the TypeScript engine (src/engines/yolo/).

Each data file runs `(window.__AI_DATA__ ||= {})[name] = {...}`; binary arrays are base64 strings,
gzip-compressed where that helps ({"gzip": ...}).
SpaceNet imagery and buildings: CC BY-SA 4.0. OpenStreetMap labels: (c) OpenStreetMap contributors, ODbL.
"""
import base64
import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.metrics import bbox_iou, compute_ap
from ultralytics.utils.nms import non_max_suppression

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "public" / "data"
FIXTURES = ROOT / "web" / "tests" / "fixtures"
SEG_NAMES = ["building", "sports_field"]
SEM_NAMES = ["background", "building", "road", "green", "water", "parking"]
CITIES = ["Vegas", "Paris", "Shanghai", "Khartoum"]
PER_CITY = 2
CONF, IOU = 0.25, 0.7
M_PER_DEGREE = 111_320  # metres per degree of latitude (WGS84, mean)


def data_dir(source: Path) -> Path:
    """YOLO_SpaceNet_data.py writes data/yolo/{seg,sem,tiles}; the first spike wrote data/{seg,sem,tiles}."""
    return source / "data" / "yolo" if (source / "data" / "yolo" / "seg").exists() else source / "data"


def runs_dir(source: Path) -> Path:
    """YOLO_SpaceNet.py trains into runs/yolo/<task>-pretrained; the spike into runs/<task>-pretrained."""
    return source / "runs" / "yolo" if (source / "runs" / "yolo").exists() else source / "runs"


def b64(data) -> str:
    raw = data if isinstance(data, bytes) else np.ascontiguousarray(data).tobytes()
    return base64.b64encode(raw).decode("ascii")


def gz(data) -> dict:
    raw = data if isinstance(data, bytes) else np.ascontiguousarray(data).tobytes()
    return {"gzip": b64(gzip.compress(raw, 9, mtime=0))}


def write_data(name: str, value) -> None:
    text = f"(window.__AI_DATA__ ||= {{}})[{json.dumps(name)}] = {json.dumps(value, separators=(',', ':'))};\n"
    (OUT / f"{name}.js").write_text(text, encoding="utf-8", newline="\n")
    print(f"web/public/data/{name}.js  {len(text) / 1e6:.2f} MB")


def metres_per_pixel(meta: dict) -> tuple[float, float]:
    """Ground size of one pixel (east-west, north-south): the tiles are in degrees, so the two differ."""
    lat = (meta["bounds"][1] + meta["bounds"][3]) / 2
    return (abs(meta["transform"][0]) * M_PER_DEGREE * np.cos(np.radians(lat)), abs(meta["transform"][4]) * M_PER_DEGREE)


def pick_samples(source: Path) -> list[str]:
    """Per city the val tiles that show the most land-cover classes, with a sports field where there is one."""
    labels = data_dir(source) / "seg" / "labels" / "val"
    masks = data_dir(source) / "sem" / "masks" / "val"
    picked = []
    for city in CITIES:
        def score(p: Path) -> tuple:
            m = cv2.imread(str(masks / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
            lines = p.read_text().splitlines()
            classes = sum(1 for c in range(1, len(SEM_NAMES)) if (m == c).mean() > 0.005)
            return (any(line.startswith("1 ") for line in lines), classes, (m == 255).mean() == 0, len(lines), p.stem)

        ranked = sorted(labels.glob(f"{city}_*.txt"), key=score, reverse=True)
        picked += [p.stem for p in ranked[:PER_CITY]]
    return picked


def samples(source: Path, names: list[str]) -> list[dict]:
    out = []
    for name in names:
        city, img = name.split("_", 1)
        meta = json.loads((data_dir(source) / "tiles" / f"{name}.json").read_text())
        polygons = []
        for line in (data_dir(source) / "seg" / "labels" / "val" / f"{name}.txt").read_text().splitlines():
            v = line.split()
            polygons.append({"cls": int(v[0]), "xy": [round(float(x), 5) for x in v[1:]]})
        mask = cv2.imread(str(data_dir(source) / "sem" / "masks" / "val" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        mx, my = metres_per_pixel(meta)
        out.append({
            "name": name, "city": city, "tile": img, "width": meta["width"], "height": meta["height"],
            "metresPerPixel": [round(mx, 5), round(my, 5)],
            "jpeg": b64((data_dir(source) / "seg" / "images" / "val" / f"{name}.jpg").read_bytes()),
            "polygons": polygons,
            "landCover": gz(mask),  # uint8 class per pixel, 255 = no image data
        })
    return out


RUNS = {  # what the Training tab shows: run folder → the curves it keeps
    "seg-pretrained": ["train/box_loss", "train/seg_loss", "train/cls_loss", "val/box_loss", "val/seg_loss", "val/cls_loss",
                       "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)", "metrics/mAP50(B)"],
    "seg-scratch": ["train/box_loss", "train/seg_loss", "train/cls_loss", "val/box_loss", "val/seg_loss", "val/cls_loss",
                    "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)", "metrics/mAP50(B)"],
    "sem-pretrained": ["train/ce_loss", "train/dice_loss", "val/ce_loss", "val/dice_loss", "metrics/mIoU", "metrics/pixel_acc"],
}
SNAPSHOT_EPOCHS = [1, 5, 10, 25, 50, 75, 100]  # wished for; epochN.pt is saved after epoch N + 1, the last is last.pt
SNAPSHOT_TILE = "Shanghai_img209"


def snapshot_epochs(source: Path, last: int) -> list[int]:
    """The epochs whose weights every run kept: the wished ones if there, else up to 7 spread over what was saved
    (YOLO_SpaceNet.py keeps every 10th epoch: 1, 11, 21, …)."""
    saved = None
    for run in RUNS:
        found = {int(p.stem[5:]) + 1 for p in (runs_dir(source) / run / "weights").glob("epoch*.pt")} | {last}
        saved = found if saved is None else saved & found
    wished = [e for e in SNAPSHOT_EPOCHS if e in saved]
    if len(wished) >= 5:
        return wished
    have = sorted(saved)
    return [have[round(i)] for i in np.linspace(0, len(have) - 1, min(7, len(have)))]


def training_settings(source: Path) -> dict:
    """The settings the instance model was trained with, as Ultralytics wrote them (args.yaml)."""
    from ultralytics.utils import YAML

    args = YAML.load(runs_dir(source) / "seg-pretrained" / "args.yaml")
    return {k: args[k] for k in ("imgsz", "batch", "epochs", "flipud", "degrees", "seed")}


def run_record(source: Path) -> dict:
    """The three training runs: curves per epoch, final scores per class, and one val tile predicted along the way."""
    import csv

    from ultralytics import YOLO

    runs = {}
    for run, columns in RUNS.items():
        rows = list(csv.DictReader((runs_dir(source) / run / "results.csv").open()))
        runs[run] = {
            "epochs": len(rows),
            "seconds": round(float(rows[-1]["time"])),
            "curves": {c: [round(float(r[c]), 5) for r in rows] for c in columns},
        }
    # final scores per class, measured again on the val set with the best weights
    for run in ("seg-pretrained", "seg-scratch"):
        m = YOLO(str(runs_dir(source) / run / "weights" / "best.pt")).val(
            data=str(data_dir(source) / "seg" / "data.yaml"), imgsz=640, batch=4, plots=False, verbose=False)
        runs[run]["final"] = {
            "maskAP50": [round(float(v), 4) for v in m.seg.ap50], "maskAP50_95": [round(float(v), 4) for v in m.seg.ap],
            "boxAP50": [round(float(v), 4) for v in m.box.ap50], "maskMAP50": round(float(m.seg.map50), 4),
            "maskMAP50_95": round(float(m.seg.map), 4), "boxMAP50": round(float(m.box.map50), 4),
            "precision": round(float(m.seg.mp), 4), "recall": round(float(m.seg.mr), 4),
        }
    m = YOLO(str(runs_dir(source) / "sem-pretrained" / "weights" / "best.pt")).val(
        data=str(data_dir(source) / "sem" / "data.yaml"), imgsz=640, batch=4, plots=False, verbose=False)
    iou = getattr(m, "_per_class_iou", None)  # SemanticMetrics keeps the IoU of every class here
    runs["sem-pretrained"]["final"] = {
        "mIoU": round(float(m.results_dict["metrics/mIoU"]), 4), "pixelAccuracy": round(float(m.results_dict["metrics/pixel_acc"]), 4),
        "iou": [round(float(v), 4) for v in iou] if iou is not None else None,
    }

    # the same val tile after 1, 5, 10, … epochs
    tile = str(data_dir(source) / "seg" / "images" / "val" / f"{SNAPSHOT_TILE}.jpg")
    snapshots = {}
    epochs = snapshot_epochs(source, runs["seg-pretrained"]["epochs"])
    for run in RUNS:
        shots = []
        for epoch in epochs:
            weights = runs_dir(source) / run / "weights" / ("last.pt" if epoch == runs[run]["epochs"] else f"epoch{epoch - 1}.pt")
            r = YOLO(str(weights)).predict(tile, imgsz=640, conf=CONF, iou=IOU, verbose=False)[0]
            if run.startswith("sem"):
                classes = r.semantic_mask.data if hasattr(r, "semantic_mask") else r.masks.data
                classes = np.asarray(classes.cpu().numpy() if hasattr(classes, "cpu") else classes).astype(np.uint8).squeeze()
                shots.append({"epoch": epoch, "classes": gz(classes), "width": int(classes.shape[1]), "height": int(classes.shape[0])})
            else:
                objects = []
                if r.masks is not None:
                    for box, cls, score, poly in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.masks.xy):
                        objects.append({"cls": int(cls), "score": round(score, 3), "box": [round(v, 1) for v in box],
                                        "xy": [round(float(v), 1) for v in np.asarray(poly).ravel()]})
                shots.append({"epoch": epoch, "objects": objects})
            print(f"  {run} epoch {epoch}: {len(shots[-1].get('objects', []))} objects")
        snapshots[run] = shots
    count = lambda split: len(list((data_dir(source) / "seg" / "images" / split).glob("*.jpg")))  # noqa: E731
    return {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "settings": training_settings(source),
        "images": {"train": count("train"), "val": count("val")},
        "runs": runs, "snapshotTile": SNAPSHOT_TILE, "snapshots": snapshots,
    }


MATRIX_CONFS = [0.001] + [round(0.01 * k, 2) for k in range(1, 96)]  # Ultralytics' val threshold, then 0.01 … 0.95
EXAMPLE_CONF, EXAMPLES_PER_CELL, CROP_SIDE = 0.25, 6, 128
CURVE_POINTS = 101


def jpeg(image: np.ndarray, side: int, quality: int) -> str:
    """An image (BGR) shrunk to at most `side` pixels and encoded as JPEG, base64."""
    scale = min(1.0, side / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return b64(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes())


def capture_val(weights: Path, data: Path) -> tuple:
    """Validates like the end of training (conf 0.001) and keeps what reaches Ultralytics' ConfusionMatrix: per image
    the detections (boxes in tile pixels, score, class) and the true boxes, so the matrix can be rebuilt at any
    threshold."""
    import tempfile

    from ultralytics import YOLO
    from ultralytics.utils.metrics import ConfusionMatrix

    records, original = [], ConfusionMatrix.process_batch

    def keep(self, detections, batch, conf=0.25, iou_thres=0.5):
        records.append(({k: detections[k].detach().cpu().clone() for k in ("bboxes", "conf", "cls")},
                        {"cls": batch["cls"].detach().cpu().clone(), "bboxes": batch["bboxes"].detach().cpu().clone(),
                         "im_file": batch["im_file"]}))
        return original(self, detections, batch, conf, iou_thres)

    ConfusionMatrix.process_batch = keep
    try:
        with tempfile.TemporaryDirectory() as tmp:  # plots=True: only then does the validator fill the matrix
            m = YOLO(str(weights)).val(data=str(data), imgsz=640, batch=8, plots=True, workers=0, verbose=False,
                                       project=tmp, name="val")
    finally:
        ConfusionMatrix.process_batch = original
    return m, records


def matrix_at(records: list, conf: float, nc: int) -> np.ndarray:
    from ultralytics.utils.metrics import ConfusionMatrix

    cm = ConfusionMatrix(names={k: str(k) for k in range(nc)})
    for d, b in records:
        cm.process_batch(d, b, conf=conf)
    return cm.matrix.astype(int)


def matches_at(d: dict, b: dict, conf: float, iou_thres: float = 0.5) -> list[tuple]:
    """(predicted class or -1, true class or -1, detection index or -1, truth index or -1) per entry of the matrix,
    with ConfusionMatrix.process_batch's greedy, class-agnostic matching (box IoU > 0.5, highest score first)."""
    from ultralytics.utils.metrics import box_iou

    keep = (d["conf"] > conf).nonzero().flatten().tolist()
    gt_cls = b["cls"].int().flatten().tolist()
    out, matched = [], set()
    if keep and gt_cls:
        iou = box_iou(b["bboxes"], d["bboxes"][keep]).numpy()
        gt_match = np.full(len(gt_cls), -1)
        scores = d["conf"][keep].numpy()
        candidates = np.flatnonzero((iou > iou_thres).any(0))
        for j in candidates[np.argsort(-scores[candidates], kind="stable")]:
            g = iou[:, j].argmax()
            if iou[g, j] > iou_thres:
                gt_match[g] = j
                iou[g] = 0
        matched = set(gt_match[gt_match >= 0].tolist())
        for g, gc in enumerate(gt_cls):
            j = gt_match[g]
            out.append((int(d["cls"][keep[j]]) if j >= 0 else -1, gc, keep[j] if j >= 0 else -1, g))
    else:
        out += [(-1, gc, -1, g) for g, gc in enumerate(gt_cls)]
    out += [(int(d["cls"][i]), -1, i, -1) for j, i in enumerate(keep) if j not in matched]
    return out


def cell_examples(records: list, nc: int) -> dict:
    """Up to six crops per cell of the matrix at EXAMPLE_CONF, from different tiles: the detection's box, the true box,
    or both, around the object with a margin."""
    rng = np.random.default_rng(0)
    cells: dict[str, list] = {}
    for k, (d, b) in enumerate(records):
        for pc, tc, di, gi in matches_at(d, b, EXAMPLE_CONF):
            cells.setdefault(f"{nc if pc < 0 else pc},{nc if tc < 0 else tc}", []).append((k, di, gi))
    out = {}
    for cell, entries in sorted(cells.items()):
        order = rng.permutation(len(entries))
        chosen, tiles = [], set()
        for i in order:  # one per tile first, then any
            if entries[i][0] not in tiles:
                chosen.append(entries[i])
                tiles.add(entries[i][0])
            if len(chosen) == EXAMPLES_PER_CELL:
                break
        items = []
        for k, di, gi in chosen:
            d, b = records[k]
            image = cv2.imread(b["im_file"])
            pred = d["bboxes"][di].tolist() if di >= 0 else None
            truth = b["bboxes"][gi].tolist() if gi >= 0 else None
            boxes = [x for x in (pred, truth) if x]
            x1, y1 = min(x[0] for x in boxes), min(x[1] for x in boxes)
            x2, y2 = max(x[2] for x in boxes), max(x[3] for x in boxes)
            side = max(x2 - x1, y2 - y1, 40) + 24  # a square around both boxes, at least 40 px plus a margin
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            h, w = image.shape[:2]
            left = int(max(0, min(w - side, cx - side / 2)))
            top = int(max(0, min(h - side, cy - side / 2)))
            crop = image[top:top + int(side), left:left + int(side)]
            scale = CROP_SIDE / max(crop.shape[:2])
            move = lambda box: [round((box[0] - left) * scale, 1), round((box[1] - top) * scale, 1),  # noqa: E731
                                round((box[2] - left) * scale, 1), round((box[3] - top) * scale, 1)] if box else None
            items.append({"tile": Path(b["im_file"]).stem, "jpeg": jpeg(crop, CROP_SIDE, 80),
                          "size": [round(crop.shape[1] * scale), round(crop.shape[0] * scale)],
                          "pred": move(pred), "truth": move(truth),
                          "score": round(float(d["conf"][di]), 3) if di >= 0 else None})
        out[cell] = {"count": len(entries), "items": items}
    return out


def thin(x) -> list:
    """A 1000-point Ultralytics curve as 101 points."""
    x = np.asarray(x, dtype=float)
    idx = np.round(np.linspace(0, x.shape[-1] - 1, CURVE_POINTS)).astype(int)
    return np.round(x[..., idx], 4).tolist()


def console_lines(log: Path) -> dict | None:
    """Lines of the training console (YOLO's own output saved to a file), without colour codes, progress redraws or the
    folder the run was in."""
    import re

    if not log.exists():
        return None
    text = log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    clean = lambda s: re.sub(r"[A-Za-z]:/(?:[^/ ]+/)*?(?=(?:data|runs)/)", "", s.replace("\\", "/"))  # noqa: E731
    find = lambda pattern, start=0: next(i for i in range(start, len(lines)) if re.search(pattern, lines[i]))  # noqa: E731
    head = find(r"^\s+Epoch\s+GPU_mem")
    out = {
        "start": [clean(lines[find(p)].strip()) for p in (r"^train: Scanning", r"^val: Scanning", r"^optimizer: AdamW")],
        "header": lines[head],
        "progress": clean(lines[find(r"^\s+1/\d+ .* 10/\d+ ")]),
        "epochs": [],
    }
    for epoch in (1, 2, 50, 100):
        i = find(rf"^\s+{epoch}/\d+ .*100% ")
        val = find(r"^\s+all\s+\d+", i)
        out["epochs"].append({"epoch": epoch, "train": lines[i], "valHeader": lines[val - 1].split(":")[0], "val": lines[val]})
    end = find(r"epochs completed in")
    out["end"] = [clean(lines[i].strip()) for i in range(end, find(r"^Validating ", end) + 1)]
    final = find(r"^\s+all\s+\d+", end)
    out["final"] = [lines[final - 1].split(":")[0], lines[final]]
    return out


def label_stats(source: Path) -> dict:
    """What labels.jpg plots: objects per class and every training box (centre and size, divided by the tile size)."""
    rows = []
    for p in sorted((data_dir(source) / "seg" / "labels" / "train").glob("*.txt")):
        for line in p.read_text().splitlines():
            v = line.split()
            xy = np.asarray(v[1:], dtype=float).reshape(-1, 2)
            (x1, y1), (x2, y2) = xy.min(0), xy.max(0)
            rows.append([int(v[0]), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
    rows = np.asarray(rows)
    q = np.clip(np.round(rows[:, 1:] * 255), 0, 255).astype(np.uint8)
    packed = np.concatenate([rows[:, :1].astype(np.uint8), q], axis=1)  # class, cx, cy, w, h (× 255)
    return {"counts": np.bincount(rows[:, 0].astype(int), minlength=len(SEG_NAMES)).tolist(), "boxes": gz(packed)}


def outputs_record(source: Path) -> dict:
    """What the Outputs tab reads: the files of runs/seg-pretrained, the console, results.csv, label statistics, sample
    batches, confusion matrices at many thresholds with example crops, the P/R/F1 curves, and the land-cover matrix."""
    import csv
    import tempfile

    from ultralytics import YOLO

    run = runs_dir(source) / "seg-pretrained"
    m, records = capture_val(run / "weights" / "best.pt", data_dir(source) / "seg" / "data.yaml")
    nc = len(SEG_NAMES)
    matrices = [matrix_at(records, c, nc).ravel().tolist() for c in MATRIX_CONFS]
    print(f"  matrix at 0.001: {matrices[0]}, at 0.25: {matrices[MATRIX_CONFS.index(0.25)]}")
    images = [sum(1 for _, b in records if (b["cls"].int() == c).any()) for c in range(nc)]
    curves = {}
    for key, result in (("B", m.box), ("M", m.seg)):
        found = dict(zip(m.curves, m.curves_results))
        pr, f1c, pc, rc = (found[f"{n}({key})"] for n in ("Precision-Recall", "F1-Confidence", "Precision-Confidence", "Recall-Confidence"))
        f1 = np.asarray(f1c[1]).mean(0)
        best = int(f1.argmax())
        curves[key] = {
            "confidence": thin(f1c[0]), "precision": thin(pc[1]), "recall": thin(rc[1]), "f1": thin(f1c[1]),
            "prRecall": thin(pr[0]), "prPrecision": thin(pr[1]),
            "bestF1": {"confidence": round(float(np.asarray(f1c[0])[best]), 3), "f1": round(float(f1[best]), 4)},
            "table": {"precision": [round(float(v), 4) for v in result.p], "recall": [round(float(v), 4) for v in result.r],
                      "ap50": [round(float(v), 4) for v in result.ap50], "ap50_95": [round(float(v), 4) for v in result.ap]},
        }
    with tempfile.TemporaryDirectory() as tmp:
        s = YOLO(str(runs_dir(source) / "sem-pretrained" / "weights" / "best.pt")).val(
            data=str(data_dir(source) / "sem" / "data.yaml"), imgsz=640, batch=4, plots=True, workers=0, verbose=False,
            project=tmp, name="val")
    sem = np.asarray(s.confusion_matrix.matrix).round().astype(np.int64)  # predicted × true pixels
    rows = list(csv.DictReader((run / "results.csv").open()))
    results = {c: [round(float(r[c]), 5) for r in rows] for c in rows[0] if c != "epoch"}
    batches = sorted(run.glob("train_batch*.jpg"), key=lambda p: int(p.stem[11:]))
    late = next(p for p in batches if int(p.stem[11:]) > 2)  # the first batch plotted when mosaic stops (close_mosaic)
    files = sorted(str(p.relative_to(run)).replace("\\", "/") for p in run.rglob("*") if p.is_file() and p.suffix != ".onnx")
    return {
        "files": [{"path": f, "bytes": (run / f).stat().st_size} for f in files],
        "console": console_lines(source / "train_seg.log"),
        "results": results,
        "labels": label_stats(source),
        "batches": {
            "first": {"name": batches[0].name, "jpeg": jpeg(cv2.imread(str(batches[0])), 960, 72)},
            "last": {"name": late.name, "jpeg": jpeg(cv2.imread(str(late)), 960, 72)},
            "valLabels": {"name": "val_batch0_labels.jpg", "jpeg": jpeg(cv2.imread(str(run / "val_batch0_labels.jpg")), 960, 72)},
            "valPred": {"name": "val_batch0_pred.jpg", "jpeg": jpeg(cv2.imread(str(run / "val_batch0_pred.jpg")), 960, 72)},
        },
        "matrix": {"names": SEG_NAMES, "confs": MATRIX_CONFS, "counts": matrices, "images": images,
                   "tiles": len(records), "exampleConf": EXAMPLE_CONF, "examples": cell_examples(records, nc)},
        "curves": curves,
        "semMatrix": {"names": SEM_NAMES, "counts": sem.tolist()},
    }


def resize_cases() -> list[dict]:
    """cv2.resize INTER_LINEAR on small random images: shrinking, enlarging, almost 1:1, different ratios."""
    rng = np.random.default_rng(0)
    cases = []
    for (h, w), (dh, dw) in [((17, 29), (64, 64)), ((70, 90), (50, 64)), ((65, 65), (64, 64)), ((40, 13), (61, 20)), ((33, 31), (8, 9)),
                              ((34, 30), (17, 15)), ((13, 11), (100, 90))]:  # exactly half: OpenCV averages 2 x 2
        src = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        dst = cv2.resize(src, (dw, dh), interpolation=cv2.INTER_LINEAR)
        cases.append({"src": b64(src), "width": w, "height": h, "dstWidth": dw, "dstHeight": dh, "dst": b64(dst)})
    return cases


def letterbox_cases() -> dict:
    """Ultralytics' LetterBox (static ONNX input: auto=False, centred, grey 114) for a few image shapes."""
    shapes = [(650, 650), (480, 640), (200, 300), (1000, 333), (641, 639), (63, 65)]
    params = []
    for h, w in shapes:
        p = LetterBox(new_shape=(640, 640), auto=False)(image=np.zeros((h, w, 3), np.uint8))
        lb = LetterBox(new_shape=(640, 640), auto=False).get_params({"img": np.zeros((h, w, 3), np.uint8)})
        params.append({"height": h, "width": w, "newWidth": lb["new_unpad"][0], "newHeight": lb["new_unpad"][1],
                       "top": lb["top"], "left": lb["left"], "gain": lb["ratio"][0], "out": list(p.shape[:2])})
    rng = np.random.default_rng(1)
    src = rng.integers(0, 256, (23, 37, 3), dtype=np.uint8)
    dst = LetterBox(new_shape=(64, 64), auto=False)(image=src)
    return {"params": params, "pixels": {"src": b64(src), "width": 37, "height": 23, "size": 64, "dst": b64(dst)}}


def model_case(source: Path, sample: str) -> dict:
    """Both models on one tile shrunk to 320 px (small enough for a fixture), with Ultralytics' own post-processing."""
    bgr = cv2.imread(str(data_dir(source) / "seg" / "images" / "val" / f"{sample}.jpg"))
    small = cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_AREA)[..., ::-1].copy()  # RGB
    x = LetterBox(new_shape=(640, 640), auto=False)(image=small)
    x = (x.transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    seg = ort.InferenceSession(str(runs_dir(source) / "seg-pretrained" / "weights" / "seg_web.onnx"), providers=["CPUExecutionProvider"])
    sem = ort.InferenceSession(str(runs_dir(source) / "sem-pretrained" / "weights" / "sem_web.onnx"), providers=["CPUExecutionProvider"])
    head, proto = (torch.from_numpy(a) for a in seg.run(None, {"images": x}))
    candidates = int((head[0, 4:4 + len(SEG_NAMES)].amax(0) > CONF).sum())  # Ultralytics keeps scores above conf
    det = non_max_suppression(head, CONF, IOU, nc=len(SEG_NAMES))[0]
    masks = ops.process_mask(proto[0], det[:, 6:], det[:, :4], (640, 640), upsample=True)
    classes = sem.run(None, {"images": x})[0][0]
    return {
        "sample": sample, "rgb": gz(small), "width": 320, "height": 320, "conf": CONF, "iou": IOU,
        "aboveConf": candidates,
        "detections": [{"cls": int(d[5]), "score": round(float(d[4]), 5), "box": [round(float(v), 3) for v in d[:4]],
                        "maskArea": int(m.sum())} for d, m in zip(det, masks)],
        "landCover": np.bincount(classes.ravel(), minlength=len(SEM_NAMES)).tolist(),
    }


def ap_cases() -> list[dict]:
    rng = np.random.default_rng(2)
    cases = []
    for n in (1, 5, 40):
        recall, precision = np.sort(rng.random(n)).round(6), rng.random(n).round(6)
        cases.append({"recall": recall.tolist(), "precision": precision.tolist(), "ap": float(compute_ap(recall, precision)[0])})
    # recalls that land exactly on the 101 levels: 7 of 20 true objects found = 0.35
    recall, precision = np.array([1, 7, 7, 14, 20]) / 20, np.array([1.0, 0.875, 0.7, 0.7, 0.5])
    cases.append({"recall": recall.tolist(), "precision": precision.tolist(), "ap": float(compute_ap(recall, precision)[0])})
    return cases


def practice_tile(source: Path) -> dict:
    """The decoded pixels of the tile the Practice experiments run on: Node cannot decode a JPEG, and the
    browsers decode it exactly as OpenCV does, so the tests see what the page sees."""
    bgr = cv2.imread(str(data_dir(source) / "seg" / "images" / "val" / f"{SNAPSHOT_TILE}.jpg"))
    return {"name": SNAPSHOT_TILE, "width": bgr.shape[1], "height": bgr.shape[0], "rgb": gz(bgr[..., ::-1])}


def iou_cases() -> list[dict]:
    """Ultralytics' bbox_iou (IoU, GIoU, DIoU, CIoU) for overlapping, touching, apart and nested boxes."""
    pairs = [([10, 10, 50, 40], [20, 15, 60, 45]), ([0, 0, 10, 10], [10, 0, 20, 10]), ([0, 0, 10, 10], [30, 5, 40, 25]),
             ([0, 0, 100, 100], [25, 25, 75, 75]), ([552.1, 312.9, 640.9, 452.6], [548.0, 320.0, 630.0, 460.0])]
    out = []
    for a, b in pairs:
        ta, tb = torch.tensor([a], dtype=torch.float64), torch.tensor([b], dtype=torch.float64)
        out.append({"a": a, "b": b, **{k: float(bbox_iou(ta, tb, xywh=False, **{flag: True} if flag else {})[0, 0])
                                         for k, flag in (("iou", None), ("giou", "GIoU"), ("diou", "DIoU"), ("ciou", "CIoU"))}})
    return out


def write_assets(source: Path, picked: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, run, file, names in [("yolo-seg", "seg-pretrained", "seg_web.onnx", SEG_NAMES), ("yolo-sem", "sem-pretrained", "sem_web.onnx", SEM_NAMES)]:
        onnx = (runs_dir(source) / run / "weights" / file).read_bytes()
        write_data(name, {"names": names, "size": 640, "onnx": gz(onnx)})
    write_data("yolo-samples", {
        "source": "SpaceNet 2 (CC BY-SA 4.0); sports fields and land cover: (c) OpenStreetMap contributors (ODbL)",
        "segNames": SEG_NAMES, "semNames": SEM_NAMES, "samples": samples(source, picked),
    })
    write_data("yolo-run", run_record(source))
    write_data("yolo-outputs", outputs_record(source))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    source = Path(args[0]).resolve()
    if "--outputs-only" in sys.argv:  # only the Outputs tab's file (yolo-outputs.js)
        OUT.mkdir(parents=True, exist_ok=True)
        write_data("yolo-outputs", outputs_record(source))
        return
    picked = pick_samples(source)
    if "--fixture-only" not in sys.argv:  # --fixture-only: the tests' fixture without rewriting the data files
        write_assets(source, picked)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    fixture = {"resize": resize_cases(), "letterbox": letterbox_cases(), "model": model_case(source, picked[0]), "ap": ap_cases(), "iou": iou_cases(),
               "practiceTile": practice_tile(source)}
    path = FIXTURES / "yolo-parity.json"
    path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"web/tests/fixtures/yolo-parity.json  {path.stat().st_size / 1e6:.2f} MB; samples: {', '.join(picked)}")


if __name__ == "__main__":
    main()
