"""
Exports what the YOLO chapter of the static site needs, as JavaScript data files a page loads with a
<script> tag (which also works when the page is opened from a file, where fetch() is not allowed).

    python web/tools/export_yolo_assets.py <source> [--fixture-only]      (from the repository root)

<source> is the folder the chapter's Python code worked in: data/seg, data/sem, data/tiles (the val tiles,
their labels and georeference) and runs/seg-pretrained, runs/sem-pretrained (the trained models, exported
by scripts/export_variants.py as seg_web.onnx and sem_web.onnx: float16 weights, float32 inputs and outputs).
Needs ultralytics, onnxruntime, opencv-python and numpy.

Writes web/public/data/:

    yolo-seg.js       the instance segmentation model (building, sports_field), gzip
    yolo-sem.js       the semantic segmentation model (background, building, road, green, water, parking), gzip
    yolo-samples.js   validation tiles (JPEG as they are), their true polygons and land-cover masks, metres per pixel
    yolo-run.js       the three training runs (results.csv curves, final scores per class) and one val tile
                      predicted after 1, 5, 10, 25, 50, 75 and 100 epochs (runs/*/weights/epoch*.pt)

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
    labels = source / "data" / "seg" / "labels" / "val"
    masks = source / "data" / "sem" / "masks" / "val"
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
        meta = json.loads((source / "data" / "tiles" / f"{name}.json").read_text())
        polygons = []
        for line in (source / "data" / "seg" / "labels" / "val" / f"{name}.txt").read_text().splitlines():
            v = line.split()
            polygons.append({"cls": int(v[0]), "xy": [round(float(x), 5) for x in v[1:]]})
        mask = cv2.imread(str(source / "data" / "sem" / "masks" / "val" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        mx, my = metres_per_pixel(meta)
        out.append({
            "name": name, "city": city, "tile": img, "width": meta["width"], "height": meta["height"],
            "metresPerPixel": [round(mx, 5), round(my, 5)],
            "jpeg": b64((source / "data" / "seg" / "images" / "val" / f"{name}.jpg").read_bytes()),
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
SNAPSHOT_EPOCHS = [1, 5, 10, 25, 50, 75, 100]  # epochN.pt is saved after epoch N + 1; the last one is last.pt
SNAPSHOT_TILE = "Shanghai_img209"


def run_record(source: Path) -> dict:
    """The three training runs: curves per epoch, final scores per class, and one val tile predicted along the way."""
    import csv

    from ultralytics import YOLO

    runs = {}
    for run, columns in RUNS.items():
        rows = list(csv.DictReader((source / "runs" / run / "results.csv").open()))
        runs[run] = {
            "epochs": len(rows),
            "seconds": round(float(rows[-1]["time"])),
            "curves": {c: [round(float(r[c]), 5) for r in rows] for c in columns},
        }
    # final scores per class, measured again on the val set with the best weights
    for run in ("seg-pretrained", "seg-scratch"):
        m = YOLO(str(source / "runs" / run / "weights" / "best.pt")).val(
            data=str(source / "data" / "seg" / "data.yaml"), imgsz=640, batch=4, plots=False, verbose=False)
        runs[run]["final"] = {
            "maskAP50": [round(float(v), 4) for v in m.seg.ap50], "maskAP50_95": [round(float(v), 4) for v in m.seg.ap],
            "boxAP50": [round(float(v), 4) for v in m.box.ap50], "maskMAP50": round(float(m.seg.map50), 4),
            "maskMAP50_95": round(float(m.seg.map), 4), "boxMAP50": round(float(m.box.map50), 4),
            "precision": round(float(m.seg.mp), 4), "recall": round(float(m.seg.mr), 4),
        }
    m = YOLO(str(source / "runs" / "sem-pretrained" / "weights" / "best.pt")).val(
        data=str(source / "data" / "sem" / "data.yaml"), imgsz=640, batch=4, plots=False, verbose=False)
    iou = getattr(m, "_per_class_iou", None)  # SemanticMetrics keeps the IoU of every class here
    runs["sem-pretrained"]["final"] = {
        "mIoU": round(float(m.results_dict["metrics/mIoU"]), 4), "pixelAccuracy": round(float(m.results_dict["metrics/pixel_acc"]), 4),
        "iou": [round(float(v), 4) for v in iou] if iou is not None else None,
    }

    # the same val tile after 1, 5, 10, … epochs
    tile = str(source / "data" / "seg" / "images" / "val" / f"{SNAPSHOT_TILE}.jpg")
    snapshots = {}
    for run in RUNS:
        shots = []
        for epoch in SNAPSHOT_EPOCHS:
            weights = source / "runs" / run / "weights" / ("last.pt" if epoch == runs[run]["epochs"] else f"epoch{epoch - 1}.pt")
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
    count = lambda split: len(list((source / "data" / "seg" / "images" / split).glob("*.jpg")))  # noqa: E731
    return {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "settings": {"imgsz": 640, "batch": 4, "epochs": 100, "flipud": 0.5, "degrees": 90, "seed": 0},
        "images": {"train": count("train"), "val": count("val")},
        "runs": runs, "snapshotTile": SNAPSHOT_TILE, "snapshots": snapshots,
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
    bgr = cv2.imread(str(source / "data" / "seg" / "images" / "val" / f"{sample}.jpg"))
    small = cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_AREA)[..., ::-1].copy()  # RGB
    x = LetterBox(new_shape=(640, 640), auto=False)(image=small)
    x = (x.transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    seg = ort.InferenceSession(str(source / "runs" / "seg-pretrained" / "weights" / "seg_web.onnx"), providers=["CPUExecutionProvider"])
    sem = ort.InferenceSession(str(source / "runs" / "sem-pretrained" / "weights" / "sem_web.onnx"), providers=["CPUExecutionProvider"])
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
        onnx = (source / "runs" / run / "weights" / file).read_bytes()
        write_data(name, {"names": names, "size": 640, "onnx": gz(onnx)})
    write_data("yolo-samples", {
        "source": "SpaceNet 2 (CC BY-SA 4.0); sports fields and land cover: (c) OpenStreetMap contributors (ODbL)",
        "segNames": SEG_NAMES, "semNames": SEM_NAMES, "samples": samples(source, picked),
    })
    write_data("yolo-run", run_record(source))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    source = Path(args[0]).resolve()
    picked = pick_samples(source)
    if "--fixture-only" not in sys.argv:  # --fixture-only: the tests' fixture without rewriting the data files
        write_assets(source, picked)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    fixture = {"resize": resize_cases(), "letterbox": letterbox_cases(), "model": model_case(source, picked[0]), "ap": ap_cases(), "iou": iou_cases()}
    path = FIXTURES / "yolo-parity.json"
    path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"web/tests/fixtures/yolo-parity.json  {path.stat().st_size / 1e6:.2f} MB; samples: {', '.join(picked)}")


if __name__ == "__main__":
    main()
