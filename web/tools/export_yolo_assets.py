"""
Exports what the YOLO chapter of the static site needs, as JavaScript data files a page loads with a
<script> tag (which also works when the page is opened from a file, where fetch() is not allowed).

    python web/tools/export_yolo_assets.py <source>      (from the repository root)

<source> is the folder the chapter's Python code worked in: data/seg, data/sem, data/tiles (the val tiles,
their labels and georeference) and runs/seg-pretrained, runs/sem-pretrained (the trained models, exported
by scripts/export_variants.py as seg_web.onnx and sem_web.onnx: float16 weights, float32 inputs and outputs).
Needs ultralytics, onnxruntime, opencv-python and numpy.

Writes web/public/data/:

    yolo-seg.js       the instance segmentation model (building, sports_field), gzip
    yolo-sem.js       the semantic segmentation model (background, building, road, green, water, parking), gzip
    yolo-samples.js   validation tiles (JPEG as they are), their true polygons and land-cover masks, metres per pixel

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
from ultralytics.utils.metrics import compute_ap
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
    (OUT / f"{name}.js").write_text(text, encoding="utf-8")
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


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    source = Path(sys.argv[1]).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, run, file, names in [("yolo-seg", "seg-pretrained", "seg_web.onnx", SEG_NAMES), ("yolo-sem", "sem-pretrained", "sem_web.onnx", SEM_NAMES)]:
        onnx = (source / "runs" / run / "weights" / file).read_bytes()
        write_data(name, {"names": names, "size": 640, "onnx": gz(onnx)})
    picked = pick_samples(source)
    write_data("yolo-samples", {
        "source": "SpaceNet 2 (CC BY-SA 4.0); sports fields and land cover: (c) OpenStreetMap contributors (ODbL)",
        "segNames": SEG_NAMES, "semNames": SEM_NAMES, "samples": samples(source, picked),
    })
    FIXTURES.mkdir(parents=True, exist_ok=True)
    fixture = {"resize": resize_cases(), "letterbox": letterbox_cases(), "model": model_case(source, picked[0]), "ap": ap_cases()}
    path = FIXTURES / "yolo-parity.json"
    path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"web/tests/fixtures/yolo-parity.json  {path.stat().st_size / 1e6:.2f} MB; samples: {', '.join(picked)}")


if __name__ == "__main__":
    main()
