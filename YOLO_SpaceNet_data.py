"""
The data of the YOLO chapter (chapter 4 of the learning path), prepared from public sources.

Downloads SpaceNet 2 satellite tiles of Las Vegas, Paris, Shanghai and Khartoum, turns their 16-bit
images into ordinary JPEGs and builds two training sets for Ultralytics YOLO:

    data/yolo/seg   instance segmentation, YOLO polygon labels:
                    0 building (SpaceNet's hand-drawn outlines), 1 sports_field (OpenStreetMap)
    data/yolo/sem   semantic segmentation, one PNG mask per tile:
                    0 background, 1 building, 2 road, 3 green, 4 water, 5 parking (255 = no image data)

Run:  python YOLO_SpaceNet_data.py                  the chapter's 746 tiles (about 1.9 GB to download, 130 MB kept)
      python YOLO_SpaceNet_data.py --per-city 20    a quick start: 20 tiles per city (about 200 MB)
      python YOLO_SpaceNet_data.py --check Paris_img637    draws one tile's labels into data/yolo/check/
Then: python YOLO_SpaceNet.py train seg            (the other file of the chapter)

Needs: pip install ultralytics rasterio shapely certifi   (OpenCV and NumPy come with ultralytics)

Licences: SpaceNet imagery and building outlines, CC BY-SA 4.0 (SpaceNet on AWS, registry.opendata.aws/spacenet);
OpenStreetMap, ODbL, (c) OpenStreetMap contributors. OpenStreetMap changes every day, so labels made today can
differ a little from the chapter's.
"""
import argparse
import json
import random
import shutil
import ssl
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import cv2
import numpy as np
import rasterio
from rasterio.transform import Affine
from shapely import STRtree, force_2d, transform as map_coords
from shapely.geometry import LineString, Polygon, box, shape
from shapely.ops import polygonize, unary_union


#--------------- Settings ---------------#
OUT = Path("data/yolo")
VAL_SHARE = 0.2        # one tile in five is kept for validation, chosen at random per city
MIN_AREA_PX = 12       # slivers left where the tile edge cuts an outline are dropped
MAX_NODATA = 0.25      # tiles with more empty (black) image than this are left out
SEED = 0
#----------------------------------------#

BUCKET = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN2_buildings/train"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
AGENT = {"User-Agent": "ai-learning-path-yolo/1.0"}
SSL = ssl.create_default_context(cafile=certifi.where())   # some Windows certificate stores are out of date

SEG_NAMES = ["building", "sports_field"]
SEM_NAMES = ["background", "building", "road", "green", "water", "parking"]
PAINT = ["green", "water", "parking", "road", "building"]   # painted in this order, later classes on top
IGNORE = 255           # Ultralytics' "ignore" value in semantic masks

# OpenStreetMap tags of each class (Overpass filters)
OSM_FILTERS = {
    "sports": ['["leisure"~"^(pitch|stadium|track|sports_centre)$"]'],
    "parking": ['["amenity"="parking"]'],
    "green": ['["leisure"~"^(park|garden|golf_course)$"]', '["landuse"~"^(grass|recreation_ground|cemetery|forest|meadow)$"]',
              '["natural"~"^(wood|scrub|grassland)$"]'],
    "water": ['["natural"="water"]', '["waterway"="riverbank"]', '["leisure"="swimming_pool"]'],
}
# roads are lines in OpenStreetMap: they are widened by a typical width for their type
ROAD_WIDTH_M = {"motorway": 20, "trunk": 16, "primary": 14, "secondary": 12, "tertiary": 10,
                "residential": 8, "unclassified": 8, "service": 5}
OSM_FILTERS["road"] = ['["highway"~"^(' + "|".join(ROAD_WIDTH_M) + ')$"]']


# ===========================================================================================
# 1. Downloading a tile
# ===========================================================================================

def get(url: str, data: bytes | None = None, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, data=data, headers=AGENT)
    return urllib.request.urlopen(request, timeout=timeout, context=SSL).read()


def to_8bit(bands: np.ndarray) -> np.ndarray:
    """Per-band 2–98 percentile stretch: the 11-bit satellite values become an ordinary photo."""
    out = np.zeros(bands.shape[1:] + (3,), np.uint8)
    for i in range(3):
        b = bands[i].astype(np.float32)
        valid = b[b > 0]                                   # 0 = no data at the edge of the satellite strip
        lo, hi = np.percentile(valid, (2, 98)) if valid.size else (0, 1)
        out[..., i] = np.clip((b - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    return out


def download_tile(city: str, img: str, folder: Path) -> int:
    """One tile: the image as a JPEG, its georeference and SpaceNet's building outlines in a JSON."""
    name = f"{city.split('_', 2)[2]}_{img}"
    jpg, meta = folder / f"{name}.jpg", folder / f"{name}.json"
    if jpg.exists() and meta.exists():
        return 0
    tif = folder / f"{name}.tif"
    tif.write_bytes(get(f"{BUCKET}/{city}/PS-RGB/SN2_buildings_train_{city}_PS-RGB_{img}.tif"))
    size = tif.stat().st_size
    with rasterio.open(tif) as src:
        rgb = to_8bit(src.read([1, 2, 3]))
        info = {"city": city, "img": img, "width": src.width, "height": src.height,
                "transform": list(src.transform)[:6], "bounds": list(src.bounds), "crs": str(src.crs)}
    tif.unlink()                                           # 2.5 MB each: keep only the JPEG
    geojson = get(f"{BUCKET}/{city}/geojson_buildings/SN2_buildings_train_{city}_geojson_buildings_{img}.geojson")
    info["buildings"] = json.loads(geojson)
    cv2.imwrite(str(jpg), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    meta.write_text(json.dumps(info))
    return size


# ===========================================================================================
# 2. OpenStreetMap: the other classes
# ===========================================================================================

def overpass(bounds: list[list[float]]) -> list[dict]:
    """Every OSM object of our classes that touches one of the tiles (lon/lat bounds), with its geometry."""
    boxes = "".join(f"way({b[1]},{b[0]},{b[3]},{b[2]});relation({b[1]},{b[0]},{b[3]},{b[2]});" for b in bounds)
    filters = "".join(f"{kind}.all{f};" for fs in OSM_FILTERS.values() for f in fs for kind in ("way", "relation"))
    query = f"[out:json][timeout:300];({boxes})->.all;({filters});out geom;"
    body = urllib.parse.urlencode({"data": query}).encode()
    for attempt in range(6):                                # public servers are often busy: wait and retry
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            return json.loads(get(url, body, timeout=400))["elements"]
        except Exception as error:
            print(f"  {url.split('/')[2]}: {error}; trying again", flush=True)
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("the Overpass API did not answer; try again later")


def classify(tags: dict) -> str | None:
    """The class of an OSM object, from the same tags as OSM_FILTERS."""
    if tags.get("highway") in ROAD_WIDTH_M: return "road"
    if tags.get("leisure") in {"pitch", "stadium", "track", "sports_centre"}: return "sports"
    if tags.get("amenity") == "parking": return "parking"
    if tags.get("natural") == "water" or tags.get("waterway") == "riverbank" or tags.get("leisure") == "swimming_pool": return "water"
    if (tags.get("leisure") in {"park", "garden", "golf_course"} or tags.get("landuse") in {"grass", "recreation_ground", "cemetery", "forest", "meadow"}
            or tags.get("natural") in {"wood", "scrub", "grassland"}): return "green"
    return None


def geometry(element: dict):
    """An OSM way or relation as a shapely shape: roads are always lines (a roundabout is a closed line, not an
    area); other closed ways are areas; a multipolygon relation is the area its outer ways enclose together
    (a large park's outline is often split into several ways)."""
    if element["type"] == "relation":
        outer = [[(p["lon"], p["lat"]) for p in m["geometry"]] for m in element.get("members", [])
                 if m.get("role") == "outer" and len(m.get("geometry", [])) >= 2]
        area = unary_union(list(polygonize([LineString(w) for w in outer]))) if outer else None
        return area if area is not None and not area.is_empty else None
    points = [(p["lon"], p["lat"]) for p in element.get("geometry", [])]
    if len(points) < 2:
        return None
    closed = len(points) >= 4 and points[0] == points[-1]
    if closed and "highway" not in element.get("tags", {}):
        return Polygon(points).buffer(0)
    return LineString(points)


# ===========================================================================================
# 3. From map coordinates to YOLO labels
# ===========================================================================================

def to_pixels(geom, meta: dict):
    """Longitude/latitude to pixel coordinates of the tile, through its affine transform."""
    inverse = ~Affine(*meta["transform"])
    return map_coords(geom, lambda xy: np.column_stack(inverse * (xy[:, 0], xy[:, 1])))


def polygons_px(geom, frame) -> list[Polygon]:
    """The polygons of a shape cut to the tile, in pixels, without slivers."""
    g = geom.buffer(0).intersection(frame)
    parts = g.geoms if hasattr(g, "geoms") else [g]
    return [p for p in parts if p.geom_type == "Polygon" and p.area >= MIN_AREA_PX]


def nodata_mask(img: np.ndarray) -> np.ndarray:
    """The empty edge of a satellite strip: near-black regions touching the tile border.

    A black roof is not empty data: it is not exactly black and it does not reach the border in one piece.
    """
    dark = (img.max(axis=2) < 4).astype(np.uint8)
    n, parts, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    h, w = dark.shape
    out = np.zeros(dark.shape, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area > 0.01 * h * w and (x == 0 or y == 0 or x + bw == w or y + bh == h):
            out |= parts == i
    return out


def mostly_visible(poly: Polygon, nodata: np.ndarray) -> bool:
    canvas = np.zeros(nodata.shape, np.uint8)
    cv2.fillPoly(canvas, [np.round(np.asarray(poly.exterior.coords)).astype(np.int32)], 1)
    inside = canvas.astype(bool)
    return inside.sum() == 0 or nodata[inside].mean() < 0.5


def yolo_line(cls: int, poly: Polygon, w: int, h: int) -> str:
    """A YOLO segmentation label: the class, then the outline's x y pairs divided by the image size."""
    return f"{cls} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly.exterior.coords[:-1])


def tile_layers(meta: dict, objects: list, roads: list, tree: STRtree, road_tree: STRtree, nodata: np.ndarray) -> dict:
    """All labelled polygons of one tile, in pixels, by class."""
    w, h = meta["width"], meta["height"]
    frame, tile = box(0, 0, w, h), box(*meta["bounds"])
    lat = (meta["bounds"][1] + meta["bounds"][3]) / 2
    metres_per_px = abs(meta["transform"][0]) * 111_320 * np.cos(np.radians(lat))   # east-west, for road widths
    layers = defaultdict(list)
    for feature in meta["buildings"]["features"]:
        g = force_2d(shape(feature["geometry"]))           # SpaceNet's points carry a z = 0
        if g.geom_type in ("Polygon", "MultiPolygon"):     # a tile with no building holds a placeholder point
            layers["building"] += polygons_px(to_pixels(g, meta), frame)
    for i in tree.query(tile, predicate="intersects"):
        cls, g = objects[i]
        layers["sports_field" if cls == "sports" else cls] += polygons_px(to_pixels(g, meta), frame)
    for i in road_tree.query(tile, predicate="intersects"):
        kind, g = roads[i]
        width_px = ROAD_WIDTH_M[kind] / metres_per_px
        layers["road"] += polygons_px(to_pixels(g, meta).buffer(width_px / 2, cap_style="flat"), frame)
    if nodata.any():                                       # objects mostly over the empty edge are not visible
        layers = defaultdict(list, {c: [p for p in ps if mostly_visible(p, nodata)] for c, ps in layers.items()})
    return layers


# ===========================================================================================
# 4. The two datasets
# ===========================================================================================

def osm_elements(city: str, ms: list[dict]) -> list[dict]:
    """The OSM objects of a city's tiles, kept in data/yolo/osm_<city>.json and fetched again when tiles were added."""
    cache = OUT / f"osm_{city}.json"
    names = sorted(m["img"] for m in ms)
    saved = json.loads(cache.read_text()) if cache.exists() else {}
    if not set(names) <= set(saved.get("tiles", [])):
        print(f"OpenStreetMap: {len(ms)} tiles of {city}", flush=True)
        saved = {"tiles": names, "elements": overpass([m["bounds"] for m in ms])}
        cache.write_text(json.dumps(saved))
    return saved["elements"]


def make_datasets(tiles: Path, names: list[str]) -> None:
    """data/yolo/seg (polygons) and data/yolo/sem (masks) from the chosen tiles, split into train and val."""
    rng = random.Random(SEED)
    seg, sem = OUT / "seg", OUT / "sem"
    for d in (seg, sem):
        shutil.rmtree(d, ignore_errors=True)
    metas = [json.loads((tiles / f"{name}.json").read_text()) for name in sorted(names)]
    by_city = defaultdict(list)
    for m in metas:
        by_city[m["city"]].append(m)
    stats = defaultdict(Counter)
    for city, ms in by_city.items():
        elements = osm_elements(city, ms)
        shapes = [(classify(el.get("tags", {})), geometry(el)) for el in elements]
        objects = [(c, g) for c, g in shapes if c and c != "road" and g is not None and g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty]
        roads = [(el["tags"]["highway"], g) for el, (c, g) in zip(elements, shapes) if c == "road" and g is not None and g.geom_type == "LineString"]
        tree, road_tree = STRtree([g for _, g in objects]), STRtree([g for _, g in roads])
        rng.shuffle(ms)
        n_val = round(len(ms) * VAL_SHARE)
        for k, meta in enumerate(ms):
            split = "val" if k < n_val else "train"
            name = f"{meta['city'].split('_', 2)[2]}_{meta['img']}"
            image = cv2.imread(str(tiles / f"{name}.jpg"))
            nodata = nodata_mask(image)
            if nodata.mean() > MAX_NODATA:
                stats[split]["skipped (no image data)"] += 1
                continue
            layers = tile_layers(meta, objects, roads, tree, road_tree, nodata)
            w, h = meta["width"], meta["height"]
            for d in (seg, sem):
                (d / "images" / split).mkdir(parents=True, exist_ok=True)
                shutil.copy(tiles / f"{name}.jpg", d / "images" / split / f"{name}.jpg")
            # instance segmentation: one line per object
            lines = [yolo_line(0, p, w, h) for p in layers["building"]] + [yolo_line(1, p, w, h) for p in layers["sports_field"]]
            (seg / "labels" / split).mkdir(parents=True, exist_ok=True)
            (seg / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            # semantic segmentation: the classes painted one over the other
            mask = np.zeros((h, w), np.uint8)
            for cls in PAINT:
                polys = layers[cls] + (layers["sports_field"] if cls == "green" else [])   # sports fields are mostly grass
                for p in polys:   # one polygon at a time: in one call, OpenCV would leave where two overlap empty
                    cv2.fillPoly(mask, [np.round(np.asarray(p.exterior.coords)).astype(np.int32)], SEM_NAMES.index(cls))
            mask[nodata] = IGNORE
            (sem / "masks" / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sem / "masks" / split / f"{name}.png"), mask)
            stats[split]["images"] += 1
            for cls, polys in layers.items():
                stats[split][cls] += len(polys)
    # the files Ultralytics reads to find the data and the class names
    (seg / "data.yaml").write_text(f"path: {seg.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
                                   + "".join(f"  {i}: {n}\n" for i, n in enumerate(SEG_NAMES)))
    (sem / "data.yaml").write_text(f"path: {sem.resolve().as_posix()}\ntrain: images/train\nval: images/val\nmasks_dir: masks\nnames:\n"
                                   + "".join(f"  {i}: {n}\n" for i, n in enumerate(SEM_NAMES)))
    for split, c in sorted(stats.items(), reverse=True):
        print(f"{split}: " + ", ".join(f"{k} {v}" for k, v in c.items()))


# ===========================================================================================
# 5. Looking at the labels
# ===========================================================================================

SEM_COLORS = np.array([[0, 0, 0], [60, 60, 230], [200, 200, 200], [60, 200, 60], [230, 140, 30], [0, 200, 230]], np.uint8)  # BGR


def check_labels(name: str) -> Path:
    """One tile with its polygons (left) and its land-cover mask (right): most label mistakes show at once."""
    for split in ("train", "val"):
        image_file = OUT / "seg" / "images" / split / f"{name}.jpg"
        if image_file.exists():
            break
    else:
        raise SystemExit(f"{name} is not in data/yolo/seg (was it skipped for missing image data?)")
    img = cv2.imread(str(image_file))
    h, w = img.shape[:2]
    left = img.copy()
    for line in (OUT / "seg" / "labels" / split / f"{name}.txt").read_text().splitlines():
        v = line.split()
        xy = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
        cv2.polylines(left, [xy], True, (0, 255, 255) if v[0] == "0" else (255, 0, 255), 2)
    mask = cv2.imread(str(OUT / "sem" / "masks" / split / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    shown = np.where(mask == IGNORE, 0, mask)
    right = np.where(shown[..., None] > 0, (0.45 * img + 0.55 * SEM_COLORS[shown]).astype(np.uint8), img)
    out = OUT / "check" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack([left, right]))
    return out


# ===========================================================================================
# The chapter's tiles: chosen so that the rarer classes have examples (tiles holding sports fields,
# water, parking and parks in OpenStreetMap first, a third at random so ordinary tiles stay in too)
# ===========================================================================================

TILES = {
    "AOI_2_Vegas": [  # 224 tiles
        81, 95, 107, 127, 136, 151, 191, 192, 241, 301, 302, 311, 361, 382, 413, 442, 458, 481, 482, 497, 507, 520, 563,
        582, 601, 603, 667, 669, 674, 676, 696, 699, 720, 734, 735, 747, 758, 761, 801, 802, 805, 806, 807, 812, 827,
        828, 829, 854, 865, 868, 870, 871, 931, 933, 934, 935, 959, 960, 961, 997, 1002, 1028, 1034, 1038, 1101, 1102,
        1131, 1187, 1196, 1200, 1222, 1267, 1310, 1322, 1323, 1385, 1386, 1419, 1442, 1509, 1510, 1552, 1557, 1571,
        1634, 1779, 1813, 1832, 1840, 1893, 1901, 2008, 2032, 2284, 2325, 2328, 2407, 2409, 2412, 2417, 2508, 2522,
        2571, 2572, 2575, 2594, 2612, 2644, 2650, 2660, 2678, 2691, 2739, 2740, 2823, 2854, 2957, 2989, 3075, 3117,
        3155, 3161, 3273, 3306, 3321, 3386, 3458, 3498, 3509, 3516, 3524, 3591, 3603, 3670, 3747, 3749, 3983, 3987,
        4007, 4011, 4012, 4013, 4017, 4062, 4067, 4086, 4093, 4108, 4172, 4218, 4238, 4248, 4249, 4250, 4329, 4373,
        4383, 4399, 4403, 4408, 4467, 4484, 4526, 4580, 4797, 4799, 4828, 4879, 4959, 4974, 5036, 5037, 5094, 5099,
        5111, 5119, 5120, 5121, 5142, 5173, 5174, 5198, 5199, 5219, 5222, 5278, 5329, 5347, 5426, 5478, 5479, 5555,
        5584, 5618, 5639, 5707, 5719, 5760, 5777, 5789, 5796, 5804, 5858, 5864, 5878, 6004, 6047, 6112, 6113, 6116,
        6152, 6191, 6193, 6225, 6266, 6274, 6283, 6288, 6331, 6332, 6342, 6348, 6350, 6409
    ],
    "AOI_3_Paris": [  # 261 tiles
        14, 23, 28, 34, 44, 49, 52, 74, 75, 85, 86, 90, 92, 95, 96, 98, 100, 101, 102, 108, 113, 119, 132, 133, 134,
        136, 147, 155, 156, 160, 164, 165, 166, 168, 181, 182, 196, 197, 200, 203, 209, 211, 215, 217, 229, 230, 240,
        242, 243, 269, 283, 294, 299, 300, 303, 305, 306, 335, 336, 345, 353, 354, 362, 373, 385, 387, 389, 411, 416,
        431, 440, 441, 443, 444, 463, 469, 472, 473, 482, 484, 486, 495, 506, 509, 511, 514, 520, 545, 552, 564, 570,
        576, 589, 594, 595, 597, 599, 604, 605, 608, 625, 637, 638, 642, 643, 648, 649, 654, 661, 672, 685, 687, 689,
        701, 703, 716, 727, 728, 730, 731, 733, 742, 746, 748, 750, 755, 767, 773, 774, 775, 786, 789, 792, 793, 817,
        819, 820, 821, 822, 825, 827, 844, 857, 864, 865, 866, 879, 909, 913, 926, 930, 936, 942, 944, 951, 953, 954,
        955, 971, 978, 997, 998, 999, 1008, 1009, 1010, 1016, 1029, 1052, 1057, 1059, 1060, 1063, 1095, 1102, 1129,
        1140, 1146, 1154, 1164, 1181, 1185, 1202, 1205, 1207, 1224, 1229, 1230, 1232, 1274, 1275, 1276, 1285, 1292,
        1308, 1314, 1315, 1318, 1320, 1359, 1371, 1378, 1395, 1405, 1425, 1444, 1449, 1455, 1459, 1463, 1464, 1496,
        1499, 1509, 1511, 1517, 1523, 1538, 1540, 1541, 1555, 1556, 1585, 1587, 1588, 1589, 1617, 1627, 1632, 1635,
        1636, 1643, 1653, 1664, 1669, 1670, 1682, 1687, 1689, 1691, 1717, 1727, 1728, 1735, 1740, 1743, 1751, 1754,
        1755, 1764, 1766, 1776, 1784, 1811, 1814, 1817, 1818, 1822, 1843, 1892, 1908
    ],
    "AOI_4_Shanghai": [  # 112 tiles
        40, 152, 179, 206, 209, 214, 221, 381, 652, 943, 1017, 1057, 1158, 1327, 1398, 1422, 1478, 1751, 1755, 1773,
        1832, 1903, 2012, 2067, 2366, 2382, 2536, 2735, 2809, 2846, 2847, 2914, 2938, 2939, 3077, 3093, 3382, 3474,
        3475, 3527, 3568, 3582, 3625, 3660, 3679, 3759, 3800, 3856, 3857, 4003, 4049, 4145, 4191, 4216, 4241, 4289,
        4348, 4383, 4458, 4476, 4486, 4573, 4759, 4808, 4853, 4937, 4938, 4965, 5010, 5032, 5041, 5308, 5339, 5340,
        5613, 5675, 5678, 5683, 5704, 5792, 5795, 5863, 5882, 5969, 6005, 6145, 6400, 6579, 6603, 6611, 6681, 6695,
        6807, 6858, 6888, 6899, 6931, 7060, 7067, 7134, 7154, 7216, 7297, 7302, 7316, 7333, 7334, 7379, 7458, 7492,
        7536, 7537
    ],
    "AOI_5_Khartoum": [  # 149 tiles
        46, 50, 74, 79, 104, 125, 126, 132, 163, 226, 268, 297, 300, 305, 309, 310, 311, 313, 314, 315, 316, 317, 318,
        325, 328, 334, 345, 351, 352, 353, 359, 361, 362, 372, 386, 455, 467, 468, 497, 508, 554, 559, 560, 564, 570,
        586, 596, 611, 618, 624, 632, 656, 691, 694, 701, 703, 713, 728, 729, 776, 785, 786, 787, 788, 792, 822, 857,
        879, 880, 885, 887, 888, 889, 890, 892, 904, 908, 952, 953, 987, 988, 990, 992, 993, 994, 1003, 1045, 1072,
        1087, 1095, 1097, 1098, 1104, 1109, 1116, 1121, 1122, 1124, 1131, 1154, 1156, 1183, 1191, 1202, 1210, 1219,
        1227, 1237, 1238, 1240, 1276, 1298, 1299, 1300, 1304, 1307, 1308, 1328, 1329, 1338, 1339, 1346, 1348, 1351,
        1378, 1379, 1404, 1428, 1429, 1445, 1446, 1454, 1455, 1456, 1483, 1518, 1542, 1543, 1546, 1580, 1612, 1614,
        1630, 1633, 1634, 1644, 1654, 1669, 1683
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-city", type=int, default=0, help="download only this many tiles per city (0: all)")
    parser.add_argument("--check", metavar="NAME", help="draw the labels of one tile, e.g. Paris_img637")
    args = parser.parse_args()
    if args.check:
        print(f"wrote {check_labels(args.check)}")
        return

    rng = random.Random(SEED)
    jobs = []
    for city, numbers in TILES.items():
        chosen = rng.sample(numbers, args.per_city) if 0 < args.per_city < len(numbers) else numbers
        jobs += [(city, f"img{n}") for n in sorted(chosen)]
    tiles = OUT / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(jobs)} tiles from SpaceNet (CC BY-SA 4.0)", flush=True)
    done = total = 0
    with ThreadPoolExecutor(8) as pool:
        for size in pool.map(lambda job: download_tile(*job, tiles), jobs):
            done += 1
            total += size
            if done % 25 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} tiles, {total / 1e9:.2f} GB", flush=True)
    make_datasets(tiles, [f"{city.split('_', 2)[2]}_{img}" for city, img in jobs])   # only these, even if more were downloaded before
    print("labels from OpenStreetMap: (c) OpenStreetMap contributors, ODbL")
    print("next: python YOLO_SpaceNet_data.py --check <tile name>, then python YOLO_SpaceNet.py train seg")


if __name__ == "__main__":
    main()
