# ----------------------------------------------------------------------
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ----------------------------------------------------------------------
"""
============================
     SimpleCNN Dashboard
============================

A small local web app to watch SimpleCNN train on MNIST in real time and to
run inference with the trained model afterwards.

    python dashboard.py              # opens http://127.0.0.1:8000 in the browser
    python dashboard.py --port 8080 --no-browser

Training tab:  stage tracker, live loss/accuracy curves, the conv1 filters as
               they learn, and predictions on fixed validation images.
Inference tab: draw a digit (or pick a test image) and see the prediction
               together with what every layer of the network produces.
Theory tab:    the CNN building blocks explained, each with a live demo that
               runs on a real MNIST image and the trained model's weights.

The model, data loading and evaluation code are reused from SimpleCNN.py, and
the web server uses only the Python standard library. With the default
settings a dashboard run gives exactly the same result as `python SimpleCNN.py`.
"""
import argparse
import base64
import io
import json
import os
import random
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    import torch
except ModuleNotFoundError:
    raise SystemExit("PyTorch is not installed for this Python interpreter.\n"
                     "Run the dashboard with the project's virtual environment:\n"
                     "    .venv\\Scripts\\python.exe dashboard.py   (Windows)\n"
                     "    .venv/bin/python dashboard.py            (macOS / Linux)") from None
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets
from torchvision.transforms import ToTensor

import SimpleCNN as cnn

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
CHECKPOINT = ROOT / "checkpoints" / "simple_cnn.pt"
# The finished run that produced the checkpoint (curves, filters, samples, log), so the
# Training tab can show it again after a restart
RUN_FILE = CHECKPOINT.with_name("simple_cnn_run.json")
RUN_FIELDS = ("config", "epoch", "batch", "batches", "step", "dataset_sizes", "batch_losses", "history",
              "test", "filters", "samples", "sample_predictions", "log", "started_at", "finished_at")
# Bumped when the page needs a newer server API; the page warns when an older server is still running
SERVER_VERSION = 4

NUM_SAMPLES = 16      # Validation images whose predictions are shown while training
FILTER_EVERY = 10     # Refresh the conv1 filter snapshot every N batches
LOG_LIMIT = 300       # Log lines kept in memory


#-------------------- Helpers --------------------#
def image_to_uint8(image: torch.Tensor) -> list[int]:
    """Converts a 0-1 image tensor to a flat list of 0-255 pixel values."""
    return (image.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).flatten().tolist()


def activation_to_uint8(values: torch.Tensor) -> list[int]:
    """Scales a non-negative activation map to 0-255 (0 stays 0, the peak becomes 255)."""
    values = values.detach().cpu().float()
    peak = values.max().item()
    if peak <= 1e-8:
        return [0] * values.numel()
    return (values / peak * 255).round().clamp(0, 255).to(torch.uint8).flatten().tolist()


def conv1_filters(model: torch.nn.Module) -> list[list[float]]:
    """Returns the 32 conv1 kernels (5x5), each scaled to [-1, 1] for display."""
    weights = model.conv1.weight.detach().cpu()[:, 0]                   # (32, 5, 5)
    scale = weights.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return (weights / scale).flatten(1).round(decimals=3).tolist()


def preprocess_drawing(data_url: str) -> torch.Tensor | None:
    """
    Turns a canvas drawing (white digit on black) into an MNIST-style input.

    MNIST digits were scaled to fit a 20x20 box and centered by their center of
    mass inside a 28x28 image, so the drawing gets the same treatment.

    Returns:
        A (1, 1, 28, 28) tensor with values in [0, 1], or None if the canvas is empty.
    """
    encoded = data_url.split(",", 1)[-1]
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("L")
    pixels = np.asarray(image, dtype=np.float32)
    ys, xs = np.nonzero(pixels > 30)
    if xs.size == 0:
        return None

    # Crop to the digit and scale its longer side to 20 pixels
    digit = image.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    width, height = digit.size
    scale = 20.0 / max(width, height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    digit = digit.resize((new_w, new_h), Image.Resampling.LANCZOS)
    digit = np.asarray(digit, dtype=np.float32) / 255.0

    # Place it so that its center of mass lands in the middle of the 28x28 image
    total = digit.sum()
    center_y = (digit.sum(axis=1) * np.arange(new_h)).sum() / total
    center_x = (digit.sum(axis=0) * np.arange(new_w)).sum() / total
    top = int(np.clip(round(14 - center_y), 0, 28 - new_h))
    left = int(np.clip(round(14 - center_x), 0, 28 - new_w))

    canvas = np.zeros((28, 28), dtype=np.float32)
    canvas[top:top + new_h, left:left + new_w] = digit
    return torch.from_numpy(np.clip(canvas, 0, 1)).view(1, 1, 28, 28)


def parse_config(body: dict) -> dict:
    """Validates the training settings sent by the browser."""
    def number(key, cast, default, low, high):
        try:
            value = cast(body.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' geçerli bir sayı değil.") from None
        if not low <= value <= high:
            raise ValueError(f"'{key}' {low} ile {high} arasında olmalı.")
        return value

    return {
        "epochs": number("epochs", int, cnn.num_epochs, 1, 50),
        "lr": number("lr", float, cnn.learning_rate, 1e-5, 1.0),
        "batch_size": number("batch_size", int, cnn.batch_size, 8, 4096),
        "seed": number("seed", int, cnn.seed, 0, 2**31 - 1),
    }


def device_name() -> str:
    if torch.cuda.is_available():
        return f"GPU · {torch.cuda.get_device_name(0)}"
    return "CPU"


#------------------- Inference -------------------#
class ModelNotReady(Exception):
    pass


class InferenceEngine:
    """
    Keeps CPU copies of the model for inference, so predictions never disturb a
    training run on the GPU. Forward hooks record what each stage produces.

    Two models are kept: the trained one and the "initial" one, i.e. the exact
    random weights training starts from, so the effect of training can be compared.
    """

    MODELS = ("trained", "initial")

    def __init__(self):
        self.lock = threading.Lock()
        self.model = cnn.SimpleCNN(num_classes=cnn.num_class).eval()
        self.initial = cnn.SimpleCNN(num_classes=cnn.num_class).eval()
        self.initial_seed = None
        self.ready = False
        self.source = None
        self.meta = None
        self.version = 0
        self._test_set = None
        self._test_lock = threading.Lock()
        self._activations = {}
        for model in (self.model, self.initial):
            self._add_hooks(model)
        self.initial_from_seed(cnn.seed)

    def _add_hooks(self, model):
        # ReLU is applied functionally in SimpleCNN.forward, so it is applied here too
        stages = {
            "conv1": (model.batch1, F.relu),   # Conv1 -> BatchNorm -> ReLU
            "pool1": (model.max1, None),       # MaxPool (Dropout is off at inference)
            "conv2": (model.batch2, F.relu),   # Conv2 -> BatchNorm -> ReLU
            "pool2": (model.max2, None),
            "fc1": (model.fc1, F.relu),        # FC1 -> ReLU
        }
        for name, (module, activation) in stages.items():
            module.register_forward_hook(self._make_hook(name, activation))

    def initial_from_seed(self, seed: int):
        """Rebuilds the weights a training run with this seed starts from (same order as SimpleCNN.py)."""
        with torch.random.fork_rng(devices=[]):   # leaves the global random state untouched
            torch.manual_seed(seed)
            fresh = cnn.SimpleCNN(num_classes=cnn.num_class)
        self.set_initial(fresh.state_dict(), seed)

    def set_initial(self, state_dict: dict, seed: int):
        cpu_state = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        with self.lock:
            self.initial.load_state_dict(cpu_state)
            self.initial.eval()
            self.initial_seed = seed

    def _pick(self, which: str):
        """The model to use; call it with the lock held."""
        if which == "initial":
            return self.initial
        if not self.ready:
            raise ModelNotReady()
        return self.model

    def _make_hook(self, name, activation):
        def hook(_module, _inputs, output):
            self._activations[name] = activation(output) if activation else output
        return hook

    def info(self) -> dict:
        with self.lock:
            return {"ready": self.ready, "source": self.source, "meta": self.meta,
                    "version": self.version, "initial_seed": self.initial_seed}

    def load_state(self, state_dict: dict, source: str, meta: dict | None = None):
        cpu_state = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
        with self.lock:
            self.model.load_state_dict(cpu_state)
            self.model.eval()
            self.ready = True
            self.source = source
            self.meta = meta
            self.version += 1

    def save_checkpoint(self, state_dict: dict, meta: dict, initial_state: dict):
        CHECKPOINT.parent.mkdir(exist_ok=True)
        cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
        torch.save({"state_dict": cpu_state, "initial_state_dict": initial_state, "meta": meta},
                   CHECKPOINT)

    def load_checkpoint(self) -> bool:
        if not CHECKPOINT.exists():
            return False
        try:
            checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
            meta = checkpoint.get("meta") or {}
            self.load_state(checkpoint["state_dict"], "Kayıtlı model", meta)
            # The run's starting weights: stored with newer checkpoints, otherwise rebuilt from its seed
            if "initial_state_dict" in checkpoint:
                self.set_initial(checkpoint["initial_state_dict"], meta.get("seed", cnn.seed))
            else:
                self.initial_from_seed(meta.get("seed", cnn.seed))
            return True
        except Exception as exc:  # A broken or outdated file should not stop the app
            print(f"Could not load {CHECKPOINT}: {exc}")
            return False

    def theory_params(self, which: str = "trained") -> dict:
        """Conv1 and BatchNorm1 weights for the Theory tab, which runs its demos in the browser."""
        def values(tensor):
            return [round(v, 6) for v in tensor.detach().flatten().tolist()]

        with self.lock:
            try:
                model = self._pick(which)
            except ModelNotReady:
                return {"ready": False}
            return {
                "ready": True,
                "model": which,
                "version": self.version,
                "conv1": {"weight": [values(w) for w in model.conv1.weight[:, 0]],   # 32 x (5*5)
                          "bias": values(model.conv1.bias)},
                "bn1": {"gamma": values(model.batch1.weight), "beta": values(model.batch1.bias),
                        "mean": values(model.batch1.running_mean),
                        "var": values(model.batch1.running_var), "eps": model.batch1.eps},
                "dropout": [cnn.dropout1_rate, cnn.dropout2_rate],
            }

    def _test_dataset(self):
        with self._test_lock:
            if self._test_set is None:
                self._test_set = datasets.MNIST(root=str(ROOT / "data"), train=False,
                                                download=True, transform=ToTensor())
            return self._test_set

    def random_test_samples(self, count: int) -> list[dict]:
        test_set = self._test_dataset()
        # A private RNG, so a training run's random sequence is never touched
        indices = random.Random().sample(range(len(test_set)), count)
        samples = []
        for index in indices:
            image, label = test_set[index]
            samples.append({"index": index, "label": int(label), "pixels": image_to_uint8(image)})
        return samples

    def test_image(self, index: int) -> tuple[torch.Tensor, int]:
        test_set = self._test_dataset()
        if not 0 <= index < len(test_set):
            raise ValueError("Geçersiz test görüntüsü.")
        image, label = test_set[index]
        return image.unsqueeze(0), int(label)

    def predict(self, image: torch.Tensor, which: str = "trained") -> dict:
        """Runs a model on one (1, 1, 28, 28) image and returns every stage's output."""
        with self.lock:
            model = self._pick(which)
            self._activations.clear()
            with torch.no_grad():
                logits = model(image)
            activations = dict(self._activations)

        probabilities = F.softmax(logits, dim=1)[0]
        layers = []
        for key in ("conv1", "pool1", "conv2", "pool2"):
            maps = activations[key][0]                                   # (C, H, W)
            layers.append({
                "key": key,
                "shape": list(maps.shape),
                "maps": [activation_to_uint8(m) for m in maps],
                "peaks": [round(v, 3) for v in maps.amax(dim=(1, 2)).tolist()],
            })
        fc1 = activations["fc1"][0]
        return {
            "prediction": int(probabilities.argmax()),
            "probabilities": [round(p, 5) for p in probabilities.tolist()],
            "logits": [round(v, 3) for v in logits[0].tolist()],
            "input": image_to_uint8(image[0, 0]),
            "layers": layers,
            "fc1": {"values": activation_to_uint8(fc1), "active": int((fc1 > 0).sum())},
            "model": which,
        }


#-------------------- Training --------------------#
class StopTraining(Exception):
    pass


class TrainingSession:
    """Runs one training job at a time in a background thread and records its progress."""

    def __init__(self, engine: InferenceEngine):
        self.engine = engine
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.run_id = 0
        self._reset(config=None)

    def _reset(self, config):
        self.config = config
        self.state = "idle"          # idle | running | done | stopped | error
        self.stage = None            # loading | training | validating | testing | saving | done
        self.message = ""
        self.epoch = 0
        self.batch = 0
        self.batches = 0
        self.step = 0
        self.dataset_sizes = None
        self.batch_losses = []
        self.history = []
        self.test = None
        self.filters = None
        self.samples = None
        self.sample_predictions = None
        self.log = []
        self.started_at = None
        self.finished_at = None
        self.restored = None         # {"trained_at": ...} when the run was loaded from RUN_FILE

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, config: dict) -> bool:
        with self.lock:
            if self.is_running():
                return False
            self.run_id += 1
            self._reset(config)
            self.state = "running"
            self.stage = "loading"
            self.started_at = time.time()
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run, args=(config,), daemon=True)
            self.thread.start()
            return True

    def stop(self):
        self.stop_event.set()

    def _log(self, text: str):
        print(text)
        with self.lock:
            self.log.append(f"[{datetime.now():%H:%M:%S}] {text}")
            del self.log[:-LOG_LIMIT]

    def _set(self, **fields):
        with self.lock:
            for key, value in fields.items():
                setattr(self, key, value)

    def _predict_samples(self, model, images, labels, device, epoch):
        model.eval()
        with torch.no_grad():
            probabilities = F.softmax(model(images.to(device)), dim=1).cpu()
        confidence, predictions = probabilities.max(dim=1)
        self._set(sample_predictions={
            "epoch": epoch,
            "predictions": predictions.tolist(),
            "confidence": [round(c, 4) for c in confidence.tolist()],
            "correct": int((predictions == torch.tensor(labels)).sum()),
        })

    def _run(self, config: dict):
        try:
            # Same order as SimpleCNN.py's main, so the default settings reproduce its result
            cnn.set_seed(config["seed"])
            model, criterion, optimizer, device = cnn._setup_model_for_training(
                cnn.num_class, config["lr"])
            initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.engine.set_initial(initial_state, config["seed"])
            self._log(f"Device: {device_name()}")
            self._log("Loading MNIST...")
            train_loader, val_loader, test_loader = cnn.get_mnist_dataloaders(
                config["batch_size"], cnn.validation_split)

            # Fixed validation images used to show how predictions change
            val_set = val_loader.dataset
            sample_images = torch.stack([val_set[i][0] for i in range(NUM_SAMPLES)])
            sample_labels = [int(val_set[i][1]) for i in range(NUM_SAMPLES)]
            sizes = {"train": len(train_loader.dataset), "validation": len(val_set),
                     "test": len(test_loader.dataset)}
            self._set(batches=len(train_loader), dataset_sizes=sizes, filters=conv1_filters(model),
                      samples={"images": [image_to_uint8(img[0]) for img in sample_images],
                               "labels": sample_labels})
            self._log(f"Data: {sizes['train']} train / {sizes['validation']} validation / "
                      f"{sizes['test']} test images, {len(train_loader)} batches per epoch")
            self._predict_samples(model, sample_images, sample_labels, device, epoch=0)

            for epoch in range(1, config["epochs"] + 1):
                self._set(stage="training", epoch=epoch, batch=0)
                train_acc, train_loss = self._train_epoch(model, criterion, optimizer,
                                                          train_loader, device)
                self._set(stage="validating")
                val_acc, val_loss = cnn.evaluate_model(model, criterion, val_loader, device)
                entry = {"epoch": epoch, "train_loss": round(train_loss, 5),
                         "train_acc": round(train_acc, 3), "val_loss": round(val_loss, 5),
                         "val_acc": round(val_acc, 3),
                         "time": round(time.time() - self.started_at, 1)}
                with self.lock:
                    self.history.append(entry)
                self._log(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Accuracy: "
                          f"{train_acc:.2f}% | Validation Loss: {val_loss:.4f}, "
                          f"Validation Accuracy: {val_acc:.2f}%")
                self._predict_samples(model, sample_images, sample_labels, device, epoch)
                self.engine.load_state(model.state_dict(),
                                       f"Epoch {epoch}/{config['epochs']} sonrası (eğitim sürüyor)")

            self._set(stage="testing")
            test_acc, test_loss = cnn.evaluate_model(model, criterion, test_loader, device)
            self._set(test={"loss": round(test_loss, 5), "acc": round(test_acc, 3)})
            self._log(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

            self._set(stage="saving")
            meta = {**config, "test_acc": round(test_acc, 3), "test_loss": round(test_loss, 5),
                    "val_acc": self.history[-1]["val_acc"], "device": device_name(),
                    "trained_at": datetime.now().isoformat(timespec="seconds")}
            self.engine.save_checkpoint(model.state_dict(), meta, initial_state)
            self.engine.load_state(model.state_dict(), "Yeni eğitilen model", meta)
            self._log(f"Model saved to {CHECKPOINT.relative_to(ROOT).as_posix()}")
            self._set(stage="done", state="done", finished_at=time.time())
            self._save_run(meta["trained_at"])
        except StopTraining:
            self._log("Training stopped by the user.")
            self._set(state="stopped", message="Eğitim durduruldu.")
        except Exception as exc:
            traceback.print_exc()
            self._log(f"Error: {type(exc).__name__}: {exc}")
            self._set(state="error", message=f"{type(exc).__name__}: {exc}")
        finally:
            self._set(finished_at=time.time())

    def _train_epoch(self, model, criterion, optimizer, data_loader, device):
        """Same steps as SimpleCNN.train_epoch, but reports progress after every batch."""
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for index, (inputs, labels) in enumerate(data_loader, start=1):
            if self.stop_event.is_set():
                raise StopTraining()
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()                 # 1. Reset gradients
            outputs = model(inputs)               # 2. Forward pass
            loss = criterion(outputs, labels)     # 3. Compute loss
            loss.backward()                       # 4. Backpropagation
            optimizer.step()                      # 5. Update parameters

            loss_value = loss.item()
            loss_sum += loss_value
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)
            with self.lock:
                self.batch = index
                self.step += 1
                self.batch_losses.append(round(loss_value, 5))
                if index % FILTER_EVERY == 0 or index == self.batches:
                    self.filters = conv1_filters(model)
        return 100 * correct / total, loss_sum / len(data_loader)

    def _save_run(self, trained_at: str):
        """Writes the finished run next to the checkpoint (atomically, so a crash leaves no half file)."""
        try:
            with self.lock:
                data = {"trained_at": trained_at, **{key: getattr(self, key) for key in RUN_FIELDS}}
            tmp = RUN_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            tmp.replace(RUN_FILE)
        except Exception as exc:  # The model is saved; losing the curves is not worth failing the run
            print(f"Could not save {RUN_FILE.name}: {exc}")

    def restore_run(self, meta: dict | None) -> bool:
        """Shows the run that produced the loaded checkpoint, if it was saved (and matches it)."""
        if not meta or not RUN_FILE.exists():
            return False
        try:
            data = json.loads(RUN_FILE.read_text(encoding="utf-8"))
            if data.get("trained_at") != meta.get("trained_at"):
                print(f"{RUN_FILE.name} belongs to another run than the checkpoint; not loaded.")
                return False
            with self.lock:
                self._reset(data["config"])
                for key in RUN_FIELDS:
                    setattr(self, key, data[key])
                self.run_id = 1
                self.state = "done"
                self.stage = "done"
                self.restored = {"trained_at": data["trained_at"]}
            self._log(f"Loaded the last run ({data['trained_at']}) from "
                      f"{RUN_FILE.relative_to(ROOT).as_posix()}")
            return True
        except Exception as exc:  # A broken file should not stop the app
            print(f"Could not load {RUN_FILE.name}: {exc}")
            return False

    def snapshot(self) -> dict:
        with self.lock:
            end = self.finished_at or time.time()
            return {
                "run_id": self.run_id,
                "state": self.state,
                "stage": self.stage,
                "running": self.is_running(),
                "message": self.message,
                "config": self.config,
                "epoch": self.epoch,
                "batch": self.batch,
                "batches": self.batches,
                "step": self.step,
                "dataset_sizes": self.dataset_sizes,
                "elapsed": round(end - self.started_at, 1) if self.started_at else None,
                "batch_losses": list(self.batch_losses),
                "history": list(self.history),
                "test": self.test,
                "filters": self.filters,
                "samples": self.samples,
                "sample_predictions": self.sample_predictions,
                "log": list(self.log),
                "restored": self.restored,
            }


#--------------------- Server ---------------------#
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml",
                 ".woff2": "font/woff2"}   # KaTeX fonts (static/vendor/katex)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets two servers share one port, so a busy port would go unnoticed
    allow_reuse_address = os.name != "nt"


def make_handler(session: TrainingSession, engine: InferenceEngine):
    defaults = {"epochs": cnn.num_epochs, "lr": cnn.learning_rate,
                "batch_size": cnn.batch_size, "seed": cnn.seed}
    device = device_name()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # The page polls often; keep the console readable
            pass

        def _send(self, body: bytes, content_type: str, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._send(body, "application/json; charset=utf-8", status)

        def _error(self, message: str, status: HTTPStatus):
            self._json({"error": message}, status)

        def _static(self, relative: str):
            target = (STATIC_DIR / relative).resolve()
            if STATIC_DIR not in target.parents or not target.is_file():
                return self._error("Bulunamadı.", HTTPStatus.NOT_FOUND)
            content_type = CONTENT_TYPES.get(target.suffix, "application/octet-stream")
            self._send(target.read_bytes(), content_type)

        def do_GET(self):
            try:
                url = urlparse(self.path)
                if url.path in ("/", "/index.html"):
                    return self._static("index.html")
                if url.path.startswith("/static/"):
                    return self._static(url.path[len("/static/"):])
                if url.path == "/api/status":
                    status = session.snapshot()
                    status.update(model=engine.info(), defaults=defaults, device=device,
                                  server_version=SERVER_VERSION)
                    return self._json(status)
                if url.path == "/api/source/SimpleCNN.py":   # read-only, for the Coding tab
                    body = (ROOT / "SimpleCNN.py").read_bytes()
                    return self._send(body, "text/plain; charset=utf-8")
                if url.path == "/api/theory/params":
                    which = parse_qs(url.query).get("model", ["trained"])[0]
                    if which not in engine.MODELS:
                        return self._error("Geçersiz model.", HTTPStatus.BAD_REQUEST)
                    return self._json(engine.theory_params(which))
                if url.path == "/api/test-samples":
                    count = int(parse_qs(url.query).get("n", ["12"])[0])
                    count = max(1, min(count, 48))
                    return self._json({"samples": engine.random_test_samples(count)})
                return self._error("Bulunamadı.", HTTPStatus.NOT_FOUND)
            except ConnectionError:
                pass
            except Exception as exc:
                traceback.print_exc()
                self._error(f"{type(exc).__name__}: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > 5_000_000:
                    return self._error("İstek çok büyük.", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    return self._error("Geçersiz JSON.", HTTPStatus.BAD_REQUEST)

                path = urlparse(self.path).path
                if path == "/api/train":
                    try:
                        config = parse_config(body)
                    except ValueError as exc:
                        return self._error(str(exc), HTTPStatus.BAD_REQUEST)
                    if not session.start(config):
                        return self._error("Eğitim zaten sürüyor.", HTTPStatus.CONFLICT)
                    return self._json({"ok": True})

                if path == "/api/stop":
                    session.stop()
                    return self._json({"ok": True})

                if path == "/api/predict":
                    which = body.get("model", "trained")
                    if which not in engine.MODELS:
                        return self._error("Geçersiz model.", HTTPStatus.BAD_REQUEST)
                    label = None
                    if "image" in body:
                        image = preprocess_drawing(str(body["image"]))
                        if image is None:
                            return self._error("Önce bir rakam çizin.", HTTPStatus.BAD_REQUEST)
                    elif "test_index" in body:
                        image, label = engine.test_image(int(body["test_index"]))
                    elif "pixels" in body:   # an already prepared 28x28 image (0-255), e.g. from the Theory tab
                        pixels = body["pixels"]
                        if not isinstance(pixels, list) or len(pixels) != 784:
                            return self._error("28×28 = 784 piksel bekleniyor.", HTTPStatus.BAD_REQUEST)
                        image = torch.tensor(pixels, dtype=torch.float32).clamp(0, 255).view(1, 1, 28, 28) / 255
                    else:
                        return self._error("Görüntü gönderilmedi.", HTTPStatus.BAD_REQUEST)
                    try:
                        result = engine.predict(image, which)
                    except ModelNotReady:
                        return self._error("Henüz eğitilmiş bir model yok.", HTTPStatus.CONFLICT)
                    result["label"] = label
                    return self._json(result)

                return self._error("Bulunamadı.", HTTPStatus.NOT_FOUND)
            except ConnectionError:
                pass
            except Exception as exc:
                traceback.print_exc()
                self._error(f"{type(exc).__name__}: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="SimpleCNN training & inference dashboard")
    parser.add_argument("--port", type=int, default=8000, help="first port to try (default 8000)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args()

    # Never let a character the console cannot show crash a print (e.g. output redirected to a file)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    os.chdir(ROOT)  # SimpleCNN.get_mnist_dataloaders uses ./data
    engine = InferenceEngine()
    if engine.load_checkpoint():
        print(f"Loaded saved model: {CHECKPOINT.relative_to(ROOT).as_posix()}")
    session = TrainingSession(engine)
    if session.restore_run(engine.meta):
        print(f"Loaded the last training run: {RUN_FILE.relative_to(ROOT).as_posix()}")

    server = None
    for port in range(args.port, args.port + 20):
        try:
            server = DashboardServer(("127.0.0.1", port), make_handler(session, engine))
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"No free port between {args.port} and {args.port + 19}.")

    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"SimpleCNN dashboard running at {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        session.stop()
        server.server_close()


if __name__ == "__main__":
    main()
