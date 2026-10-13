import gc
import logging
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER

LOGGER.setLevel(logging.CRITICAL)

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "configs" / "yolov8n.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

DATA_YAML = config["data"]
RUNS_DIR = config["experiment"]["project"]
RUN_NAME = config["experiment"]["name"]

model = YOLO(config["model"])

_seen = set()


def on_fit_epoch_end(trainer):
    epoch = trainer.epoch + 1
    total = trainer.epochs

    if epoch in _seen or epoch > total:
        return

    _seen.add(epoch)

    train_loss = (
        trainer.loss.detach().item()
        if hasattr(trainer.loss, "detach")
        else float(trainer.loss)
    )

    val_map50 = 0.0
    val_map5095 = 0.0

    if hasattr(trainer, "validator") and trainer.validator is not None:
        metrics = getattr(trainer.validator, "metrics", None)

        if metrics and hasattr(metrics, "box"):
            val_map50 = metrics.box.map50 * 100
            val_map5095 = metrics.box.map * 100

    print(
        f"Epoch [{epoch}/{total}] "
        f"Train Loss: {train_loss:.4f}  "
        f"Val mAP50: {val_map50:.1f}%  "
        f"Val mAP50-95: {val_map5095:.1f}%"
    )

    gc.collect()


model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

print("Starting training pipeline v3 (yolov8n + v2 hyperparams)...")

model.train(
    data=DATA_YAML,
    **config["training"],
    project=RUNS_DIR,
    name=RUN_NAME,
    exist_ok=config["experiment"]["exist_ok"],
)

best_model_path = Path(RUNS_DIR) / RUN_NAME / "weights" / "best.pt"

print(f"\nTraining complete. Best model saved at {best_model_path}")

best_model = YOLO(str(best_model_path))

metrics = best_model.val(
    data=DATA_YAML,
    split="val",
    verbose=False,
)

print(
    f"\nValidation mAP50: {metrics.box.map50 * 100:.1f}%  "
    f"mAP50-95: {metrics.box.map * 100:.1f}%"
)