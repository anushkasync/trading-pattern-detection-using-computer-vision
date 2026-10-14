
import gc
import logging
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER

LOGGER.setLevel(logging.CRITICAL)

# --------------------------------------------------
# 1. Load configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "configs" / "yolov8n.yaml"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

DATA_YAML = config["data"]
RUNS_DIR = config["experiment"]["project"]
RUN_NAME = config["experiment"]["name"]
RUN_DIR = Path(RUNS_DIR) / RUN_NAME

# --------------------------------------------------
# 2. Configure MLflow
# --------------------------------------------------

# Store MLflow tracking data inside this project.
mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
)

# Group related training runs under one experiment.
mlflow.set_experiment("NiftyCV Object Detection")

model = YOLO(config["model"])

# --------------------------------------------------
# 3. Log metrics after each epoch
# --------------------------------------------------

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

    # Record metrics in MLflow, indexed by epoch.
    mlflow.log_metrics(
        {
            "train_loss": float(train_loss),
            "val_map50_percent": float(val_map50),
            "val_map50_95_percent": float(val_map5095),
        },
        step=epoch,
    )

    print(
        f"Epoch [{epoch}/{total}] "
        f"Train Loss: {train_loss:.4f}  "
        f"Val mAP50: {val_map50:.1f}%  "
        f"Val mAP50-95: {val_map5095:.1f}%"
    )

    gc.collect()


model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

# --------------------------------------------------
# 4. Train and track this experiment
# --------------------------------------------------

print("Starting YOLO training with MLflow tracking...")

with mlflow.start_run(run_name=RUN_NAME) as run:

    # Log configuration parameters.
    mlflow.log_param("model", config["model"])
    mlflow.log_param("data", DATA_YAML)

    mlflow.log_params(
        {
            key: str(value)
            for key, value in config["training"].items()
        }
    )

    # Keep a copy of the requested configuration.
    mlflow.log_artifact(str(CONFIG_PATH), artifact_path="config")
    mlflow.log_artifact(str(ROOT_DIR / "uv.lock"))
    print("Lockfile logged")

    # Run your existing training pipeline.
    model.train(
        data=DATA_YAML,
        **config["training"],
        project=RUNS_DIR,
        name=RUN_NAME,
        exist_ok=config["experiment"]["exist_ok"],
    )

    best_model_path = RUN_DIR / "weights" / "best.pt"

    print(f"\nTraining complete. Best model: {best_model_path}")

    # Evaluate the best checkpoint on the validation split.
    best_model = YOLO(str(best_model_path))

    metrics = best_model.val(
        data=DATA_YAML,
        split="val",
        verbose=False,
    )

    final_metrics = {
        "final_val_map50_percent": float(metrics.box.map50 * 100),
        "final_val_map50_95_percent": float(metrics.box.map * 100),
        "final_val_precision_percent": float(metrics.box.mp * 100),
        "final_val_recall_percent": float(metrics.box.mr * 100),
    }

    mlflow.log_metrics(final_metrics)

    print("\nFinal validation metrics:")
    for name, value in final_metrics.items():
        print(f"{name}: {value:.2f}")

    # Log selected training and evaluation outputs.
    artifact_names = [
        "results.csv",
        "args.yaml",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
    ]

    for filename in artifact_names:
        artifact_path = RUN_DIR / filename
        if artifact_path.is_file():
            mlflow.log_artifact(str(artifact_path), artifact_path="training_outputs")

    # Preserve the best checkpoint with this MLflow run.
    if best_model_path.is_file():
        mlflow.log_artifact(
            str(best_model_path),
            artifact_path="model_checkpoint",
        )

    print(f"\nMLflow run ID: {run.info.run_id}")
    print("Training parameters, metrics, and selected artifacts logged.")