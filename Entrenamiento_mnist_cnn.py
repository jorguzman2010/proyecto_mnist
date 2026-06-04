"""
Pipeline de entrenamiento CNN para MNIST
========================================

Este script implementa un pipeline completo y reutilizable:

1. Configuración de hiperparámetros
2. Carga y preparación del dataset MNIST
3. Construcción de una CNN
4. Compilación del modelo
5. Entrenamiento con callbacks
6. Evaluación en test
7. Guardado de artefactos:
   - mejor modelo .keras
   - historial de entrenamiento .csv
   - métricas .json
   - matriz de confusión .png
   - reporte de clasificación .txt

Uso recomendado:
    python training_pipeline_mnist_cnn.py

Uso con parámetros:
    python training_pipeline_mnist_cnn.py --epochs 15 --batch_size 128 --learning_rate 0.001
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class TrainingConfig:
    """Configuración central del pipeline de entrenamiento."""
    model_name: str = "mnist_cnn"
    artifacts_dir: str = "artifacts_mnist_cnn"
    random_state: int = 42
    validation_size: float = 0.10
    image_height: int = 28
    image_width: int = 28
    channels: int = 1
    num_classes: int = 10
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 10
    patience: int = 3
    monitor_metric: str = "val_loss"


def set_seed(seed: int) -> None:
    """Fija semillas para favorecer reproducibilidad."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_artifacts_dir(config: TrainingConfig) -> Path:
    """Crea la carpeta donde se guardarán los resultados del pipeline."""
    artifacts_path = Path(config.artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    return artifacts_path


def load_and_prepare_data(
    config: TrainingConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga MNIST, normaliza imágenes, agrega canal y separa validación.

    Retorna:
        x_train, x_val, x_test, y_train, y_val, y_test
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalización: de 0-255 a 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Agregar canal: (n, 28, 28) -> (n, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=y_train,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def build_model(config: TrainingConfig) -> keras.Model:
    """Construye una CNN simple para clasificación multiclase en MNIST."""
    input_shape = (config.image_height, config.image_width, config.channels)

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),

            layers.Conv2D(32, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(64, kernel_size=3, activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=2),

            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.25),

            layers.Dense(config.num_classes, activation="softmax"),
        ],
        name=config.model_name,
    )

    return model


def compile_model(model: keras.Model, config: TrainingConfig) -> keras.Model:
    """Compila el modelo con Adam y sparse categorical crossentropy."""
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_callbacks(config: TrainingConfig, artifacts_path: Path) -> list:
    """Define callbacks de entrenamiento."""
    checkpoint_path = artifacts_path / "best_model.keras"

    early_stopping = keras.callbacks.EarlyStopping(
        monitor=config.monitor_metric,
        patience=config.patience,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=config.monitor_metric,
        save_best_only=True,
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=config.monitor_metric,
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    )

    return [early_stopping, checkpoint, reduce_lr]


def train_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainingConfig,
    artifacts_path: Path,
) -> keras.callbacks.History:
    """Entrena el modelo y guarda el historial."""
    callbacks = get_callbacks(config, artifacts_path)

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(artifacts_path / "training_history.csv", index=False)

    return history


def plot_training_curves(history: keras.callbacks.History, artifacts_path: Path) -> None:
    """Guarda curvas de pérdida y accuracy."""
    history_df = pd.DataFrame(history.history)
    epochs_range = range(1, len(history_df) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_df["loss"], label="Pérdida entrenamiento")
    plt.plot(epochs_range, history_df["val_loss"], label="Pérdida validación")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title("Curva de pérdida")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_path / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_df["accuracy"], label="Accuracy entrenamiento")
    plt.plot(epochs_range, history_df["val_accuracy"], label="Accuracy validación")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")
    plt.title("Curva de accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts_path / "accuracy_curve.png", dpi=150)
    plt.close()


def evaluate_model(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    artifacts_path: Path,
) -> dict:
    """Evalúa el modelo y guarda métricas, matriz de confusión y reporte."""
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_text = classification_report(y_test, y_pred)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
    }

    with open(artifacts_path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    with open(artifacts_path / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Matriz de confusión - MNIST")
    plt.xlabel("Etiqueta predicha")
    plt.ylabel("Etiqueta real")
    plt.colorbar()

    tick_marks = np.arange(10)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(artifacts_path / "confusion_matrix.png", dpi=150)
    plt.close()

    return metrics


def save_config(config: TrainingConfig, artifacts_path: Path) -> None:
    """Guarda la configuración del experimento."""
    with open(artifacts_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=4, ensure_ascii=False)


def run_training_pipeline(config: TrainingConfig) -> dict:
    """Ejecuta el pipeline completo de entrenamiento."""
    set_seed(config.random_state)
    artifacts_path = create_artifacts_dir(config)
    save_config(config, artifacts_path)

    print("1) Cargando y preparando datos...")
    x_train, x_val, x_test, y_train, y_val, y_test = load_and_prepare_data(config)

    print("2) Construyendo modelo...")
    model = build_model(config)
    model = compile_model(model, config)
    model.summary()

    print("3) Entrenando modelo...")
    history = train_model(model, x_train, y_train, x_val, y_val, config, artifacts_path)

    print("4) Guardando curvas de entrenamiento...")
    plot_training_curves(history, artifacts_path)

    print("5) Evaluando modelo...")
    metrics = evaluate_model(model, x_test, y_test, artifacts_path)

    final_model_path = artifacts_path / "final_model.keras"
    model.save(final_model_path)

    print("\nPipeline finalizado correctamente.")
    print(f"Artefactos guardados en: {artifacts_path.resolve()}")
    print("Métricas:", metrics)

    return metrics


def parse_args() -> TrainingConfig:
    """Permite ejecutar el pipeline desde consola con parámetros."""
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento CNN para MNIST")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--artifacts_dir", type=str, default="artifacts_mnist_cnn")
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    return TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        artifacts_dir=args.artifacts_dir,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    cfg = parse_args()
    run_training_pipeline(cfg)
