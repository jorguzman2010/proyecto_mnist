"""
Aplicación Streamlit para dibujar un número con el mouse y consultarlo en un modelo CNN MNIST.

Uso:
    streamlit run streamlit_app_mnist_dibujo.py

Requisito esperado:
    Tener entrenado el modelo y guardado como:
        artifacts_mnist_cnn/best_model.keras

Instalación:
    pip install streamlit tensorflow pillow numpy pandas streamlit-drawable-canvas
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model


DEFAULT_MODEL_PATH = "artifacts_mnist_cnn/best_model.keras"
CLASS_NAMES = [str(i) for i in range(10)]

st.set_page_config(
    page_title="Dibujar número MNIST",
    page_icon="✍️",
    layout="centered",
)


@st.cache_resource
def cargar_modelo(model_path: str):
    """Carga el modelo Keras desde disco y lo mantiene en caché."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {path.resolve()}")
    return load_model(path)


def extraer_digito(arr_gray: np.ndarray, margen: int = 20) -> Image.Image:
    """
    Recorta el área dibujada, la redimensiona manteniendo proporción y la centra en 28x28.
    El arreglo de entrada debe estar en escala de grises con fondo negro y trazo claro.
    """
    # Detectar pixeles dibujados. Umbral bajo para capturar bordes suavizados.
    coords = np.argwhere(arr_gray > 10)

    if coords.size == 0:
        # Imagen vacía: devolver un lienzo negro 28x28.
        return Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode="L")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    h, w = arr_gray.shape
    x_min = max(x_min - margen, 0)
    y_min = max(y_min - margen, 0)
    x_max = min(x_max + margen, w - 1)
    y_max = min(y_max + margen, h - 1)

    recorte = arr_gray[y_min : y_max + 1, x_min : x_max + 1]
    img = Image.fromarray(recorte.astype(np.uint8), mode="L")

    # Redimensionar a máximo 20x20 y centrar en 28x28, similar al formato MNIST.
    img.thumbnail((20, 20), Image.Resampling.LANCZOS)
    lienzo = Image.new("L", (28, 28), 0)
    pos_x = (28 - img.width) // 2
    pos_y = (28 - img.height) // 2
    lienzo.paste(img, (pos_x, pos_y))

    return lienzo


def preparar_canvas_para_mnist(image_rgba: np.ndarray) -> Tuple[np.ndarray, Image.Image]:
    """
    Convierte el resultado del canvas en entrada para MNIST.
    Retorna:
    - tensor con shape (1, 28, 28, 1)
    - imagen procesada 28x28 para visualización
    """
    img = Image.fromarray(image_rgba.astype("uint8"), mode="RGBA").convert("L")

    # El canvas se configura con fondo negro y trazo blanco.
    arr = np.array(img).astype(np.uint8)
    procesada = extraer_digito(arr)

    entrada = np.array(procesada).astype("float32") / 255.0
    entrada = np.expand_dims(entrada, axis=(0, -1))
    return entrada, procesada


def predecir(modelo, entrada: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """Retorna clase predicha, confianza y vector de probabilidades."""
    probs = modelo.predict(entrada, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confianza = float(np.max(probs))
    return pred_idx, confianza, probs


st.title("✍️ Dibujar un número y consultar el modelo")
st.write(
    "Dibuja un dígito del 0 al 9 con el mouse. El sistema lo transforma a formato MNIST y consulta el modelo entrenado."
)

with st.sidebar:
    st.header("Configuración")
    model_path = st.text_input("Ruta del modelo", value=DEFAULT_MODEL_PATH)
    stroke_width = st.slider("Grosor del lápiz", min_value=8, max_value=35, value=18, step=1)
    canvas_size = st.slider("Tamaño del lienzo", min_value=220, max_value=420, value=280, step=20)
    st.caption("Primero entrena el modelo con `python training_pipeline_mnist_cnn.py`.")

try:
    modelo = cargar_modelo(model_path)
    st.sidebar.success("Modelo cargado correctamente")
except Exception:
    st.sidebar.error("No se pudo cargar el modelo")
    st.error(
        "No encontré el modelo. Primero ejecuta el pipeline de entrenamiento o corrige la ruta en la barra lateral."
    )
    st.code("python training_pipeline_mnist_cnn.py", language="bash")
    st.stop()

st.subheader("Lienzo")
st.caption("Dibuja con trazo blanco sobre fondo negro. Usa el botón de abajo para borrar.")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=canvas_size,
    width=canvas_size,
    drawing_mode="freedraw",
    key="canvas_mnist",
)

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    consultar = st.button("Consultar modelo", type="primary", use_container_width=True)
with col_btn2:
    st.write("Para borrar: usa el ícono de papelera del lienzo o recarga la página.")

if consultar:
    if canvas_result.image_data is None:
        st.warning("Primero dibuja un número en el lienzo.")
        st.stop()

    entrada, procesada = preparar_canvas_para_mnist(canvas_result.image_data)
    pred_idx, confianza, probs = predecir(modelo, entrada)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen procesada 28x28")
        st.image(procesada.resize((140, 140), Image.Resampling.NEAREST), caption="Entrada real al modelo")
    with col2:
        st.subheader("Resultado")
        st.metric("Predicción", CLASS_NAMES[pred_idx])
        st.metric("Confianza", f"{confianza:.2%}")

    st.subheader("Probabilidades por clase")
    df_probs = pd.DataFrame({"Dígito": CLASS_NAMES, "Probabilidad": probs})
    st.bar_chart(df_probs, x="Dígito", y="Probabilidad")

    with st.expander("Ver probabilidades exactas"):
        st.dataframe(df_probs, use_container_width=True, hide_index=True)
else:
    st.info("Dibuja un número y presiona **Consultar modelo**.")

st.divider()
st.caption(
    "Consejo: dibuja el número grande, centrado y con un trazo grueso. MNIST funciona mejor con números claros sobre fondo negro."
)
