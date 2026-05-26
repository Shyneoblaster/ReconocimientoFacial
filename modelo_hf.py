"""
MÓDULO 02 — Extracción de Embeddings
Modelo: gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
Librería: timm (Hugging Face)

MEJORAS v2:
  - preprocesar_rostro(): expande el ROI del Haar Cascade con margen del 20%
    para capturar mejor el rostro completo sin cortar cejas ni barbilla.
  - Normalización de iluminación con CLAHE antes de extraer el embedding,
    lo que reduce la diferencia entre fotos de registro y cámara en tiempo real.
  - extraer_embedding() acepta opcionalmente (x, y, w, h) para aplicar
    el margen directamente sobre el frame completo.
"""

import timm
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2


# ── Constantes del modelo ──────────────────────────────────────
MODELO_ID  = "hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3"
INPUT_SIZE = 112       # el modelo requiere exactamente 112x112 px
EMBED_DIM  = 512       # dimensión del vector de salida

# Margen que se añade alrededor del ROI detectado por Haar Cascade.
# 0.20 = 20% extra en cada lado → captura mejor el rostro completo.
MARGEN_ROI = 0.20


def cargar_modelo():
    """
    Descarga (primera vez) y carga el modelo desde Hugging Face Hub.
    Retorna el modelo en modo evaluación listo para inferencia.
    """
    print(f"[modelo_hf] Cargando modelo: {MODELO_ID}")
    modelo = timm.create_model(MODELO_ID, pretrained=True).eval()
    print(f"[modelo_hf] Modelo cargado — output: {EMBED_DIM} dims")
    return modelo


def preprocesar_rostro(frame_bgr: np.ndarray,
                       x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Recorta el rostro del frame completo aplicando un margen del 20%
    para evitar que el recorte sea demasiado ajustado.

    El Haar Cascade devuelve un bounding box que a veces corta cejas,
    barbilla o incluye mucho cuello. Expandir el ROI mejora la calidad
    del embedding y reduce las distancias euclidiana en tiempo real.

    Args:
        frame_bgr: imagen completa de la cámara (BGR, numpy array)
        x, y, w, h: coordenadas del bounding box del Haar Cascade

    Returns:
        numpy array BGR del rostro recortado con margen
    """
    alto_frame, ancho_frame = frame_bgr.shape[:2]

    margen_x = int(w * MARGEN_ROI)
    margen_y = int(h * MARGEN_ROI)

    x1 = max(0, x - margen_x)
    y1 = max(0, y - margen_y)
    x2 = min(ancho_frame, x + w + margen_x)
    y2 = min(alto_frame,  y + h + margen_y)

    return frame_bgr[y1:y2, x1:x2]


def normalizar_iluminacion(imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)
    al canal de luminosidad para reducir el efecto de la iluminación
    desigual entre las fotos de registro y la cámara en tiempo real.

    Args:
        imagen_bgr: imagen BGR (numpy array)

    Returns:
        imagen BGR con iluminación normalizada
    """
    lab   = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Checar clipLimit para pruebas: 1.0, 1.2, 1.5, 2.0
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))
    l_eq  = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocesar_imagen(imagen_bgr: np.ndarray) -> torch.Tensor:
    """
    Convierte un recorte de rostro BGR (OpenCV) al tensor que espera el modelo.

    Pasos:
      1. Normalizar iluminación con CLAHE
      2. BGR → RGB
      3. Redimensionar a 112x112
      4. Normalizar al rango [-1, 1]  (requerido por CosFace/MS1MV3)
      5. Convertir a tensor (1, 3, 112, 112)

    Args:
        imagen_bgr: array numpy HxWx3 en formato BGR (salida de OpenCV)

    Returns:
        tensor de forma (1, 3, 112, 112)
    """
    # Normalizar iluminación antes de convertir
    imagen_bgr = normalizar_iluminacion(imagen_bgr)

    imagen_rgb = imagen_bgr[:, :, ::-1].copy()          # BGR → RGB
    pil_img    = Image.fromarray(imagen_rgb).resize(
        (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR
    )
    arr    = np.array(pil_img).astype(np.float32) / 255.0
    arr    = (arr - 0.5) / 0.5                           # normalización [-1, 1]
    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 112, 112)
    return tensor


def extraer_embedding(modelo, imagen_bgr: np.ndarray) -> np.ndarray:
    """
    Extrae el vector de características (embedding) de un rostro.

    Args:
        modelo:      modelo cargado con cargar_modelo()
        imagen_bgr:  recorte del rostro en formato BGR (numpy array)

    Returns:
        vector numpy de forma (512,) normalizado L2
    """
    tensor = preprocesar_imagen(imagen_bgr)
    with torch.no_grad():
        embedding = modelo(tensor)                      # (1, 512)
        embedding = F.normalize(embedding, dim=1)       # normalización L2
    return embedding.numpy().flatten()                  # (512,)