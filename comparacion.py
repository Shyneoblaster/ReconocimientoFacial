"""
MÓDULO 03 — Comparación Vectorial
Implementación propia con NumPy (sin funciones automáticas de librerías)

Métrica principal: DISTANCIA EUCLIDIANA
  - Rango: 0 a 2 (con vectores L2-normalizados)
  - 0 = idénticos | mayor = más distintos
  - Se busca el valor MENOR (más cercano)

Umbral dinámico:
  - Se calcula como el promedio de las distancias entre
    las dos fotos de registro de cada alumno multiplicado por un factor.
  - Guardado en embeddings/umbral.pkl al momento de registrar alumnos.

MEJORAS v2:
  - UMBRAL_DEFAULT bajado a 0.80 (era 1.10) para el modelo gaunernst
    cuyos embeddings L2-normalizados producen distancias en el rango 0–2.
  - UMBRAL_MIN bajado a 0.50 (era 0.50, se documenta explícitamente).
  - FACTOR_MARGEN reducido a 1.8 (era 2.5): el preprocesamiento mejorado
    (CLAHE + margen de ROI) reduce la variabilidad entre fotos, por lo
    que ya no se necesita un factor tan permisivo.
  - Se agrega UMBRAL_MAX = 1.00 para evitar umbrales demasiado altos
    que aceptarían a cualquier persona.
"""

import numpy as np
import pickle
import os


# ── Rutas ──────────────────────────────────────────────────────
RUTA_BD     = os.path.join(os.path.dirname(__file__), "embeddings", "base_datos.pkl")
RUTA_UMBRAL = os.path.join(os.path.dirname(__file__), "embeddings", "umbral.pkl")

# ── Límites del umbral ─────────────────────────────────────────
# Con vectores L2-normalizados y el modelo gaunernst/vit_small:
#   - dist < 0.60: prácticamente la misma foto
#   - dist 0.60–0.85: misma persona, condiciones distintas  ← zona de trabajo
#   - dist > 1.00: personas diferentes
UMBRAL_DEFAULT = 0.80   # umbral si no hay umbral calculado o es inválido
UMBRAL_MIN     = 0.80   # no aceptar umbrales por debajo de esto
UMBRAL_MAX     = 1.00   # no aceptar umbrales por encima de esto


# ──────────────────────────────────────────────────────────────
# MÉTRICA PRINCIPAL: DISTANCIA EUCLIDIANA
# Fórmula: d(A,B) = sqrt( sum( (Ai - Bi)^2 ) )
# ──────────────────────────────────────────────────────────────
def distancia_euclidiana(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distancia euclidiana implementada manualmente con NumPy.
    Fórmula: d(A,B) = sqrt( sum_i( (A_i - B_i)^2 ) )
    Rango: 0 a 2 con vectores L2-normalizados (0 = idénticos)

    Args:
        a: vector numpy de forma (512,)
        b: vector numpy de forma (512,)

    Returns:
        float >= 0
    """
    a = a.flatten()
    b = b.flatten()
    return float(np.sqrt(np.sum((a - b) ** 2)))


# ──────────────────────────────────────────────────────────────
# MÉTRICA ALTERNATIVA: SIMILITUD COSENO (para comparación en reporte)
# Fórmula: cos(A,B) = (A · B) / (||A|| * ||B||)
# ──────────────────────────────────────────────────────────────
def similitud_coseno(a: np.ndarray, b: np.ndarray) -> float:
    """
    Similitud coseno implementada manualmente con NumPy.

    Args:
        a: vector numpy de forma (512,)
        b: vector numpy de forma (512,)

    Returns:
        float entre 0 y 1  (1 = idénticos)
    """
    a = a.flatten()
    b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ──────────────────────────────────────────────────────────────
# UMBRAL DINÁMICO
# ──────────────────────────────────────────────────────────────
def calcular_umbral(bd: dict) -> float:
    """
    Calcula el umbral automáticamente como el promedio de las
    distancias entre foto1 y foto2 de cada alumno, multiplicado
    por un factor de margen para tolerar variación de iluminación.

    Con el preprocesamiento mejorado (CLAHE + margen de ROI), la
    distancia entre las dos fotos de un mismo alumno suele ser
    0.25–0.50. El factor 1.8 da un umbral ~0.45–0.90, que queda
    dentro de los límites [UMBRAL_MIN, UMBRAL_MAX].

    Fórmula:
        umbral = clip(promedio(dist(foto1, foto2)) * FACTOR, MIN, MAX)

    Args:
        bd: diccionario con embeddings de dos fotos por alumno

    Returns:
        float — umbral calculado y limitado al rango válido
    """
    FACTOR_MARGEN = 1.8

    distancias = []
    for nombre, fotos in bd.items():
        if "foto1" in fotos and "foto2" in fotos:
            ref1  = fotos["foto1"]
            ref2  = fotos["foto2"]
            ref   = ref1 + ref2
            norma = np.linalg.norm(ref)
            ref   = ref / norma if norma > 0 else ref

            n1    = np.linalg.norm(ref1)
            ref1n = ref1 / n1 if n1 > 0 else ref1
            dist  = distancia_euclidiana(ref1n, ref)
            distancias.append(dist)
            print(f"[comparacion] dist({nombre} foto1↔ref) = {dist:.4f}")

    if not distancias:
        print("[comparacion] ⚠️  No hay pares de fotos — usando umbral por defecto")
        return UMBRAL_DEFAULT

    promedio = float(np.mean(distancias))
    umbral   = promedio * FACTOR_MARGEN
    umbral   = float(np.clip(umbral, UMBRAL_MIN, UMBRAL_MAX))
    print(f"[comparacion] Promedio distancias: {promedio:.4f} × {FACTOR_MARGEN} = {umbral:.4f}")
    return umbral


def guardar_umbral(umbral: float):
    """
    Guarda el umbral calculado en embeddings/umbral.pkl.
    Aplica el clip [UMBRAL_MIN, UMBRAL_MAX] antes de guardar.
    """
    umbral_final = float(np.clip(umbral, UMBRAL_MIN, UMBRAL_MAX))
    if umbral_final != umbral:
        print(f"[comparacion] Umbral ajustado {umbral:.4f} → {umbral_final:.4f} "
              f"(límites [{UMBRAL_MIN}, {UMBRAL_MAX}])")
    os.makedirs(os.path.dirname(RUTA_UMBRAL), exist_ok=True)
    with open(RUTA_UMBRAL, "wb") as f:
        pickle.dump(umbral_final, f)
    print(f"[comparacion] Umbral guardado: {umbral_final:.4f}")


def cargar_umbral() -> float:
    """
    Retorna el umbral a usar para la identificación.
    Si existe umbral.pkl y está dentro del rango válido, lo usa.
    Si no, usa UMBRAL_DEFAULT.
    """
    if os.path.exists(RUTA_UMBRAL):
        with open(RUTA_UMBRAL, "rb") as f:
            umbral = pickle.load(f)
        if UMBRAL_MIN <= umbral <= UMBRAL_MAX:
            print(f"[comparacion] Umbral cargado: {umbral:.4f}")
            return float(umbral)
        else:
            print(f"[comparacion] Umbral guardado ({umbral:.4f}) fuera de rango "
                  f"[{UMBRAL_MIN}, {UMBRAL_MAX}] — usando default {UMBRAL_DEFAULT}")
            return UMBRAL_DEFAULT
    print(f"[comparacion] Usando umbral por defecto: {UMBRAL_DEFAULT}")
    return UMBRAL_DEFAULT


# ──────────────────────────────────────────────────────────────
# BASE DE DATOS
# ──────────────────────────────────────────────────────────────
def cargar_base_datos() -> dict:
    """
    Carga el archivo .pkl con los embeddings registrados.

    Returns:
        dict: { "Nombre": { "foto1": np.ndarray(512,), "foto2": np.ndarray(512,) }, ... }
    """
    if not os.path.exists(RUTA_BD):
        print("[comparacion] ⚠️  No existe base_datos.pkl — ejecuta captura_alumnos.py primero")
        return {}
    with open(RUTA_BD, "rb") as f:
        bd = pickle.load(f)
    print(f"[comparacion] Base de datos cargada — {len(bd)} alumnos registrados")
    return bd


def guardar_base_datos(bd: dict):
    """
    Guarda el diccionario de embeddings en el archivo .pkl.

    Args:
        bd: dict { "Nombre": { "foto1": np.ndarray, "foto2": np.ndarray }, ... }
    """
    os.makedirs(os.path.dirname(RUTA_BD), exist_ok=True)
    with open(RUTA_BD, "wb") as f:
        pickle.dump(bd, f)
    print(f"[comparacion] Base de datos guardada — {len(bd)} alumnos")


# ──────────────────────────────────────────────────────────────
# IDENTIFICACIÓN
# ──────────────────────────────────────────────────────────────
def identificar(embedding_detectado: np.ndarray, bd: dict, umbral: float):
    """
    Compara el embedding del rostro detectado contra el promedio
    de las dos fotos registradas de cada alumno.

    Proceso:
      1. Para cada alumno calcular su embedding representativo
         como promedio de foto1 y foto2 (renormalizado L2)
      2. Calcular distancia euclidiana al embedding detectado
      3. Encontrar el alumno con MENOR distancia
      4. Si dist < umbral → alumno identificado
      5. Si dist >= umbral → "Desconocido"

    Args:
        embedding_detectado: vector (512,) del rostro en cámara
        bd:                  diccionario con embeddings registrados
        umbral:              distancia máxima para aceptar

    Returns:
        nombre (str):     nombre del alumno o "Desconocido"
        distancia (float): distancia euclidiana del mejor candidato
    """
    if not bd:
        return "Desconocido", 999.0

    mejor_nombre    = "Desconocido"
    menor_distancia = float("inf")

    for nombre, fotos in bd.items():
        # Promedio de ambas fotos + renormalización L2 obligatoria
        embedding_ref = fotos["foto1"] + fotos["foto2"]
        norma = np.linalg.norm(embedding_ref)
        if norma > 0:
            embedding_ref = embedding_ref / norma   # renormalizar L2

        dist = distancia_euclidiana(embedding_detectado, embedding_ref)

        if dist < menor_distancia:
            menor_distancia = dist
            mejor_nombre    = nombre

    if menor_distancia < umbral:
        return mejor_nombre, menor_distancia
    else:
        return "Desconocido", menor_distancia