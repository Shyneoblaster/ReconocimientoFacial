========================================================
  SISTEMA DE ASISTENCIA POR RECONOCIMIENTO FACIAL
  Unidad 4 - Aplicaciones con Técnicas de IA
========================================================

MODELO UTILIZADO
  gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3
  Librería: timm (Hugging Face)
  Dataset:  MS1MV3 (5.17M imágenes, 93K identidades)
  Output:   512 dimensiones, métrica: Distancia Euclidiana

INSTALACIÓN
  pip install timm torch torchvision opencv-python openpyxl numpy Pillow huggingface_hub

ESTRUCTURA DEL PROYECTO
  main.py              → Punto de entrada. Integra todos los módulos.
  modelo_hf.py         → Módulo 02: carga del modelo y extracción de embeddings.
  comparacion.py       → Módulo 03: similitud coseno y euclidiana (NumPy manual).
  registro_excel.py    → Módulo 04: escritura en Excel con openpyxl.
  captura_alumnos.py   → Script de registro previo de alumnos.
  embeddings/          → Base de datos de vectores (base_datos.pkl)
  fotos_registro/      → Fotos de referencia de cada alumno.
  asistencias/         → Archivos Excel generados por sesión.

CÓMO USAR
  PASO 1 — Registrar alumnos (hacer UNA sola vez):
    python captura_alumnos.py
    - Escribe el nombre del alumno y presiona ENTER
    - Mira a la cámara y presiona ESPACIO para capturar
    - Repite para cada alumno (mínimo 10)
    - Escribe 'fin' para terminar

  PASO 2 — Iniciar el sistema de asistencia:
    python main.py
    - El sistema detecta rostros automáticamente
    - Muestra nombre y similitud en pantalla
    - Registra la asistencia en asistencias/asistencia_YYYY-MM-DD.xlsx
    - Presiona Q para salir

NOTAS TÉCNICAS
  - El modelo se descarga automáticamente la primera vez (~80MB)
  - El umbral de aceptación se ajusta en comparacion.py (UMBRAL_COSENO)
  - La similitud coseno va de 0 a 1 (1 = personas idénticas)
  - Umbral recomendado: 0.75 (ajustar experimentalmente)
