"""
CAPTURA DE ALUMNOS — Sistema híbrido de registro
Dos modos:
  1. Cargar fotos desde directorio (archivos ya almacenados)
  2. Registrar nuevo alumno con formulario + cámara en vivo

Ejecutar antes de usar main.py:
    python captura_alumnos.py
"""

import cv2
import pickle
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading

from modelo_hf   import cargar_modelo, extraer_embedding, preprocesar_rostro
from comparacion import (guardar_base_datos, guardar_umbral,
                         calcular_umbral, distancia_euclidiana, RUTA_BD)

CARPETA_FOTOS = os.path.join(os.path.dirname(__file__), "fotos_registro")
DETECTOR_CARA = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ──────────────────────────────────────────────────────────────
# VENTANA PRINCIPAL — Menú de selección de modo
# ──────────────────────────────────────────────────────────────
class MenuPrincipal:
    def __init__(self, modelo, bd):
        self.modelo = modelo
        self.bd     = bd

        self.root = tk.Tk()
        self.root.title("Sistema de Asistencia — Registro de Alumnos")
        self.root.geometry("500x530")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        self._build_ui()

    def _build_ui(self):
        tk.Label(self.root, text="Face Recognition",
                 font=("Arial", 22, "bold"), fg="#e94560", bg="#1a1a2e").pack(pady=20)
        tk.Label(self.root, text="Sistema de Registro de Alumnos",
                 font=("Arial", 12), fg="#a8a8b3", bg="#1a1a2e").pack()

        self.lbl_info = tk.Label(
            self.root,
            text=f"Alumnos registrados: {len(self.bd)}",
            font=("Arial", 11), fg="#f5f5f5", bg="#1a1a2e"
        )
        self.lbl_info.pack(pady=15)

        frame_btns = tk.Frame(self.root, bg="#1a1a2e")
        frame_btns.pack(pady=10)

        btn_style = {"font": ("Arial", 12, "bold"), "width": 28,
                     "pady": 10, "cursor": "hand2", "bd": 0, "relief": "flat"}

        tk.Button(frame_btns, text="📂  Cargar fotos desde directorio",
                  bg="#16213e", fg="#f5f5f5",
                  command=self.abrir_carga_directorio,
                  **btn_style).pack(pady=8)

        tk.Button(frame_btns, text="📝  Registrar nuevo alumno",
                  bg="#e94560", fg="#ffffff",
                  command=self.abrir_registro_nuevo,
                  **btn_style).pack(pady=8)

        tk.Button(frame_btns, text="🗑️  Ver / Eliminar alumnos",
                  bg="#16213e", fg="#f5f5f5",
                  command=self.abrir_lista_alumnos,
                  **btn_style).pack(pady=8)

        tk.Button(frame_btns, text="⚠️  Eliminar TODOS los registros",
                  bg="#7b0000", fg="#ffffff",
                  command=self.eliminar_todos,
                  **btn_style).pack(pady=8)

        tk.Button(frame_btns, text="🔧  Recalcular umbral",
                  bg="#16213e", fg="#f5f5f5",
                  command=self.recalcular_umbral,
                  **btn_style).pack(pady=8)

        tk.Button(self.root, text="Salir",
                  bg="#1a1a2e", fg="#a8a8b3",
                  font=("Arial", 10), bd=0, cursor="hand2",
                  command=self.root.destroy).pack(pady=10)

    def actualizar_info(self):
        self.lbl_info.config(text=f"Alumnos registrados: {len(self.bd)}")

    def abrir_carga_directorio(self):
        VentanaCargaDirectorio(self.root, self.modelo, self.bd, self)

    def abrir_registro_nuevo(self):
        VentanaRegistroNuevo(self.root, self.modelo, self.bd, self)

    def abrir_lista_alumnos(self):
        VentanaListaAlumnos(self.root, self.bd, self)

    def eliminar_todos(self):
        if not self.bd:
            messagebox.showinfo("Aviso", "No hay alumnos registrados")
            return
        n = len(self.bd)
        if not messagebox.askyesno(
            "⚠️ Confirmar eliminación",
            f"¿Eliminar TODOS los {n} alumnos registrados?\n\n"
            "Esta acción no se puede deshacer.\n"
            "Se borrarán base_datos.pkl y umbral.pkl"
        ):
            return

        # Limpiar base de datos en memoria
        self.bd.clear()

        # Borrar archivos pkl
        import os as _os
        from comparacion import RUTA_BD, RUTA_UMBRAL
        for ruta in [RUTA_BD, RUTA_UMBRAL]:
            if _os.path.exists(ruta):
                _os.remove(ruta)

        self.actualizar_info()
        messagebox.showinfo("✅ Listo", f"Se eliminaron {n} alumnos correctamente.")

    def recalcular_umbral(self):
        if not self.bd:
            messagebox.showwarning("Aviso", "No hay alumnos registrados")
            return
        from comparacion import calcular_umbral, guardar_umbral
        umbral = calcular_umbral(self.bd)
        guardar_umbral(umbral)
        messagebox.showinfo(
            "Umbral actualizado",
            f"Nuevo umbral: {umbral:.4f}"
            f"(promedio distancias × 1.8)"
        )

    def run(self):
        self.root.mainloop()


# ──────────────────────────────────────────────────────────────
# MODO 1 — Cargar fotos desde directorio
# ──────────────────────────────────────────────────────────────
class VentanaCargaDirectorio:
    def __init__(self, parent, modelo, bd, menu):
        self.modelo = modelo
        self.bd     = bd
        self.menu   = menu

        self.win = tk.Toplevel(parent)
        self.win.title("Cargar fotos desde directorio")
        self.win.geometry("520x400")
        self.win.configure(bg="#1a1a2e")
        self.win.grab_set()

        self._build_ui()

    def _build_ui(self):
        tk.Label(self.win, text="Cargar fotos desde directorio",
                 font=("Arial", 14, "bold"), fg="#e94560", bg="#1a1a2e").pack(pady=15)

        tk.Label(self.win,
                 text="El directorio debe tener subcarpetas con el nombre\n"
                      "de cada alumno, cada una con al menos 2 fotos.",
                 font=("Arial", 10), fg="#a8a8b3", bg="#1a1a2e",
                 justify="center").pack(pady=5)

        tk.Label(self.win,
                 text="Ejemplo:\n"
                      "  fotos/\n"
                      "    Juan Perez/foto1.jpg, foto2.jpg\n"
                      "    Maria Lopez/foto1.jpg, foto2.jpg",
                 font=("Courier", 9), fg="#f5f5f5", bg="#16213e",
                 justify="left", padx=10, pady=8).pack(padx=20, fill="x")

        frame_dir = tk.Frame(self.win, bg="#1a1a2e")
        frame_dir.pack(pady=12, padx=20, fill="x")

        self.var_dir = tk.StringVar()
        tk.Entry(frame_dir, textvariable=self.var_dir,
                 font=("Arial", 10), bg="#16213e", fg="#f5f5f5",
                 insertbackground="white", relief="flat").pack(side="left", fill="x", expand=True, ipady=6)
        tk.Button(frame_dir, text="Buscar",
                  bg="#e94560", fg="white", font=("Arial", 10),
                  bd=0, cursor="hand2", padx=10,
                  command=self._seleccionar_dir).pack(side="left", padx=(8, 0))

        self.log = tk.Text(self.win, height=7, bg="#16213e", fg="#a8ffb0",
                           font=("Courier", 9), relief="flat", state="disabled")
        self.log.pack(padx=20, fill="x")

        frame_btns = tk.Frame(self.win, bg="#1a1a2e")
        frame_btns.pack(pady=12)

        tk.Button(frame_btns, text="Procesar fotos",
                  bg="#e94560", fg="white", font=("Arial", 11, "bold"),
                  bd=0, cursor="hand2", padx=20, pady=8,
                  command=self._procesar).pack(side="left", padx=8)

        tk.Button(frame_btns, text="Cerrar",
                  bg="#16213e", fg="#a8a8b3", font=("Arial", 11),
                  bd=0, cursor="hand2", padx=20, pady=8,
                  command=self.win.destroy).pack(side="left", padx=8)

    def _seleccionar_dir(self):
        d = filedialog.askdirectory(title="Selecciona el directorio de fotos")
        if d:
            self.var_dir.set(d)

    def _log(self, texto):
        self.log.config(state="normal")
        self.log.insert("end", texto + "\n")
        self.log.see("end")
        self.log.config(state="disabled")
        self.win.update()

    def _procesar(self):
        directorio = self.var_dir.get().strip()
        if not directorio or not os.path.isdir(directorio):
            messagebox.showerror("Error", "Selecciona un directorio válido")
            return

        self._log("Procesando fotos...")
        os.makedirs(CARPETA_FOTOS, exist_ok=True)
        procesados = 0

        for nombre_alumno in os.listdir(directorio):
            ruta_alumno = os.path.join(directorio, nombre_alumno)
            if not os.path.isdir(ruta_alumno):
                continue

            fotos = sorted([f for f in os.listdir(ruta_alumno)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))])

            if len(fotos) < 2:
                self._log(f"  ⚠️  {nombre_alumno}: necesita al menos 2 fotos — omitido")
                continue

            embeddings_alumno = []
            for foto in fotos:          # TODAS las fotos disponibles
                img_bgr = cv2.imread(os.path.join(ruta_alumno, foto))
                if img_bgr is None:
                    continue

                # Intentar detectar y recortar el rostro
                gris  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                caras = DETECTOR_CARA.detectMultiScale(gris, 1.1, 5, minSize=(40,40))

                if len(caras) > 0:
                    x, y, w, h = caras[0]
                    # Usar preprocesar_rostro para expandir el ROI con margen
                    rostro = preprocesar_rostro(img_bgr, x, y, w, h)
                    self._log(f"    → {foto} — rostro detectado ✅")
                else:
                    # Sin detección: usar imagen completa
                    rostro = img_bgr
                    self._log(f"    → {foto} — sin detección, usando imagen completa ⚠️")

                emb = extraer_embedding(self.modelo, rostro)
                embeddings_alumno.append(emb)

            if len(embeddings_alumno) >= 2:
                # Dividir en dos mitades y promediar cada una
                mitad = len(embeddings_alumno) // 2
                foto1 = np.mean(embeddings_alumno[:mitad], axis=0)
                foto2 = np.mean(embeddings_alumno[mitad:], axis=0)
                self.bd[nombre_alumno] = {"foto1": foto1, "foto2": foto2}
                self._log(f"  ✅ {nombre_alumno} registrado ({len(embeddings_alumno)} fotos promediadas)")
                procesados += 1
            else:
                self._log(f"  ❌ {nombre_alumno}: error al leer fotos")

        if procesados > 0:
            guardar_base_datos(self.bd)
            umbral = calcular_umbral(self.bd)
            guardar_umbral(umbral)
            self._log(f"\n✅ {procesados} alumno(s) procesados")
            self._log(f"📊 Umbral calculado: {umbral:.4f}")
            self.menu.actualizar_info()
        else:
            self._log("⚠️  No se procesó ningún alumno")


# ──────────────────────────────────────────────────────────────
# MODO 2 — Registro nuevo: solo Nombre + Cámara + Botones
# ──────────────────────────────────────────────────────────────
class VentanaRegistroNuevo:
    def __init__(self, parent, modelo, bd, menu):
        self.modelo    = modelo
        self.bd        = bd
        self.menu      = menu
        self.cap       = None
        self.corriendo = False
        self.fotos_capturadas = {}

        self.win = tk.Toplevel(parent)
        self.win.title("Registrar nuevo alumno")
        self.win.geometry("460x560")
        self.win.configure(bg="#ffffff")
        self.win.resizable(False, False)
        self.win.grab_set()
        self.win.protocol("WM_DELETE_WINDOW", self._cerrar)

        self._build_ui()
        self._iniciar_camara()

    def _build_ui(self):
        # Header
        header = tk.Frame(self.win, bg="#1a1a2e", pady=14)
        header.pack(fill="x")
        tk.Label(header, text="Face Recognition",
                 font=("Arial", 16, "bold"), fg="#e94560", bg="#1a1a2e").pack()
        tk.Label(header, text="Register",
                 font=("Arial", 13, "bold"), fg="#ffffff", bg="#1a1a2e").pack()

        # Campo nombre
        frame_form = tk.Frame(self.win, bg="#ffffff", padx=30)
        frame_form.pack(fill="x", pady=14)

        self.var_nombre = tk.StringVar()
        self.entry_nombre = tk.Entry(
            frame_form, textvariable=self.var_nombre,
            font=("Arial", 12), relief="solid", bd=1,
            fg="#aaaaaa", bg="#f9f9f9"
        )
        self.entry_nombre.pack(fill="x", ipady=9)
        self.entry_nombre.insert(0, "Full name")
        self.entry_nombre.bind("<FocusIn>",  self._on_focus_in)
        self.entry_nombre.bind("<FocusOut>", self._on_focus_out)

        # Cámara
        frame_cam = tk.Frame(self.win, bg="#ffffff", padx=30)
        frame_cam.pack(fill="x")

        self.lbl_camara = tk.Label(frame_cam, bg="#000000", cursor="hand2")
        self.lbl_camara.pack(fill="x", ipady=0)
        self.lbl_camara.bind("<Button-1>", self._capturar_foto)

        self.lbl_instruccion = tk.Label(
            frame_cam,
            text="Click video to capture face  (foto 1/2)",
            font=("Arial", 9), fg="#e94560", bg="#ffffff"
        )
        self.lbl_instruccion.pack(pady=4)

        # Indicadores de fotos
        frame_fotos = tk.Frame(frame_cam, bg="#ffffff")
        frame_fotos.pack()
        self.lbl_foto1 = tk.Label(frame_fotos, text="● Foto 1",
                                   font=("Arial", 9), fg="#cccccc", bg="#ffffff")
        self.lbl_foto1.pack(side="left", padx=12)
        self.lbl_foto2 = tk.Label(frame_fotos, text="● Foto 2",
                                   font=("Arial", 9), fg="#cccccc", bg="#ffffff")
        self.lbl_foto2.pack(side="left", padx=12)

        # Botones Cancel / Submit
        frame_btns = tk.Frame(self.win, bg="#ffffff", padx=30)
        frame_btns.pack(fill="x", pady=16)

        tk.Button(
            frame_btns, text="Cancel",
            bg="#dddddd", fg="#333333", font=("Arial", 11),
            bd=0, cursor="hand2", pady=10,
            command=self._cerrar
        ).pack(side="left", expand=True, fill="x", padx=(0, 6))

        tk.Button(
            frame_btns, text="Submit",
            bg="#f0c040", fg="#333333", font=("Arial", 11, "bold"),
            bd=0, cursor="hand2", pady=10,
            command=self._submit
        ).pack(side="left", expand=True, fill="x", padx=(6, 0))

    def _on_focus_in(self, e):
        if self.entry_nombre.get() == "Full name":
            self.entry_nombre.delete(0, "end")
            self.entry_nombre.config(fg="#333333")

    def _on_focus_out(self, e):
        if not self.entry_nombre.get():
            self.entry_nombre.insert(0, "Full name")
            self.entry_nombre.config(fg="#aaaaaa")

    def _iniciar_camara(self):
        self.cap       = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.corriendo = True
        threading.Thread(target=self._loop_camara, daemon=True).start()

    def _loop_camara(self):
        while self.corriendo:
            ret, frame = self.cap.read()
            if not ret:
                break

            gris  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            caras = DETECTOR_CARA.detectMultiScale(gris, 1.1, 5)
            for (x, y, w, h) in caras:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (233, 69, 96), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (400, 260))
            img       = ImageTk.PhotoImage(Image.fromarray(frame_rgb))

            try:
                self.lbl_camara.config(image=img)
                self.lbl_camara.image = img
                self._ultimo_frame    = frame.copy()
            except:
                break
            time.sleep(0.03)

    def _capturar_foto(self, event=None):
        if not hasattr(self, "_ultimo_frame"):
            return

        frame = self._ultimo_frame
        gris  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = DETECTOR_CARA.detectMultiScale(gris, 1.1, 5)

        if len(caras) == 0:
            messagebox.showwarning("Sin rostro",
                                   "No se detectó ningún rostro.\n"
                                   "Asegúrate de estar frente a la cámara.")
            return

        x, y, w, h = caras[0]
        # Usar preprocesar_rostro para expandir el ROI con margen
        rostro = preprocesar_rostro(frame, x, y, w, h)
        emb    = extraer_embedding(self.modelo, rostro)
        os.makedirs(CARPETA_FOTOS, exist_ok=True)
        nombre_archivo = self.var_nombre.get().strip().replace(" ", "_")

        if "foto1" not in self.fotos_capturadas:
            self.fotos_capturadas["foto1"] = emb
            cv2.imwrite(os.path.join(CARPETA_FOTOS, f"{nombre_archivo}_foto1.jpg"), rostro)
            self.lbl_foto1.config(text="✅ Foto 1", fg="#27ae60")
            self.lbl_instruccion.config(
                text="¡Foto 1 lista! Cambia posición y captura foto 2")

        elif "foto2" not in self.fotos_capturadas:
            self.fotos_capturadas["foto2"] = emb
            cv2.imwrite(os.path.join(CARPETA_FOTOS, f"{nombre_archivo}_foto2.jpg"), rostro)
            self.lbl_foto2.config(text="✅ Foto 2", fg="#27ae60")
            self.lbl_instruccion.config(
                text="✅ Ambas fotos listas — presiona Submit")

        else:
            messagebox.showinfo("Fotos completas",
                                "Ya tienes 2 fotos.\nPresiona Submit para registrar.")

    def _submit(self):
        nombre = self.var_nombre.get().strip()

        if not nombre or nombre == "Full name":
            messagebox.showerror("Error", "Ingresa el nombre completo del alumno")
            return
        if "foto1" not in self.fotos_capturadas or "foto2" not in self.fotos_capturadas:
            messagebox.showerror("Error", "Captura las 2 fotos antes de continuar")
            return
        if nombre in self.bd:
            if not messagebox.askyesno("Confirmar",
                                        f"'{nombre}' ya existe.\n¿Sobreescribir?"):
                return

        self.bd[nombre] = {
            "foto1": self.fotos_capturadas["foto1"],
            "foto2": self.fotos_capturadas["foto2"]
        }
        guardar_base_datos(self.bd)
        umbral = calcular_umbral(self.bd)
        guardar_umbral(umbral)

        dist = distancia_euclidiana(
            self.fotos_capturadas["foto1"],
            self.fotos_capturadas["foto2"]
        )
        messagebox.showinfo(
            "✅ Registrado",
            f"{nombre} registrado exitosamente\n"
            f"dist(foto1, foto2) = {dist:.4f}\n"
            f"Umbral actualizado: {umbral:.4f}"
        )
        self.menu.actualizar_info()
        self._cerrar()

    def _cerrar(self):
        self.corriendo = False
        if self.cap:
            self.cap.release()
        self.win.destroy()


# ──────────────────────────────────────────────────────────────
# LISTA DE ALUMNOS — Ver y eliminar
# ──────────────────────────────────────────────────────────────
class VentanaListaAlumnos:
    def __init__(self, parent, bd, menu):
        self.bd   = bd
        self.menu = menu

        self.win = tk.Toplevel(parent)
        self.win.title("Alumnos registrados")
        self.win.geometry("400x380")
        self.win.configure(bg="#1a1a2e")
        self.win.grab_set()

        self._build_ui()

    def _build_ui(self):
        tk.Label(self.win, text=f"Alumnos registrados ({len(self.bd)})",
                 font=("Arial", 13, "bold"), fg="#e94560", bg="#1a1a2e").pack(pady=12)

        frame_lista = tk.Frame(self.win, bg="#1a1a2e")
        frame_lista.pack(fill="both", expand=True, padx=20)

        scrollbar = tk.Scrollbar(frame_lista)
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            frame_lista, yscrollcommand=scrollbar.set,
            bg="#16213e", fg="#f5f5f5", font=("Arial", 11),
            selectbackground="#e94560", relief="flat"
        )
        self.listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        for nombre in sorted(self.bd.keys()):
            self.listbox.insert("end", f"  {nombre}")

        frame_btns = tk.Frame(self.win, bg="#1a1a2e")
        frame_btns.pack(pady=10)

        tk.Button(frame_btns, text="🗑️  Eliminar seleccionado",
                  bg="#e94560", fg="white", font=("Arial", 10),
                  bd=0, cursor="hand2", padx=15, pady=6,
                  command=self._eliminar).pack(side="left", padx=8)

        tk.Button(frame_btns, text="Cerrar",
                  bg="#16213e", fg="#a8a8b3", font=("Arial", 10),
                  bd=0, cursor="hand2", padx=15, pady=6,
                  command=self.win.destroy).pack(side="left", padx=8)

    def _eliminar(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("Aviso", "Selecciona un alumno primero")
            return
        nombre = self.listbox.get(sel[0]).strip()
        if messagebox.askyesno("Confirmar", f"¿Eliminar a '{nombre}'?"):
            del self.bd[nombre]
            guardar_base_datos(self.bd)
            if self.bd:
                guardar_umbral(calcular_umbral(self.bd))
            self.listbox.delete(sel[0])
            self.menu.actualizar_info()
            messagebox.showinfo("Listo", f"'{nombre}' eliminado")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    if os.path.exists(RUTA_BD):
        with open(RUTA_BD, "rb") as f:
            bd = pickle.load(f)
    else:
        bd = {}

    print("[captura_alumnos] Cargando modelo...")
    modelo = cargar_modelo()
    print("[captura_alumnos] Iniciando interfaz...")

    MenuPrincipal(modelo, bd).run()


if __name__ == "__main__":
    main()