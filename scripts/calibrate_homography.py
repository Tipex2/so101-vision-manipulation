"""
calibrate_homography.py — Calibración mano-ojo (píxel → mm) para el SO-101.

Calcula la matriz de homografía 3x3 que traduce coordenadas en píxel de la
cámara cenital ("overhead") a coordenadas reales en milímetros sobre el plano
de la mesa, con origen en la proyección vertical del eje shoulder_pan del
brazo seguidor.

Por qué homografía (y no cinemática 3D completa):
    Los objetos que vamos a manipular (cubos) están apoyados en un plano fijo
    (la mesa). Cualquier transformación entre "puntos en un plano visto por una
    cámara" y "puntos en ese plano en el mundo real" se puede describir con una
    única matriz 3x3. No hace falta modelar la posición 3D de la cámara ni la
    lente — solo pares de correspondencias (píxel, mm) suficientes. RANSAC
    hace el ajuste robusto y descarta puntos mal clicados.

Sistema de coordenadas del mundo real:
    - Origen (0, 0) = proyección vertical del eje shoulder_pan sobre la mesa.
      Ese punto queda tapado por el propio robot y no se calibra directamente;
      es solo la referencia desde la que se miden los demás.
    - X positivo = hacia la derecha del robot (visto desde el punto de vista
      físico del observador, NO desde la imagen de la cámara).
    - Y positivo = hacia adelante, en dirección al área de trabajo.
    - Unidades: milímetros.

Uso:
    # Calibración interactiva desde cero
    python scripts/calibrate_homography.py --camera /dev/video0 --fourcc MJPG

    # Verificar una homografía ya guardada, sin recalibrar
    python scripts/calibrate_homography.py --camera /dev/video0 --fourcc MJPG \
        --verify configs/homography.npy

Controles durante la calibración:
    - Clic izquierdo sobre una marca: registra ese píxel para el punto actual.
    - 'u': deshace el último clic (por si te equivocas).
    - 'r': reinicia toda la calibración desde el primer punto.
    - 'q': aborta sin guardar.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 1. Puntos de calibración
# --------------------------------------------------------------------------
# Cada tupla es (nombre_legible, x_mm, y_mm) en el sistema del mundo real.
# El orden en esta lista es el ORDEN EN QUE EL SCRIPT PEDIRÁ LOS CLICS.
# Se ha elegido para minimizar el riesgo de confusión: fila por fila desde la
# más alejada del robot (Y=300) hasta la más cercana (Y=0), y dentro de cada
# fila de izquierda a derecha SEGÚN EL MUNDO REAL (X negativo → X positivo),
# ignorando cómo aparezcan en la imagen de la cámara (que puede estar rotada).
CALIBRATION_POINTS_MM = [
    ("Fila lejana - izquierda",   -150, 300),
    ("Fila lejana - centro",         0, 300),
    ("Fila lejana - derecha",      150, 300),
    ("Fila media - izquierda",    -150, 150),
    ("Fila media - centro",          0, 150),
    ("Fila media - derecha",       150, 150),
    ("Fila cercana - izquierda",  -150,   0),
    ("Fila cercana - derecha",     150,   0),
    # (0, 0) NO se incluye: es la base del robot, tapada por el propio brazo,
    # y no necesitamos calibrarla directamente para que la homografía funcione.
]

DEFAULT_OUTPUT_PATH = Path("configs/homography.npy")


# --------------------------------------------------------------------------
# 2. Cámara (idéntico a detect_color.py, para mantener el mismo comportamiento
#    que ya sabemos que funciona con la overhead: backend V4L2 + fourcc MJPG)
# --------------------------------------------------------------------------
def resolve_camera_source(raw):
    return int(raw) if raw.isdigit() else raw


def open_camera(source, width, height, fps, fourcc):
    """Abre la cámara forzando el backend V4L2 (no dejar en ANY, que puede
    escoger FFMPEG y romper cap.set() en cámaras UVC — ver notas-hardware.md)."""
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara '{source}'. Verifica con "
            "`v4l2-ctl --list-devices` que el índice/ruta es correcto y que "
            "ningún otro proceso la tiene abierta."
        )
    return cap


# --------------------------------------------------------------------------
# 3. Utilidades de dibujo
# --------------------------------------------------------------------------
# Colores BGR (formato nativo de OpenCV). Usamos verde para clics ya hechos
# y amarillo brillante para la marca objetivo pendiente, así siempre queda
# claro visualmente qué toca clicar ahora.
COLOR_DONE = (0, 200, 0)         # verde: ya clicado
COLOR_PENDING = (0, 255, 255)    # amarillo: el que toca ahora


def draw_overlay(frame, clicks, current_index, total_points):
    """Dibuja overlay compacto abajo del frame, no arriba, para no tapar los
    puntos de la fila lejana. También reduce grosor/tamaño de fuente para
    minimizar la zona ocultada."""

    h, w = frame.shape[:2]

    # Puntos ya registrados (marcador pequeño y sin círculo exterior grande)
    for i, (px, py) in enumerate(clicks):
        cv2.circle(frame, (px, py), 4, COLOR_DONE, -1)
        cv2.putText(frame, str(i + 1), (px + 8, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_DONE, 1)

    # Instrucciones del punto pendiente (arriba solo indicador diminuto)
    if current_index < total_points:
        name, x_mm, y_mm = CALIBRATION_POINTS_MM[current_index]
        instr = f"Clica: {name}  ({x_mm:+}, {y_mm:+}) mm"
    else:
        instr = "Todos registrados. Pulsa una tecla para calcular."

    # Barra inferior compacta (~40 px de alto) con fondo semitransparente
    bar_h = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, instr, (8, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PENDING, 1)
    cv2.putText(frame, f"{current_index}/{total_points}   "
                       "[u]deshacer  [r]reiniciar  [q]abortar",
                (8, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


# --------------------------------------------------------------------------
# 4. Bucle interactivo de clics
# --------------------------------------------------------------------------
class ClickCollector:
    """Encapsula el estado de la calibración (lista de píxeles clicados)
    y expone un callback para el ratón que OpenCV puede llamar."""

    def __init__(self, total_points):
        self.clicks = []
        self.total_points = total_points

    def on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicks) < self.total_points:
            self.clicks.append((x, y))
            name, x_mm, y_mm = CALIBRATION_POINTS_MM[len(self.clicks) - 1]
            print(f"  [{len(self.clicks)}/{self.total_points}] {name}: "
                  f"pixel=({x}, {y}) → real=({x_mm}, {y_mm}) mm")


def collect_clicks(cap, window):
    total = len(CALIBRATION_POINTS_MM)
    collector = ClickCollector(total)
    cv2.setMouseCallback(window, collector.on_mouse)

    print(f"\nRecogida de {total} puntos de calibración.")
    print("Sigue las instrucciones que aparecen en la ventana (fila lejana → cercana,")
    print("izquierda → derecha SEGÚN EL MUNDO REAL, no según la imagen).\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Aviso: no se pudo leer un frame, deteniendo.", file=sys.stderr)
            return None

        draw_overlay(frame, collector.clicks, len(collector.clicks), total)
        cv2.imshow(window, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Calibración abortada por el usuario.")
            return None

        if key == ord("u") and collector.clicks:
            removed = collector.clicks.pop()
            print(f"  Deshecho el último clic {removed}.")

        if key == ord("r"):
            collector.clicks.clear()
            print("  Calibración reiniciada.")

        # Si ya están todos los puntos, cualquier otra tecla confirma
        if len(collector.clicks) == total and key not in (255, ord("u"), ord("r")):
            return collector.clicks


# --------------------------------------------------------------------------
# 5. Cálculo y validación de la homografía
# --------------------------------------------------------------------------
def compute_homography(pixel_points, world_points_mm):
    """Calcula H tal que  world_mm ≈ H · pixel  (en coordenadas homogéneas).

    Usa RANSAC: si algún clic fue impreciso (por ejemplo, se clicó justo al
    borde de una marca en vez de en su centro), el ajuste lo detecta como
    outlier y no lo usa para el cálculo final. Esto es más robusto que la
    versión exacta cuando hay ≥4 puntos.
    """
    src = np.array(pixel_points, dtype=np.float32)  # píxeles
    dst = np.array(world_points_mm, dtype=np.float32)  # mm reales

    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC,
                                  ransacReprojThreshold=5.0)

    if H is None:
        raise RuntimeError(
            "cv2.findHomography no consiguió calcular la matriz. "
            "Suele pasar si hay clics muy desalineados o si todos los puntos "
            "son casi colineales. Revisa la posición física de las marcas."
        )
    return H, mask.ravel().astype(bool)


def apply_homography(H, px, py):
    """Aplica la homografía a un píxel individual y devuelve (x_mm, y_mm).

    La homografía trabaja en coordenadas homogéneas: un píxel (px, py) se
    representa como (px, py, 1), se multiplica por H, y el resultado (x', y', w')
    se divide por w' para volver a coordenadas cartesianas (x'/w', y'/w').
    """
    src = np.array([px, py, 1.0])
    dst = H @ src
    return float(dst[0] / dst[2]), float(dst[1] / dst[2])


def validate_homography(H, pixel_points, world_points_mm, inlier_mask):
    """Reproyecta cada píxel a mm y compara con el valor real medido.

    Un error de pocos milímetros indica una buena calibración. Errores de
    centímetros indican clic impreciso, marca mal medida, o distorsión de
    lente significativa que la homografía no puede compensar por sí sola.
    """
    print("\nValidación de la homografía:")
    print(f"  {'Punto':<28} {'Real (mm)':<16} {'Reproyectado (mm)':<20} "
          f"{'Error (mm)':<12} {'Usado':<8}")
    print("  " + "-" * 88)

    errors = []
    for i, ((px, py), (x_real, y_real)) in enumerate(zip(pixel_points, world_points_mm)):
        x_rep, y_rep = apply_homography(H, px, py)
        err = np.hypot(x_rep - x_real, y_rep - y_real)
        errors.append(err)

        name = CALIBRATION_POINTS_MM[i][0]
        used = "sí" if inlier_mask[i] else "NO (outlier)"
        print(f"  {name:<28} ({x_real:>4}, {y_real:>4})   "
              f"({x_rep:>6.1f}, {y_rep:>6.1f})    {err:>6.2f}       {used}")

    errors = np.array(errors)
    print(f"\n  Error medio: {errors.mean():.2f} mm | "
          f"Máximo: {errors.max():.2f} mm | Puntos usados: {inlier_mask.sum()}/{len(errors)}")

    if errors.max() > 15:
        print("\n  ⚠️  Error máximo > 15 mm. Recomendaciones:")
        print("     - Comprueba que ningún clic quedó lejos del centro de la marca.")
        print("     - Verifica con regla que las coordenadas reales son correctas.")
        print("     - Considera corregir la distorsión de lente antes (ver docs).")
    elif errors.mean() < 3:
        print("\n  ✅ Calibración de alta calidad (error medio < 3 mm).")
    else:
        print("\n  ✅ Calibración aceptable. Guardando.")


# --------------------------------------------------------------------------
# 6. Modo verificación (comprueba una H ya guardada sin recalibrar)
# --------------------------------------------------------------------------
def run_verification(cap, homography_path):
    """Carga una homografía existente y superpone en vivo la coordenada real
    del centro de la imagen y de dondequiera que muevas el ratón. Útil para
    comprobar visualmente que una calibración guardada sigue siendo válida
    (por ejemplo, tras reiniciar el sistema)."""
    H = np.load(homography_path)
    print(f"Homografía cargada desde {homography_path}. Forma: {H.shape}")
    print("Mueve el ratón sobre la imagen: se mostrará la coordenada real correspondiente.")
    print("Pulsa 'q' para salir.")

    mouse_state = {"x": 0, "y": 0}

    def track_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_state["x"], mouse_state["y"] = x, y

    window = "Verificacion homografia"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, track_mouse)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mx, my = mouse_state["x"], mouse_state["y"]
        x_mm, y_mm = apply_homography(H, mx, my)

        cv2.drawMarker(frame, (mx, my), (0, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, f"Pixel ({mx}, {my}) → Real ({x_mm:+.1f}, {y_mm:+.1f}) mm",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# --------------------------------------------------------------------------
# 7. CLI y main
# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibración de homografía cámara→mesa para el SO-101."
    )
    parser.add_argument("--camera", default="0",
                         help="Índice (0, 1...) o ruta (/dev/video0) de la cámara overhead.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default=None,
                         help="Forzar formato (MJPG para la cámara overhead).")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH),
                         help="Ruta donde guardar la matriz de homografía (.npy).")
    parser.add_argument("--verify", default=None,
                         help="En vez de calibrar, verifica una homografía existente.")
    return parser.parse_args()


def main():
    args = parse_args()
    source = resolve_camera_source(args.camera)
    cap = open_camera(source, args.width, args.height, args.fps, args.fourcc)

    try:
        if args.verify:
            run_verification(cap, Path(args.verify))
            return

        window = "Calibracion homografia - SO-101"
        cv2.namedWindow(window)

        pixel_points = collect_clicks(cap, window)
        if pixel_points is None:
            return

        world_points_mm = [(x, y) for (_name, x, y) in CALIBRATION_POINTS_MM]
        H, inlier_mask = compute_homography(pixel_points, world_points_mm)
        validate_homography(H, pixel_points, world_points_mm, inlier_mask)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, H)
        print(f"\nHomografía guardada en {output_path.resolve()}")
        print("\nMatriz H (3x3):")
        print(H)
        print("\nSiguientes pasos:")
        print("  1. Verifícala en vivo con:")
        print(f"     python scripts/calibrate_homography.py --camera {args.camera} "
              f"{'--fourcc ' + args.fourcc if args.fourcc else ''} --verify {output_path}")
        print("  2. Cárgala en tu pipeline de pick-and-place con np.load().")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()