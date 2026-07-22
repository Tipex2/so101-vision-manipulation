"""
detect_color.py — Baseline de visión clásica para el SO-101.

Detecta cubos de colores en el feed de una cámara usando umbralización en el
espacio de color HSV (Hue, Saturation, Value) con OpenCV.

Por qué HSV y no BGR/RGB:
    En BGR, el color de un píxel mezcla "matiz" e "iluminación" en los 3 canales
    a la vez, así que un cambio de luz ambiental desplaza los 3 valores y rompe
    cualquier umbral fijo. HSV separa el matiz (H, "qué color es") de la
    saturación (S, "qué tan puro/vivo es") y el brillo (V, "qué tan iluminado
    está"). Esto permite fijar un rango de H estrecho (el color en sí) y rangos
    de S/V más anchos (tolerantes a sombras/brillos), que es justo lo que
    necesitamos para reconocer "el cubo azul" bajo luz de mesa variable.

Uso:
    # Detección en vivo (objetivo = azul, distractores = rojo y amarillo)
    python scripts/detect_color.py --camera 0

    # Cámara overhead identificada por ruta y con MJPG forzado (ver notas-hardware.md)
    python scripts/detect_color.py --camera /dev/video2 --fourcc MJPG

    # Modo calibración: ajustar en vivo el rango HSV de un color con sliders
    python scripts/detect_color.py --camera 0 --calibrate azul

Controles en cualquier modo: pulsar 'q' con la ventana de vídeo activa para salir.
"""

import argparse
import sys

import cv2
import numpy as np

# --------------------------------------------------------------------------
# 1. Configuración de colores
# --------------------------------------------------------------------------
# En OpenCV, HSV usa rangos H:[0,179] S:[0,255] V:[0,255] (H va de 0 a 179,
# no de 0 a 360, porque un canal de 8 bits solo llega a 255 y así cabe /2).
#
# Cada color es una LISTA de rangos (low, high) en vez de uno solo, porque el
# rojo es especial: su matiz está justo en el punto donde la rueda de color da
# la vuelta (0° y 360° son el mismo rojo). En la escala 0-179 de OpenCV eso
# corresponde a los extremos 0 y 179, así que el rojo necesita DOS rangos
# (uno pegado a 0 y otro pegado a 179) para cubrirlo sin "cortarlo" por la mitad.

COLOR_RANGES = {
    "azul": [
        ((84, 110, 45), (119, 248, 155)),
    ],
    "rojo": [
        ((127, 134, 87), (179, 255, 255)),
    ],
    "amarillo": [
        ((15, 70, 109), (73, 255, 255)),
    ],
}

# Color objetivo de esta fase del proyecto y los que actúan como distractores.
# Cambiar solo estas dos líneas basta para reenfocar el script a otro color.
TARGET_COLOR = "azul"
DISTRACTOR_COLORS = ["rojo", "amarillo"]

# Color BGR (formato nativo de OpenCV para dibujar) usado para las anotaciones
# de cada color detectado, así el overlay en pantalla es legible de un vistazo.
DRAW_COLOR_BGR = {
    "azul": (255, 0, 0),
    "rojo": (0, 0, 255),
    "amarillo": (0, 255, 255),
}

# Área mínima (en píxeles) para que un blob cuente como detección real y no
# como ruido (reflejos, motas de polvo, compresión JPEG, etc.).
DEFAULT_MIN_AREA_PX = 300


# --------------------------------------------------------------------------
# 2. Procesado de imagen: máscara -> limpieza -> blobs
# --------------------------------------------------------------------------
def build_mask(hsv_frame, ranges):
    """Construye una máscara binaria (0/255) combinando uno o varios rangos HSV.

    cv2.inRange marca con 255 los píxeles dentro del rango y 0 el resto.
    Si un color necesita varios rangos (el rojo), los combinamos con OR a
    nivel de bit: un píxel pertenece al color si cae en CUALQUIERA de los rangos.
    """
    mask = None
    for low, high in ranges:
        low_arr = np.array(low, dtype=np.uint8)
        high_arr = np.array(high, dtype=np.uint8)
        partial = cv2.inRange(hsv_frame, low_arr, high_arr)
        mask = partial if mask is None else cv2.bitwise_or(mask, partial)
    return mask


def clean_mask(mask):
    """Elimina ruido de la máscara con operaciones morfológicas.

    - OPEN (erosionar y luego dilatar): borra puntos sueltos pequeños que no
      son el objeto real, sin encoger mucho el blob principal.
    - CLOSE (dilatar y luego erosionar): rellena pequeños huecos dentro del
      blob (p.ej. un reflejo de luz en la cara del cubo que rompe la máscara).
    """
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detect_blobs(mask, min_area):
    """Encuentra regiones conectadas ("blobs") en la máscara y devuelve sus datos.

    cv2.findContours devuelve el contorno (borde) de cada región blanca.
    Para cada contorno calculamos:
      - área: para descartar ruido por debajo de `min_area`.
      - bounding box (x, y, w, h): rectángulo que encierra el blob, para dibujar.
      - centroide (cx, cy) vía momentos de imagen: es el "centro de masa" del
        blob. Este punto es el que más adelante alimentará la calibración
        mano-ojo (homografía) para convertir píxel -> coordenada real de mesa.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        moments = cv2.moments(contour)
        if moments["m00"] == 0:  # evita división por cero en contornos degenerados
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        blobs.append({"cx": cx, "cy": cy, "x": x, "y": y, "w": w, "h": h, "area": area})

    return blobs


# --------------------------------------------------------------------------
# 3. Cámara
# --------------------------------------------------------------------------
def resolve_camera_source(raw):
    """Traduce el argumento --camera a lo que espera cv2.VideoCapture.

    Acepta tanto un índice entero ("0", "1"...) como una ruta de dispositivo
    ("/dev/video2"), porque en este proyecto los nodos /dev/videoN cambian
    según el orden de conexión de las cámaras (ver notas-hardware.md).
    """
    return int(raw) if raw.isdigit() else raw


def open_camera(source, width, height, fps, fourcc):
    """Abre y configura la cámara.

    El orden importa: fijamos el FOURCC (formato de compresión) ANTES que la
    resolución. La cámara "overhead" de este proyecto solo funciona en MJPG
    (comprimido) a 1080p-derivadas de resolución; en el formato YUYV sin
    comprimir por defecto, el ancho de banda USB no alcanza y falla con
    "ioctl(VIDIOC_STREAMON): Protocol error". La cámara "wrist" funciona igual
    sin forzar nada, así que --fourcc es opcional y solo hace falta para la
    overhead.
    """
    cap = cv2.VideoCapture(source)

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara '{source}'. Verifica con "
            "`v4l2-ctl --list-devices` que el índice/ruta es correcto y que "
            "ningún otro proceso (cheese, guvcview, otro script) la tiene abierta."
        )

    return cap


# --------------------------------------------------------------------------
# 4. Modo detección: el modo normal de uso del script
# --------------------------------------------------------------------------
def run_detection(cap, target, distractors, min_area):
    print(f"Detección en vivo — objetivo: {target.upper()} | distractores: "
          f"{', '.join(c.upper() for c in distractors)}")
    print("Pulsa 'q' sobre la ventana de vídeo para salir.")

    all_colors = [target] + list(distractors)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Aviso: no se pudo leer un frame de la cámara, deteniendo.", file=sys.stderr)
            break

        # Un desenfoque suave antes de umbralizar reduce ruido de compresión
        # JPEG/sensor que si no genera pequeños blobs falsos en los bordes.
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        for color in all_colors:
            mask = build_mask(hsv, COLOR_RANGES[color])
            mask = clean_mask(mask)
            blobs = detect_blobs(mask, min_area)

            is_target = color == target
            box_color = DRAW_COLOR_BGR[color]
            thickness = 3 if is_target else 1  # el objetivo resalta más que los distractores

            for blob in blobs:
                top_left = (blob["x"], blob["y"])
                bottom_right = (blob["x"] + blob["w"], blob["y"] + blob["h"])
                cv2.rectangle(frame, top_left, bottom_right, box_color, thickness)
                cv2.circle(frame, (blob["cx"], blob["cy"]), 4, box_color, -1)

                label = f"{color}{' [OBJETIVO]' if is_target else ''}"
                cv2.putText(frame, label, (blob["x"], blob["y"] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                if is_target:
                    # Esta es la coordenada de píxel que más adelante se
                    # convertirá a coordenada real de mesa (homografía).
                    print(f"[{color}] centro_px=({blob['cx']}, {blob['cy']}) "
                          f"area={int(blob['area'])}")

        cv2.imshow("Deteccion de color - SO-101", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# --------------------------------------------------------------------------
# 5. Modo calibración: ajustar los rangos HSV con sliders en vivo
# --------------------------------------------------------------------------
def _trackbar_noop(_value):
    """Callback vacío requerido por la API de cv2.createTrackbar (necesita
    una función aunque no hagamos nada con el valor en el propio callback;
    lo leemos activamente cada frame con getTrackbarPos en su lugar)."""
    pass


def run_calibration(cap, color_name):
    window = f"Calibracion HSV - {color_name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Trackbar = slider dentro de la ventana de OpenCV. Los inicializamos con
    # el rango que ya tengamos en COLOR_RANGES como punto de partida, si existe.
    existing = COLOR_RANGES.get(color_name, [((0, 0, 0), (179, 255, 255))])[0]
    (h0, s0, v0), (h1, s1, v1) = existing

    cv2.createTrackbar("H min", window, h0, 179, _trackbar_noop)
    cv2.createTrackbar("H max", window, h1, 179, _trackbar_noop)
    cv2.createTrackbar("S min", window, s0, 255, _trackbar_noop)
    cv2.createTrackbar("S max", window, s1, 255, _trackbar_noop)
    cv2.createTrackbar("V min", window, v0, 255, _trackbar_noop)
    cv2.createTrackbar("V max", window, v1, 255, _trackbar_noop)

    print(f"Calibrando '{color_name}'. Mueve los sliders hasta que en el panel "
          "derecho ('mask') SOLO queden en blanco los cubos de ese color.")
    print("Pulsa 'q' para salir e imprimir el rango final por terminal.")

    h_min = h_max = s_min = s_max = v_min = v_max = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Aviso: no se pudo leer un frame de la cámara, deteniendo.", file=sys.stderr)
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H min", window)
        h_max = cv2.getTrackbarPos("H max", window)
        s_min = cv2.getTrackbarPos("S min", window)
        s_max = cv2.getTrackbarPos("S max", window)
        v_min = cv2.getTrackbarPos("V min", window)
        v_max = cv2.getTrackbarPos("V max", window)

        low = np.array([h_min, s_min, v_min], dtype=np.uint8)
        high = np.array([h_max, s_max, v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, low, high)

        # bitwise_and con la propia máscara para "recortar" del frame original
        # solo lo que pasa el umbral: ayuda a ver visualmente qué se está
        # capturando de verdad, no solo la máscara en blanco y negro.
        result = cv2.bitwise_and(frame, frame, mask=mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # para poder unirla al resto en color

        # Panel triple: original | máscara | resultado recortado, uno junto al otro.
        combined = np.hstack([frame, mask_bgr, result])
        cv2.imshow(window, combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"\nRango final para '{color_name}':")
    print(f'    "{color_name}": [(({h_min}, {s_min}, {v_min}), ({h_max}, {s_max}, {v_max}))],')
    print("Copia esa línea dentro de COLOR_RANGES en este mismo archivo.")


# --------------------------------------------------------------------------
# 6. CLI
# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Deteccion de cubos por color (HSV) para el SO-101."
    )
    parser.add_argument("--camera", default="0",
                         help="Indice (0, 1...) o ruta (/dev/video2) de la camara.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default=None,
                         help="Forzar formato, p.ej. MJPG (necesario para la camara overhead).")
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_AREA_PX,
                         help="Area minima en pixeles para contar una deteccion como valida.")
    parser.add_argument("--calibrate", choices=list(COLOR_RANGES.keys()), default=None,
                         help="En vez de detectar, abre el modo de calibracion HSV para este color.")
    return parser.parse_args()


def main():
    args = parse_args()
    source = resolve_camera_source(args.camera)
    cap = open_camera(source, args.width, args.height, args.fps, args.fourcc)

    try:
        if args.calibrate:
            run_calibration(cap, args.calibrate)
        else:
            run_detection(cap, TARGET_COLOR, DISTRACTOR_COLORS, args.min_area)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
