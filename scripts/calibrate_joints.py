"""
calibrate_joints.py — Calibración de rejilla articular (mm -> ángulos) para el SO-101.

Complementa a calibrate_homography.py: aquel script traduce píxel -> mm de mesa;
este traduce mm de mesa -> ángulos de las 6 articulaciones del brazo SEGUIDOR,
en los MISMOS 8 puntos físicos (se importan directamente de
`calibrate_homography.CALIBRATION_POINTS_MM`, para que ambas calibraciones
describan exactamente la misma rejilla y no se puedan desincronizar).

Por qué esto y no cinemática inversa (IK):
    LeRobot trae un solver de IK completo (placo + URDF), pero exige instalar
    una dependencia bastante delicada (placo/Pinocchio con versiones muy
    pinneadas) y descargar a mano el URDF del SO-101. Nuestro caso es más
    simple: los cubos están siempre sobre el mismo plano de mesa y el gripper
    siempre agarra apuntando hacia abajo con la misma orientación. Con eso
    basta guardar, en un puñado de puntos conocidos, qué ángulos tenía el
    brazo al alcanzar cada uno, e INTERPOLAR entre esos puntos para cualquier
    posición intermedia (ver pick_and_place.py, siguiente paso). Es la misma
    idea que una homografía, pero en espacio de articulaciones en vez de en mm.

Cómo funciona:
    Teleoperas con normalidad — el líder mueve al seguidor en vivo, igual que
    con lerobot-teleoperate — mientras ves el feed de la cámara overhead con
    una marca de qué punto de la rejilla toca visitar ahora. Cuando el gripper
    del SEGUIDOR está justo encima de esa marca física en la mesa, pulsas 'c'
    para capturar los 6 ángulos articulares actuales. Repites para los 8 puntos.

Uso:
    python scripts/calibrate_joints.py \
        --follower-port /dev/ttyACM0 --leader-port /dev/ttyACM1 \
        --camera /dev/video0 --fourcc MJPG

Controles (con la ventana de vídeo activa):
    'c' — capturar el punto actual (el gripper debe estar sobre la marca)
    'u' — deshacer la última captura, por si no quedó bien centrado
    'r' — reiniciar toda la calibración desde el primer punto
    'q' — abortar sin guardar
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from calibrate_homography import CALIBRATION_POINTS_MM, open_camera, resolve_camera_source
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------
# Los 6 motores del seguidor, en el mismo orden que expone get_observation()
# (claves "<motor>.pos"). Se captura también 'gripper' por uniformidad, aunque
# el script de pick-and-place normalmente lo ignorará y controlará el gripper
# explícitamente (abrir/cerrar), ya que su valor aquí es solo el que tenía el
# líder en el momento de la captura, no algo calibrado a propósito.
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

DEFAULT_OUTPUT_PATH = Path("configs/joint_grid.json")
TELEOP_FPS = 60  # misma frecuencia que usáis a diario en lerobot-teleoperate

COLOR_DONE = (0, 200, 0)
COLOR_PENDING = (0, 255, 255)


# --------------------------------------------------------------------------
# Overlay visual (mismo estilo que calibrate_homography.py, adaptado)
# --------------------------------------------------------------------------
def draw_overlay(frame, captured, current_index, total):
    h, w = frame.shape[:2]

    if current_index < total:
        name, x_mm, y_mm = CALIBRATION_POINTS_MM[current_index]
        instr = f"Mueve el gripper a: {name}  ({x_mm:+}, {y_mm:+}) mm  -  pulsa 'c' para capturar"
    else:
        instr = "Los 8 puntos capturados. Pulsa una tecla (no u/r) para guardar."

    bar_h = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, instr, (8, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PENDING, 1)
    cv2.putText(frame, f"{len(captured)}/{total} capturados   "
                       "['c']capturar  ['u']deshacer  ['r']reiniciar  ['q']abortar",
                (8, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


# --------------------------------------------------------------------------
# Bucle principal: teleoperación en vivo + captura por teclado
# --------------------------------------------------------------------------
def run_calibration(cap, follower, leader, window):
    total = len(CALIBRATION_POINTS_MM)
    captured = []  # cada elemento: {"name", "x_mm", "y_mm", "joints": {...}}

    print(f"\nCalibración articular: {total} puntos (los mismos de calibrate_homography.py).")
    print("Teleopera con normalidad; cuando el gripper esté sobre la marca, pulsa 'c'.\n")

    while True:
        loop_start = time.perf_counter()

        # --- Teleoperación normal: líder manda, seguidor obedece ---
        # (Sin pipelines de procesado intermedios: con la config por defecto son
        # identidad pura, así que llamar directo es equivalente y más legible.)
        action = leader.get_action()
        follower.send_action(action)

        ok, frame = cap.read()
        if ok:
            draw_overlay(frame, captured, len(captured), total)
            cv2.imshow(window, frame)
        else:
            print("Aviso: no se pudo leer un frame de la cámara.", file=sys.stderr)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Calibración abortada por el usuario.")
            return None

        elif key == ord("u") and captured:
            removed = captured.pop()
            print(f"  Deshecha la captura de '{removed['name']}'.")

        elif key == ord("r"):
            captured.clear()
            print("  Calibración reiniciada.")

        elif key == ord("c") and len(captured) < total:
            # Leemos el ángulo REAL de los encoders (get_observation), no la
            # acción que acabamos de enviar: así capturamos la posición
            # efectiva del brazo, no solo el objetivo que se le pidió.
            obs = follower.get_observation()
            joints = {m: obs[f"{m}.pos"] for m in MOTOR_NAMES}
            name, x_mm, y_mm = CALIBRATION_POINTS_MM[len(captured)]
            captured.append({"name": name, "x_mm": x_mm, "y_mm": y_mm, "joints": joints})
            print(f"  [{len(captured)}/{total}] {name}: {joints}")

        elif len(captured) == total and key not in (255, ord("u"), ord("r")):
            return captured

        # Mantener ~60Hz como en la teleoperación normal, para que el líder
        # se sienta igual de responsivo mientras calibras.
        dt = time.perf_counter() - loop_start
        time.sleep(max(1.0 / TELEOP_FPS - dt, 0.0))


# --------------------------------------------------------------------------
# CLI y main
# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibración de rejilla articular (mm -> angulos) para el SO-101."
    )
    parser.add_argument("--follower-port", required=True, help="Puerto del seguidor, p.ej. /dev/ttyACM0.")
    parser.add_argument("--follower-id", default="follower_arm")
    parser.add_argument("--leader-port", required=True, help="Puerto del lider, p.ej. /dev/ttyACM1.")
    parser.add_argument("--leader-id", default="leader_arm")
    parser.add_argument("--camera", default="0", help="Indice o ruta de la camara overhead.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default=None, help="Forzar formato, p.ej. MJPG.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main():
    args = parse_args()

    source = resolve_camera_source(args.camera)
    cap = open_camera(source, args.width, args.height, args.fps, args.fourcc)

    follower = SO101Follower(SO101FollowerConfig(port=args.follower_port, id=args.follower_id))
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))

    window = "Calibracion articular - SO-101"
    cv2.namedWindow(window)

    follower.connect()
    leader.connect()

    try:
        captured = run_calibration(cap, follower, leader, window)
    finally:
        leader.disconnect()
        follower.disconnect()
        cap.release()
        cv2.destroyAllWindows()

    if not captured or len(captured) < len(CALIBRATION_POINTS_MM):
        print("Calibración incompleta: no se ha guardado nada.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captured, f, indent=2, ensure_ascii=False)

    print(f"\nGuardado en {output_path.resolve()}")
    print("Siguiente paso: usar esta rejilla para interpolar el objetivo articular "
          "de un pick a partir de la posición (x_mm, y_mm) que da la homografía.")


if __name__ == "__main__":
    main()
