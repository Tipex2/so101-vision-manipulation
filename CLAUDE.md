# SO-101 Vision-Guided Manipulation — Contexto del proyecto

## Quién soy y qué sé
Estudiante de Informática Industrial y Robótica (3er año acabado). Cómodo con Python,
ROS2, simulación (CoppeliaSim, FlexSim), RL básico (DQN), embebidos (ESP32/MQTT).
Nuevo en: LeRobot, imitation learning, visión clásica aplicada a robots reales.

## Objetivo del proyecto
Brazo SO-101 (impreso en 3D, montado y calibrado) que aprende a manipular objetos
por criterio visual: primero color, luego defectos/deformidades. Doble enfoque:
visión clásica (OpenCV) como baseline, e imitation learning (LeRobot: ACT/Diffusion
Policy) como corazón del proyecto. Objetivo final: pieza de portfolio para prácticas.

## Stack
- SO: Kubuntu 24.04, dual-boot con Windows, disco dedicado SanDisk 465GB
- GPU: RTX 3070, drivers NVIDIA vía ubuntu-drivers (no .run manual)
- Entorno: conda env `lerobot`, Python 3.12 (LeRobot exige >=3.12)
- Librería robot: LeRobot (huggingface), instalada en ~/Proyectos/lerobot con
  `pip install -e ".[feetech]"`
- Repo de este proyecto: ~/Proyectos/so101-vision-manipulation (separado del repo lerobot)

## Estado del hardware — YA HECHO
- Ambos brazos (líder + seguidor) montados físicamente
- Motores configurados: seguidor 6x STS3215 30kg·cm/12V; líder 6x STS3215 7.4V con
  distinta reducción por articulación (ver docs/notas-hardware.md)
- Ambos brazos calibrados (`lerobot-calibrate`), IDs guardados como `follower_arm`
  y `leader_arm`
- Teleoperación líder-seguidor funcionando a 60Hz, verificada con --display_data=true (Rerun)
- 2 cámaras Innomaker U20CAM-1080p-S1 (overhead + wrist) integradas y funcionando
  - IMPORTANTE: la cámara "overhead" requiere forzar fourcc=MJPG en la config de
    OpenCV o falla con "ioctl(VIDIOC_STREAMON): Protocol error"
  - Los nodos /dev/videoN cambian según orden de conexión — siempre reverificar con
    `v4l2-ctl --list-devices` antes de lanzar comandos
- Las dos placas controladoras son físicamente idénticas — etiquetadas a mano
  (líder/seguidor), los puertos /dev/ttyACMx también cambian según orden de conexión

## Estado del software — EN PROGRESO
- [ ] Objetos 3D: imprimiendo cubos de colores (verde=objetivo + distractores rojo/
      azul/amarillo, ~2.5cm). Pendiente: piezas "buena/defectuosa" (bulto, asimetría)
- [ ] scripts/detect_color.py — detección HSV con OpenCV (siguiente paso inmediato)
- [ ] Calibración mano-ojo (homografía cv2.findHomography, plano de mesa 2D, no
      cinemática 3D completa)
- [ ] Baseline pick-and-place clásico end-to-end
- [ ] Recogida de demostraciones con lerobot-record
- [ ] Entrenamiento de política (lerobot-train, ACT o Diffusion Policy)
- [ ] Evaluación con métricas de tasa de éxito

## Convenciones del proyecto
- Todo el código nuevo va en scripts/, configs/ para JSON/YAML de cámaras y robot,
  docs/ para notas de hardware y decisiones
- Commits pequeños y frecuentes, mensajes descriptivos en español
- Ver docs/setup.md para reproducir el entorno completo desde cero
- Ver docs/notas-hardware.md para gotchas específicos de esta unidad de hardware

## Preferencias de trabajo
- Explicaciones detalladas paso a paso, sin asumir conocimiento previo de LeRobot
- Verificar/diagnosticar antes de lanzar comandos que puedan dañar hardware
  (alimentación, IDs de servos)
- Priorizar claridad sobre velocidad — este es un proyecto de aprendizaje además
  de portfolio
