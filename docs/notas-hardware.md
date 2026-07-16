# Notas de hardware — SO-101

## Cámaras
- Modelo: Innomaker U20CAM-1080p-S1 (x2)
- Cámara "overhead" (vista cenital): requiere forzar `fourcc: MJPG` en la config de OpenCV.
  Sin esto, falla con `ioctl(VIDIOC_STREAMON): Protocol error` — problema de negociación
  de ancho de banda con formato YUYV sin comprimir.
- Cámara "wrist" (muñeca): funciona con configuración por defecto (YUYV).
- Los nodos /dev/videoN cambian según orden de conexión. Verificar siempre con:
  `v4l2-ctl --list-devices` antes de lanzar teleoperación.

## Servos
- Seguidor: 6x STS3215, 30kg·cm, 12V — intercambiables entre sí.
- Líder: 6x STS3215, 7.4V, con distinta reducción según articulación:
  - Motor 2 (Shoulder Lift): 1/345 (19.5 kg·cm)
  - Motores 1 y 3 (Base, Elbow): 1/191 (16 kg·cm)
  - Motores 4, 5, 6 (Wrist Flex, Wrist Roll, Gripper): 1/147 (14.4 kg·cm)

## Alimentación
- Líder: fuente 7.4V
- Seguidor: fuente 12V
- NUNCA cruzar — daña los servos del líder.

## Calibración
- wrist_roll es rotación continua, no aparece en la tabla MIN/MAX de calibración
  (comportamiento esperado, no error).
