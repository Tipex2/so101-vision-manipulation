# SO-101 Vision-Guided Manipulation

Proyecto de brazo robótico SO-101 (impreso en 3D) con manipulación guiada por visión.

## Objetivo
Enseñar al brazo a distinguir objetos por color y, posteriormente, por defectos/deformidades, combinando:
- Visión clásica (OpenCV)
- Aprendizaje por imitación (LeRobot / Diffusion Policy / ACT)

## Hardware
- Brazo SO-101 (líder + seguidor), impreso en 3D
- 2x cámara Innomaker UVC (vista cenital + muñeca)
- RTX 3070 para entrenamiento

## Estado actual
- [x] Montaje mecánico de ambos brazos
- [x] Configuración y calibración de motores
- [x] Teleoperación líder-seguidor funcional (60Hz)
- [x] Ambas cámaras integradas (una requiere fourcc=MJPG, ver /docs/notas-hardware.md)
- [ ] Baseline de visión clásica (detección por color)
- [ ] Calibración mano-ojo
- [ ] Recogida de demostraciones
- [ ] Entrenamiento de política de imitación

## Setup
Ver [docs/setup.md](docs/setup.md) para instrucciones completas de instalación y configuración de hardware.

## Estructura del repo

so101-vision-manipulation/
├── docs/ # notas de hardware, setup, decisiones
├── scripts/ # scripts propios (detección de color, calibración, etc.)
├── configs/ # configuraciones de cámaras y robot
└── README.md

