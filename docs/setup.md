# Setup — SO-101 Vision-Guided Manipulation

Guía completa para reproducir el entorno de este proyecto desde cero.

## 1. Sistema operativo

- Kubuntu 24.04 LTS (dual-boot con Windows).
- Drivers NVIDIA instalados vía `ubuntu-drivers autoinstall` (no el .run manual de NVIDIA).

## 2. Entorno Python (conda/miniforge)

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
```

> Nota: LeRobot requiere Python >=3.12 desde ciertas versiones. Si `pip install -e ".[feetech]"`
> falla con un error de versión de Python, recrea el entorno con 3.12.

## 3. PyTorch con soporte CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verificación:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Debe devolver `True` y el nombre de tu GPU (en este proyecto: RTX 3070).

## 4. LeRobot

```bash
git clone https://github.com/huggingface/lerobot.git ~/Proyectos/lerobot
cd ~/Proyectos/lerobot
pip install -e ".[feetech]"
```

El extra `[feetech]` es necesario para los servos STS3215 del SO-101.

## 5. Permisos de puerto serie (USB)

```bash
sudo usermod -aG dialout $USER
```

Cerrar sesión y volver a entrar para que el cambio de grupo se aplique. Verificar con `groups`.

## 6. Dependencias del sistema para cámaras

```bash
sudo apt install -y ffmpeg v4l-utils cheese guvcview libgl1 libglib2.0-0
```

## 7. Identificar puertos de las placas (líder / seguidor)

Con solo una placa conectada a la vez:
```bash
lerobot-find-port
```

Las dos placas controladoras son físicamente idénticas — etiquetar físicamente cada una
(líder / seguidor) es imprescindible, ya que el sistema les asigna `/dev/ttyACMx` según
orden de conexión, no por identidad fija.

## 8. Asignar IDs a los servos

Seguidor (12V, servos 30kg·cm):
```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACMx
```

Líder (7.4V, servos de distinta reducción según articulación — ver `notas-hardware.md`):
```bash
lerobot-setup-motors --teleop.type=so101_leader --teleop.port=/dev/ttyACMx
```

Sigue el proceso interactivo, conectando cada servo de uno en uno según el orden que
pida el script. Etiquetar físicamente cada servo con su ID conforme se asigna.

## 9. Calibración

Seguidor:
```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACMx --robot.id=follower_arm
```

Líder:
```bash
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACMx --teleop.id=leader_arm
```

> `wrist_roll` no aparece en la tabla final de calibración — es esperado, ya que es una
> articulación de rotación continua sin límites físicos que calibrar.

## 10. Cámaras

Identificar nodos de vídeo (con ambas cámaras conectadas):
```bash
v4l2-ctl --list-devices
```

Ver notas específicas de configuración de cada cámara en [notas-hardware.md](notas-hardware.md)
(una de las dos requiere forzar `fourcc: MJPG`).

## 11. Teleoperación completa (verificación final)

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACMx \
    --robot.id=follower_arm \
    --robot.cameras='{"overhead": {"type": "opencv", "index_or_path": "/dev/videoX", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"}, "wrist": {"type": "opencv", "index_or_path": "/dev/videoY", "width": 640, "height": 480, "fps": 30}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACMz \
    --teleop.id=leader_arm \
    --display_data=true
```

Si el líder mueve el seguidor en tiempo real y ambas cámaras muestran vídeo fluido sin
errores de lectura, el sistema está completamente operativo.
