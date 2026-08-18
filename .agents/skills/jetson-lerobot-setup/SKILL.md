---
name: jetson-lerobot-setup
description: Guide and troubleshoot setting up Hugging Face LeRobot (v0.5.2+) with Python 3.12 and PyTorch on NVIDIA Jetson Orin Nano (JetPack 6.x). Use when working on Jetson Orin Nano environment configuration, resolving PyTorch/CUDA/GLIBC/pip dependencies, patching Python 3.12 LeRobot code, optimizing storage, or managing motor/teleoperation setup on edge devices.
---

# Jetson Orin Nano LeRobot Setup and Troubleshooting

A comprehensive guide and reference for configuring Hugging Face LeRobot 0.5.2+ on NVIDIA Jetson Orin Nano running JetPack 6.2.1 (L4T 36.4.4 / Ubuntu 22.04 LTS), resolving ARM64 dependency bottlenecks, and managing edge teleoperation workflows.

## 1. System Specifications and Architecture

### 1.1 Hardware and Software Matrix

| Category | Specification / Version | Notes |
| --- | --- | --- |
| **Target Device (Edge)** | NVIDIA Jetson Orin Nano Developer Kit | ARM64 (`aarch64`) architecture |
| **OS / JetPack** | JetPack 6.2.1 / L4T R36.4.4 | Ubuntu 22.04 LTS (`jammy`) |
| **NVIDIA Driver** | 540.4.0 (CUDA 12.6 driver support) | System level |
| **CUDA Toolkit** | CUDA 12.6.68 (`nvcc 12.6`) | System level |
| **Python Runtime** | Python 3.12.13 | Conda virtual environment (`lerobot312`) |
| **LeRobot Framework** | `v0.5.2` | Local repository clone (`~/lerobot`) |
| **PyTorch / TorchVision** | PyTorch 2.11.0 / TorchVision 0.26.0 | `conda-forge` CPU build (self-contained `.so`) |

### 1.2 Edge vs Server Role Division

```
[SO-100 / SO-101 Arm Motors] + [UVC Cameras (Top / Wrist; Side/Belly unused)]
                        ▲
                        │ (USB Serial & UVC Video)
                        ▼
┌───────────────────────────────────────────────────────────────┐
│ Jetson Orin Nano (Edge Client)                                │
│  - Motor calibration (`lerobot-calibrate`)                    │
│  - Real-time leader-follower teleoperation (`lerobot-teleop`) │
│  - Video and joint dataset logging (`lerobot-record`)         │
│  - Lightweight CPU tensor packing, safety clamp & serial I/O  │
└───────────────────────────────────────────────────────────────┘
                        ▲
                        │ (Network / Tailscale async stream)
                        ▼
┌───────────────────────────────────────────────────────────────┐
│ Remote GPU Server / Workstation                               │
│  - Policy model training (ACT, SmolVLA, Diffusion)            │
│  - Heavy neural network async inference (`policy_server`)     │
└───────────────────────────────────────────────────────────────┘
```

> **Rationale for CPU-mode PyTorch on Jetson:**
> - The edge device only reads/writes motor angles (tensors), runs safety limiters, captures video frames, and streams observations. These tasks are 100% CPU and serial/network I/O bound.
> - Heavy model training and policy inference are handled by the remote GPU server. Using CPU-mode PyTorch avoids brittle PyPI/CUDA 13.0 wheel mismatches on Jetson ARM64.

---

## 2. Root Cause Analysis of Known Failures (Troubleshooting Archive)

### ① Python 3.10 vs 3.12 Syntax Incompatibility (`SyntaxError`)
- **Symptom:** Running LeRobot 0.5.2 on Python 3.10 fails during import with:
  ```python
  def deserialize_json_into_object[T: JsonLike](...) -> T:
                                  ^
  SyntaxError: invalid syntax
  ```
- **Cause:** LeRobot 0.5.2 codebase (`src/lerobot/utils/io_utils.py`, etc.) utilizes PEP 695 generic function syntax introduced in Python 3.12.
- **Resolution:** Python 3.12 is strictly required. Do not use Python 3.10 environments for LeRobot 0.5.2+.

### ② PyPI `pip` CUDA 13.0 Wheel and Missing `.so` Libraries
- **Symptom:** Running `pip install torch` in Python 3.12 downloads PyPI `torch 2.11.0+cu130` wheels, which immediately crash on `import torch` with missing symbols (`libnccl.so.2`, `libnvshmem_host.so.3`).
- **Cause:** PyPI defaults to CUDA 13.0 binaries that are incompatible with JetPack 6.2.1's CUDA 12.6 driver and lack bundled ARM64 NCCL/NVSHMEM runtimes.
- **Resolution:** Install the self-contained CPU binary from `conda-forge`: `conda install -c conda-forge pytorch=2.11.0 torchvision -y`.

### ③ GLIBC Version Mismatch (`GLIBC 2.38` vs `2.35`) and Source Build OOM
- **Symptom:** Third-party aarch64 Python 3.12 wheels refuse to load due to `GLIBC_2.38 not found`, while compiling PyTorch from source causes Jetson Orin Nano to crash/hang due to Out-Of-Memory (OOM).
- **Cause:** JetPack 6.2.1 (Ubuntu 22.04) ships with `GLIBC 2.35`. Wheels compiled against Ubuntu 23.10+ fail to link. Source compilation exceeds memory limits.
- **Resolution:** `conda-forge` packages are strictly built against compatible glibc baselines and include required runtime libraries.

### ④ Destructive Overwrites by `pip install -e .`
- **Symptom:** Executing bare `pip install -e .` inside `~/lerobot` triggers `pyproject.toml` dependency resolution, uninstalling the functional Conda PyTorch and reinstalling broken PyPI wheels.
- **Resolution:** Always use `pip install --no-deps -e .` to register the LeRobot package without touching the conda-managed PyTorch installation.

### ⑤ PyAV Python 3.12 Eager Type Evaluation (`AttributeError: module 'av' has no attribute 'option'`)
- **Symptom:** Running `lerobot-record --help` fails with:
  ```text
  File ".../lerobot/datasets/pyav_utils.py", line 44, in <module>
      def _get_codec_options_by_name(vcodec: str) -> dict[str, av.option.Option]:
                                                                ^^^^^^^^^
  AttributeError: module 'av' has no attribute 'option'
  ```
- **Cause:** In Python 3.12, function type annotations are evaluated eagerly unless postponed evaluation is enabled. `av.option` is a subpackage not loaded at top-level import.
- **Resolution:** Reinstall `av` from `conda-forge` and insert `from __future__ import annotations` at line 1 of `src/lerobot/datasets/pyav_utils.py`.

---

## 3. Step-by-Step Setup Workflow

### Step 1: Create Conda Environment and Install PyTorch
```bash
conda create -n lerobot312 python=3.12 -y
conda activate lerobot312
conda install -c conda-forge pytorch=2.11.0 torchvision -y
```

### Step 2: Install LeRobot in Isolated Mode
```bash
cd ~/lerobot
# Register LeRobot without overwriting conda dependencies
pip install --no-deps -e .

# Install required runtime dependencies individually
pip install datasets pyarrow fsspec pandas h5py zarr rerun-sdk draccus huggingface_hub diffusers cmake numba einops tqdm opencv-python imageio termcolor

# Remove incompatible ARM64 CUDA artifacts if present
pip uninstall -y nvidia-cusparselt-cu13
```

### Step 3: Patch PyAV and Enable Postponed Evaluation
```bash
# Reinstall clean PyAV binary from conda-forge
pip uninstall -y av
conda install -c conda-forge av -y

# Insert postponed evaluation into pyav_utils.py
sed -i '1s/^/from __future__ import annotations\n/' ~/lerobot/src/lerobot/datasets/pyav_utils.py
```

### Step 4: Verification and Sanity Checks
```bash
# Verify PyTorch CPU and version
python3 -c "import torch; print('PyTorch OK:', torch.__version__)"

# Verify LeRobot package import
python3 -c "import lerobot; print('LeRobot version:', lerobot.__version__)"

# Verify PyAV import
python3 -c "import av; print('PyAV Version:', av.__version__)"

# Verify dependency consistency
pip check

# Test LeRobot CLI entry points
lerobot-calibrate --help
lerobot-teleoperate --help
lerobot-record --help
```

---

## 4. Jetson Disk Optimization (Reclaim 20GB+)

When setting up multiple environments on Jetson Orin Nano, storage quickly runs out. Clean caches and unused environments:

```bash
# 1. Purge pip wheel download cache (~4.8 GB)
pip cache purge

# 2. Clean conda tarballs, index, and package cache (~7.0 GB)
conda clean --all -y

# 3. Remove obsolete virtual environments (~8.0 GB+)
rm -rf ~/miniforge3/envs/lerobot ~/miniforge3/envs/lerobot310 ~/miniforge3/envs/'lerobot_gpu '

# 4. Remove residual CUDA 13 site-packages (~1.7 GB)
rm -rf ~/miniforge3/envs/lerobot312/lib/python3.12/site-packages/nvidia/cu13
```

Confirm active environments:
```bash
conda env list
# base
# lerobot312  <-- Primary working environment
```

---

## 5. Hardware Setup and Follow-up Actions

### 5.1 Serial Permissions and Motor SDK
```bash
# Add current user to dialout group for serial access
sudo usermod -aG dialout $USER

# Install Feetech motor SDK for STS3215 servos (SO-100 / SO-101)
pip install feetech-servo-sdk
```

### 5.2 Device Alias Configuration
Ensure persistent udev rules exist for serial ports and cameras:
- Follower arm: `/dev/so101_follower`
- Leader arm: `/dev/so101_leader`
- Cameras: `/dev/cam_top`, `/dev/cam_wrist` (Side/Belly camera `/dev/cam_belly` is unused)

### 5.3 JetPack 7.2 Upgrade Fallback
If CPU-only execution is insufficient for customized real-time edge processing:
- Upgrade to **JetPack 7.2** which provides official Python 3.12 and modern CUDA wheel support.
- **Crucial Note:** JetPack 7.2+ does **not** support microSD card flashing on Orin Nano DevKit; flashing must be performed via a **USB drive (ISO image)** or SDK Manager.
