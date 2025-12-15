````markdown
This repository provides the **inference-stage implementation** for our work:

> **"Lightweight vision architecture deployed in the terminal for safety monitoring and early warning of transmission lines"**

It supports:
- **PC inference** (image/video) with Python
- **Terminal deployment** on **RKNN** devices (ONNX → RKNN → edge inference)

---

# Transmission_line_vision

## NEWS
Due to the amount of research involved, the time to maintain this implementation code is very limited. We will compile the complete code and upload it to this repository as soon as possible.

---

## Overview

<p align="center">
  <img src="./video.gif" width="700"/>
</p>

This project provides an inference pipeline for **hazard detection and safety early warning** in transmission line corridors, including:
- Image/video hazard detection inference
- Depth/ranging-related inference utilities (see `trans_depth.py` and `parameters/`)
- RKNN deployment demo for terminal chips (see `RKNN/demo_rknn.py`)

---

## Quick Start

- **PC inference**: run `detect.py` on `data/test.jpg` or `data/video.mp4`
- **RKNN inference**: run `export.py` → convert ONNX to RKNN → run `RKNN/demo_rknn.py`

---

# 1) PC Inference

## 1.1 Environment
- OS: Windows 10/11 or Ubuntu 18.04/20.04/22.04
- Python: **3.8+** (recommended: 3.8 / 3.9)
- Dependencies: see `requirements.txt`

> Tip: Use a virtual environment to avoid conflicts.

---

## 1.2 Installation

### (Optional) Create and activate a virtual environment

**Windows**
```bash
python -m venv .venv
.\.venv\Scripts\activate
````

**Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies (both commands are provided)

```bash
pip install -r requirements.txt
```

```bash
python -m pip install -r requirements.txt
```

---

## 1.3 GPU support (optional)

If you have an NVIDIA GPU, please install a CUDA-enabled PyTorch build according to the official PyTorch instructions for your CUDA version.
(Then install the remaining packages in `requirements.txt` as above.)

---

## 1.4 Inference

### 1) Image demo (`data/test.jpg`)

**Input**

<p align="center">
  <img src="./data/test.jpg" width="520"/>
</p>

**Run**

```bash
python detect.py --source data/test.jpg
```

**Output**

* The result will be saved under:

  * `runs/detect/`
* Example output file:

  * `runs/detect/test.jpg`

**Example result**

<p align="center">
  <img src="./runs/detect/test.jpg" width="520"/>
</p>

---

### 2) Video demo (`data/video.mp4`)

**Run**

```bash
python detect.py --source data/video.mp4
```

**Output**

* The result will be saved under:

  * `runs/detect/`
* Example output file:

  * `runs/detect/video.mp4`

> The visual effect of the output video is consistent with the example shown in the revised manuscript (**Supplementary Video 1.mp4**) as a reference demo.

---

## 1.5 Notes

* If you run inference multiple times, the output directory may be auto-incremented (e.g., `runs/detect/exp`, `runs/detect/exp2`, ...), depending on your script settings.
* The depth/ranging-related parameters are provided under:

  * `parameters/`
* You may need to ensure the corresponding parameter files exist before running the full pipeline:

  * `parameters/hk2_depth.bin`
  * `parameters/hk2_segments.npz`
  * `parameters/parameters_hk2.py`

---

# 2) RKNN Inference (Terminal Deployment)

This section describes the typical workflow:

1. Export model to **ONNX** using `export.py`
2. Convert **ONNX → RKNN**
3. Run **RKNN inference** on a terminal chip using `RKNN/demo_rknn.py`

---

## 2.1 Recommended Runtime Environment

* Device: RK-based terminal chip (e.g., RK3588 class devices)
* OS: Linux (vendor system image or Ubuntu-based)
* Python: **3.8** (commonly used for RKNN toolchains)
* RKNN runtime/toolkit: installed on the device or host (depending on your conversion strategy)

> The RKNN demo code is located in `RKNN/`.

---

## 2.2 Step A — Export to ONNX (on PC)

**Run**

```bash
python export.py --imgsz 640 --output model.onnx
```

**Expected output**

* `model.onnx`

> If your `export.py` requires additional flags (e.g., weights path), please follow the arguments defined in your script.

---

## 2.3 Step B — Convert ONNX to RKNN

There are two common options:

### Option 1 (Recommended): Convert on PC/Host

* Install the RKNN conversion toolkit on the host environment
* Convert ONNX to RKNN
* Copy the `.rknn` model to the device

**Example conversion command (template)**

```bash
python RKNN/convert_onnx_to_rknn.py --onnx model.onnx --rknn model.rknn
```

**Expected output**

* `model.rknn`

> If you do not have `convert_onnx_to_rknn.py`, you can integrate your own RKNN conversion script using the official RKNN toolkit API.

---

### Option 2: Convert on Device

* Install the RKNN toolkit on the device
* Convert ONNX to RKNN directly on the device

This option depends on the device environment and toolchain availability.

---

## 2.4 Step C — Run RKNN demo on terminal chip

### Prepare files

Copy the following to your device:

* `RKNN/demo_rknn.py`
* `RKNN/RK_anchors_mul.txt` (if required by your demo)
* `model.rknn` (or place it under `RKNN/`)
* Test data:

  * `data/test.jpg` and/or `data/video.mp4`
* Parameters (if required by the pipeline):

  * `RKNN/parameters/hk2_depth.bin`
  * `RKNN/parameters/hk2_segments.npz`
  * `RKNN/parameters/parameters_hk2.py`

### Run image inference on RKNN

```bash
python RKNN/demo_rknn.py --model model.rknn --source data/test.jpg
```

### Run video inference on RKNN

```bash
python RKNN/demo_rknn.py --model model.rknn --source data/video.mp4
```

**Output**

* Inference results are typically saved under:

  * `runs/detect/`
* Output filenames may be device/demo dependent (e.g., `test_rknn.jpg`, `video_rknn.mp4`).

---

## 2.5 Notes for RKNN deployment

* If you enable quantization during ONNX→RKNN conversion, you may need a **calibration dataset** for better accuracy.
* For stable FPS on terminal devices:

  * Ensure adequate cooling (heat sink / fan)
  * Use performance mode if supported by the OS
  * Consider reducing input resolution (`--imgsz`) if your demo supports it

---

# License

See `LICENSE`.

---

# Acknowledgements

This repository is released to support academic reproducibility and engineering deployment for transmission line corridor safety monitoring and early warning.

```
```
