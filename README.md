This package provides an implementation of the prediction and analysis of our paper work **"Lightweight vision architecture deployed in the terminal for safety monitoring and early warning of transmission lines"**.

This repository is organized to satisfy the editor’s request by providing the **inference-stage code** of the proposed method. The usage is divided into two parts: **(I) PC-side inference** and **(II) terminal deployment on RKNN (e.g., RK3588)**.

---

## Part I — Run on PC (Inference)

### 1) Recommended environment

* OS: Windows / Linux
* Python: 3.9 (recommended)
* PyTorch: 2.0.1 (recommended for `.pt` checkpoint compatibility)
* OpenCV: opencv-python

> Notes (Windows): If image loading fails (e.g., “Image Not Found”), please avoid non-ASCII paths and move the project/data to an English-only directory such as `C:\repo\Transmission_line_vision\`.

### 2) Installation

Install dependencies using either command:

python -m pip install -r requirements.txt

### 3) GPU support (optional)

If you have an NVIDIA GPU, install a CUDA-enabled PyTorch build according to the official PyTorch instructions for your CUDA version, then install the remaining packages as above.

### 4) Image demo (test.jpg)

Run inference on the demo image:

python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --img-size 640 --source data/test.jpg

Output:

* The prediction result will be saved under `runs/detect/` by default.
* Example output file: `runs/detect/test.jpg`.

Homepage example display (input first, output below):

Input (data/test.jpg):
![Input](data/test.jpg)

Output (runs/detect/test.jpg):
![Output](runs/detect/test.jpg)

### 5) Video demo (video.mp4)

Run inference on the demo video:

python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --img-size 640 --source data/video.mp4

Output:

* The output video will be saved under `runs/detect/` by default.
* The output visualization is consistent with the results shown in the revised manuscript (**Supplementary Video 1.mp4**).

---

## Part II — Run on Terminal Device (RKNN / RK3588)

This section describes the typical terminal deployment workflow:

1. Export the PyTorch model to ONNX using `export.py`
2. Convert ONNX to RKNN using RKNN toolkit
3. Run the RKNN inference demo on the terminal device using `RKNN/demo_rknn.py`

### 1) Export to ONNX (on PC)

Export command (example for RK3588):

python export.py --rknpu RK3588 --weight runs/train/exp/weights/best.pt

After export, an ONNX model file (e.g., `model.onnx`) will be generated (the exact output filename depends on `export.py`).

### 2) Convert ONNX to RKNN (host-side conversion)

Recommended toolchain:

* rknn-toolkit2 (ONNX → RKNN conversion)

Typical conversion pipeline:
best.pt  →  export.py  →  model.onnx  →  rknn-toolkit2  →  model.rknn

Suggested host environment (example):

* OS: Linux (recommended)
* Python: 3.8/3.9 (depending on rknn-toolkit2 version)
* Follow the official Rockchip installation instructions for rknn-toolkit2.

### 3) Run on RKNN terminal device (RK3588)

On the terminal device, enter the RKNN folder and run the demo:

cd RKNN
python demo_rknn.py

Notes:

* Please ensure the RKNN runtime is installed on the terminal device (rknn-runtime).
* Required parameter files are provided under `RKNN/parameters/` (depth/segmentation calibration files), and will be used by the demo script.

### 4) RKNN runtime environment (example)

* Device: RK3588 (or compatible Rockchip NPU platform)
* OS: Linux (Ubuntu-based distributions commonly used)
* Python: 3.8/3.9 (depending on runtime/toolkit)
* Dependencies:

  * rknn-runtime
  * numpy, opencv-python (or headless variants)
  * other dependencies required by `demo_rknn.py`

---

## License

This project is released under the license specified in the LICENSE file.
