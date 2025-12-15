This package provides an implementation of the **inference-stage prediction and analysis code** for our paper:

**“Lightweight vision architecture deployed in the terminal for safety monitoring and early warning of transmission lines.”**

To meet the editor’s request, this repository releases the **complete inference pipeline** of the proposed method. The usage is divided into two parts: **PC-side inference** and **terminal deployment based on RKNN**.

---

## Part I — PC-Side Inference

### Environment

* OS: Windows / Linux
* Python: 3.9 (recommended)
* PyTorch: 2.0.1 (recommended for checkpoint compatibility)
* OpenCV: opencv-python

GPU acceleration is supported if a CUDA-enabled PyTorch version is installed.

---

### Installation

Install all required dependencies with:

python -m pip install -r requirements.txt

---

### Image Inference

Run inference on a single image:

python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --img-size 640 --source data/test.jpg

The inference result will be saved automatically to:

runs/detect/test.jpg

**Example visualization**

Input image (data/test.jpg):

![Input](data/test.jpg)

Output image (runs/detect/test.jpg):

![Output](runs/detect/test.jpg)

---

### Video Inference

Run inference on a video:

python detect.py --weights runs/train/exp/weights/best.pt --conf 0.25 --img-size 640 --source data/video.mp4

The output video will be saved under:

runs/detect/

The resulting visualization and temporal behavior are consistent with those shown in the revised manuscript (**Supplementary Video 1.mp4**).

---

## Part II — Terminal Deployment (RKNN / RK3588)

This part provides a practical deployment pipeline for terminal devices equipped with Rockchip NPUs (e.g., RK3588).

The workflow consists of:

1. Exporting the PyTorch model to ONNX
2. Converting the ONNX model to RKNN format
3. Running inference on the terminal device

---

### Export to ONNX

Export the trained model using:

python export.py --rknpu RK3588 --weight runs/train/exp/weights/best.pt

After execution, an ONNX model file will be generated (the exact filename depends on the export configuration).

---

### ONNX to RKNN Conversion

Use **rknn-toolkit2** to convert the ONNX model into an RKNN model.

Typical pipeline:

best.pt → export.py → model.onnx → rknn-toolkit2 → model.rknn

Recommended host environment:

* OS: Linux
* Python: 3.8 / 3.9 (depending on toolkit version)

Please follow the official Rockchip documentation to install and configure rknn-toolkit2.

---

### Terminal Inference on RK3588

On the terminal device, navigate to the RKNN directory and run:

cd RKNN
python demo_rknn.py

The demo script performs inference using the RKNN model and the provided calibration and parameter files.

---

### Recommended RKNN Runtime Environment

* Device: RK3588 (or compatible Rockchip NPU platform)
* OS: Linux (Ubuntu-based distributions commonly used)
* Python: 3.8 / 3.9
* Runtime dependencies:

  * rknn-runtime
  * numpy
  * opencv-python

---

## License

This project is released under the license specified in the LICENSE file.

---

## Citation

If you use this code for academic or industrial research, please cite our paper accordingly.
