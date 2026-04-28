# 🎨 DDPM — Denoising Diffusion Probabilistic Model on CelebA-HQ

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4%20x2-20BEFF?style=flat-square&logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Course](https://img.shields.io/badge/Course-AI4009%20Generative%20AI-purple?style=flat-square)

**A from-scratch PyTorch implementation of DDPM trained on the CelebA-HQ 256×256 face dataset.**
*No Hugging Face. No wrappers. Just raw math and PyTorch.*

</div>

---


## 📋 Overview

This project implements a **Denoising Diffusion Probabilistic Model (DDPM)** from scratch as part of Assignment 4 for the Generative AI course (AI4009), Spring 2026.

The model learns to generate realistic human face images by:
1. **Forward process** — Gradually corrupting real images with Gaussian noise over T=300 timesteps
2. **Reverse process** — Training a U-Net to denoise step-by-step and generate new faces from pure noise

---

## ⚙️ Architecture

```
Input (noisy image + timestep t)
        │
   ┌────▼────┐
   │  Down1  │  3 → 64 channels (DoubleConv)
   └────┬────┘
        │ skip connection ─────────────────────────┐
   ┌────▼────┐                                     │
   │  Down2  │  64 → 128 channels (DoubleConv)     │
   └────┬────┘                                     │
        │ skip connection ──────────────┐           │
   ┌────▼────┐                         │           │
   │  Down3  │  128 → 256 channels     │           │
   └────┬────┘ (bottleneck)            │           │
        │                              │           │
   ┌────▼────┐                         │           │
   │   Up1   │  ConvTranspose2d ◄──────┘           │
   └────┬────┘                                     │
        │                                          │
   ┌────▼────┐                                     │
   │   Up2   │  ConvTranspose2d ◄──────────────────┘
   └────┬────┘
        │
   ┌────▼────┐
   │  Output │  1×1 Conv → predicted noise (3 channels)
   └─────────┘
```

---

## 📊 Training Details & Results

| Parameter | Value |
|-----------|-------|
| Dataset | CelebA-HQ 256×256 |
| Image size (training) | 128×128 |
| Timesteps (T) | 300 |
| Beta schedule | Linear (1e-4 → 0.02) |
| Optimizer | Adam (lr=1e-4) |
| Batch size | 16 |
| Epochs | 10 |
| Loss function | MSE (noise prediction) |
| Gradient clipping | 1.0 |
| Platform | Kaggle T4 × 2 GPU |

| Metric | Score |
|--------|-------|
| PSNR | *1.9098005482008016* dB |
| SSIM | *0.022120968* |

---

## 🗂️ Repository Structure

```
ddpm-celeba-pytorch/
├── ddpm_training.ipynb      # Full training + sampling notebook
├── assets/
│   ├── noising_viz.png          # Forward diffusion visualization
│   ├── generated_samples.png    # Generated face samples
│   └── loss_curve.png           # Training loss curve
├── weights/
│   └── .gitkeep                 # Model weights hosted externally (see below)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ddpm-celeba-pytorch.git
cd ddpm-celeba-pytorch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

The CelebA-HQ dataset is available on Kaggle:
```
https://www.kaggle.com/datasets/denislukovnikov/celebahq256-images-only
```

### 4. Run the notebook

Open `notebooks/ddpm_training.ipynb` in Kaggle or Jupyter and run all cells.

### 5. Download pretrained weights *(optional)*

> Model weights (`ddpm_model.pth`) are hosted externally due to GitHub's file size limit.
>
> 📥 **[Download from Google Drive / Kaggle / Hugging Face]** ← *(update this link)*

---

## 🔬 Key Concepts

**Noise Schedule**
```python
T = 300
betas     = torch.linspace(1e-4, 0.02, T)
alphas    = 1. - betas
alpha_hat = torch.cumprod(alphas, dim=0)
```

**Forward Process** (add noise to any timestep in one step)
```python
def add_noise(x, t):
    noise = torch.randn_like(x)
    x_noisy = sqrt(alpha_hat[t]) * x + sqrt(1 - alpha_hat[t]) * noise
    return x_noisy, noise
```

**Training Objective** — predict the noise added at each timestep
```python
pred_noise = model(noisy_images, t)
loss = MSELoss(pred_noise, noise)
```

**Sampling** — reverse diffusion from pure Gaussian noise
```python
x = torch.randn((n, 3, 128, 128))   # start from noise
for t in reversed(range(T)):
    x = denoise_step(model, x, t)    # remove noise step by step
```

---

## 🖥️ Gradio Demo

A simple web demo is included in the notebook:

```python
import gradio as gr

def generate():
    img = sample(model, 1)
    img = (img.clamp(-1, 1) + 1) / 2
    return img[0].permute(1, 2, 0).cpu().numpy()

gr.Interface(fn=generate, inputs=[], outputs="image").launch()
```

---

## 📦 Requirements

```
torch
torchvision
gradio
numpy
pillow
```
---

## 👤 Author

**Syed Ghani**
Generative AI — AI4009, Spring 2026
FAST-NUCES

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
