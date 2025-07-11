# 🎨 Pix2Pix – From Scratch (Architecture Only)

This project contains a **from-scratch PyTorch implementation** of the architecture Pix2Pix

> ⚠️ **Note**: This implementation currently includes **only the model architecture** (Generator + Discriminator) as described in the paper.  
> Training loop, loss functions, and dataset loading are **not yet implemented**.

---

## 🧠 What is Pix2Pix?

Pix2Pix is a **supervised image-to-image translation** method that uses a **conditional GAN**.  
It learns a mapping from input image **A** to output image **B**, given **paired training examples**.

Typical use cases include:
- Sketch → Photo
- Map → Aerial view
- Black-and-white → Color

---

## 🏗️ Implementation Details

### 🔧 Generator — U-Net Architecture
- The generator is a **U-Net**, which consists of:
  - A **contracting path** (encoder) with downsampling via Conv-BatchNorm-LeakyReLU.
  - An **expanding path** (decoder) with upsampling via TransposedConv-BatchNorm-ReLU.
  - **Skip connections** between encoder and decoder layers of the same resolution.

### 🔍 Discriminator — PatchGAN
- The discriminator is a **70×70 PatchGAN**, which classifies each 70×70 patch in the input image as real or fake.
- Architecture: Several Conv-BatchNorm-LeakyReLU layers with increasing filters, ending in a 1-channel output map.
- Instead of a binary decision, it outputs a **grid of probabilities**, encouraging high-frequency correctness.

### 🧮 Loss Functions (Planned)
- **Adversarial Loss**: Binary Cross-Entropy (BCEWithLogitsLoss)
- **L1 Loss**: For pixel-level similarity between generated and target images

---

