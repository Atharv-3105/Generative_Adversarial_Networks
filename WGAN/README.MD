# 🌊 Wasserstein GAN (WGAN) – From Scratch (MNIST)

This project implements a **Wasserstein GAN (WGAN)** from scratch using **PyTorch**.

> ⚠️ **Note**: This implementation is complete, but the model has **not yet been trained** on any dataset. Training and results will be added in future updates.

---

## 🎯 Objective

Train a GAN that generates realistic 28×28 handwritten digits using:
- A **generator** that maps noise vectors to image space.
- A **critic** (not a discriminator) that scores real and fake images using the Wasserstein distance.

---

## 🧱 Key Differences from Vanilla GAN

- Uses a **critic** instead of a discriminator (no sigmoid activation).
- No binary cross-entropy loss — uses Wasserstein loss.
- Enforces a **Lipschitz constraint** via **weight clipping**.
- More stable training, especially with low-dimensional data.

---
