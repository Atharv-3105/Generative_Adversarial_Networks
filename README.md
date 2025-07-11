# 🧠 Generative Adversarial Networks – From Scratch

This repository contains clean, well-structured **from-scratch implementations** of various Generative Adversarial Networks (GANs) using PyTorch. The goal is to provide a learning-focused and fully transparent codebase for understanding how different GAN architectures work under the hood.

---

## 📂 Implemented Architectures

### ✅ 1. [Vanilla GAN (Original GAN Paper)](https://arxiv.org/abs/1406.2661)
> *Generative Adversarial Networks* by Ian Goodfellow et al., 2014

- Objective: Minimize the Jensen-Shannon divergence between real and generated data.
- Implemented using basic Network Layers.

---

### ✅ 2. [Deep Convolutional GAN (DCGAN)](https://arxiv.org/abs/1511.06434)
> *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* by Radford et al., 2015

- Generator and Discriminator are built using CNNs.
- Works well with image datasets like MNIST, CIFAR-10.

---

### ✅ 3. [Wasserstein GAN (WGAN)](https://arxiv.org/abs/1701.07875)
> *Wasserstein GAN* by Arjovsky et al., 2017

- Improves GAN training stability by using Earth-Mover (Wasserstein) distance.
- Replaces sigmoid + BCE loss with critic + Wasserstein loss.
- Includes weight clipping to enforce the Lipschitz constraint.

---

### ✅ 4. [Pix2Pix (Conditional GAN)](https://arxiv.org/abs/1611.07004)
> *Image-to-Image Translation with Conditional Adversarial Networks* by Isola et al., 2017

- A supervised image-to-image translation framework.
- Trained on paired datasets like edges-to-photo or sketches-to-image.
- Generator learns a mapping from input image to target output.

---

### ✅ 5. [CycleGAN](https://arxiv.org/abs/1703.10593)
> *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* by Zhu et al., 2017

- An unsupervised image-to-image translation model for **unpaired** datasets.
- Uses cycle consistency loss to translate between domains (e.g., horses ↔ zebras, photos ↔ Van Gogh paintings).
- Implements two generators and two discriminators with identity and cycle-consistency loss.

---
