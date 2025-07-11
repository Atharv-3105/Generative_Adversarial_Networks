# 🔄 CycleGAN – From Scratch (VanGogh ↔ Realistic Translation)

This project contains a **from-scratch PyTorch implementation** of [CycleGAN](https://arxiv.org/abs/1703.10593), trained on the **VanGogh2Realistic** dataset. It performs **unpaired image-to-image translation** between artistic and realistic domains.

> ⚠️ **Note**: The model has been trained for only a **small number of epochs** due to resource limitations.  
> The results below are from early-stage training and will improve with longer training and tuning.

---

## 🧠 What is CycleGAN?

CycleGAN allows translation between two unpaired image domains using:
- Two generators (`G: X → Y` and `F: Y → X`)
- Two discriminators (`D_X` and `D_Y`)
- **Cycle-consistency loss** to ensure invertibility
- **Identity loss** to retain color/content structure

---

## 🏗️ Model Overview

- **Generators**: ResNet-based (9 blocks)
- **Discriminators**: 70×70 PatchGANs
- **Losses**: Adversarial + Cycle-consistency + Identity
- **Training Technique**: Mixed precision (`torch.cuda.amp`)
- **Data**: VanGogh2Realistic dataset (unpaired)

---

## 🖼️ Results

Below are samples from the model trained for limited epochs:

### 🎨 Van Gogh → Realistic

| Input (Van Gogh) | Output (Fake Realistic) | Cycle Reconstructed |
|------------------|--------------------------|----------------------|
| ![](saved_imgs/real_vangogh_800.png) | ![](saved_imgs/fake_realistic_800.png) | ![](saved_imgs/cycle_vangogh_800.png) |

---

### 🏞️ Realistic → Van Gogh

| Input (Realistic) | Output (Fake Van Gogh) | Cycle Reconstructed |
|-------------------|------------------------|----------------------|
| ![](saved_imgs/real_realistic_800.png) | ![](saved_imgs/fake_vangogh_800.png) | ![](saved_imgs/cycle_realistic_800.png) |


> _Note: Images are saved every 200 iterations and are available in `saved_images/`._

---
