# importancescore

A lightweight Python package for **dynamic LoRA fine-tuning** using **importance-based adaptive rank allocation**, built on PyTorch.

---

## 🚀 Features

- Computes **importance scores** from gradients × activations:
  \[
  s_i = \frac{1}{K} \sum_{k=1}^K \left| \frac{\partial L}{\partial h_i^{(k)}} \cdot h_i^{(k)} \right|
  \]
- Dynamically allocates LoRA ranks via softmax:
  \[
  r_i = R \cdot \frac{e^{s_i/\tau}}{\sum_j e^{s_j/\tau}}
  \]
- Recomputes importance and reallocates ranks during training.
- Fully modular design — can plug into custom or transformer models.

---

## 🧱 Installation

```bash
pip install .
