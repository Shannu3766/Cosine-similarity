# ============================================================
# adaptive_lora_gradient/importance.py
# Gradient × Activation based Importance (s_i)
# ============================================================

import torch
from torch.utils.data import DataLoader
from typing import Dict
import logging
from .utils import get_lora_layers

logger = logging.getLogger(__name__)

def compute_gradient_importance_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 2
) -> Dict[str, float]:
    """
    Computes per-layer importance scores using:
        s_i = (1/K) * Σ_k [ (∂L/∂h_i^(k)) · h_i^(k) ]
    
    where h_i^(k) is the activation output of LoRA layer i for sample k.

    Args:
        model: PEFT-wrapped model (with LoRA layers).
        dataloader: Validation DataLoader (small subset).
        device: torch.device (GPU or CPU).
        num_batches: number of batches to sample for averaging (K).
    
    Returns:
        Dict[str, float]: Normalized importance score per layer.
    """
    model.eval()
    lora_layers = get_lora_layers(model)
    if not lora_layers:
        logger.warning("No LoRA layers found. Returning empty scores.")
        return {}

    # Storage for activations and hooks
    activations = {name: [] for name in lora_layers.keys()}
    hooks = []

    # ============================================================
    # ✅ Hook: retain gradients on original outputs (no detach)
    # ============================================================
    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                out.retain_grad()
                activations[name].append(out)
            elif isinstance(out, (tuple, list)):
                for elem in out:
                    if isinstance(elem, torch.Tensor):
                        elem.retain_grad()
                        activations[name].append(elem)
                        break
        return hook

    for name, layer in lora_layers.items():
        try:
            hooks.append(layer.register_forward_hook(make_hook(name)))
        except Exception as e:
            logger.warning(f"Failed to register hook for {name}: {e}")

    data_iter = iter(dataloader)
    model.zero_grad(set_to_none=True)
    batches_used = 0

    # ============================================================
    # 🔁 Forward + Backward on few batches
    # ============================================================
    with torch.enable_grad():
        for _ in range(min(num_batches, len(dataloader))):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            # Move tensors to device
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Ensure model outputs loss
            outputs = model(**batch)
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            elif hasattr(outputs, "loss"):
                loss = outputs.loss
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                loss = outputs[0]
            else:
                logger.error("Model output has no loss. Ensure 'labels' in batch.")
                continue

            model.zero_grad(set_to_none=True)
            loss.backward(retain_graph=False)
            batches_used += 1

    # ============================================================
    # 📊 Compute per-layer (grad · act) averages
    # ============================================================
    raw_scores = {}
    for name, acts in activations.items():
        if not acts:
            raw_scores[name] = 0.0
            continue

        per_layer_vals = []
        for act in acts:
            if act.grad is None:
                continue

            # Move to CPU for safety
            a = act.detach().cpu()
            g = act.grad.detach().cpu()

            # Flatten last dim (hidden)
            if a.dim() < 2:
                continue
            hidden_dim = a.size(-1)
            a_flat = a.view(-1, hidden_dim)
            g_flat = g.view(-1, hidden_dim)

            # Compute dot product per token, then mean
            dots = (a_flat * g_flat).sum(dim=1)
            per_layer_vals.append(dots.mean().item())

            # Free memory early
            del a, g, a_flat, g_flat, dots
            torch.cuda.empty_cache()

        raw_scores[name] = float(sum(per_layer_vals) / len(per_layer_vals)) if per_layer_vals else 0.0

    # Remove hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    if batches_used == 0 or not raw_scores:
        logger.warning("No batches processed or no raw scores computed. Returning zeros.")
        return {k: 0.0 for k in lora_layers.keys()}

    # ============================================================
    # 🔄 Normalize to [0, 1]
    # ============================================================
    s_vals = torch.tensor(list(raw_scores.values()), dtype=torch.float32)
    eps = 1e-8
    s_min, s_max = s_vals.min(), s_vals.max()

    if (s_max - s_min) < eps:
        normed = {k: 0.0 for k in raw_scores.keys()}
    else:
        s_norm = (s_vals - s_min) / (s_max - s_min)
        normed = {k: float(v) for k, v in zip(raw_scores.keys(), s_norm)}

    model.train()
    return normed
