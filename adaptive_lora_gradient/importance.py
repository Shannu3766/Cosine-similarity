import torch
from torch.utils.data import DataLoader
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def compute_gradient_importance_scores(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, num_batches: int = 3) -> Dict[str, float]:
    """
    Compute importance scores s_i using the formula:
        s_i = (1/K) * sum_{k=1..K} (dL/dh_i^{(k)} · h_i^{(k)})
    where the dot is the elementwise inner product across hidden dimension for each token/item,
    averaged over tokens/items and batches.

    Args:
        model: PEFT-wrapped model (must accept labels in the batch to produce loss).
        dataloader: DataLoader that yields dicts of tensors (including 'labels').
        device: device to run the computation on.
        num_batches: number of batches from dataloader to use (K). Keep small for speed/memory.

    Returns:
        dict mapping layer_name -> normalized importance score in [0,1].
    """
    model.eval()

    # Find LoRA layers (same util you used earlier)
    from .utils import get_lora_layers
    lora_layers = get_lora_layers(model)
    if not lora_layers:
        logger.warning("No LoRA layers found. Returning empty scores.")
        return {}

    # Storage for activations and hooks
    activations = {name: [] for name in lora_layers.keys()}
    hooks = []

    # Forward hook: save activation and call retain_grad() so grad appears after backward
    def make_forward_hook(name):
        def hook(module, inp, out):
            # out is typically a Tensor (e.g., [B, S, H]) or tuple; handle Tensor only
            if isinstance(out, torch.Tensor):
                # Ensure gradient will be retained on non-leaf tensors
                out = out.detach()
                out.requires_grad_(True)
                out.retain_grad()
                activations[name].append(out)
            else:
                # If out is tuple, try first Tensor element
                for x in out:
                    if isinstance(x, torch.Tensor):
                        x = x.detach()
                        x.requires_grad_(True)
                        x.retain_grad()
                        activations[name].append(x)
                        break
        return hook

    # Register hooks
    for name, layer in lora_layers.items():
        try:
            hooks.append(layer.register_forward_hook(make_forward_hook(name)))
        except Exception as e:
            logger.warning(f"Failed to register hook on {name}: {e}")

    # Iterate a few batches and compute gradients w.r.t. activations
    batches_used = 0
    # Ensure model grads are zeroed
    model.zero_grad(set_to_none=True)

    data_iter = iter(dataloader)
    with torch.enable_grad():
        for _ in range(min(num_batches, len(dataloader))):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            # Move tensors to device (only tensors)
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Forward pass with labels so model returns loss
            try:
                outputs = model(**batch)
            except Exception as e:
                logger.error(f"Model forward failed during BI computation: {e}")
                break

            # model outputs might be a dict with 'loss' or a tuple; try to extract loss
            loss = None
            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                # some models return (loss, logits, ...)
                loss = outputs[0] if isinstance(outputs[0], torch.Tensor) else None

            if loss is None:
                logger.error("Could not obtain loss from model outputs. Ensure the batch contains 'labels'.")
                break

            # Backpropagate to populate activation gradients (no optimizer step)
            model.zero_grad(set_to_none=True)  # clear previous grads
            loss.backward(retain_graph=False)
            batches_used += 1

    # Now compute the per-layer score using collected activations and their grads
    raw_scores = {}
    for name, acts in activations.items():
        if not acts:
            logger.warning(f"No activations captured for layer {name}. Skipping.")
            continue

        # For each captured activation in this layer, get its grad and compute inner product
        per_batch_values = []
        for act in acts:
            # act is on device (we created it on device), grad should be populated on same device
            if act.grad is None:
                # sometimes grad isn't stored (e.g., if nothing depended on it) — skip safely
                continue

            # Move to CPU to avoid keeping GPU memory
            a = act.detach().cpu()
            g = act.grad.detach().cpu()

            # Shape handling: expect [..., hidden_dim]
            # Flatten leading dims into N and keep last dim as hidden
            if a.dim() < 1 or g.dim() < 1:
                continue

            hidden_dim = a.size(-1)
            a_flat = a.view(-1, hidden_dim)   # [N, H]
            g_flat = g.view(-1, hidden_dim)   # [N, H]

            # Per-token/item dot product (g · a) for each row, then average across rows
            # This yields a scalar per activation tensor
            dots = (g_flat * a_flat).sum(dim=1)  # [N]
            mean_dot = dots.mean().item()        # scalar
            per_batch_values.append(mean_dot)

        if not per_batch_values:
            # If nothing valid was computed, fall back to 0
            raw_scores[name] = 0.0
        else:
            # Average across all captured activation tensors in this layer
            raw_scores[name] = float(sum(per_batch_values) / len(per_batch_values))

    # Clean up hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    # If no batches were used, return zeros
    if batches_used == 0 or not raw_scores:
        logger.warning("No batches processed or no raw scores computed. Returning empty dict.")
        return {}

    # Normalize raw_scores to [0,1] (same normalization as before)
    s_vals = torch.tensor(list(raw_scores.values()), dtype=torch.float32)
    eps = 1e-8
    s_min, s_max = float(s_vals.min()), float(s_vals.max())
    if (s_max - s_min) < eps:
        # nearly uniform: return zeros or small constants
        normed = {k: 0.0 for k in raw_scores.keys()}
    else:
        normed_list = ((s_vals - s_min) / (s_max - s_min)).tolist()
        normed = {k: float(v) for k, v in zip(raw_scores.keys(), normed_list)}

    # Put model back to train mode
    model.train()
    return normed
