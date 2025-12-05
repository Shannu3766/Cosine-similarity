# import torch
# from torch.utils.data import DataLoader
# from typing import Dict, Optional
# import logging
# from tqdm.auto import tqdm
# from .utils import get_lora_layers

# logger = logging.getLogger(__name__)

# def compute_gradient_importance_scores(
#     model: torch.nn.Module,
#     dataloader: DataLoader,
#     device: torch.device,
#     num_batches: Optional[int] = None, # None = Process All
#     batch_size: int = 8                # Controls memory usage per step
# ) -> Dict[str, float]:
#     """
#     Computes per-layer importance scores iteratively (Memory Safe).
#     Calculates (Activation * Gradient) per batch and accumulates the sum.
#     """
#     model.eval()
#     lora_layers = get_lora_layers(model)
#     if not lora_layers:
#         logger.warning("No LoRA layers found. Returning empty scores.")
#         return {}

#     # 1. Adjust Batch Size if needed
#     if batch_size != dataloader.batch_size:
#         logger.info(f"Adjusting DataLoader batch size from {dataloader.batch_size} to {batch_size}")
#         dataloader = DataLoader(
#             dataset=dataloader.dataset,
#             batch_size=batch_size,
#             shuffle=False, 
#             collate_fn=dataloader.collate_fn, 
#             num_workers=dataloader.num_workers,
#             pin_memory=dataloader.pin_memory
#         )

#     # Dictionary to hold the running total of importance scores
#     # accumulated_scores[layer_name] = cumulative_score
#     accumulated_scores = {name: 0.0 for name in lora_layers.keys()}
    
#     # Temporary storage for the CURRENT batch's activations
#     current_batch_activations = {}

#     # 2. Register Hooks (Overwrites storage per batch)
#     def make_hook(name):
#         def hook(module, inp, out):
#             # We only care about the output tensor
#             if isinstance(out, torch.Tensor):
#                 out.retain_grad() # Crucial: enables .grad on non-leaf tensor
#                 current_batch_activations[name] = out
#             elif isinstance(out, (tuple, list)):
#                 for elem in out:
#                     if isinstance(elem, torch.Tensor):
#                         elem.retain_grad()
#                         current_batch_activations[name] = elem
#                         break
#         return hook

#     hooks = []
#     for name, layer in lora_layers.items():
#         try:
#             hooks.append(layer.register_forward_hook(make_hook(name)))
#         except Exception as e:
#             logger.warning(f"Failed to register hook for {name}: {e}")

#     # Determine loop length
#     total_steps = len(dataloader)
#     if num_batches is not None:
#         total_steps = min(num_batches, total_steps)

#     batches_processed = 0

#     # 3. Iterative Processing Loop
#     model.zero_grad(set_to_none=True)
    
#     with torch.enable_grad():
#         for step, batch in tqdm(enumerate(dataloader), total=total_steps, desc="Computing Importance", leave=False):
            
#             if num_batches is not None and step >= num_batches:
#                 break

#             # --- A. Move Batch to Device ---
#             if hasattr(batch, "items"):
#                 batch_input = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
#                 outputs = model(**batch_input)
#             elif isinstance(batch, (list, tuple)):
#                 batch_input = [b.to(device) for b in batch if isinstance(b, torch.Tensor)]
#                 outputs = model(*batch_input)
#             else: # Tensor
#                 outputs = model(batch.to(device))

#             # --- B. Get Loss ---
#             if isinstance(outputs, dict) and "loss" in outputs:
#                 loss = outputs["loss"]
#             elif hasattr(outputs, "loss"):
#                 loss = outputs.loss
#             elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
#                 loss = outputs[0]
#             else:
#                 continue

#             # --- C. Backward Pass ---
#             # This populates .grad on the tensors stored in current_batch_activations
#             loss.backward()

#             # --- D. Compute Score for this Batch IMMEDIATEY ---
#             # We calculate and add to total, then discard tensors to free memory.
#             for name, act in current_batch_activations.items():
#                 if act.grad is None:
#                     continue

#                 # Move to CPU to save GPU memory during math, or keep on GPU if speed is priority
#                 # Using GPU (device) is faster, CPU is safer for VRAM.
#                 # Here we keep on device for speed, assuming batch_size is small enough.
                
#                 # Flatten: [Batch, Seq_Len, Hidden] -> [N, Hidden]
#                 a_flat = act.detach().view(-1, act.size(-1))
#                 g_flat = act.grad.detach().view(-1, act.size(-1))

#                 # Dot product: (Activation * Gradient)
#                 # Sum across dimensions, then average over the batch
#                 importance = (a_flat * g_flat).sum(dim=1).mean().item()
                
#                 # Add to running total
#                 accumulated_scores[name] += importance

#             # --- E. Cleanup for Next Batch ---
#             # Clear references to release memory
#             current_batch_activations.clear()
#             model.zero_grad(set_to_none=True)
#             batches_processed += 1
            
#             # Explicit cache clear (optional, helps if VRAM is very tight)
#             # torch.cuda.empty_cache() 

#     # Cleanup Hooks
#     for h in hooks:
#         h.remove()

#     if batches_processed == 0:
#         return {k: 0.0 for k in lora_layers.keys()}

#     # 4. Average and Normalize
#     # Divide cumulative sum by number of batches to get the average
#     final_scores = {k: v / batches_processed for k, v in accumulated_scores.items()}

#     s_vals = torch.tensor(list(final_scores.values()), dtype=torch.float32)
#     eps = 1e-8
#     s_min, s_max = s_vals.min(), s_vals.max()

#     if (s_max - s_min) < eps:
#         normed = {k: 0.0 for k in final_scores.keys()}
#     else:
#         s_norm = (s_vals - s_min) / (s_max - s_min)
#         normed = {k: float(v) for k, v in zip(final_scores.keys(), s_norm)}

#     model.train()
#     return normed


# file: importance.py  (replace function body)
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging
from tqdm.auto import tqdm
from .utils import get_lora_layers

logger = logging.getLogger(__name__)

def _safe_tensor_from_output(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for elem in out:
            if isinstance(elem, torch.Tensor):
                return elem
    if isinstance(out, dict):
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
    return None

def compute_gradient_importance_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: Optional[int] = None, # None = Process All
    batch_size: int = 8                # Controls memory usage per step
) -> Dict[str, float]:
    """
    Implements s_i = (1/K) * sum_k | <dL/dh_i^{(k)}, h_i^{(k)}> |
    where <.,.> is the dot product across hidden units for each example/position.
    """
    model.eval()
    lora_layers = get_lora_layers(model)
    if not lora_layers:
        logger.warning("No LoRA layers found. Returning empty scores.")
        return {}

    # If caller requested a smaller batch for importance pass, create a new DataLoader
    if hasattr(dataloader, "batch_size") and batch_size != dataloader.batch_size:
        dataloader = DataLoader(
            dataset=dataloader.dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=getattr(dataloader, "collate_fn", None),
            num_workers=getattr(dataloader, "num_workers", 0),
            pin_memory=getattr(dataloader, "pin_memory", False)
        )

    # accumulate a scalar importance per layer per batch, then average later
    accumulated_scores = {name: 0.0 for name in lora_layers.keys()}
    hooks = []
    current_batch_activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            t = _safe_tensor_from_output(out)
            if t is None:
                return
            # retain_grad so .grad is populated on this tensor after backward()
            t.retain_grad()
            current_batch_activations[name] = t
        return hook

    for name, layer in lora_layers.items():
        try:
            hooks.append(layer.register_forward_hook(make_hook(name)))
        except Exception as e:
            logger.warning(f"Failed to register hook for {name}: {e}")

    total_steps = len(dataloader)
    if num_batches is not None:
        total_steps = min(num_batches, total_steps)

    batches_processed = 0
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        for step, batch in tqdm(enumerate(dataloader), total=total_steps, desc="Computing Importance", leave=False):
            if num_batches is not None and step >= num_batches:
                break

            # Move input tensors to device and call model
            if hasattr(batch, "items"):
                batch_input = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = model(**batch_input)
            elif isinstance(batch, (list, tuple)):
                batch_input = [b.to(device) for b in batch if isinstance(b, torch.Tensor)]
                outputs = model(*batch_input)
            else:
                outputs = model(batch.to(device))

            # Extract loss (prefer trainer-provided loss)
            loss = None
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            elif hasattr(outputs, "loss"):
                loss = outputs.loss
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                # In some cases outputs[0] might be loss but usually it's logits.
                # If it's not a loss tensor with grad_fn, we skip the batch.
                loss = outputs[0] if hasattr(outputs[0], "grad_fn") else None

            if loss is None:
                # Can't backprop without a scalar loss -> skip batch
                current_batch_activations.clear()
                continue

            # Backprop to populate .grad on retained activation tensors
            loss.backward()

            # For each registered activation, compute per-position dot across hidden,
            # then take absolute and average (this matches the formula in the screenshot).
            for name, act in list(current_batch_activations.items()):
                if act is None or act.grad is None:
                    continue

                # detach activation and gradient to avoid further autograd linkage
                a = act.detach()
                g = act.grad.detach()

                # We want per-example/per-position dot across the last (hidden) dim.
                # Ensure we treat last dimension as hidden if tensor has >=2 dims.
                if a.ndim >= 2:
                    hidden = a.shape[-1]
                    a_flat = a.reshape(-1, hidden)   # shape: [N, H]
                    g_flat = g.reshape(-1, hidden)   # shape: [N, H]
                else:
                    # fallback for unusual shapes
                    a_flat = a.reshape(-1, 1)
                    g_flat = g.reshape(-1, 1)

                # per-position dot: sum over hidden units -> shape [N]
                per_row_dot = (a_flat * g_flat).sum(dim=1)

                # take absolute value per example then average across positions
                batch_importance = per_row_dot.abs().mean().cpu().item()

                # accumulate scalar
                accumulated_scores[name] += batch_importance

            # cleanup for next batch
            current_batch_activations.clear()
            model.zero_grad(set_to_none=True)
            batches_processed += 1

    # remove hooks safely
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    if batches_processed == 0:
        return {k: 0.0 for k in lora_layers.keys()}

    # average across batches -> this is the s_i before normalization
    final_scores = {k: v / batches_processed for k, v in accumulated_scores.items()}

    # Min-max normalize to [0,1] across layers (same as earlier)
    s_vals = torch.tensor(list(final_scores.values()), dtype=torch.float32)
    eps = 1e-8
    s_min, s_max = s_vals.min(), s_vals.max()
    if (s_max - s_min) < eps:
        normed = {k: 0.0 for k in final_scores.keys()}
    else:
        s_norm = (s_vals - s_min) / (s_max - s_min)
        normed = {k: float(v) for k, v in zip(final_scores.keys(), s_norm)}

    model.train()
    return normed
