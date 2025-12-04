import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging
from tqdm.auto import tqdm
from .utils import get_lora_layers

logger = logging.getLogger(__name__)

def compute_gradient_importance_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: Optional[int] = None, # <--- Modified: None means "Process All"
    batch_size: int = 8
) -> Dict[str, float]:
    """
    Computes per-layer importance scores using the dataset.
    If num_batches is None, processes the ENTIRE dataset.
    """
    model.eval()
    lora_layers = get_lora_layers(model)
    if not lora_layers:
        logger.warning("No LoRA layers found. Returning empty scores.")
        return {}

    # 1. Adjust Batch Size if needed
    if batch_size != dataloader.batch_size:
        logger.info(f"Adjusting DataLoader batch size from {dataloader.batch_size} to {batch_size}")
        dataloader = DataLoader(
            dataset=dataloader.dataset,
            batch_size=batch_size,
            shuffle=False, 
            collate_fn=dataloader.collate_fn, 
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory
        )

    # Storage for activations
    activations = {name: [] for name in lora_layers.keys()}
    hooks = []

    # 2. Register Hooks
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

    model.zero_grad(set_to_none=True)
    batches_used = 0

    # ============================================================
    # 🆕 Logic: Determine Total Steps for Progress Bar
    # ============================================================
    if num_batches is None:
        total_steps = len(dataloader)
    else:
        total_steps = min(num_batches, len(dataloader))

    # 3. Forward + Backward Loop
    with torch.enable_grad():
        # 🆕 Iterate directly over dataloader with enumerate
        for step, batch in tqdm(enumerate(dataloader), total=total_steps, desc="Computing Importance", leave=False):
            
            # Stop if we hit the user-defined limit (if any)
            if num_batches is not None and step >= num_batches:
                break

            # Handle BatchEncoding / Dict / List
            if hasattr(batch, "items"):
                batch_input = {
                    k: v.to(device) for k, v in batch.items() 
                    if isinstance(v, torch.Tensor)
                }
                outputs = model(**batch_input)
            
            elif isinstance(batch, (list, tuple)):
                batch_input = [b.to(device) for b in batch if isinstance(b, torch.Tensor)]
                outputs = model(*batch_input)
            
            else:
                if isinstance(batch, torch.Tensor):
                    batch_input = batch.to(device)
                    outputs = model(batch_input)
                else:
                    logger.error(f"Unknown batch type: {type(batch)}. Skipping.")
                    continue

            # Extract loss
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

    # 4. Compute Scores
    # ... (Rest of your existing scoring logic remains identical) ...
    raw_scores = {}
    for name, acts in activations.items():
        if not acts:
            raw_scores[name] = 0.0
            continue

        per_layer_vals = []
        for act in acts:
            if act.grad is None:
                continue

            a = act.detach().cpu()
            g = act.grad.detach().cpu()

            if a.dim() < 2:
                continue
            
            hidden_dim = a.size(-1)
            a_flat = a.view(-1, hidden_dim)
            g_flat = g.view(-1, hidden_dim)

            dots = (a_flat * g_flat).sum(dim=1)
            per_layer_vals.append(dots.mean().item())

            del a, g, a_flat, g_flat, dots
            torch.cuda.empty_cache()

        raw_scores[name] = float(sum(per_layer_vals) / len(per_layer_vals)) if per_layer_vals else 0.0

    # Cleanup
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    if batches_used == 0 or not raw_scores:
        logger.warning("No batches processed. Returning zeros.")
        return {k: 0.0 for k in lora_layers.keys()}

    # 5. Normalize
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