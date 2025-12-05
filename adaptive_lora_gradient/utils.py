import os
import csv
from typing import Dict, Any
import torch
from peft.tuners.lora import LoraLayer
import logging

logger = logging.getLogger(__name__)

def get_lora_layers(model: torch.nn.Module) -> Dict[str, LoraLayer]:
    """
    Finds all modules in the model that are instances of peft.tuners.lora.LoraLayer.

    Args:
        model: The PEFT model.

    Returns:
    
        A dictionary mapping qualified layer names to the LoraLayer module.
    """
    # Note: We look for LoraLayer, which is the base class for peft.lora.Linear,
    # peft.lora.Embedding, peft.lora.Conv2d, etc.
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoraLayer)
    }

def save_epoch_log(
    log_file: str, 
    epoch: int, 
    ranks: Dict[str, int], 
    scores: Dict[str, float]
):
    """
    Appends the rank allocation results for the current epoch to a CSV log file.
    
    Args:
        log_file: Path to the CSV file.
        epoch: The current epoch number.
        ranks: Dictionary of {layer_name: allocated_rank}.
        scores: Dictionary of {layer_name: importance_score}.
    """
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    fieldnames = ['epoch', 'layer_name', 'importance_score', 'allocated_rank']
    
    # Check if file exists to write header
    file_exists = os.path.isfile(log_file)
    
    try:
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
                
            for layer_name in ranks.keys():
                writer.writerow({
                    'epoch': epoch,
                    'layer_name': layer_name,
                    'importance_score': scores.get(layer_name, 0.0),
                    'allocated_rank': ranks.get(layer_name, 0)
                })
    except IOError as e:
        logger.error(f"Failed to write to log file {log_file}: {e}")



# Keep your existing imports... 
# (peft.tuners.lora import LoraLayer should already be there)

def resize_lora_layer_svd(
    layer: LoraLayer, 
    new_rank: int, 
    lora_alpha: int, 
    adapter_name: str = "default",
    **kwargs
):
    """
    Resizes a LoRA layer using SVD to preserve the learned weights.
    
    1. Computes effective weight W = B_old @ A_old * scaling_old
    2. Performs SVD on W
    3. Truncates to new_rank
    4. Updates layer structure
    5. Initializes new A and B with truncated SVD components
    """
    
    # 1. Capture current weights and scaling
    # peft stores weights as: A=(rank, in), B=(out, rank)
    # scaling = alpha / rank
    with torch.no_grad():
        if adapter_name not in layer.lora_A:
            return  # Skip if adapter doesn't exist
            
        old_r = layer.r[adapter_name]
        old_alpha = layer.lora_alpha[adapter_name]
        old_scaling = old_alpha / old_r
        
        # Get weights on the correct device
        A_old = layer.lora_A[adapter_name].weight
        B_old = layer.lora_B[adapter_name].weight
        
        # Compute the effective delta weight: (out, in)
        # Matrix multiplication: (out, old_r) @ (old_r, in)
        W_delta = (B_old @ A_old) * old_scaling
        
        # 2. Perform SVD
        # U: (out, out), S: (min_dim,), Vh: (in, in)
        # Use float32 for stability during SVD even if training in bf16
        dtype = A_old.dtype
        U, S, Vh = torch.linalg.svd(W_delta.float(), full_matrices=False)
        
        # 3. Truncate to new rank
        # If new_rank > current rank, we effectively zero-pad the extra dimensions
        # because S will run out of values.
        k = new_rank
        
        # Handle cases where new_rank is larger than matrix dimensions
        k = min(k, S.size(0))
        
        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]
        
        # 4. Calculate new A and B
        # We split sqrt(S) between A and B
        sqrt_S = torch.diag(torch.sqrt(S_k))
        
        # Reconstruct B (out, k) and A (k, in)
        B_new = (U_k @ sqrt_S).to(dtype)
        A_new = (sqrt_S @ Vh_k).to(dtype)
        
        # 5. Adjust for NEW scaling
        # The standard LoRA forward pass is: (B @ A) * (alpha / r)
        # We want: (B_new @ A_new) * (alpha / new_rank) == W_delta
        # Our current B_new @ A_new == W_delta.
        # So we must scale B_new and A_new down by sqrt(new_scaling).
        
        new_scaling = lora_alpha / new_rank
        scale_correction = 1.0 / (new_scaling ** 0.5)
        
        B_new *= scale_correction
        A_new *= scale_correction
        
    # 6. Update the layer structure (This resets weights to random!)
    # We pass **kwargs to handle all other config flags (dropout, dora, etc.)
    layer.update_layer(
        adapter_name=adapter_name,
        r=new_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=True, # We will overwrite this immediately
        **kwargs
    )
    
    # 7. Overwrite with our SVD-preserved weights
    with torch.no_grad():
        # Note: If new_rank was clamped (k < new_rank), we might need to pad.
        # But usually new_rank << min(in, out), so k == new_rank.
        if k < new_rank:
             # If strictly needed, pad with zeros here. 
             # PEFT initializes with zeros for B and random for A usually.
             # We just overwrite the top-left submatrix.
             layer.lora_A[adapter_name].weight.data.zero_()
             layer.lora_B[adapter_name].weight.data.zero_()
             
             layer.lora_A[adapter_name].weight.data[:k, :] = A_new
             layer.lora_B[adapter_name].weight.data[:, :k] = B_new
        else:
             layer.lora_A[adapter_name].weight.data = A_new
             layer.lora_B[adapter_name].weight.data = B_new