import torch
from typing import Dict
import logging

logger = logging.getLogger(__name__)

def _largest_remainder_rounding(
    raw_ranks: torch.Tensor, 
    budget: int
) -> torch.Tensor:
    """
    Rounds raw rank allocations to integers while perfectly preserving the sum.
    This implements "rounding top residuals" (Algorithm 2, line 14). 
    
    Args:
        raw_ranks: A tensor of floating-point rank allocations.
        budget: The exact integer total sum that must be preserved.

    Returns:
        A tensor of integer ranks.
    """
    # This is r_i = floor(R * alpha_i) (Algorithm 2, line 13)
    int_ranks = torch.floor(raw_ranks).int()
    
    # This is the adjustment step (Algorithm 2, line 14)
    remainder = budget - int_ranks.sum()
    
    if remainder < 0:
        # This can happen due to floating point inaccuracies
        logger.warning(f"Rounding remainder is negative ({remainder}). Adjusting...")
        _, top_indices = torch.topk(raw_ranks, k=int(torch.abs(remainder).item()))
        int_ranks[top_indices] -= 1
        return int_ranks
        
    elif remainder > 0:
        # Get the "residuals" (the fractional parts)
        residuals = raw_ranks - int_ranks
        
        # "rounding top residuals"
        _, top_indices = torch.topk(residuals, k=int(remainder.item()))
        int_ranks[top_indices] += 1
            
    return int_ranks

# def allocate_ranks_bi(
#     scores: Dict[str, float], 
#     total_rank: int, 
#     tau: float = 1.0
# ) -> Dict[str, int]:
#     """
#     Allocates ranks to layers based on their BI importance scores,
#     as defined in Algorithm 2 of the paper. 

#     Args:
#         scores: Dictionary of {layer_name: importance_score}. 
#         total_rank: The total rank budget R to distribute. 
#         tau: Temperature parameter τ. 

#     Returns:
#         A dictionary of {layer_name: allocated_rank}.
#     """
#     if not scores:
#         return {}

#     num_layers = len(scores)
    
#     if total_rank < 0:
#         raise ValueError("Total rank must be non-negative.")
        
#     if total_rank < num_layers:
#          logger.warning(
#             f"Total rank {total_rank} is less than the number of layers ({num_layers}). "
#             f"Some layers will necessarily be assigned a rank of 0."
#          )
    
#     layer_names = list(scores.keys())
#     s = torch.tensor([scores[name] for name in layer_names], dtype=torch.float32)

#     # --- Algorithm 2, Line 12 ---
#     # Compute softmax weights: alpha_i = exp(s_i / tau) / sum(...)
#     s_temp = s / tau
#     probs = torch.softmax(s_temp, dim=0) # This is alpha_i 
    
#     # --- Algorithm 2, Line 13 & 14 ---
#     # Allocate preliminary ranks: r_i = R * alpha_i
#     raw_ranks = probs * total_rank
    
#     # Round to integers, ensuring sum is exactly R
#     # (This implements Line 13 and 14) 
#     final_ranks = _largest_remainder_rounding(raw_ranks, total_rank)
        
#     return {name: rank.item() for name, rank in zip(layer_names, final_ranks)}


def allocate_ranks_bi(
    scores: Dict[str, float],
    total_rank: int,
    tau: float = 0.3,        # make distribution sharper
    eps: float = 1e-8
) -> Dict[str, int]:
    """
    Allocates ranks to layers based on their BI importance scores.
    Adds normalization and temperature sharpening to avoid uniform ranks.
    """
    if not scores:
        return {}

    layer_names = list(scores.keys())
    s = torch.tensor([scores[name] for name in layer_names], dtype=torch.float32)

    # --- 1️⃣ Normalize scores to [0, 1]
    s_min, s_max = s.min(), s.max()
    if (s_max - s_min) < eps:
        # all scores nearly equal → assign rank 1
        return {name: max(1, total_rank // len(scores)) for name in layer_names}
    s_norm = (s - s_min) / (s_max - s_min)

    # --- 2️⃣ Sharpen using temperature (smaller tau -> sharper)
    s_temp = s_norm / tau
    probs = torch.softmax(s_temp, dim=0)

    # --- 3️⃣ Allocate and round ranks
    raw_ranks = probs * total_rank
    int_ranks = torch.floor(raw_ranks).int()

    remainder = total_rank - int_ranks.sum()
    if remainder > 0:
        residuals = raw_ranks - int_ranks
        _, top_indices = torch.topk(residuals, k=int(remainder.item()))
        int_ranks[top_indices] += 1

    # --- 4️⃣ Ensure no layer gets zero rank
    int_ranks = torch.clamp(int_ranks, min=1)

    return {name: rank.item() for name, rank in zip(layer_names, int_ranks)}
