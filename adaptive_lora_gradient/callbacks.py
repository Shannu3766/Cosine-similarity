import os
import logging
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from .importance import compute_gradient_importance_scores
from .allocation import allocate_ranks_bi
from .utils import get_lora_layers, save_epoch_log

logger = logging.getLogger(__name__)

class AdaptiveLoRACallback(TrainerCallback):
    """
    Adaptive LoRA callback that:
    - Computes Block Influence (BI) scores *before each epoch*.
    - Allocates new ranks before training that epoch.
    - Logs and saves rank evolution after each epoch.

    Works across Causal LM, Classification, and QA tasks.
    """

    def __init__(
        self,
        total_rank: int,
        val_dataloader,
        tau: float = 1.0,
        log_path: str = "./logs",
        verbose: bool = True,
        lora_alpha:int=4,
        validate_batch_size:int=8,
        min_rank:int=4
    ):
        self.total_rank = total_rank
        self.val_dataloader = val_dataloader
        self.tau = tau
        self.verbose = verbose
        self.log_file = os.path.join(log_path, "adaptive_lora_epoch_logs.csv")
        self.lora_alpha=lora_alpha
        self.validate_batch_size=validate_batch_size
        self.min_rank=min_rank

        os.makedirs(log_path, exist_ok=True)

        # For storing the latest scores/ranks per epoch
        self.latest_scores = {}
        self.latest_ranks = {}

    # ============================================================
    # 🔁 EPOCH-BEGIN: Compute and apply ranks before training
    # ============================================================
    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs
    ):
        # Handle pre-training (state.epoch is None before training starts)
        epoch = int(state.epoch) + 1 if state.epoch is not None else 0

        if self.verbose:
            print(f"\n--- AdaptiveLoRA: Preparing ranks for Epoch {epoch} ---")

        device = next(model.parameters()).device

        # 1️⃣ Compute BI scores BEFORE training
        if self.verbose:
            print("Computing BI importance scores (pre-training)...")
        scores = compute_gradient_importance_scores(model, self.val_dataloader, device,batch_size=self.validate_batch_size)
        if not scores:
            if self.verbose:
                print("⚠️ No LoRA layers or BI scores found. Skipping rank update.")
            return

        # 2️⃣ Allocate new ranks
        if self.verbose:
            print("Allocating new ranks based on BI scores...")
        new_ranks = allocate_ranks_bi(scores, self.total_rank, self.tau,min_rank=self.min_rank)

        # 3️⃣ Apply new ranks to LoRA layers
        if self.verbose:
            print("Applying new ranks to LoRA modules for this epoch...")

        lora_layers = get_lora_layers(model)
        config = model.peft_config.get("default")
        if not config:
            logger.error("❌ PEFT config not found. Skipping update.")
            return

        # Extract config flags
        init_lora_weights = getattr(config, "init_lora_weights", False)
        use_rslora = getattr(config, "use_rslora", False)
        use_dora = getattr(config, "use_dora", False)
        use_qalora = getattr(config, "use_qalora", False)
        lora_bias = getattr(config, "bias", "none")
        qalora_group_size = getattr(config, "qalora_group_size", 64)

        for name, layer in lora_layers.items():
            new_rank = new_ranks.get(name)
            if new_rank is None:
                continue

            current_rank = layer.r.get("default", 0)
            score = scores.get(name, 0.0)

            # Print all layers (even unchanged ones)
            if current_rank != new_rank:
                if self.verbose:
                    print(f"  - {name}: r={current_rank} → {new_rank} (Score: {score:.4f})")
            else:
                if self.verbose:
                    print(f"  - {name}: r={new_rank} (Unchanged, Score: {score:.4f})")

            # Update rank if different
            if hasattr(layer, "update_layer") and current_rank != new_rank:
                lora_dropout_p = 0.0
                if hasattr(layer, "lora_dropout") and "default" in layer.lora_dropout:
                    lora_dropout_p = layer.lora_dropout["default"].p

                layer.update_layer(
                    adapter_name="default",
                    r=new_rank,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=lora_dropout_p,
                    init_lora_weights=init_lora_weights,
                    # init_lora_weights=False,
                    use_rslora=use_rslora,
                    use_dora=use_dora,
                    use_qalora=use_qalora,
                    lora_bias=lora_bias,
                    qalora_group_size=qalora_group_size,
                )

        # Save for logging after training
        self.latest_scores = scores
        self.latest_ranks = new_ranks

        if self.verbose:
            print(f"✅ AdaptiveLoRA: Rank setup for Epoch {epoch} complete.\n")

    # ============================================================
    # 📊 EPOCH-END: Log ranks and scores
    # ============================================================
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs
    ):
        epoch = int(state.epoch) if state.epoch is not None else -1

        if self.latest_ranks and self.latest_scores:
            save_epoch_log(self.log_file, epoch, self.latest_ranks, self.latest_scores)
            if self.verbose:
                print(
                    f"📄 Epoch {epoch}: Rank allocations logged to {self.log_file}\n"
                )
# import os
# import logging
# import torch
# import torch.nn as nn
# from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
# from .importance import compute_gradient_importance_scores
# from .allocation import allocate_ranks_bi
# from .utils import get_lora_layers, save_epoch_log

# logger = logging.getLogger(__name__)

# class AdaptiveLoRACallback(TrainerCallback):
#     """
#     High-Performance Adaptive LoRA callback.
#     - Uses naive weight transfer (padding/truncation) to prevent forgetting.
#     - Manually swaps layer pointers to avoid 'update_layer' overhead.
#     """

#     def __init__(
#         self,
#         total_rank: int,
#         val_dataloader,
#         tau: float = 1.0,
#         log_path: str = "./logs",
#         verbose: bool = True,
#         lora_alpha: int = 4,
#         validate_batch_size: int = 8,
#         min_rank: int = 4
#     ):
#         self.total_rank = total_rank
#         self.val_dataloader = val_dataloader
#         self.tau = tau
#         self.verbose = verbose
#         self.log_file = os.path.join(log_path, "adaptive_lora_epoch_logs.csv")
#         self.lora_alpha_multiplier = lora_alpha # Multiplier relative to rank (alpha = r * multiplier)
#         self.validate_batch_size = validate_batch_size
#         self.min_rank = min_rank

#         os.makedirs(log_path, exist_ok=True)

#         self.latest_scores = {}
#         self.latest_ranks = {}

#     def on_epoch_begin(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         model,
#         **kwargs
#     ):
#         # Handle pre-training state
#         epoch = int(state.epoch) + 1 if state.epoch is not None else 0

#         if self.verbose:
#             print(f"\n--- AdaptiveLoRA: Preparing ranks for Epoch {epoch} ---")

#         # 1. Compute Scores
#         device = next(model.parameters()).device
#         if self.verbose:
#             print("Computing BI importance scores...")
            
#         scores = compute_gradient_importance_scores(
#             model, 
#             self.val_dataloader, 
#             device,
#             batch_size=self.validate_batch_size
#         )
        
#         if not scores:
#             if self.verbose: print("⚠️ No scores found. Skipping.")
#             return

#         # 2. Allocate Ranks
#         if self.verbose: print("Allocating new ranks...")
#         new_ranks = allocate_ranks_bi(scores, self.total_rank, self.tau, min_rank=self.min_rank)

#         # 3. Fast Weight Transfer & Layer Swap
#         if self.verbose: print("Applying rank updates (Fast Swap)...")
        
#         lora_layers = get_lora_layers(model)
#         updates_count = 0

#         with torch.no_grad(): 
#             for name, layer in lora_layers.items():
#                 new_rank = new_ranks.get(name)
#                 if new_rank is None: continue

#                 current_rank = layer.r.get("default", 0)
#                 if current_rank == new_rank: continue

#                 updates_count += 1
                
#                 # --- A. Grab Old Weights ---
#                 # Clone to ensure we don't lose data during swap
#                 old_A = layer.lora_A["default"].weight.data
#                 old_B = layer.lora_B["default"].weight.data
                
#                 # Device/Dtype for new layers
#                 target_device = old_A.device
#                 target_dtype = old_A.dtype

#                 # --- B. Prepare New Weights (Pad/Truncate) ---
#                 # Handle Matrix A [rank, in_dim]
#                 if new_rank > current_rank:
#                     # Pad rows
#                     pad_rows = new_rank - current_rank
#                     new_A_weight = torch.cat([
#                         old_A, 
#                         torch.zeros((pad_rows, old_A.shape[1]), device=target_device, dtype=target_dtype)
#                     ], dim=0)
#                 else:
#                     # Truncate rows
#                     new_A_weight = old_A[:new_rank, :]

#                 # Handle Matrix B [out_dim, rank]
#                 if new_rank > current_rank:
#                     # Pad cols
#                     pad_cols = new_rank - current_rank
#                     new_B_weight = torch.cat([
#                         old_B, 
#                         torch.zeros((old_B.shape[0], pad_cols), device=target_device, dtype=target_dtype)
#                     ], dim=1)
#                 else:
#                     # Truncate cols
#                     new_B_weight = old_B[:, :new_rank]

#                 # --- C. Scale Adjustment ---
#                 # Scale weights so (B*scale)*(A*scale) == BA * old_scale
#                 scale_adjustment = (new_rank / current_rank) ** 0.5
#                 new_A_weight.mul_(scale_adjustment)
#                 new_B_weight.mul_(scale_adjustment)

#                 # --- D. Manual Module Replacement (Fastest) ---
#                 # Instead of update_layer(), we create nn.Linear manually
                
#                 # Create new A
#                 new_linear_A = nn.Linear(new_A_weight.shape[1], new_rank, bias=False)
#                 new_linear_A.weight.data = new_A_weight
#                 new_linear_A.to(device=target_device, dtype=target_dtype)
                
#                 # Create new B
#                 new_linear_B = nn.Linear(new_rank, new_B_weight.shape[0], bias=False)
#                 new_linear_B.weight.data = new_B_weight
#                 new_linear_B.to(device=target_device, dtype=target_dtype)

#                 # Swap Modules
#                 layer.lora_A["default"] = new_linear_A
#                 layer.lora_B["default"] = new_linear_B
                
#                 # Update PEFT Metadata
#                 layer.r["default"] = new_rank
#                 layer.lora_alpha["default"] = new_rank * self.lora_alpha_multiplier
#                 layer.scaling["default"] = layer.lora_alpha["default"] / new_rank

#         # Explicit cleanup to prevent VRAM fragmentation
#         torch.cuda.empty_cache()
        
#         self.latest_scores = scores
#         self.latest_ranks = new_ranks

#         if self.verbose:
#             print(f"✅ Updated {updates_count} layers. Rank setup complete.\n")

#     def on_epoch_end(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         model,
#         **kwargs
#     ):
#         epoch = int(state.epoch) if state.epoch is not None else -1
#         if self.latest_ranks and self.latest_scores:
#             save_epoch_log(self.log_file, epoch, self.latest_ranks, self.latest_scores)
#             if self.verbose:
#                 print(f"📄 Epoch {epoch}: Rank allocations logged to {self.log_file}\n")