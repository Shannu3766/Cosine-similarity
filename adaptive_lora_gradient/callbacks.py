# import os
# import logging
# from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
# from .importance import compute_gradient_importance_scores
# from .allocation import allocate_ranks_bi
# from .utils import get_lora_layers, save_epoch_log

# logger = logging.getLogger(__name__)

# class AdaptiveLoRACallback(TrainerCallback):
#     """
#     Adaptive LoRA callback that:
#     - Computes Block Influence (BI) scores *before each epoch*.
#     - Allocates new ranks before training that epoch.
#     - Logs and saves rank evolution after each epoch.

#     Works across Causal LM, Classification, and QA tasks.
#     """

#     def __init__(
#         self,
#         total_rank: int,
#         val_dataloader,
#         tau: float = 1.0,
#         log_path: str = "./logs",
#         verbose: bool = True,
#         lora_alpha:int=4,
#         validate_batch_size:int=8,
#         min_rank:int=4
#     ):
#         self.total_rank = total_rank
#         self.val_dataloader = val_dataloader
#         self.tau = tau
#         self.verbose = verbose
#         self.log_file = os.path.join(log_path, "adaptive_lora_epoch_logs.csv")
#         self.lora_alpha=lora_alpha
#         self.validate_batch_size=validate_batch_size
#         self.min_rank=min_rank

#         os.makedirs(log_path, exist_ok=True)

#         # For storing the latest scores/ranks per epoch
#         self.latest_scores = {}
#         self.latest_ranks = {}

#     # ============================================================
#     # 🔁 EPOCH-BEGIN: Compute and apply ranks before training
#     # ============================================================
#     def on_epoch_begin(
#         self,
#         args: TrainingArguments,
#         state: TrainerState,
#         control: TrainerControl,
#         model,
#         **kwargs
#     ):
#         # Handle pre-training (state.epoch is None before training starts)
#         epoch = int(state.epoch) + 1 if state.epoch is not None else 0

#         if self.verbose:
#             print(f"\n--- AdaptiveLoRA: Preparing ranks for Epoch {epoch} ---")

#         device = next(model.parameters()).device

#         # 1️⃣ Compute BI scores BEFORE training
#         if self.verbose:
#             print("Computing BI importance scores (pre-training)...")
#         scores = compute_gradient_importance_scores(model, self.val_dataloader, device,batch_size=self.validate_batch_size)
#         if not scores:
#             if self.verbose:
#                 print("⚠️ No LoRA layers or BI scores found. Skipping rank update.")
#             return

#         # 2️⃣ Allocate new ranks
#         if self.verbose:
#             print("Allocating new ranks based on BI scores...")
#         new_ranks = allocate_ranks_bi(scores, self.total_rank, self.tau,min_rank=self.min_rank)

#         # 3️⃣ Apply new ranks to LoRA layers
#         if self.verbose:
#             print("Applying new ranks to LoRA modules for this epoch...")

#         lora_layers = get_lora_layers(model)
#         config = model.peft_config.get("default")
#         if not config:
#             logger.error("❌ PEFT config not found. Skipping update.")
#             return

#         # Extract config flags
#         init_lora_weights = getattr(config, "init_lora_weights", True)
#         use_rslora = getattr(config, "use_rslora", False)
#         use_dora = getattr(config, "use_dora", False)
#         use_qalora = getattr(config, "use_qalora", False)
#         lora_bias = getattr(config, "bias", "none")
#         qalora_group_size = getattr(config, "qalora_group_size", 64)

#         for name, layer in lora_layers.items():
#             new_rank = new_ranks.get(name)
#             if new_rank is None:
#                 continue

#             current_rank = layer.r.get("default", 0)
#             score = scores.get(name, 0.0)

#             # Print all layers (even unchanged ones)
#             if current_rank != new_rank:
#                 if self.verbose:
#                     print(f"  - {name}: r={current_rank} → {new_rank} (Score: {score:.4f})")
#             else:
#                 if self.verbose:
#                     print(f"  - {name}: r={new_rank} (Unchanged, Score: {score:.4f})")

#             # Update rank if different
#             if hasattr(layer, "update_layer") and current_rank != new_rank:
#                 lora_dropout_p = 0.0
#                 if hasattr(layer, "lora_dropout") and "default" in layer.lora_dropout:
#                     lora_dropout_p = layer.lora_dropout["default"].p

#                 layer.update_layer(
#                     adapter_name="default",
#                     r=new_rank,
#                     lora_alpha=self.lora_alpha,
#                     lora_dropout=lora_dropout_p,
#                     # init_lora_weights=init_lora_weights,
#                     init_lora_weights=False,
#                     use_rslora=use_rslora,
#                     use_dora=use_dora,
#                     use_qalora=use_qalora,
#                     lora_bias=lora_bias,
#                     qalora_group_size=qalora_group_size,
#                 )

#         # Save for logging after training
#         self.latest_scores = scores
#         self.latest_ranks = new_ranks

#         if self.verbose:
#             print(f"✅ AdaptiveLoRA: Rank setup for Epoch {epoch} complete.\n")

#     # ============================================================
#     # 📊 EPOCH-END: Log ranks and scores
#     # ============================================================
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
#                 print(
#                     f"📄 Epoch {epoch}: Rank allocations logged to {self.log_file}\n"
#                 )



import os
import logging
import torch
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
    - Uses SVD to transfer learned weights to the new rank dimensions (prevents forgetting).
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
        lora_alpha: int = 4,
        validate_batch_size: int = 8,
        min_rank: int = 4
    ):
        self.total_rank = total_rank
        self.val_dataloader = val_dataloader
        self.tau = tau
        self.verbose = verbose
        self.log_file = os.path.join(log_path, "adaptive_lora_epoch_logs.csv")
        self.lora_alpha = lora_alpha
        self.validate_batch_size = validate_batch_size
        self.min_rank = min_rank

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
        scores = compute_gradient_importance_scores(model, self.val_dataloader, device, batch_size=self.validate_batch_size)
        if not scores:
            if self.verbose:
                print("⚠️ No LoRA layers or BI scores found. Skipping rank update.")
            return

        # 2️⃣ Allocate new ranks
        if self.verbose:
            print("Allocating new ranks based on BI scores...")
        new_ranks = allocate_ranks_bi(scores, self.total_rank, self.tau, min_rank=self.min_rank)

        # 3️⃣ Apply new ranks to LoRA layers with SVD Weight Transfer
        if self.verbose:
            print("Applying new ranks to LoRA modules for this epoch...")

        lora_layers = get_lora_layers(model)
        config = model.peft_config.get("default")
        if not config:
            logger.error("❌ PEFT config not found. Skipping update.")
            return

        # Extract config flags
        init_lora_weights = getattr(config, "init_lora_weights", True)
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

            # Skip if rank hasn't changed
            if current_rank == new_rank:
                if self.verbose:
                    print(f"  - {name}: r={new_rank} (Unchanged, Score: {score:.4f})")
                continue

            if self.verbose:
                print(f"  - {name}: r={current_rank} → {new_rank} (Score: {score:.4f}) [Transferring Weights]")

            # -------------------------------------------------------------
            # STEP A: CAPTURE OLD WEIGHTS & SCALING
            # -------------------------------------------------------------
            # Get current LoRA matrices
            # A shape: [r_old, d_in], B shape: [d_out, r_old]
            old_A = layer.lora_A["default"].weight.data.clone()
            old_B = layer.lora_B["default"].weight.data.clone()
            
            # Calculate the full learned delta matrix: W = B @ A
            # We must account for the scaling factor (alpha / r) to preserve magnitude
            scaling_factor = layer.scaling["default"] # typically alpha / r_old
            
            # Reconstruct the effective weight update: W_delta = scaling * (B @ A)
            delta_weight = scaling_factor * (old_B @ old_A) # Shape: [d_out, d_in]

            # -------------------------------------------------------------
            # STEP B: UPDATE THE LAYER ARCHITECTURE
            # -------------------------------------------------------------
            lora_dropout_p = 0.0
            if hasattr(layer, "lora_dropout") and "default" in layer.lora_dropout:
                lora_dropout_p = layer.lora_dropout["default"].p

            # Update the layer to the new rank
            # CRITICAL: We set init_lora_weights=False to prevent random initialization overwriting our work
            layer.update_layer(
                adapter_name="default",
                r=new_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=lora_dropout_p,
                init_lora_weights=False, 
                use_rslora=use_rslora,
                use_dora=use_dora,
                use_qalora=use_qalora,
                lora_bias=lora_bias,
                qalora_group_size=qalora_group_size,
            )

            # -------------------------------------------------------------
            # STEP C: SVD DECOMPOSITION & PROJECTION
            # -------------------------------------------------------------
            # We want: new_scaling * (B_new @ A_new) ≈ delta_weight
            # So: B_new @ A_new ≈ (1 / new_scaling) * delta_weight
            
            new_scaling = layer.scaling["default"] # Now updated to alpha / new_rank
            target_matrix = (1 / new_scaling) * delta_weight

            # Perform SVD on the target weight matrix
            # U: [d_out, d_out], S: [min_dim], Vh: [d_in, d_in]
            try:
                U, S, Vh = torch.linalg.svd(target_matrix, full_matrices=False)
            except RuntimeError:
                print(f"⚠️ SVD failed for {name}. Fallback to random init.")
                layer.reset_lora_parameters("default")
                continue

            # -------------------------------------------------------------
            # STEP D: TRUNCATE AND ASSIGN NEW WEIGHTS
            # -------------------------------------------------------------
            # Keep top-k components where k = new_rank
            # If we are increasing rank, the extra dimensions will be padded with zeros
            
            k = min(new_rank, S.size(0))
            U_trunc = U[:, :k]          # [d_out, k]
            S_trunc = S[:k]             # [k]
            Vh_trunc = Vh[:k, :]        # [k, d_in]
            
            # Compute new weights using the truncated SVD components
            # We split S (singular values) equally between A and B
            sqrt_S = torch.diag(torch.sqrt(S_trunc))
            B_new_weight = U_trunc @ sqrt_S
            A_new_weight = sqrt_S @ Vh_trunc

            # Handle dimensions if new_rank > available rank (padding with zeros for growth)
            if new_rank > k:
                pad_r = new_rank - k
                # Pad A at the bottom (extra rows)
                A_new_weight = torch.cat([A_new_weight, torch.zeros(pad_r, A_new_weight.size(1), device=device)], dim=0)
                # Pad B at the right (extra columns)
                B_new_weight = torch.cat([B_new_weight, torch.zeros(B_new_weight.size(0), pad_r, device=device)], dim=1)

            # Assign to the layer (ensuring correct shape and dtype)
            layer.lora_A["default"].weight.data = A_new_weight.to(dtype=old_A.dtype)
            layer.lora_B["default"].weight.data = B_new_weight.to(dtype=old_B.dtype)

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