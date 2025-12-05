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
#         init_lora_weights = getattr(config, "init_lora_weights", False)
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
#                     init_lora_weights=init_lora_weights,
#                     # init_lora_weights=False,
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
    - Uses Naive Weight Transfer (Pad/Truncate) to preserve learned weights.
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
        scores = compute_gradient_importance_scores(
            model, 
            self.val_dataloader, 
            device,
            batch_size=self.validate_batch_size
        )
        
        if not scores:
            if self.verbose:
                print("⚠️ No LoRA layers or BI scores found. Skipping rank update.")
            return

        # 2️⃣ Allocate new ranks
        if self.verbose:
            print("Allocating new ranks based on BI scores...")
        new_ranks = allocate_ranks_bi(scores, self.total_rank, self.tau, min_rank=self.min_rank)

        # 3️⃣ Apply new ranks to LoRA layers with Naive Weight Transfer
        if self.verbose:
            print("Applying new ranks to LoRA modules for this epoch...")

        lora_layers = get_lora_layers(model)
        config = model.peft_config.get("default")
        if not config:
            logger.error("❌ PEFT config not found. Skipping update.")
            return

        # Extract config flags
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

            # Print status
            if current_rank != new_rank:
                if self.verbose:
                    print(f"  - {name}: r={current_rank} → {new_rank} (Score: {score:.4f})")
            else:
                if self.verbose:
                    print(f"  - {name}: r={new_rank} (Unchanged, Score: {score:.4f})")
                continue # Skip update if rank is unchanged

            # ----------------------------------------------------------------
            # STEP 1: SAVE OLD WEIGHTS
            # ----------------------------------------------------------------
            # Clone them so they persist after layer update
            old_A = layer.lora_A["default"].weight.data.clone()
            old_B = layer.lora_B["default"].weight.data.clone()

            # ----------------------------------------------------------------
            # STEP 2: UPDATE LAYER TOPOLOGY
            # ----------------------------------------------------------------
            lora_dropout_p = 0.0
            if hasattr(layer, "lora_dropout") and "default" in layer.lora_dropout:
                lora_dropout_p = layer.lora_dropout["default"].p

            layer.update_layer(
                adapter_name="default",
                r=new_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=lora_dropout_p,
                init_lora_weights=False,  # CRITICAL: Prevent random initialization
                use_rslora=use_rslora,
                use_dora=use_dora,
                use_qalora=use_qalora,
                lora_bias=lora_bias,
                qalora_group_size=qalora_group_size,
            )

            # ----------------------------------------------------------------
            # STEP 3: NAIVE WEIGHT TRANSFER (PAD OR TRUNCATE)
            # ----------------------------------------------------------------
            with torch.no_grad():
                # --- Handle Matrix A [rank, dim] ---
                if new_rank > current_rank:
                    # PAD: Keep old A, fill new rows with zeros
                    pad_rows = new_rank - current_rank
                    new_A_weight = torch.cat([
                        old_A, 
                        torch.zeros((pad_rows, old_A.shape[1]), device=device, dtype=old_A.dtype)
                    ], dim=0)
                else:
                    # TRUNCATE: Keep top 'new_rank' rows
                    new_A_weight = old_A[:new_rank, :]

                # --- Handle Matrix B [dim, rank] ---
                if new_rank > current_rank:
                    # PAD: Keep old B, fill new cols with zeros
                    pad_cols = new_rank - current_rank
                    new_B_weight = torch.cat([
                        old_B, 
                        torch.zeros((old_B.shape[0], pad_cols), device=device, dtype=old_B.dtype)
                    ], dim=1)
                else:
                    # TRUNCATE: Keep left 'new_rank' cols
                    new_B_weight = old_B[:, :new_rank]

                # ----------------------------------------------------------------
                # STEP 4: CORRECT FOR SCALING FACTOR CHANGE
                # ----------------------------------------------------------------
                # Formula: scale = r_new / r_old
                # We multiply weights by sqrt(scale) so that (B*sqrt) * (A*sqrt) = BA * scale
                # This compensates for PEFT's internal scaling (alpha/r) changing when r changes.
                scale_adjustment = (new_rank / current_rank) ** 0.5
                
                new_A_weight = new_A_weight * scale_adjustment
                new_B_weight = new_B_weight * scale_adjustment

                # ----------------------------------------------------------------
                # STEP 5: ASSIGN NEW WEIGHTS
                # ----------------------------------------------------------------
                layer.lora_A["default"].weight.copy_(new_A_weight)
                layer.lora_B["default"].weight.copy_(new_B_weight)

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