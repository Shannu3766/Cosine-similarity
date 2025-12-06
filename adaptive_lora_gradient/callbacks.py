import os
import logging
from typing import Dict, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from .importance import compute_gradient_importance_scores
from .allocation import allocate_ranks_bi
from .utils import get_lora_layers, save_epoch_log, resize_lora_layer_svd

logger = logging.getLogger(__name__)

class AdaptiveLoRACallback(TrainerCallback):
    """
    Adaptive LoRA callback with Stability Improvements:
    - SVD Resizing (Preserves weights).
    - EMA Smoothing (Prevents rank thrashing).
    - Warmup/Cooldown/Intervals (Stabilizes training).
    """

    def __init__(
        self,
        total_rank: int,
        val_dataloader,
        tau: float = 1.0,
        log_path: str = "./logs",
        verbose: bool = True,
        lora_alpha: int = 16, # Increased default alpha for stability
        validate_batch_size: int = 8,
        min_rank: int = 4,
        # --- NEW STABILITY HYPERPARAMETERS ---
        score_smoothing_beta: float = 0.85, # 0.0 = No history, 0.9 = Heavy smoothing
        update_interval: int = 2,           # Update ranks every N epochs
        warmup_epochs: int = 1,             # Don't adapt for first N epochs
        cooldown_epochs: int = 2            # Stop adapting N epochs before end
    ):
        self.total_rank = total_rank
        self.val_dataloader = val_dataloader
        self.tau = tau
        self.verbose = verbose
        self.log_file = os.path.join(log_path, "adaptive_lora_epoch_logs.csv")
        self.lora_alpha = lora_alpha
        self.validate_batch_size = validate_batch_size
        self.min_rank = min_rank
        
        # Stability params
        self.score_smoothing_beta = score_smoothing_beta
        self.update_interval = update_interval
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs

        os.makedirs(log_path, exist_ok=True)

        self.latest_scores = {}
        self.latest_ranks = {}
        
        # Store EMA scores here
        self.ema_scores: Optional[Dict[str, float]] = None

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        **kwargs
    ):
        # Current epoch (1-based for logic)
        epoch = int(state.epoch) + 1 if state.epoch is not None else 1
        total_epochs = args.num_train_epochs

        # --- 1. SCHEDULING CHECKS ---
        # Skip if in warmup
        if epoch <= self.warmup_epochs:
            if self.verbose:
                print(f"⏳ AdaptiveLoRA: Warmup (Epoch {epoch}). Skipping update.")
            return

        # Skip if in cooldown (near end of training)
        if epoch > (total_epochs - self.cooldown_epochs):
            if self.verbose:
                print(f"🔒 AdaptiveLoRA: Cooldown (Epoch {epoch}). Architecture frozen.")
            return

        # Skip if not the right interval
        if (epoch - self.warmup_epochs) % self.update_interval != 0:
            if self.verbose:
                print(f"⏭️ AdaptiveLoRA: Interval skip (Epoch {epoch}). Keeping ranks.")
            return

        if self.verbose:
            print(f"\n--- AdaptiveLoRA: Adapting Ranks for Epoch {epoch} ---")

        device = next(model.parameters()).device

        # --- 2. COMPUTE RAW SCORES ---
        raw_scores = compute_gradient_importance_scores(
            model, 
            self.val_dataloader, 
            device, 
            batch_size=self.validate_batch_size
        )
        
        if not raw_scores:
            logger.warning("No scores computed. Skipping.")
            return

        # --- 3. APPLY EMA SMOOTHING ---
        # This is the key to accuracy: Don't react to noise, react to trends.
        if self.ema_scores is None:
            self.ema_scores = raw_scores
        else:
            for name, score in raw_scores.items():
                prev = self.ema_scores.get(name, 0.0)
                # EMA Formula: New = Beta * Old + (1-Beta) * Current
                self.ema_scores[name] = (
                    self.score_smoothing_beta * prev + 
                    (1.0 - self.score_smoothing_beta) * score
                )
        
        # Use smoothed scores for allocation
        final_scores = self.ema_scores

        # --- 4. ALLOCATE RANKS ---
        new_ranks = allocate_ranks_bi(
            final_scores, 
            self.total_rank, 
            self.tau, 
            min_rank=self.min_rank
        )

        # --- 5. APPLY UPDATES (SVD) ---
        lora_layers = get_lora_layers(model)
        config = model.peft_config.get("default")
        
        update_kwargs = {
            "use_rslora": getattr(config, "use_rslora", False),
            "use_dora": getattr(config, "use_dora", False),
            "use_qalora": getattr(config, "use_qalora", False),
            "lora_bias": getattr(config, "bias", "none"),
            "qalora_group_size": getattr(config, "qalora_group_size", 64),
        }

        changes_count = 0
        for name, layer in lora_layers.items():
            new_rank = new_ranks.get(name)
            if new_rank is None: continue

            current_rank = layer.r.get("default", 0)
            
            if current_rank != new_rank:
                changes_count += 1
                lora_dropout_p = 0.0
                if hasattr(layer, "lora_dropout") and "default" in layer.lora_dropout:
                    lora_dropout_p = layer.lora_dropout["default"].p

                # Use SVD Resize
                resize_lora_layer_svd(
                    layer=layer,
                    new_rank=new_rank,
                    lora_alpha=self.lora_alpha,
                    adapter_name="default",
                    lora_dropout=lora_dropout_p,
                    **update_kwargs
                )

        # Save for logging
        self.latest_scores = final_scores
        self.latest_ranks = new_ranks

        if self.verbose:
            print(f"✅ AdaptiveLoRA: Updated {changes_count} layers. (Smoothed Score used)\n")

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