from xml.parsers.expat import model

from zmq import device
import torch
from tqdm import tqdm
from .importance import compute_bi_importance_eq13
from .allocation import bi_allocate_ranks
from .lora_injector import inject_adaptive_lora

def freeze_base_model(model):
    for n, p in model.named_parameters():
        p.requires_grad = False

def enable_lora_params(model):
    for module in model.modules():
        if module.__class__.__name__ == 'LoRALinear':
            if hasattr(module, 'A') and module.A is not None:
                module.A.requires_grad = True
            if hasattr(module, 'B') and module.B is not None:
                module.B.requires_grad = True

def get_lora_parameters(model):
    params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
    return params
def evaluate(model, dataloader, device):
    """General evaluation that supports both LM and QA tasks."""
    model.eval()
    losses = []
    correct, total = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            # Move tensors to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Case 1: Question Answering models
            if "start_positions" in batch and "end_positions" in batch:
                outputs = model(**batch)
                loss = outputs.loss
                losses.append(loss.item())

                # crude span accuracy check
                start_preds = torch.argmax(outputs.start_logits, dim=-1)
                end_preds = torch.argmax(outputs.end_logits, dim=-1)
                correct += ((start_preds == batch["start_positions"]) & (end_preds == batch["end_positions"])).float().sum().item()
                total += batch["start_positions"].numel()

            # Case 2: LM / classification with labels
            elif "labels" in batch:
                labels = batch["labels"]
                inputs = {k: v for k, v in batch.items() if k != "labels"}
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                losses.append(loss.item())
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).float().sum().item()
                total += labels.numel()

    acc = correct / total if total > 0 else 0
    return {"loss": sum(losses) / len(losses), "accuracy": acc}

def enable_grad_for_linear_layers(model):
    """Temporarily enable gradients for all Linear layers for BI computation."""
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            for p in m.parameters():
                p.requires_grad = True

def fine_tune_lora_dynamic(
    model,
    train_loader,
    val_loader=None,
    device='cuda',
    total_R=64,
    tau=0.5,
    epochs=3,
    lr=1e-4,
    weight_decay=0.0,
    alpha=16,
    dropout=0.0,
    max_batches_for_bi=8,
    recompute_every=1,
    fast_mode=False,
    save_path=None,
    log_every=10
):
    model.to(device)
    freeze_base_model(model)

    for epoch in range(1, epochs+1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        if (epoch-1) % recompute_every == 0:
            print("Computing BI importance (Eq.1.3)...")
            enable_grad_for_linear_layers(model)
            module_names, scores = compute_bi_importance_eq13(
                model, val_loader if val_loader is not None else train_loader, device=device,
                max_batches=max_batches_for_bi,
                target_module_name_substrings=None
            )
            ranks = bi_allocate_ranks(scores, total_R, tau=tau)
            print("Allocated ranks (sample):")
            for n,r in list(zip(module_names, ranks))[:10]:
                print(f"  {n} -> r={r}")
            patched = inject_adaptive_lora(model, module_names, ranks, alpha=alpha, dropout=dropout)
            model.to(device)
            print(f"Patched {len(patched)} modules with LoRA adapters.")

        enable_lora_params(model)
        lora_params = get_lora_parameters(model)
        if len(lora_params) == 0:
            raise RuntimeError("No LoRA params to train. Did injection succeed?")

        optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Training epoch {epoch}")
        for step, batch in enumerate(pbar):
            # --- Flexible batch processing for both classification and QA ---
            if "labels" in batch:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)
                outputs = model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = criterion(logits, labels)
# Question Answering datasets (like SQuAD)
            elif "start_positions" in batch and "end_positions" in batch:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
            else:
                raise KeyError("Batch must contain either 'labels' or 'start_positions'/'end_positions'")

            # inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            # labels = batch['labels'].to(device)
            # outputs = model(**inputs)
            # logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            # loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (step+1) % log_every == 0:
                pbar.set_postfix({'avg_loss': total_loss / (step+1)})

        stats = evaluate(model, val_loader, device) if val_loader is not None else {}
        if stats:
            print(f"After epoch {epoch}: val loss={stats['loss']:.4f}, acc={stats['accuracy']:.4f}")
        if save_path:
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")
    return model
