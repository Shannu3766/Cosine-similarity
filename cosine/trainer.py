# examples/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from cosine.lora_injector import inject_lora_to_linears, extract_lora_parameters
from cosine.importance import compute_scores_over_dataloader
from cosine.allocation import allocate_by_importance
from collections import OrderedDict
import time

def fine_tune_lora_dynamic(
    model,
    train_loader,
    val_loader,
    device,
    total_R=32,
    tau=0.5,
    epochs=2,
    lr=1e-5,
    weight_decay=0.01,
    max_batches_for_bi=2,
    recompute_every=1,
    fast_mode=True,
    r_min=0,
):
    """
    Dynamic LoRA fine-tuning loop.
    - model: torch.nn.Module (uninitialized LoRA)
    - train_loader, val_loader: torch.utils.data.DataLoader
    - device: 'cpu' or 'cuda'
    - total_R: total LoRA rank budget (int)
    - tau: temperature for softmax allocation
    - epochs: number of training epochs
    - lr, weight_decay: optimizer params
    - max_batches_for_bi: how many val batches to use when computing importance
    - recompute_every: recompute allocation every `recompute_every` epochs
    - fast_mode: if True, uses small subset; else uses full val_loader
    - r_min: minimum rank per layer
    """
    device = torch.device(device)
    model.to(device)

    # initially set ranks to 0 for all linear modules (we will inject LoRA)
    rank_map = {}
    for name, m in model.named_modules():
        # consider only nn.Linear modules
        from torch import nn as _nn
        if isinstance(m, _nn.Linear):
            rank_map[name] = 0

    # inject LoRA placeholders
    lora_modules = inject_lora_to_linears(model, rank_map)

    # training setup: optimizer only for LoRA params
    def get_lora_param_list():
        return [p for p in extract_lora_parameters(model) if p is not None and p.requires_grad]

    optimizer = optim.AdamW(get_lora_param_list(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # utility to recompute importance scores and update ranks
    def recompute_and_apply_ranks():
        # select module map to pass to importance computation (original LoRALinear modules)
        module_map = {name: mod.orig if hasattr(mod, "orig") else mod for name, mod in lora_modules.items()}
        # compute importance scores on val set
        batches = max(1, min(max_batches_for_bi, len(val_loader))) if not fast_mode else max(1, min(1, max_batches_for_bi))
        scores = compute_scores_over_dataloader(model, val_loader, device=device, module_name_map=module_map, max_batches=batches, loss_fn=loss_fn)
        # allocate ranks
        allocation = allocate_by_importance(scores, total_R=total_R, tau=tau, r_min=r_min)
        # apply ranks to LoRA modules
        for name, rank in allocation.items():
            lmod = lora_modules.get(name)
            if lmod is not None:
                lmod.update_rank(rank)
        # rebuild optimizer with new LoRA params
        nonlocal optimizer
        optimizer = optim.AdamW(get_lora_param_list(), lr=lr, weight_decay=weight_decay)
        return allocation

    # initial allocation
    print("Computing initial allocation...")
    alloc = recompute_and_apply_ranks()
    print("Initial allocation:", alloc)

    # training loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            _, pred = outputs.max(dim=1)
            correct += (pred == targets).sum().item()

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # optionally recompute allocation
        if epoch % recompute_every == 0:
            print(f"Epoch {epoch}: recomputing allocation...")
            alloc = recompute_and_apply_ranks()

        # eval
        model.eval()
        val_loss = 0.0
        vtotal = 0
        vcorrect = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                vtotal += inputs.size(0)
                _, pred = outputs.max(dim=1)
                vcorrect += (pred == targets).sum().item()
                if fast_mode and batch_idx >= 4:
                    break
        val_loss = val_loss / max(1, vtotal)
        val_acc = vcorrect / max(1, vtotal)

        t1 = time.time()
        print(f"Epoch {epoch}/{epochs} - time {t1-t0:.1f}s | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        print("Current allocation:", alloc)

    return model, alloc
