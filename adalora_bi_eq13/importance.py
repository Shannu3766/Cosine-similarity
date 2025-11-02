import torch
from tqdm import tqdm


def compute_bi_importance_eq13(
    model,
    dataloader,
    device="cuda",
    max_batches=2,
    target_module_name_substrings=None,
):
    """Compute BI importance (Eq.1.3): s_i = (1/K) Σ ∂L/∂h_i · h_i"""
    model.eval()
    importance_scores = {}
    handles = []

    def save_activation(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                importance_scores[name]["h"] = out.detach().clone().float()
        return hook

    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple) and grad_output[0] is not None:
                importance_scores[name]["grad"] = grad_output[0].detach().clone().float()
        return hook

    # register hooks on all Linear layers (or filtered subset)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if (target_module_name_substrings is None) or any(
                k in name for k in target_module_name_substrings
            ):
                importance_scores[name] = {}
                handles.append(module.register_forward_hook(save_activation(name)))
                handles.append(module.register_full_backward_hook(save_gradient(name)))

    # compute gradients
    loss_fn = torch.nn.CrossEntropyLoss()
    for i, batch in enumerate(tqdm(dataloader, desc="Computing BI importance")):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else None

        if loss is None:
            continue
        loss.backward()

        # accumulate ∂L/∂h · h for each layer
        for name, tensors in importance_scores.items():
            if "grad" in tensors and "h" in tensors:
                g, h = tensors["grad"], tensors["h"]
                score = torch.sum(g * h).item() / (h.numel() + 1e-8)
                importance_scores[name]["value"] = (
                    importance_scores[name].get("value", 0.0) + score
                )

    for h in handles:
        h.remove()

    # average over batches
    module_names, scores = [], []
    for name, tensors in importance_scores.items():
        if "value" in tensors:
            avg_score = tensors["value"] / max_batches
            module_names.append(name)
            scores.append(avg_score)

    # diagnostics
    if len(scores) > 0:
        print(f"\n[DEBUG] BI score stats → min={min(scores):.6e}, max={max(scores):.6e}, mean={sum(scores)/len(scores):.6e}")
    else:
        print("\n[WARNING] No BI scores computed! Check gradient flow or hooks.")

    return module_names, scores
