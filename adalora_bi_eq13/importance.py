import torch
import numpy as np

def compute_bi_importance_eq13(model, dataloader, device='cuda', max_batches=8, target_module_name_substrings=None):
    """Compute importance s_i using Eq. (1.3):
    s_i = (1/K) sum_k | (dL/dh_i^k) · h_i^k |
    Returns: module_names (list) and scores (list of floats)
    """
    model.to(device)
    model.eval()

    # select modules
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if target_module_name_substrings:
                if any(sub in name for sub in target_module_name_substrings):
                    modules.append((name, module))
            else:
                if any(tok in name for tok in ('q_proj','k_proj','v_proj','out_proj','q_lin','k_lin','v_lin','out_lin','ffn','lin1','lin2','dense','fc','proj')):
                    modules.append((name, module))
    if not modules:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                modules.append((name, module))
        modules = modules[:20]

    module_names = [n for n,_ in modules]
    activations = {n: [] for n in module_names}
    grads = {n: [] for n in module_names}
    hooks = []

    # forward hook to capture activations
    def make_fwd(name):
        def hook(module, inp, out):
            x = out.detach().cpu()
            if x.ndim > 1:
                vec = x.mean(dim=tuple(range(0, x.ndim-1)))
            else:
                vec = x.view(-1)
            activations[name].append(vec.view(-1))
        return hook

    # backward hook to capture gradient w.r.t. output
    def make_bwd(name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0].detach().cpu()
            if g.ndim > 1:
                vec = g.mean(dim=tuple(range(0, g.ndim-1)))
            else:
                vec = g.view(-1)
            grads[name].append(vec.view(-1))
        return hook

    for name, module in modules:
        try:
            hooks.append(module.register_forward_hook(make_fwd(name)))
            if hasattr(module, 'register_full_backward_hook'):
                hooks.append(module.register_full_backward_hook(make_bwd(name)))
            else:
                hooks.append(module.register_backward_hook(make_bwd(name)))
        except Exception:
            continue

    loss_fn = torch.nn.CrossEntropyLoss()
    num_batches = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        to_dev = {}
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                to_dev[k] = v.to(device)
            else:
                to_dev[k] = v
        try:
            outputs = model(**to_dev)
        except TypeError:
            kwargs = {}
            for k in ('input_ids','attention_mask','inputs_embeds'):
                if k in to_dev:
                    kwargs[k] = to_dev[k]
            outputs = model(**kwargs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        if 'labels' in to_dev:
            labels = to_dev['labels']
            loss = loss_fn(logits, labels)
        else:
            continue
        model.zero_grad()
        loss.backward()
        num_batches += 1

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    scores = []
    for name in module_names:
        A = activations.get(name, [])
        G = grads.get(name, [])
        M = min(len(A), len(G))
        if M == 0:
            scores.append(0.0)
            continue
        vals = []
        for j in range(M):
            a = A[j].numpy().ravel()
            g = G[j].numpy().ravel()
            n = min(a.size, g.size)
            a = a[:n]; g = g[:n]
            vals.append(abs(float((g * a).sum())))
        scores.append(float(sum(vals)/len(vals)))
    arr = np.array(scores, dtype=float)
    if arr.max() - arr.min() > 1e-12:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
    else:
        arr = arr * 0.0
    return module_names, arr.tolist()
