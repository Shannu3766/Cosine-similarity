# examples/importance.py
import torch
from collections import defaultdict

class ActivationGradRecorder:
    """
    Records activations and gradients for modules of interest.
    Use register on modules (e.g., nn.Linear) to capture forward activations
    and backward gradients w.r.t. those activations.
    """

    def __init__(self):
        self._acts = {}
        self._grads = {}
        self._hooks = []

    def register_module(self, name, module):
        # forward hook to save activation
        def forward_hook(mod, inp, out):
            # out may be tensor or tuple
            self._acts[name] = out.detach().cpu()
        fh = module.register_forward_hook(forward_hook)
        self._hooks.append(fh)

        # backward hook: gradient w.r.t. output activation
        def backward_hook(mod, grad_in, grad_out):
            # grad_out is tuple
            go = grad_out[0].detach().cpu() if isinstance(grad_out, tuple) else grad_out.detach().cpu()
            self._grads[name] = go
        # PyTorch >= 1.8: register_full_backward_hook or register_backward_hook
        try:
            bh = module.register_full_backward_hook(backward_hook)
        except Exception:
            bh = module.register_backward_hook(backward_hook)
        self._hooks.append(bh)

    def clear(self):
        self._acts.clear()
        self._grads.clear()

    def remove(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []

    def compute_importance_scores(self):
        """
        For each recorded module name, compute mean(|grad * act|) over elements.
        Returns dict: name -> scalar score
        """
        scores = {}
        for name in self._acts:
            act = self._acts.get(name)
            grad = self._grads.get(name)
            if act is None or grad is None:
                # missing either activation or grad -> score 0
                scores[name] = 0.0
                continue
            # ensure shapes align: shape of grad and act must match
            try:
                # compute absolute elementwise product, then mean
                prod = (grad * act).abs()
                scores[name] = float(prod.mean().item())
            except Exception:
                # fallback: flatten both and compute mean of elementwise
                a = act.reshape(-1)
                g = grad.reshape(-1)
                L = min(a.numel(), g.numel())
                if L == 0:
                    scores[name] = 0.0
                else:
                    scores[name] = float((a[:L] * g[:L]).abs().mean().item())
        return scores

def compute_scores_over_dataloader(model, val_loader, device, module_name_map, max_batches=2, loss_fn=None):
    """
    model: torch.nn.Module
    val_loader: DataLoader providing (input, target) pairs
    module_name_map: dict name->module for which to record activations
    max_batches: how many batches to use for the aggregate
    loss_fn: loss function; default CrossEntropy for classification if None
    returns: dict name->score (averaged across batches)
    """
    recorder = ActivationGradRecorder()
    for name, module in module_name_map.items():
        recorder.register_module(name, module)

    model.eval()
    counts = defaultdict(int)
    sums = defaultdict(float)
    loss_fn = loss_fn or torch.nn.CrossEntropyLoss()

    device_cpu = torch.device(device) if isinstance(device, str) else device

    with torch.no_grad():
        # we need gradients, so temporarily enable grad by removing no_grad context.
        pass

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        inputs, targets = batch
        inputs = inputs.to(device_cpu)
        targets = targets.to(device_cpu)

        model.zero_grad()
        # forward (with gradient enabled)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        # backward to fill gradients wrt activations
        loss.backward()

        # compute scores for this batch
        batch_scores = recorder.compute_importance_scores()
        for k, v in batch_scores.items():
            sums[k] += v
            counts[k] += 1

        recorder.clear()

    # average
    final_scores = {}
    for name in module_name_map.keys():
        if counts[name] > 0:
            final_scores[name] = sums[name] / counts[name]
        else:
            final_scores[name] = 0.0

    recorder.remove()
    return final_scores
