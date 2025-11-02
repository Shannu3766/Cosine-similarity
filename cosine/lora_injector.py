# examples/lora_injector.py
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """
    A simple LoRA-adapted Linear layer wrapping an existing nn.Linear.
    delta_W = A @ B, where A: out_features x r, B: r x in_features
    Forward: original(x) + (x @ B^T) @ A^T
    """

    def __init__(self, orig_linear: nn.Linear, rank: int = 0, alpha: float = 1.0):
        super().__init__()
        self.orig = orig_linear
        self.rank = int(rank)
        self.alpha = alpha
        # freeze base weights by default (typical LoRA)
        for p in self.orig.parameters():
            p.requires_grad = False

        in_f = self.orig.in_features
        out_f = self.orig.out_features

        if self.rank > 0:
            # initialize low-rank factors
            self.A = nn.Parameter(torch.zeros(out_f, self.rank))
            self.B = nn.Parameter(torch.zeros(self.rank, in_f))
            # init like LoRA: B ~ N(0, 0.02), A zeros
            nn.init.normal_(self.B, std=0.02)
            nn.init.zeros_(self.A)
        else:
            self.A = None
            self.B = None

    def update_rank(self, new_rank):
        new_rank = int(new_rank)
        if new_rank == self.rank:
            return
        in_f = self.orig.in_features
        out_f = self.orig.out_features
        old_rank = self.rank
        oldA, oldB = (self.A, self.B) if self.A is not None else (None, None)

        if new_rank > 0:
            A = torch.zeros(out_f, new_rank, device=self.orig.weight.device)
            B = torch.zeros(new_rank, in_f, device=self.orig.weight.device)
            # copy old if exists
            if old_rank and oldA is not None and oldB is not None:
                r_copy = min(old_rank, new_rank)
                A[:, :r_copy] = oldA.data[:, :r_copy]
                B[:r_copy, :] = oldB.data[:r_copy, :]
            else:
                # init B small
                nn.init.normal_(B, std=0.02)
            self.A = nn.Parameter(A)
            self.B = nn.Parameter(B)
            # ensure orig weights frozen
            for p in self.orig.parameters():
                p.requires_grad = False
        else:
            # remove adapters
            self.A = None
            self.B = None

        self.rank = new_rank

    def forward(self, x):
        out = self.orig(x)
        if self.rank and self.A is not None and self.B is not None:
            # LoRA delta: x -> (B @ x.T) -> (A @ that). More efficient: (x @ B.T) @ A.T
            delta = (x @ self.B.t()) @ self.A.t()
            out = out + (self.alpha * delta)
        return out

def inject_lora_to_linears(model, rank_map):
    """
    Walks model modules and replaces nn.Linear with LoRALinear using ranks from rank_map (module full name -> rank).
    rank_map: dict of module_name (as returned by named_modules) -> rank int
    Returns mapping name -> module (LoRALinear)
    """
    name_to_module = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            r = rank_map.get(name, 0)
            parent, attr = _find_parent_module(model, name)
            if parent is None:
                continue
            orig = getattr(parent, attr)
            lora = LoRALinear(orig, rank=r)
            setattr(parent, attr, lora)
            name_to_module[name] = lora
    return name_to_module

def _find_parent_module(model, full_name):
    """
    Given full module name like 'encoder.layers.0.ffn.linear1', find parent module and attribute.
    """
    parts = full_name.split(".")
    if len(parts) == 1:
        parent = model
        attr = parts[0]
    else:
        parent = model
        for p in parts[:-1]:
            if not hasattr(parent, p):
                return None, None
            parent = getattr(parent, p)
        attr = parts[-1]
    return parent, attr

def extract_lora_parameters(model):
    """
    Return list of parameters belonging to LoRA adapters (A and B).
    """
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if m.A is not None:
                params.append(m.A)
            if m.B is not None:
                params.append(m.B)
    return params
