import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, orig, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.orig = orig
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.scaling = self.alpha / self.r if self.r > 0 else 1.0

        if r > 0:
            # Match dtype and device with original layer weights
            dtype = orig.weight.dtype
            device = orig.weight.device
            self.A = nn.Parameter(torch.randn(r, self.in_features, dtype=dtype, device=device) * 0.01)
            self.B = nn.Parameter(torch.randn(self.out_features, r, dtype=dtype, device=device) * 0.01)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        out = self.orig(x)
        if self.r > 0:
            # Ensure dtype consistency
            A_t = self.A.t().to(x.dtype)
            B_t = self.B.t().to(x.dtype)
            lora_mid = torch.matmul(x, A_t)
            lora_out = torch.matmul(lora_mid, B_t)
            lora_out = self.dropout(lora_out) * self.scaling
            out = out + lora_out
        return out


def _find_module_by_name(model, name):
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None, None, None
        parent = getattr(parent, p)
    attr = parts[-1]
    if not hasattr(parent, attr):
        return None, None, None
    return parent, attr, getattr(parent, attr)

def inject_adaptive_lora(model, module_names, ranks, alpha=16, dropout=0.0, replace_linear=True):
    patched = []
    for name, r in zip(module_names, ranks):
        parent, attr, module = _find_module_by_name(model, name)
        if module is None:
            found = False
            for n, m in model.named_modules():
                if name in n and isinstance(m, nn.Linear):
                    parent, attr, module = _find_module_by_name(model, n)
                    found = True
                    name = n
                    break
            if not found:
                continue
        if isinstance(module, nn.Linear) and replace_linear:
            wrapped = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, wrapped)
            patched.append(name)
        else:
            for child_name, child_module in module.named_modules():
                if isinstance(child_module, nn.Linear):
                    child_parent, child_attr, _ = _find_module_by_name(module, child_name)
                    wrapped = LoRALinear(child_module, r=r, alpha=alpha, dropout=dropout)
                    setattr(child_parent, child_attr, wrapped)
                    patched.append(f"{name}.{child_name}")
                    break
    return patched
