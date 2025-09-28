import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def grad_l2_norm(model) -> float:
    """
    Compute total L2 norm of gradients for all parameters in a model.
    Returns 0.0 if no gradients are present.
    """
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total_sq += float(g.pow(2).sum().item())
    return float(total_sq ** 0.5)

def denorm_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor in [-1, 1] float to uint8 in [0, 255].
    Accepts shape (N, C, H, W). Returns tensor on same device, dtype uint8.
    """
    x = (x.clamp(-1, 1) + 1.0) / 2.0
    x = (x * 255.0).round().clamp(0, 255)
    return x.to(torch.uint8)