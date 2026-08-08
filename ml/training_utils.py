import torch
import torch.nn.functional as F

def stage1_loss(
    logits: torch.Tensor,               # (B, C) raw logits from model.stage1_logits
    targets: torch.Tensor,              # (B,) Long indices in [0..C-1] (members + silence)
    *,
    example_weights, # (B,) optional
    class_weights,   # (C,) optional
    label_smoothing: float = 0.0,       # e.g., 0.05
    temperature: float = 1.0,           # e.g., 1.0
    reduction: str = "mean"             # "mean" | "sum" | "none"
) -> torch.Tensor:
    """
    Unified Stage-1 loss for mutually-exclusive (members + silence) classification.
    Works for both training and validation by toggling label_smoothing/example_weights.
    """
    assert reduction in ("mean", "sum", "none")
    assert temperature > 0.0
    B, C = logits.shape

    if targets.ndim == 1:
        if targets.max().item() >= C or targets.min().item() < 0:
            raise ValueError(f"targets indices out of range: max={targets.max().item()} vs C={C}")
    else:
        if targets.shape[1] != C:
            raise ValueError(f"targets one-hot has wrong C: {targets.shape[1]} vs logits C={C}")

    if temperature != 1.0:
        logits = logits / float(temperature)
        
    if targets.ndim == 1:
        # hard -> one-hot
        soft = F.one_hot(targets.to(torch.long), num_classes=C).to(logits.dtype)
    else:
        # already (B,C) distribution/one-hot
        soft = targets.to(logits.dtype)
        # quick sanity: clamp tiny negatives
        soft = soft.clamp_min(0.0)

    # Label smoothing applied directly on provided one-hot
    if label_smoothing > 0.0:
        eps  = float(label_smoothing)
        soft = soft * (1.0 - eps) + eps / C          # (B,C)

    # CE with soft targets
    logits_f32 = logits.float()
    logp = torch.nn.functional.log_softmax(logits_f32, dim=1)
    ce = -(soft * logp).sum(dim=1)   

    # Class weights (per-sample) via expectation under 'soft'
    if class_weights is not None:
        cw = class_weights.to(logits.device, logits.dtype)      # (C,)
        cw_per_sample = (soft * cw.unsqueeze(0)).sum(dim=1)     # (B,)
        ce = ce * cw_per_sample

    # Example weights (normalize to mean=1 to keep scale stable)
    if example_weights is not None:
        ew = example_weights.to(logits.device, logits.dtype)
        ce = ce * (ew / ew.mean().clamp_min(1e-8))

    if reduction == "mean":
        return ce.mean()
    elif reduction == "sum":
        return ce.sum()
    else:
        return ce  # (B,)
    
@torch.no_grad()
def stage1_metrics(logits: torch.Tensor, targets: torch.Tensor):
    """
    Convenience metrics for quick debugging/printing.
    Returns a dict with acc (float), pred_mean (C,), tgt_mean (C,)
    """
    probs = torch.softmax(logits, dim=1)                           # (B,C)
    preds = probs.argmax(dim=1)                                    # (B,)
    acc = (preds == targets).float().mean().item()

    C = logits.size(1)
    tgt_oh = F.one_hot(targets, num_classes=C).to(probs.dtype)     # (B,C)
    return {
        "acc": acc,
        "pred_mean": probs.mean(dim=0),                             # (C,)
        "tgt_mean": tgt_oh.mean(dim=0),                            # (C,)
    }    