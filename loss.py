import torch

def dice_metric(pred, target, smooth=1e-8):
    """
    Computes the Dice Score/Coefficient for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W) - logits or probabilities
        target: Tensor of ground truth (batch_size, 1, H, W) - binary masks
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Scalar Dice Score (higher is better)
    """
    # Apply sigmoid to convert logits to probabilities if needed
    pred = torch.nn.functional.sigmoid(pred)
    
    # Flatten tensors for easier computation (o1ptional but common)
    pred_flat = pred.view(pred.size(0), -1)  # (batch_size, H*W)
    target_flat = target.view(target.size(0), -1)  # (batch_size, H*W)
    
    # Calculate intersection
    intersection = (pred_flat * target_flat).sum(dim=1)  # Per batch
    
    # Calculate Dice coefficient
    # Formula: 2 * |A ∩ B| / (|A| + |B|)
    dice_denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2. * intersection + smooth) / (dice_denominator + smooth)
    
    return dice.mean()  # Average across batch

def dice_loss(pred: torch.tensor, target:torch.tensor, smooth: float = 1e-8):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W) - logits
        target: Tensor of ground truth (batch_size, 1, H, W) - binary masks
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Scalar Dice Loss (lower is better)
    """
    dice_score = dice_metric(pred, target, smooth)
    return 1 - dice_score
