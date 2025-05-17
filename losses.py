import torch
import torch.nn as nn
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

class MaskedMSELoss(nn.Module):
    """
    Computes MSE loss, **excluding** (a) stations (hard-enforced), and (b) grid points outside NY (mask=0).
    
    Usage:
        criterion = MaskedMSELoss(mask)  # mask: torch tensor [H,W]
        loss = criterion(output, target, station_mask)
    """
    def __init__(self, mask_2d):
        """
        mask_2d: torch.Tensor [H, W] (1=inside NY, 0=outside NY)
        """
        super().__init__()
        self.register_buffer("mask_2d", mask_2d.float())  # persists on .cuda()/.cpu(), such that the mask_2d devie is used. 

    def forward(self, output, target, station_mask, reduction='mean'):
        """
        output: [B, 1, H, W]
        target: [B, 1, H, W]
        station_mask: [B, 1, H, W]  (1=station, 0=else)
        """
        B, _, H, W = output.shape
        mask = self.mask_2d.unsqueeze(0).unsqueeze(0)       # [1, 1, H, W], device-matched
        station_mask = station_mask.float()     # [B, 1, H, W]

        # Only inside NY and not at stations
        valid_mask = (mask == 1) & (station_mask == 0)       # [B, 1, H, W]
        valid_mask = valid_mask.float()

        # Compute masked squared error
        se = (output - target) ** 2
        masked_se = se * valid_mask

        mse_sum_per_sample = masked_se.view(B, -1).sum(dim=1)
        valid_counts = valid_mask.view(B, -1).sum(dim=1).clamp(min=1.0)
        mse_per_sample = mse_sum_per_sample / valid_counts  # shape: [B]

        if reduction == 'none':
            return mse_per_sample  # [B]
        elif reduction == 'mean':
            return mse_per_sample.mean()  # scalar
        elif reduction == 'global':
            total_se = mse_sum_per_sample.sum()
            total_valid = valid_counts.sum().clamp(min=1.0)
            return total_se / total_valid  # scalar

class MaskedRMSELoss(nn.Module):
    """
    Computes RMSE loss, **excluding** (a) stations (hard-enforced), and (b) grid points outside NY (mask=0).
    
    Usage:
        criterion = MaskedRMSELoss(mask)  # mask: torch tensor [H,W]
        loss = criterion(output, target, station_mask)
    """
    def __init__(self, mask_2d):
        """
        mask_2d: torch.Tensor [H, W] (1=inside NY, 0=outside NY)
        """
        super().__init__()
        self.register_buffer("mask_2d", mask_2d.float())  # persists on .cuda()/.cpu(), such that the mask_2d devie is used.

    def forward(self, output, target, station_mask):
        """
        output: [B, 1, H, W]
        target: [B, 1, H, W]
        station_mask: [B, 1, H, W]  (1=station, 0=else)
        """
        B, _, H, W = output.shape
        mask = self.mask_2d.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
        station_mask = station_mask.float()  # [B, 1, H, W]

        # Only inside NY and not at stations
        valid_mask = (mask == 1) & (station_mask == 0)       # [B, 1, H, W]
        valid_mask = valid_mask.float()

        # Compute masked squared error
        se = (output - target) ** 2
        masked_se = se * valid_mask

        mse = masked_se.sum() / valid_mask.sum().clamp(min=1.0)
        rmse = torch.sqrt(mse)
        return rmse
    
class MaskedTVLoss(nn.Module):
    """
    Total Variation Loss, computed **only** at valid locations (inside NY, not at stations).
    """
    def __init__(self, mask_2d, tv_loss_weight=1.0, beta=0.5):
        """
        mask_2d: torch.Tensor [H, W] (1=inside NY, 0=outside NY)
        tv_loss_weight: scaling factor for TV loss
        beta: degree of smoothness penalty
        """
        super().__init__()
        self.register_buffer("mask_2d", mask_2d.float())    # persists on .cuda()/.cpu(), such that the mask_2d devie is used.
        self.tv_loss_weight = tv_loss_weight
        self.beta = beta

    def forward(self, x, station_mask):
        """
        x: [B, 1, H, W] (predicted field)
        station_mask: [B, 1, H, W]  (1=station, 0=else)
        """
        B, _, H, W = x.shape
        mask = self.mask_2d.unsqueeze(0).unsqueeze(0)       # [1, 1, H, W]
        station_mask = station_mask.float()     # [B, 1, H, W]
        valid_mask = (mask == 1) & (station_mask == 0)       # [B, 1, H, W]

        # Horizontal and vertical TV only for valid locations
        dh = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs() ** self.beta  # [B, 1, H, W-1]
        dw = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs() ** self.beta  # [B, 1, H-1, W]

        # valid_mask for dh (last col missing), dw (last row missing)
        valid_mask_h = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
        valid_mask_w = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]

        tv = ((dh * valid_mask_h).sum() + (dw * valid_mask_w).sum()) / (
            valid_mask_h.sum() + valid_mask_w.sum()
        ).clamp(min=1.0)
        return self.tv_loss_weight * tv

class MaskedCharbonnierLoss(nn.Module):
    """
    Charbonnier Loss, only over valid (masked) locations.
    """
    def __init__(self, mask_2d, eps=1e-3):
        """
        mask_2d: torch.Tensor [H, W] (1=inside NY, 0=outside NY)
        eps: Charbonnier smoothing factor
        """
        super().__init__()
        self.register_buffer("mask_2d", mask_2d.float())   # persists on .cuda()/.cpu(), such that the mask_2d devie is used.
        self.eps = eps

    def forward(self, x, y, station_mask):
        """
        x: [B, 1, H, W] (prediction)
        y: [B, 1, H, W] (target)
        station_mask: [B, 1, H, W]  (1=station, 0=else)
        """
        B, _, H, W = x.shape
        mask = self.mask_2d.unsqueeze(0).unsqueeze(0)       # [1, 1, H, W]
        station_mask = station_mask.float()     # [B, 1, H, W]
        valid_mask = (mask == 1) & (station_mask == 0)       # [B, 1, H, W]

        diff = x - y
        charbonnier = torch.sqrt(diff**2 + self.eps**2)
        masked_charb = charbonnier * valid_mask
        loss = masked_charb.sum() / valid_mask.sum().clamp(min=1.0)
        return loss

class MaskedPSNR(nn.Module):
    """
    Peak Signal-to-Noise Ratio (PSNR) loss, only over valid (masked) locations.
    """
    def __init__(self, mask_2d):
        """
        mask_2d: torch.Tensor [H, W] (1=inside NY, 0=outside NY)
        """
        super().__init__()
        self.register_buffer("mask_2d", mask_2d.float())   # persists on .cuda()/.cpu(), such that the mask_2d devie is used.
        self.psnr = PeakSignalNoiseRatio()

    def forward(self, x, y, station_mask):
        """
        x: [B, 1, H, W] (prediction)
        y: [B, 1, H, W] (target)
        station_mask: [B, 1, H, W]  (1=station, 0=else)
        """
        B, _, H, W = x.shape
        mask = self.mask_2d.unsqueeze(0).unsqueeze(0)       # [1, 1, H, W]
        station_mask = station_mask.float()     # [B, 1, H, W]
        valid_mask = (mask == 1) & (station_mask == 0)       # [B, 1, H, W]

        x = x * mask
        y = y * mask
        return self.psnr(x, y)
    
class MaskedSSIM(nn.Module):
    """
    Structural Similarity Index Measure (SSIM) loss, only over valid (masked) locations.
    """
    def __init__(self, mask_2d):
        """
        mask_2d: torch.Tensor [H, W] (1=inside NY, 0=outside NY)
        """
        super().__init__()
        self.register_buffer("mask_2d", mask_2d.float())   # persists on .cuda()/.cpu(), such that the mask_2d devie is used.
        self.ssim = StructuralSimilarityIndexMeasure()

    def forward(self, x, y, station_mask):
        """
        x: [B, 1, H, W] (prediction)
        y: [B, 1, H, W] (target)
        station_mask: [B, 1, H, W]  (1=station, 0=else)
        """
        B, _, H, W = x.shape
        mask = self.mask_2d.unsqueeze(0).unsqueeze(0)       # [1, 1, H, W]
        station_mask = station_mask.float()     # [B, 1, H, W]
        valid_mask = (mask == 1) & (station_mask == 0)       # [B, 1, H, W]

        x = x * valid_mask
        y = y * valid_mask
        return self.ssim(x, y)