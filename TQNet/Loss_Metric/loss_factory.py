import torch
import torch.nn as nn
import logging
from scipy.signal import find_peaks, correlate
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from Loss_Metric.hsic import HSIC
import torch.nn.functional as F
from tslearn.metrics import SoftDTWLossPyTorch

# input-->(forecast [batch, pred_len, feature], ground truth[batch, pred_len, feature]) 
# output-->scalar

# ===================== =====================

def real_corr(x, y):
    M, T = x.shape
    N = 2 * T - 1
    # smallest power-of-two >= N
    L = 1 << ((N - 1).bit_length())
    fx = torch.fft.rfft(x, n=L, dim=-1)
    fy = torch.fft.rfft(y, n=L, dim=-1)
    corr = torch.fft.irfft(torch.conj(fx) * fy, n=L, dim=-1)
    return corr[:, :N]
def peak_align_low_lag_details(x: torch.Tensor, trend: torch.Tensor):
    """Return the original PALL alignment plus auditable lag diagnostics.

    The alignment calculation is intentionally the same one used by the
    training loss.  Extra return values do not alter the selected lag.
    """
    B, T, C = x.shape
    device = x.device

    # # [B, T, C]
    x_cent = x - x.mean(dim=1, keepdim=True)
    t_cent = trend - trend.mean(dim=1, keepdim=True)


    x_conv = x_cent.permute(0, 2, 1).reshape(B * C, T)
    t_conv = t_cent.permute(0, 2, 1).reshape(B * C, T)
    corr = real_corr(x_conv, t_conv)   # (M, 2T-1)
    max_index = torch.argmax(corr, dim=1)
    shift = -max_index



    # 
    pos = torch.arange(T, device=device)
    idx = pos - shift.view(B * C, 1)
    idx = idx.clamp(0, T - 1)

    trend_aligned = trend.permute(0, 2, 1).reshape(B * C, T)
    trend_aligned = torch.gather(trend_aligned, dim=1, index=idx)
    trend_aligned = (
        trend_aligned.view(B, C, T)
        .permute(0, 2, 1)
        .contiguous()
    )

    lag = shift.view(B, C)
    search_boundary = ((max_index == 0) | (max_index == corr.shape[1] - 1)).view(B, C)
    if corr.shape[1] > 1:
        top_two = torch.topk(corr, k=2, dim=1).values
        ambiguity_gap = (
            (top_two[:, 0] - top_two[:, 1]) /
            top_two[:, 0].abs().clamp_min(torch.finfo(corr.dtype).eps)
        ).view(B, C)
    else:
        ambiguity_gap = torch.zeros((B, C), dtype=corr.dtype, device=device)
    return trend_aligned, lag, search_boundary, ambiguity_gap


def peak_align_low_lag(x: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
    return peak_align_low_lag_details(x, trend)[0]

# ==========================================
def trend_guided_reweight(pred, true, true_trend, q, alpha, beta):
    
    diff = pred - true

    mask_above_trend = (true > true_trend).float()
    mask_below_trend = (true <= true_trend).float()
    mask_pos_diff = (diff > 0).float()
    mask_neg_diff = (diff <= 0).float()

    c1 = mask_above_trend * mask_pos_diff
    c2 = mask_above_trend * mask_neg_diff
    c3 = mask_below_trend * mask_pos_diff
    c4 = mask_below_trend * mask_neg_diff

    if alpha == 1:
        w1_pos, w1_neg = 1.0 - q, q
    else:
        w1_pos, w1_neg = q, 1.0 - q

    if beta == 1:
        w2_pos, w2_neg = 1.0 - q, q
    else:
        w2_pos, w2_neg = q, 1.0 - q

    return diff, c1, c2, c3, c4, w1_pos, w1_neg, w2_pos, w2_neg

# ================================================================
class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """
    def __init__(self, ema_alpha):
        super(EMA, self).__init__()

        if not (0 < ema_alpha < 1):
            raise ValueError(f"EMA alpha must be between 0 and 1, got {ema_alpha}")
        self.ema_alpha = ema_alpha

    def forward(self, x):
        _, t, _ = x.shape

        device = x.device
        

        powers = torch.flip(torch.arange(t, dtype=torch.double, device=device), dims=(0,))
        

        weights = torch.pow((1 - self.ema_alpha), powers)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.ema_alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)
#======================

class EMADECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ema_alpha):
        super(EMADECOMP, self).__init__()
        self.ma = EMA(ema_alpha)

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average

# 3. peak_align ======================
class PALL_EMA(nn.Module):
    def __init__(self, ema_alpha):
        super().__init__()
        self.ma = EMA(ema_alpha)

    def forward(self, x):
        moving_average = self.ma(x)
        trend_aligned = peak_align_low_lag(x, moving_average)
        res = x - trend_aligned
        return res, trend_aligned


#======================
class Mean_Decomp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, T, C = x.shape
        global_mean = x.mean(dim=1, keepdim=True)  # shape: [B, 1, C]
        mean_full = global_mean.expand(-1, T, -1)
        res = x - mean_full
        return res, mean_full


# =======================================================================

class EMA_TQA_MAELoss(nn.Module):
    """
    xiaorong
    """
    def __init__(self, alpha=None, beta=None, gamma=None, q=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ema_alpha = 0.3
        self.q = float(gamma)
        self.decomp = EMADECOMP(self.ema_alpha)

    def forward(self, pred, true):

        if pred.device != true.device:
            true = true.to(pred.device)

        _, true_trend = self.decomp(true)
        diff, c1, c2, c3, c4, w1_pos, w1_neg, w2_pos, w2_neg = \
            trend_guided_reweight(pred, true, true_trend, self.q, self.alpha, self.beta)
        
        abs_diff = torch.abs(diff)

        loss = abs_diff * (w1_pos*c1 + w1_neg*c2 + w2_pos*c3 + w2_neg*c4)
        return loss.mean()

class TQA_MAELoss(nn.Module):
    def __init__(self, alpha=None, beta=None, gamma=None, q=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ema_alpha = 0.3
        self.q = float(gamma)
        self.decomp = Mean_Decomp()

    def forward(self, pred, true):
        if pred.device != true.device:
            true = true.to(pred.device)

        _, true_trend = self.decomp(true)
        diff, c1, c2, c3, c4, w1_pos, w1_neg, w2_pos, w2_neg = \
            trend_guided_reweight(pred, true, true_trend, self.q, self.alpha, self.beta)
        
        abs_diff = torch.abs(diff)

        loss = abs_diff * (w1_pos*c1 + w1_neg*c2 + w2_pos*c3 + w2_neg*c4)
        return loss.mean()

class PALL_EMA_TQA_MAELoss(nn.Module):
    def __init__(self, alpha=None, beta=None, gamma=None):
        super().__init__()
        if gamma is None or not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma is used as q and must be in [0, 1], got {gamma}")
        self.alpha = alpha
        self.beta = beta
        self.ema_alpha = 0.3
        self.q = float(gamma)
        self.decomp = PALL_EMA(self.ema_alpha)

    def forward(self, pred, true):
        if pred.device != true.device:
            true = true.to(pred.device)

        _, true_trend = self.decomp(true)
        diff, c1, c2, c3, c4, w1_pos, w1_neg, w2_pos, w2_neg = \
            trend_guided_reweight(pred, true, true_trend, self.q, self.alpha, self.beta)
        
        abs_diff = torch.abs(diff)

        loss = abs_diff * (w1_pos*c1 + w1_neg*c2 + w2_pos*c3 + w2_neg*c4)
        
        return loss.mean()

class DBLoss(nn.Module):

    def __init__(self, alpha=0.2, beta=0.5):
        super().__init__()
        self.decomp = EMADECOMP(alpha)
        self.beta = beta
        self.mse = nn.MSELoss(reduction="mean")
        self.mae = nn.L1Loss(reduction="mean")

    def forward(self, pred, target):
        pred_season, pred_trend = self.decomp(pred)
        target_season, target_trend = self.decomp(target)

        season_loss = self.mse(pred_season, target_season)
        trend_loss = self.mae(pred_trend, target_trend)
        trend_loss = trend_loss * (season_loss / (trend_loss + 1e-8)).detach()
        return self.beta * season_loss + (1 - self.beta) * trend_loss

class CARDLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()  
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred, target) -> torch.Tensor:
        prelen = pred.shape[1]
        ratio = 1.0 / torch.sqrt(torch.arange(1, prelen + 1, dtype=torch.float32, device=pred.device)).unsqueeze(-1)
        outputs = pred * ratio
        batch_y = target * ratio
        loss = self.mse(outputs, batch_y)
        return loss

class xPatchLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()  
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred, target) -> torch.Tensor:
        prelen = pred.shape[1]

        ratio = [-1 * math.atan(i+1) + math.pi/4 + 1 for i in range(prelen)]
        ratio = torch.tensor(ratio, dtype=torch.float32, device=pred.device)
        ratio = ratio.unsqueeze(-1)

        outputs = pred * ratio
        batch_y = target * ratio
        loss = self.mse(outputs, batch_y)
        return loss
    
class FreDFLoss(nn.Module):
    def __init__(self,rec_lambda=0.1) -> None:
        super().__init__()  
        self.rec_lambda = rec_lambda
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred, target):
        loss_rec = self.mse(pred, target)
        loss = self.rec_lambda * loss_rec

        loss_auxi = torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)
        loss_auxi = loss_auxi.abs().mean()
        loss += (1- self.rec_lambda) * loss_auxi
        return loss

class LinexLoss(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        if alpha == 0:
            raise ValueError("LinexLoss requires a non-zero loss_alpha")
        self.alpha = alpha

    def forward(self, pred, target):
        if pred.device != target.device:
            target = target.to(pred.device)

        error = pred - target
        scaled_error = self.alpha * error
        return (torch.exp(scaled_error) - scaled_error - 1).mean()

class QuantileLoss(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        if not 0 < alpha < 1:
            raise ValueError("QuantileLoss requires 0 < loss_alpha < 1")
        self.alpha = alpha

    def forward(self, pred, target):
        if pred.device != target.device:
            target = target.to(pred.device)

        error = target - pred
        return torch.maximum(
            self.alpha * error,
            (self.alpha - 1) * error,
        ).mean()

class StableLinexLoss(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        if not math.isfinite(alpha) or alpha == 0:
            raise ValueError("StableLinexLoss requires a finite, non-zero loss_alpha")
        self.alpha = float(alpha)
        self.max_exp_input = 8.0

    def forward(self, pred, target):
        if pred.device != target.device:
            target = target.to(pred.device)

        error = pred - target
        work_error = error.float() if error.dtype in (torch.float16, torch.bfloat16) else error
        scaled_error = self.alpha * work_error
        max_exp_input = scaled_error.new_tensor(self.max_exp_input)
        safe_scaled_error = torch.minimum(scaled_error, max_exp_input)

        linex_core = torch.expm1(safe_scaled_error) - safe_scaled_error
        tail_slope = torch.expm1(max_exp_input)
        linear_tail = torch.relu(scaled_error - max_exp_input) * tail_slope
        return (linex_core + linear_tail).mean()

class AsymmetricHuberLoss(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("AsymmetricHuberLoss requires 0 < loss_alpha < 1")
        self.alpha = float(alpha)
        self.delta = 1.0

    def forward(self, pred, target):
        if pred.device != target.device:
            target = target.to(pred.device)

        error = target - pred
        abs_error = error.abs()
        quadratic = torch.minimum(abs_error, abs_error.new_tensor(self.delta))
        linear = abs_error - quadratic
        huber = 0.5 * quadratic.square() + self.delta * linear

        weight = torch.where(error >= 0, self.alpha, 1.0 - self.alpha)
        return (2.0 * weight * huber).mean()

class ExpectileLoss(nn.Module):

    def __init__(self, alpha):
        super().__init__()
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("ExpectileLoss requires 0 < loss_alpha < 1")
        self.alpha = float(alpha)

    def forward(self, pred, target):
        if pred.device != target.device:
            target = target.to(pred.device)

        error = target - pred
        weight = torch.where(error >= 0, self.alpha, 1.0 - self.alpha)
        return (weight * error.square()).mean()

class AsymmetricPeakAwareLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        if not math.isfinite(alpha) or alpha < 1:
            raise ValueError("AsymmetricPeakAwareLoss requires loss_alpha >= 1")
        if not math.isfinite(beta) or beta < 1:
            raise ValueError("AsymmetricPeakAwareLoss requires loss_beta >= 1")
        if not math.isfinite(gamma) or not 0 < gamma < 1:
            raise ValueError("AsymmetricPeakAwareLoss requires 0 < loss_gamma < 1")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def forward(self, pred, target):
        if pred.device != target.device:
            target = target.to(pred.device)

        error = pred - target
        asym_weight = torch.where(error < 0, self.alpha, 1.0)

        target_max = target.amax(dim=1, keepdim=True)
        peak_mask = target >= self.gamma * target_max
        peak_mask = peak_mask | (target == target_max)
        peak_weight = torch.where(peak_mask, self.beta, 1.0)

        return (error.abs() * asym_weight * peak_weight).mean()

class PSLoss(nn.Module):
    def __init__(self):
        """
        """
        super(PSLoss, self).__init__()
        # self.model = model 
        self.mse = nn.MSELoss(reduction='mean')
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.patch_len_threshold = 24  
        # self.ps_lambda = 0.3

    def create_patches(self, x, patch_len, stride):
        if stride <= 0:
            stride = 1  # 兜底
        x = x.permute(0, 2, 1) # [B, C, L] -> [B, L, C]
        B, C, L = x.shape
        
        num_patches = (L - patch_len) // stride + 1
        patches = x.unfold(2, patch_len, stride)
        patches = patches.reshape(B, C, num_patches, patch_len)
        
        return patches

    def fouriour_based_adaptive_patching(self, true, pred):


        # Get patch length an stride
        true_fft = torch.fft.rfft(true, dim=1)
        frequency_list = torch.abs(true_fft).mean(0).mean(-1)
        frequency_list[:1] = 0.0
        top_index = torch.argmax(frequency_list)
        period = (true.shape[1] // top_index)
        # print(period,true.shape[1] ,top_index)
        patch_len = min(period // 2, self.patch_len_threshold)
        stride = patch_len // 2
        
        # Patching
        true_patch = self.create_patches(true, patch_len, stride=stride)
        pred_patch = self.create_patches(pred, patch_len, stride=stride)

        return true_patch, pred_patch
    
    def patch_wise_structural_loss(self, true_patch, pred_patch):
        eps = 1e-8

        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)

        true_patch_var = torch.clamp(torch.var(true_patch, dim=-1, keepdim=True, unbiased=False), min=eps)
        pred_patch_var = torch.clamp(torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False), min=eps)

        true_patch_std = torch.sqrt(true_patch_var + eps)
        pred_patch_std = torch.sqrt(pred_patch_var + eps)

        true_pred_patch_cov = torch.mean((true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), dim=-1, keepdim=True)

        patch_linear_corr = (true_pred_patch_cov + eps) / (true_patch_std * pred_patch_std + eps)
        patch_linear_corr = torch.clamp(patch_linear_corr, -1.0 + eps, 1.0 - eps)
        linear_corr_loss = (1.0 - patch_linear_corr).mean()

        true_patch_softmax = torch.softmax(true_patch / (torch.max(true_patch) - torch.min(true_patch) + eps), dim=-1)
        pred_patch_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_softmax, true_patch_softmax).sum(dim=-1).mean()
 
        mean_loss = torch.abs(true_patch_mean - pred_patch_mean).mean()
        return linear_corr_loss, var_loss, mean_loss

    
    def ps_loss(self, true, pred,model):

        # Fourior based adaptive patching
        true_patch, pred_patch = self.fouriour_based_adaptive_patching(true, pred)
        
        # Pacth-wise structural loss
        corr_loss, var_loss, mean_loss = self.patch_wise_structural_loss(true_patch, pred_patch)

        # Gradient based dynamic weighting
        alpha, beta, gamma = self.gradient_based_dynamic_weighting(true, pred, corr_loss, var_loss, mean_loss,model)

        # Final PS loss
        ps_loss = alpha * corr_loss + beta * var_loss + gamma * mean_loss
        
        return ps_loss
    def gradient_based_dynamic_weighting(self, true, pred, corr_loss, var_loss, mean_loss,model):
       
        eps = 1e-8

        params = list(model.output_proj.parameters())

        def get_full_grad(loss):
            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            return torch.cat([g.detach().flatten() if g is not None else torch.zeros(1, device=g.device) for g in grads])
        corr_grad = get_full_grad(corr_loss)
        var_grad = get_full_grad(var_loss)
        mean_grad = get_full_grad(mean_loss)

        def safe_norm(g):
            return g.norm() + eps
        grad_avg = (corr_grad + var_grad + mean_grad) / 3.0
        alpha = safe_norm(grad_avg) / safe_norm(corr_grad)
        beta = safe_norm(grad_avg) / safe_norm(var_grad)
        gamma = safe_norm(grad_avg) / safe_norm(mean_grad)

        return alpha, beta, gamma
    

    def forward(self, pred, target,model):
        loss = self.mse(pred, target)
                         
        ps_loss = self.ps_loss(pred, target,model)
        loss += ps_loss * 0.3

        return loss
    
class RankMSELoss(nn.Module):
    def __init__(self, rank_weight=3.0, mse_weight=1.0):
        super(RankMSELoss, self).__init__()
        self.rank_weight = rank_weight
        self.mse_weight = mse_weight

    def forward(self, pred, target):
        mse_loss = (pred - target).pow(2).mean()

        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)
    
        rank_loss = F.relu(- (pred_diff * target_diff))
        
        rank_loss = rank_loss.sum(dim=[1])
        rank_loss = rank_loss.mean()

        combined_loss = self.rank_weight * rank_loss + self.mse_weight * mse_loss
        
        return combined_loss
    
class RIloss(nn.Module):
    def __init__(self):
        super(RIloss, self).__init__()

    def forward(self, pred, target):
        x = (target - pred).permute(0, 2, 1).contiguous() 
        x = x.reshape(-1, x.size(-1))
        # print(x.shape)

        pred_1 = torch.rand_like(x)

        corr_loss = torch.tensor(0.).cuda()
        corr_loss = HSIC(x, pred_1)

        loss = 10 * torch.exp(-1 * corr_loss) + torch.mean((target - pred)**2)
        return loss

class SoftDTWLoss(nn.Module):
    
    def __init__(self, gamma=1.0, normalize=True, reduction='mean') -> None:
        super().__init__()
        
        self.soft_dtw = SoftDTWLossPyTorch(
            gamma=gamma,
            normalize=normalize,
            dist_func=None  
        )
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        
        loss_per_sample = self.soft_dtw(pred, target)
        
        if self.reduction == 'mean':
            return loss_per_sample.mean()
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:  # 'none'
            return loss_per_sample

class TimeSQLLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.05, gamma=0.05, c=0.08, reduction='mean') -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c = c
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        squared_error = error ** 2

        smooth_l2_term = self.alpha * squared_error / (squared_error + self.c)
        
        l1_term = (1 - self.alpha) * torch.abs(error)
        
        pred_l1_reg = self.beta * torch.abs(pred)
        
        pred_l2_reg = self.gamma * (pred ** 2)

        total_loss = smooth_l2_term + l1_term + pred_l1_reg + pred_l2_reg


        if self.reduction == 'mean':

            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:  
            return total_loss

# ========================================================
LOSS_CONFIG = {

    'mse': (nn.MSELoss, []),
    'mae': (nn.L1Loss, []),
    'huber': (nn.HuberLoss, ['loss_alpha']),

    # 'card': (CARDLoss, []), 
    'ema_tqa_mae': (EMA_TQA_MAELoss, ['loss_alpha', 'loss_beta', 'loss_gamma']),
    'tqa_mae': (TQA_MAELoss, ['loss_alpha', 'loss_beta', 'loss_gamma']),
    'taq': (PALL_EMA_TQA_MAELoss, ['loss_alpha', 'loss_beta', 'loss_gamma']),
    'softdtw': (SoftDTWLoss,[]),
    'timesql': (TimeSQLLoss, []),
    'dbloss': (DBLoss,[]),
    'cardloss': (CARDLoss, []),
    'xpatchloss': (xPatchLoss, []),
    'fredfloss': (FreDFLoss, []),
    'slinex': (StableLinexLoss, ['loss_alpha']),
    'linex_legacy': (LinexLoss, ['loss_alpha']),
    'quantile': (QuantileLoss, ['loss_alpha']),
    'ahuber': (AsymmetricHuberLoss, ['loss_alpha']),
    'expectile': (ExpectileLoss, ['loss_alpha']),
    'apal': (AsymmetricPeakAwareLoss, ['loss_alpha', 'loss_beta', 'loss_gamma']),
    'psloss': (PSLoss, []),
    'riloss': (RIloss,[]),
    'rankmseloss': (RankMSELoss, []),

}

# ===============================================
def get_loss_function(args,model=None):


    loss_type = args.loss.lower()

    if loss_type not in LOSS_CONFIG:
        raise ValueError(
            f"Unknown loss type: {args.loss}\n"
            f"Supported loss types: {list(LOSS_CONFIG.keys())}"
        )


    loss_cls, required_params = LOSS_CONFIG[loss_type]
    loss_kwargs = {}
    for param in required_params:
        if not hasattr(args, param):
            raise ValueError(f"{param}no find")

        key = param.replace("loss_", "")
        loss_kwargs[key] = getattr(args, param)

    try:
        print(f"loss: {loss_type}, Args:{loss_kwargs}")
        return loss_cls(**loss_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize loss function '{loss_type}' with kwargs {loss_kwargs}. Error: {str(e)}")
    
# =================================================
def Residual_vis(pred, true, loss_type,  batch_idx, save_dir_base):
    """
    Residual_vis(
                        pred=pd, 
                        true=gt, 
                        loss_type=self.args.loss.lower(), 
                        batch_idx=i,
                        save_dir_base=folder_path
                    )
    """
    plt.figure(figsize=(6,6))

        
    T = len(true)
    x = np.arange(T)

    plt.plot(x, true, color="#1f77b4", linewidth=1, label='True')

    plt.plot(x, pred, color="#ff7f0e", linewidth=1, label='Pred')
    x = range(len(true))

    plt.fill_between(
        x,
        true,
        pred,
        where=pred >= true, 
        color="#2ca02c",                
        label='Overestimation'
    )


    plt.fill_between(
        x,
        true,
        pred,
        where=pred < true,
        color="#d62728",             
        label='Underestimation'
    )


    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir_base, exist_ok=True)
    save_path = os.path.join(save_dir_base, f'batch_{batch_idx}.pdf')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
