import torch
import torch.nn as nn
import logging
from scipy.signal import find_peaks, correlate
import os
import matplotlib.pyplot as plt
import numpy as np
import math

def real_corr(x, y):
    M, T = x.shape
    N = 2 * T - 1
    L = 1 << ((N - 1).bit_length())
    fx = torch.fft.rfft(x, n=L, dim=-1)
    fy = torch.fft.rfft(y, n=L, dim=-1)
    corr = torch.fft.irfft(torch.conj(fx) * fy, n=L, dim=-1)
    return corr[:, :N]
def peak_align_low_lag(x: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape
    device = x.device

    x_cent = x - x.mean(dim=1, keepdim=True)
    t_cent = trend - trend.mean(dim=1, keepdim=True)


    x_conv = x_cent.permute(0, 2, 1).reshape(B * C, T)
    t_conv = t_cent.permute(0, 2, 1).reshape(B * C, T)
    corr = real_corr(x_conv, t_conv)   
    shift = - torch.argmax(corr, dim=1)

    pos = torch.arange(T, device=device)
    idx = pos - shift.view(B * C, 1)
    idx = idx.clamp(0, T - 1)

    trend_aligned = trend.reshape(B * C, T)
    trend_aligned = torch.gather(trend_aligned, dim=1, index=idx)
    trend_aligned = trend_aligned.view(B, T, C)
    return trend_aligned


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

class PALL_EMA(nn.Module):
    def __init__(self, ema_alpha):
        super().__init__()
        self.ma = EMA(ema_alpha)

    def forward(self, x):
        moving_average = self.ma(x)
        trend_aligned = peak_align_low_lag(x, moving_average)
        res = x - trend_aligned
        return res, trend_aligned


class Mean_Decomp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, T, C = x.shape
        global_mean = x.mean(dim=1, keepdim=True)   
        mean_full = global_mean.expand(-1, T, -1)
        res = x - mean_full
        return res, mean_full


class EMA_TQA_MSELoss(nn.Module):
    """
    xiaorong
    """
    def __init__(self, alpha=None, beta=None, gamma=None, q=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ema_alpha = gamma
        self.q = q
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

class TQA_MSELoss(nn.Module):

    def __init__(self, alpha=None, beta=None, gamma=None, q=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ema_alpha = gamma
        self.q = q
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

    def __init__(self, alpha=None, beta=None, gamma=None, q=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ema_alpha = gamma
        self.q = q
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

LOSS_CONFIG = {

    'mse': (nn.MSELoss, []),
    'mae': (nn.L1Loss, []),
    'ema_tqa_mse': (EMA_TQA_MSELoss, ['TQALoss_alpha', 'TQALoss_beta', 'TQALoss_gamma', 'TQALoss_q']),
    'tqa_mae': (TQA_MSELoss, ['TQALoss_alpha', 'TQALoss_beta', 'TQALoss_gamma', 'TQALoss_q']),
    'pall_ema_tqa_mae': (PALL_EMA_TQA_MAELoss, ['TQALoss_alpha', 'TQALoss_beta', 'TQALoss_gamma', 'TQALoss_q']),
}


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
            raise ValueError(f" {param}no find")

        key = param.replace("TQALoss_", "")
        loss_kwargs[key] = getattr(args, param)
    
    try:
        return loss_cls(**loss_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize loss function '{loss_type}' with kwargs {loss_kwargs}. Error: {str(e)}")
    
def trend_mean_vis(pred, true, loss_type, gamma,batch_idx, save_dir_base):

    T = len(true)
    true_tensor = torch.tensor(true, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    decomp = PALL_EMA(ema_alpha=gamma)

    _, true_trend_tensor = decomp(true_tensor[:, 96:, :])
    true_trend = true_trend_tensor.squeeze(0).squeeze(-1).detach().cpu().numpy()

    mean_decomp = Mean_Decomp()
    _, true_mean_tensor = mean_decomp(true_tensor[:, 96:, :])
    true_mean = true_mean_tensor.squeeze(0).squeeze(-1).detach().cpu().numpy()

    line4_vis(pred, true, true_trend, true_mean, loss_type,  batch_idx, save_dir_base)


def line4_vis(pred, true, true_trend, true_mean, loss_type,  batch_idx, save_dir_base):
        plt.figure(figsize=(10,10))


        
        T = len(true)-96
        x = np.arange(T)

        plt.figure(figsize=(4,4))

        plt.plot(x, true_mean, color='gray', linestyle=':', linewidth=2, label='Mean')
        plt.plot(x, true_trend, color='#2ca02c', linestyle='--', linewidth=2, label='Trend')
        plt.plot(x, true[96:], color="#1f77b4", linewidth=2, label='True')
        if len(pred) >= 96:
            plt.plot(x, pred[96:], color="#ff7f0e", linewidth=2, label='Pred')

        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        os.makedirs(save_dir_base, exist_ok=True)
        save_path = os.path.join(save_dir_base, f'batch_{batch_idx}.pdf')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()