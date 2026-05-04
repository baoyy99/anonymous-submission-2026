import torch
import torch.nn as nn
import logging
from scipy.signal import find_peaks, correlate
import os
import matplotlib.pyplot as plt
import numpy as np
import math
#所有的损失函数都是以下的输入输出：
# input-->(forecast [batch, pred_len, feature], ground truth[batch, pred_len, feature]) 
# output-->scalar

# ===================== 一、通用零滞后对齐工具 =====================
# 不是整个batch进行对齐，而是对batch中的每个T序列逐个进行对齐，不会被其他变量的值引导
def real_corr(x, y):
    M, T = x.shape
    N = 2 * T - 1
    # smallest power-of-two >= N
    L = 1 << ((N - 1).bit_length())
    fx = torch.fft.rfft(x, n=L, dim=-1)
    fy = torch.fft.rfft(y, n=L, dim=-1)
    corr = torch.fft.irfft(torch.conj(fx) * fy, n=L, dim=-1)
    return corr[:, :N]
def peak_align_low_lag(x: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
    B, T, C = x.shape
    device = x.device

    # # 中心化 [B, T, C]
    x_cent = x - x.mean(dim=1, keepdim=True)
    t_cent = trend - trend.mean(dim=1, keepdim=True)


    x_conv = x_cent.permute(0, 2, 1).reshape(B * C, T)
    t_conv = t_cent.permute(0, 2, 1).reshape(B * C, T)
    corr = real_corr(x_conv, t_conv)   # (M, 2T-1)
    # shift = torch.argmax(corr, dim=1) - (T - 1)
    shift = - torch.argmax(corr, dim=1)

    # x_conv = x_cent.permute(0, 2, 1).reshape(B * C, 1, T)
    # t_conv = t_cent.permute(0, 2, 1).reshape(B * C,1, T)
    # corr = torch.conv1d(x_conv, torch.conj(t_conv), padding=T - 1)
    # # print(corr)
    # corr = corr.view(B*C, -1)
    # shift = torch.argmax(corr, dim=1) - (T - 1)
    # shift = shift.long()
    # print(shift)


    # 向量化 gather 对齐
    pos = torch.arange(T, device=device)
    idx = pos - shift.view(B * C, 1)
    idx = idx.clamp(0, T - 1)

    trend_aligned = trend.reshape(B * C, T)
    trend_aligned = torch.gather(trend_aligned, dim=1, index=idx)
    trend_aligned = trend_aligned.view(B, T, C)
    # print(idx)

    return trend_aligned

# ===================== 二、通用趋势感知掩码与权重工具 =====================
def trend_guided_reweight(pred, true, true_trend, q, alpha, beta):
    """
    所有 Q_ZL_*_TQA_*Loss 共用的条件掩码 + 权重计算
    输入：pred, true, true_trend [B, T, C]
    输出：c1~c4, w1_pos, w1_neg, w2_pos, w2_neg
    """
    # 1. 误差
    diff = pred - true

    # 2. 条件掩码
    mask_above_trend = (true > true_trend).float()
    mask_below_trend = (true <= true_trend).float()
    mask_pos_diff = (diff > 0).float()
    mask_neg_diff = (diff <= 0).float()

    # 3. 四个区域
    c1 = mask_above_trend * mask_pos_diff
    c2 = mask_above_trend * mask_neg_diff
    c3 = mask_below_trend * mask_pos_diff
    c4 = mask_below_trend * mask_neg_diff

    # 4. 分位数权重（alpha/beta 控制不对称）
    if alpha == 1:
        w1_pos, w1_neg = 1.0 - q, q
    else:
        w1_pos, w1_neg = q, 1.0 - q

    if beta == 1:
        w2_pos, w2_neg = 1.0 - q, q
    else:
        w2_pos, w2_neg = q, 1.0 - q

    return diff, c1, c2, c3, c4, w1_pos, w1_neg, w2_pos, w2_neg

# ===================== 三、求趋势 =====================
# 目前实验结果中峰值对齐的最好的就是PALL EMA，用的超参数是0.3
# 1. 移动平均 ======================
class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """
    def __init__(self, ema_alpha):
        super(EMA, self).__init__()
        # 检查 alpha 范围，防止数值不稳定
        if not (0 < ema_alpha < 1):
            raise ValueError(f"EMA alpha must be between 0 and 1, got {ema_alpha}")
        self.ema_alpha = ema_alpha

    def forward(self, x):
        _, t, _ = x.shape
        # 【修复1】动态获取设备，严禁硬编码 cuda
        device = x.device
        
        # 生成时间步幂次 [t, t-1, ..., 1] 
        powers = torch.flip(torch.arange(t, dtype=torch.double, device=device), dims=(0,))
        
        # 计算权重: (1-alpha)^powers
        weights = torch.pow((1 - self.ema_alpha), powers)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.ema_alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)
# 2. 未对齐 Moving Average (EMA) ======================

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
    """
    峰值对齐 EMA 分解
    输入输出长度完全一致
    """
    def __init__(self, ema_alpha):
        super().__init__()
        self.ma = EMA(ema_alpha)

    def forward(self, x):
        moving_average = self.ma(x)
        trend_aligned = peak_align_low_lag(x, moving_average)
        res = x - trend_aligned
        return res, trend_aligned


# 4.均值分解======================
class Mean_Decomp(nn.Module):
    """
    全局均值分解：无滑动平均，直接计算整个序列的全局平均值作为趋势项
    输入输出格式与 moving_avg_zero_lag 完全一致
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, T, C = x.shape
        global_mean = x.mean(dim=1, keepdim=True)  # shape: [B, 1, C]
        mean_full = global_mean.expand(-1, T, -1)
        res = x - mean_full
        return res, mean_full


# ============================= 四、损失函数 ==========================================
# 简单的模拟数据上ema更好
# todo检查是否对每一维求损失
class PALL_EMA_TQA_MSELoss(nn.Module):
    """
    Args:
        alpha: 惩罚系数 1 (for pred > trend)
        beta: 惩罚系数 2 (for pred <= trend)
        gamma: 本方法的ema_alpha (0 < ema_alpha < 1)，用于提取趋势，推荐范围 [0.1,0.3],原文最佳值0.3
        q: 分位数q,取值范围 [0.5,0.9]，推荐[0.5,0.65]
    """
    def __init__(self, alpha=None, beta=None, gamma=None, q=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ema_alpha = gamma
        self.q = q
        self.decomp = PALL_EMA(self.ema_alpha)

    def forward(self, pred, true):
        # 确保输入在同一设备（通常由 DataLoader 保证，但防御性编程是好习惯）
        if pred.device != true.device:
            true = true.to(pred.device)

        # 1. 分解真实值获取趋势
        _, true_trend = self.decomp(true)
        diff, c1, c2, c3, c4, w1_pos, w1_neg, w2_pos, w2_neg = \
            trend_guided_reweight(pred, true, true_trend, self.q, self.alpha, self.beta)
        
        # 2.L2 损失
        squared_diff = diff ** 2

        # 3. 应用权重
        loss = squared_diff * (w1_pos*c1 + w1_neg*c2 + w2_pos*c3 + w2_neg*c4)
        return loss.mean()

class PALL_EMA_TQA_MAELoss(nn.Module):
    """
    Args:
        alpha: 惩罚系数 1 (for pred > trend)
        beta: 惩罚系数 2 (for pred <= trend)
        gamma: 本方法的ema_alpha (0 < ema_alpha < 1)，用于提取趋势，推荐范围 [0.1,0.3],原文最佳值0.3
        q: 分位数q,取值范围 [0.5,0.9]，推荐[0.5,0.65]
    """
    def __init__(self, alpha=None, beta=None, gamma=None, q=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ema_alpha = gamma
        self.q = q
        self.decomp = PALL_EMA(self.ema_alpha)

    def forward(self, pred, true):
        # 确保输入在同一设备（通常由 DataLoader 保证，但防御性编程是好习惯）
        if pred.device != true.device:
            true = true.to(pred.device)

        # 1. 分解真实值获取趋势
        _, true_trend = self.decomp(true)
        diff, c1, c2, c3, c4, w1_pos, w1_neg, w2_pos, w2_neg = \
            trend_guided_reweight(pred, true, true_trend, self.q, self.alpha, self.beta)
        
        # 2.L2 损失
        abs_diff = torch.abs(diff)

        # 3. 应用权重
        loss = abs_diff * (w1_pos*c1 + w1_neg*c2 + w2_pos*c3 + w2_neg*c4)
        
        return loss.mean()

class DBLoss(nn.Module):
    """自定义分解损失函数（趋势+季节双损失）"""

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

class xPatchLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()  
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred, target) -> torch.Tensor:
        prelen = pred.shape[1]

        # 纯 torch，无 numpy，自动对齐设备
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
        # 均值
        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)
        # 方差（强制非负）
        true_patch_var = torch.clamp(torch.var(true_patch, dim=-1, keepdim=True, unbiased=False), min=eps)
        pred_patch_var = torch.clamp(torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False), min=eps)
        # 标准差
        true_patch_std = torch.sqrt(true_patch_var + eps)
        pred_patch_std = torch.sqrt(pred_patch_var + eps)
        # 协方差
        true_pred_patch_cov = torch.mean((true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), dim=-1, keepdim=True)
        # 相关系数（全加 eps）
        patch_linear_corr = (true_pred_patch_cov + eps) / (true_patch_std * pred_patch_std + eps)
        patch_linear_corr = torch.clamp(patch_linear_corr, -1.0 + eps, 1.0 - eps)
        linear_corr_loss = (1.0 - patch_linear_corr).mean()
        # KL 散度（数值稳定）
        true_patch_softmax = torch.softmax(true_patch / (torch.max(true_patch) - torch.min(true_patch) + eps), dim=-1)
        pred_patch_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_softmax, true_patch_softmax).sum(dim=-1).mean()
        # 均值损失
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
        
        # true = true.permute(0, 2, 1)
        # pred = pred.permute(0, 2, 1)
        # true_mean = torch.mean(true, dim=-1, keepdim=True)
        # pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        # true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
        # pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
        # true_std = torch.sqrt(true_var)
        # pred_std = torch.sqrt(pred_var)
        # true_pred_cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
        # linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
        # linear_sim = (1.0 + linear_sim) * 0.5
        # var_sim = (2*true_std*pred_std + 1e-5) / (true_var + pred_var + 1e-5)
   
        # #原方法，但是有的project层会有多个输出，不是只有一个线性层，所以要改成展平
        # # print(model.output_proj.parameters())
        # corr_gradient = torch.autograd.grad(corr_loss, model.output_proj.parameters(), create_graph=True)[0]
        # var_gradient = torch.autograd.grad(var_loss, model.output_proj.parameters(), create_graph=True)[0]
        # mean_gradient = torch.autograd.grad(mean_loss, model.output_proj.parameters(), create_graph=True)[0]

        # gradiant_avg = (corr_gradient + var_gradient + mean_gradient) / 3.0

        # aplha = gradiant_avg.norm().detach() / corr_gradient.norm().detach()
        # beta =  gradiant_avg.norm().detach() /  var_gradient.norm().detach()
        # gamma = gradiant_avg.norm().detach() / mean_gradient.norm().detach()
        # gamma = gamma * torch.mean(linear_sim*var_sim).detach()
        
        # return aplha, beta, gamma
        eps = 1e-8
        # 正确取全部参数梯度
        params = list(model.output_proj.parameters())
        # 梯度拼接（防止维度不匹配）
        def get_full_grad(loss):
            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            return torch.cat([g.detach().flatten() if g is not None else torch.zeros(1, device=g.device) for g in grads])
        corr_grad = get_full_grad(corr_loss)
        var_grad = get_full_grad(var_loss)
        mean_grad = get_full_grad(mean_loss)
        # 范数防除零
        def safe_norm(g):
            return g.norm() + eps
        grad_avg = (corr_grad + var_grad + mean_grad) / 3.0
        alpha = safe_norm(grad_avg) / safe_norm(corr_grad)
        beta = safe_norm(grad_avg) / safe_norm(var_grad)
        gamma = safe_norm(grad_avg) / safe_norm(mean_grad)
        # 后续计算...
        return alpha, beta, gamma
    

    def forward(self, pred, target,model):
        loss = self.mse(pred, target)
                         
        ps_loss = self.ps_loss(pred, target,model)
        # loss += ps_loss * self.ps_lambda——这里简写成0.3
        loss += ps_loss * 0.3

        return loss

class DBLoss_taq(nn.Module):
    """自定义分解损失函数（趋势+季节双损失）"""

    def __init__(self, alpha=0.2, beta=0.5):
        super().__init__()
        self.decomp = EMADECOMP(alpha)
        self.beta = beta
        # self.mse = nn.MSELoss(reduction="mean")
        # self.mae = nn.L1Loss(reduction="mean")
        self.timeloss = PALL_EMA_TQA_MAELoss(0.0,1.0, 0.1,0.6)

    def forward(self, pred, target):
        pred_season, pred_trend = self.decomp(pred)
        target_season, target_trend = self.decomp(target)

        # season_loss = self.mse(pred_season, target_season)
        # trend_loss = self.mae(pred_trend, target_trend)
        season_loss = self.timeloss(pred_season, target_season)
        trend_loss = self.timeloss(pred_trend, target_trend)

        trend_loss = trend_loss * (season_loss / (trend_loss + 1e-8)).detach()
        return self.beta * season_loss + (1 - self.beta) * trend_loss

class xPatchLoss_taq(nn.Module):
    def __init__(self) -> None:
        super().__init__()  
        # self.mse = nn.MSELoss(reduction='mean')
        self.timeloss = PALL_EMA_TQA_MAELoss(0.0,1.0, 0.1,0.6)

    def forward(self, pred, target) -> torch.Tensor:
        prelen = pred.shape[1]

        # 纯 torch，无 numpy，自动对齐设备
        ratio = [-1 * math.atan(i+1) + math.pi/4 + 1 for i in range(prelen)]
        ratio = torch.tensor(ratio, dtype=torch.float32, device=pred.device)
        ratio = ratio.unsqueeze(-1)

        outputs = pred * ratio
        batch_y = target * ratio
        # loss = self.mse(outputs, batch_y)
        loss = self.timeloss(outputs, batch_y)
        return loss
    
class FreDFLoss_taq(nn.Module):
    def __init__(self,rec_lambda=0.1) -> None:
        super().__init__()  
        self.rec_lambda = rec_lambda
        self.mse = nn.MSELoss(reduction='mean')
        self.timeloss = PALL_EMA_TQA_MAELoss(0.0,1.0, 0.1,0.6)

    def forward(self, pred, target):
        # loss_rec = self.mse(pred, target)
        loss_rec = self.timeloss(pred, target)
        loss = self.rec_lambda * loss_rec

        loss_auxi = torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)
        loss_auxi = loss_auxi.abs().mean()
        loss += (1- self.rec_lambda) * loss_auxi
        return loss

class PSLoss_taq(nn.Module):
    """时域MSE损失替换成TAQ"""
    def __init__(self):
        """
        """
        super(PSLoss_taq, self).__init__()
        # alpha=None, beta=None, gamma=None, q=None
        self.timeloss = PALL_EMA_TQA_MAELoss(0.0,1.0, 0.1,0.6)
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
        # 均值
        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)
        # 方差（强制非负）
        true_patch_var = torch.clamp(torch.var(true_patch, dim=-1, keepdim=True, unbiased=False), min=eps)
        pred_patch_var = torch.clamp(torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False), min=eps)
        # 标准差
        true_patch_std = torch.sqrt(true_patch_var + eps)
        pred_patch_std = torch.sqrt(pred_patch_var + eps)
        # 协方差
        true_pred_patch_cov = torch.mean((true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), dim=-1, keepdim=True)
        # 相关系数（全加 eps）
        patch_linear_corr = (true_pred_patch_cov + eps) / (true_patch_std * pred_patch_std + eps)
        patch_linear_corr = torch.clamp(patch_linear_corr, -1.0 + eps, 1.0 - eps)
        linear_corr_loss = (1.0 - patch_linear_corr).mean()
        # KL 散度（数值稳定）
        true_patch_softmax = torch.softmax(true_patch / (torch.max(true_patch) - torch.min(true_patch) + eps), dim=-1)
        pred_patch_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_softmax, true_patch_softmax).sum(dim=-1).mean()
        # 均值损失
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
        
        # true = true.permute(0, 2, 1)
        # pred = pred.permute(0, 2, 1)
        # true_mean = torch.mean(true, dim=-1, keepdim=True)
        # pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        # true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
        # pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
        # true_std = torch.sqrt(true_var)
        # pred_std = torch.sqrt(pred_var)
        # true_pred_cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
        # linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
        # linear_sim = (1.0 + linear_sim) * 0.5
        # var_sim = (2*true_std*pred_std + 1e-5) / (true_var + pred_var + 1e-5)
   
        # #原方法，但是有的project层会有多个输出，不是只有一个线性层，所以要改成展平
        # # print(model.output_proj.parameters())
        # corr_gradient = torch.autograd.grad(corr_loss, model.output_proj.parameters(), create_graph=True)[0]
        # var_gradient = torch.autograd.grad(var_loss, model.output_proj.parameters(), create_graph=True)[0]
        # mean_gradient = torch.autograd.grad(mean_loss, model.output_proj.parameters(), create_graph=True)[0]

        # gradiant_avg = (corr_gradient + var_gradient + mean_gradient) / 3.0

        # aplha = gradiant_avg.norm().detach() / corr_gradient.norm().detach()
        # beta =  gradiant_avg.norm().detach() /  var_gradient.norm().detach()
        # gamma = gradiant_avg.norm().detach() / mean_gradient.norm().detach()
        # gamma = gamma * torch.mean(linear_sim*var_sim).detach()
        
        # return aplha, beta, gamma
        eps = 1e-8
        # 正确取全部参数梯度
        params = list(model.output_proj.parameters())
        # 梯度拼接（防止维度不匹配）
        def get_full_grad(loss):
            grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            return torch.cat([g.detach().flatten() if g is not None else torch.zeros(1, device=g.device) for g in grads])
        corr_grad = get_full_grad(corr_loss)
        var_grad = get_full_grad(var_loss)
        mean_grad = get_full_grad(mean_loss)
        # 范数防除零
        def safe_norm(g):
            return g.norm() + eps
        grad_avg = (corr_grad + var_grad + mean_grad) / 3.0
        alpha = safe_norm(grad_avg) / safe_norm(corr_grad)
        beta = safe_norm(grad_avg) / safe_norm(var_grad)
        gamma = safe_norm(grad_avg) / safe_norm(mean_grad)
        # 后续计算...
        return alpha, beta, gamma
    

    def forward(self, pred, target,model):
        loss = self.timeloss(pred, target)
                         
        ps_loss = self.ps_loss(pred, target,model)
        # loss += ps_loss * self.ps_lambda——这里简写成0.3
        loss += ps_loss * 0.3

        return loss

# ========================= 五、 损失配置表 ===============================
LOSS_CONFIG = {
    # PyTorch 原生损失
    'mse': (nn.MSELoss, []),
    'mae': (nn.L1Loss, []),
    # 'huber': (nn.HuberLoss, ['huber_delta']),
    # 自定义损失
    # 'card': (CARDLoss, []), 
    # # 常规
    # 峰值对齐，PALL_EMA最好
    'pall_ema_tqa_mse': (PALL_EMA_TQA_MSELoss, ['TQALoss_alpha', 'TQALoss_beta', 'TQALoss_gamma', 'TQALoss_q']),
    'pall_ema_tqa_mae': (PALL_EMA_TQA_MAELoss, ['TQALoss_alpha', 'TQALoss_beta', 'TQALoss_gamma', 'TQALoss_q']),
    'dbloss': (DBLoss,[]),
    'xpatchloss': (xPatchLoss, []),
    'fredfloss': (FreDFLoss, []),
    'psloss': (PSLoss, []),
    'dbloss_taq': (DBLoss_taq,[]),
    'xpatchloss_taq': (xPatchLoss_taq, []),
    'fredfloss_taq': (FreDFLoss_taq, []),
    'psloss_taq': (PSLoss_taq, []),
}

# ========================= 六、核心工厂方法：从args自动生成损失函数 ======================
def get_loss_function(args,model=None):
    """
    从args中自动选择并初始化损失函数，同时打印损失参数配置日志
    :param args: 训练的args对象
    :return: 初始化完成的损失函数实例
    """

    # 1. 获取损失类型（统一小写）
    loss_type = args.loss.lower()

    # 2. 检查损失类型是否支持
    if loss_type not in LOSS_CONFIG:
        raise ValueError(
            f"Unknown loss type: {args.loss}\n"
            f"Supported loss types: {list(LOSS_CONFIG.keys())}"
        )

    # 3. 获取损失函数类 + 所需参数列表
    loss_cls, required_params = LOSS_CONFIG[loss_type]
    loss_kwargs = {}
    for param in required_params:
        if not hasattr(args, param):
            raise ValueError(f"损失函数需要参数 {param}，但 args 中未找到！")
        # 直接去掉 TQALoss_ 前缀，对应损失类的 alpha/beta/gamma/q
        key = param.replace("TQALoss_", "")
        loss_kwargs[key] = getattr(args, param)
    
    # 5. 初始化并返回损失函数
    try:
        print(f"初始化损失函数 {loss_type}，参数：{loss_kwargs}")
        return loss_cls(**loss_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize loss function '{loss_type}' with kwargs {loss_kwargs}. Error: {str(e)}")
    
# =========================七、可视化函数========================
def trend_mean_vis(pred, true, loss_type, gamma,batch_idx, save_dir_base):
    """
    可视化 Pred, True, True_Trend, True_Mean
    Args:
        batch_idx: int, 当前 batch 索引
        save_dir_base: str, 保存目录
        gamma: EMA 的 alpha 
    # 1、调用 PALL_EMA 获取 True_Trend
    # 3、调用 Mean_Decomp 获取 True_Mean
    # 2、调用line4_vis画图并保存
    # 输入已经被预处理过，只有一纬，可以直接画图：true = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)、pred = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
    """

    # 转为 tensor [1, T, 1] 适配分解器
    T = len(true)
    true_tensor = torch.tensor(true, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    # 1. 趋势分解器，只传序号96之后的部分
    decomp = PALL_EMA(ema_alpha=gamma)
    # print(gamma)
    # decomp = PALL_EMA(0.1)
    _, true_trend_tensor = decomp(true_tensor[:, 96:, :])#[B, T-96, C]
    true_trend = true_trend_tensor.squeeze(0).squeeze(-1).detach().cpu().numpy()

    # 3. 计算均值，只传序号96之后的部分
    mean_decomp = Mean_Decomp()
    _, true_mean_tensor = mean_decomp(true_tensor[:, 96:, :])#[B,1,C]
    true_mean = true_mean_tensor.squeeze(0).squeeze(-1).detach().cpu().numpy()

    # 4. 调用四线图
    line4_vis(pred, true, true_trend, true_mean, loss_type,  batch_idx, save_dir_base)


# def line4_vis(pred, true, true_trend, true_mean, loss_type,  batch_idx, save_dir_base):
#         """pre+true"""
#         plt.figure(figsize=(10, 6.18))
#         # true画完整的，pred、True_Trend、True_Mean只画序号为96及以后的部分，
#         # 画线的先后顺序1、True_Mean，color='gray', linestyle=':'
#         # 2、True_Trend，绿色，color='#2ca02c', linestyle='--'，
#         # 3、True，蓝色，color="#1f77b4"
#         # 4、Pred，橙色， color="#ff7f0e"

        
#         T = len(true)
#         x = np.arange(T)

#         plt.figure(figsize=(10, 6.18))
#         plt.axvline(x=96,color="black", linewidth=1,alpha=0.8 )

#         # 1. 均值（从96开始，true_mean没有0～95对应的值）
#         plt.plot(x[96:], true_mean, color='gray', linestyle=':', linewidth=2, label='Mean')
#         # 2. 趋势（从96开始，true_trend没有0～95对应的值）
#         plt.plot(x[96:], true_trend, color='#2ca02c', linestyle='--', linewidth=2, label='Trend')
#         # 3. 真实（从96开始）
#         plt.plot(x, true, color="#1f77b4", linewidth=2, label='True')
#         # 4. 预测（从96开始）
#         if len(pred) >= 96:
#             plt.plot(x[96:], pred[96:], color="#ff7f0e", linewidth=2, label='Pred')

#         plt.title(f'Batch {batch_idx}', fontsize=12)
#         plt.xlabel('Time Step')
#         plt.ylabel('Value')
#         plt.legend(loc='best')
#         plt.grid(alpha=0.3)
#         plt.tight_layout()

#         # 保存
#         os.makedirs(save_dir_base, exist_ok=True)
#         save_path = os.path.join(save_dir_base, f'batch_{batch_idx}.png')
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()

def line4_vis(pred, true, true_trend, true_mean, loss_type,  batch_idx, save_dir_base):
        plt.figure(figsize=(10,10))
        # true、pred、True_Trend、True_Mean只画序号为96及以后的部分，
        # 画线的先后顺序1、True_Mean，color='gray', linestyle=':'
        # 2、True_Trend，绿色，color='#2ca02c', linestyle='--'，
        # 3、True，蓝色，color="#1f77b4"
        # 4、Pred，橙色， color="#ff7f0e"

        
        T = len(true)-96
        x = np.arange(T)

        plt.figure(figsize=(4,4))
        # plt.axvline(x=96,color="black", linewidth=1,alpha=0.8 )

        # 1. 均值（从96开始，true_mean没有0～95对应的值）
        plt.plot(x, true_mean, color='gray', linestyle=':', linewidth=2, label='Mean')
        # 2. 趋势（从96开始，true_trend没有0～95对应的值）
        plt.plot(x, true_trend, color='#2ca02c', linestyle='--', linewidth=2, label='Trend')
        # 3. 真实（从96开始）
        plt.plot(x, true[96:], color="#1f77b4", linewidth=2, label='True')
        # 4. 预测（从96开始）
        if len(pred) >= 96:
            plt.plot(x, pred[96:], color="#ff7f0e", linewidth=2, label='Pred')

        # plt.title(f'Batch {batch_idx}', fontsize=12)
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # 保存
        os.makedirs(save_dir_base, exist_ok=True)
        save_path = os.path.join(save_dir_base, f'batch_{batch_idx}.pdf')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()