"""Trend-aware regional metrics for time-series forecasting.

The ground-truth trend is extracted with :class:`PALL_EMA`.  The masks are
created by ``trend_guided_reweight`` so that metric regions use exactly the
same definitions as the corresponding loss implementation.
"""

from typing import Tuple, Union

import numpy as np
import torch

from Loss_Metric.loss_factory import PALL_EMA, trend_guided_reweight


MetricResult = Tuple[
    float, float, float, float, float,
    float, float, float, float, float,
]


class Metrics:
    """Accumulate regional MAE/MSE plus overall MAE/MSE.

    The returned order is::

        S_A_U_MAE, S_A_U_MSE,
        S_A_O_MAE, S_A_O_MSE,
        S_B_U_MAE, S_B_U_MSE,
        S_B_O_MAE, S_B_O_MSE,
        MAE, MSE

    where ``A``/``B`` mean that the ground truth is above/below its PALL-EMA
    trend, and ``U``/``O`` mean underestimation/overestimation.  MAE and MSE
    are computed separately over the elements in each region.

    Empty regions return ``0.0`` instead of NaN.

    Args:
        ema_alpha: PALL-EMA smoothing coefficient.
        q: Quantile passed to ``trend_guided_reweight``.  Its returned weights
            are not used by these unweighted evaluation metrics.
        alpha: Above-trend asymmetry flag passed to
            ``trend_guided_reweight``.
        beta: Below-trend asymmetry flag passed to
            ``trend_guided_reweight``.
    """

    def __init__(
        self,
        ema_alpha: float = 0.3,
        q: float = 0.5,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        if not 0.0 < ema_alpha < 1.0:
            raise ValueError(
                f"ema_alpha must be between 0 and 1, got {ema_alpha}"
            )
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be between 0 and 1, got {q}")

        self.ema_alpha = float(ema_alpha)
        self.q = float(q)
        self.alpha = alpha
        self.beta = beta
        self.decomp = PALL_EMA(self.ema_alpha)
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated sums and counts."""
        self.region_abs_sums = torch.zeros(4, dtype=torch.float64)
        self.region_squared_sums = torch.zeros(4, dtype=torch.float64)
        self.region_counts = torch.zeros(4, dtype=torch.float64)
        self.total_abs_sum = 0.0
        self.total_squared_sum = 0.0
        self.total_count = 0

    @staticmethod
    def _as_float_tensor(
        value: Union[np.ndarray, torch.Tensor],
        *,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        tensor = value.detach() if torch.is_tensor(value) else torch.as_tensor(value)
        if not tensor.is_floating_point():
            tensor = tensor.float()
        if device is not None or dtype is not None:
            tensor = tensor.to(
                device=device if device is not None else tensor.device,
                dtype=dtype if dtype is not None else tensor.dtype,
            )
        return tensor

    @staticmethod
    def _result_from_sums(
        region_abs_sums: torch.Tensor,
        region_squared_sums: torch.Tensor,
        region_counts: torch.Tensor,
        total_abs_sum: float,
        total_squared_sum: float,
        total_count: int,
    ) -> MetricResult:
        safe_counts = region_counts.clamp_min(1.0)
        regional_mae = region_abs_sums / safe_counts
        regional_mse = region_squared_sums / safe_counts
        regional_mae = torch.where(
            region_counts > 0, regional_mae, torch.zeros_like(regional_mae)
        )
        regional_mse = torch.where(
            region_counts > 0, regional_mse, torch.zeros_like(regional_mse)
        )

        if total_count == 0:
            mae = 0.0
            mse = 0.0
        else:
            mae = total_abs_sum / total_count
            mse = total_squared_sum / total_count

        regional_pairs = tuple(
            value
            for mae_value, mse_value in zip(
                regional_mae.tolist(), regional_mse.tolist()
            )
            for value in (mae_value, mse_value)
        )
        return regional_pairs + (float(mae), float(mse))

    def __call__(
        self,
        pred: Union[np.ndarray, torch.Tensor],
        true: Union[np.ndarray, torch.Tensor],
    ) -> MetricResult:
        """Accumulate one batch and return that batch's ten metrics."""
        pred_tensor = self._as_float_tensor(pred)
        true_tensor = self._as_float_tensor(
            true, device=pred_tensor.device, dtype=pred_tensor.dtype
        )

        if pred_tensor.shape != true_tensor.shape:
            raise ValueError(
                f"pred and true must have the same shape, got "
                f"{tuple(pred_tensor.shape)} and {tuple(true_tensor.shape)}"
            )
        if pred_tensor.ndim != 3:
            raise ValueError(
                f"pred and true must have shape [B, T, C], got "
                f"{tuple(pred_tensor.shape)}"
            )

        with torch.no_grad():
            self.decomp = self.decomp.to(pred_tensor.device)
            _, true_trend = self.decomp(true_tensor)
            diff, c1, c2, c3, c4, *_ = trend_guided_reweight(
                pred_tensor,
                true_tensor,
                true_trend,
                self.q,
                self.alpha,
                self.beta,
            )

            # trend_guided_reweight defines:
            # c1=A,O; c2=A,U; c3=B,O; c4=B,U.
            masks = torch.stack((c2, c1, c4, c3), dim=0)
            abs_diff = diff.abs()
            squared_diff = diff.square()

            region_abs_sums = (masks * abs_diff.unsqueeze(0)).sum(dim=(1, 2, 3))
            region_squared_sums = (
                masks * squared_diff.unsqueeze(0)
            ).sum(dim=(1, 2, 3))
            region_counts = masks.sum(dim=(1, 2, 3))

            batch_total_abs = abs_diff.sum().item()
            batch_total_squared = squared_diff.sum().item()
            batch_total_count = diff.numel()

        batch_result = self._result_from_sums(
            region_abs_sums.double().cpu(),
            region_squared_sums.double().cpu(),
            region_counts.double().cpu(),
            batch_total_abs,
            batch_total_squared,
            batch_total_count,
        )

        self.region_abs_sums += region_abs_sums.double().cpu()
        self.region_squared_sums += region_squared_sums.double().cpu()
        self.region_counts += region_counts.double().cpu()
        self.total_abs_sum += batch_total_abs
        self.total_squared_sum += batch_total_squared
        self.total_count += batch_total_count

        return batch_result

    def finalize(self) -> MetricResult:
        """Return dataset-level metrics accumulated across all batches."""
        return self._result_from_sums(
            self.region_abs_sums,
            self.region_squared_sums,
            self.region_counts,
            self.total_abs_sum,
            self.total_squared_sum,
            self.total_count,
        )
