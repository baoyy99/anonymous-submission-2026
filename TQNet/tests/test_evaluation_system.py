import csv
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from evaluation.artifacts import EvaluationRun
from evaluation.chunked import iter_sample_slices
from evaluation.config import load_evaluation_config
from evaluation.diagnostics import run_gradient_diagnostics
from evaluation.evaluator import (
    UnifiedEvaluator, _window_reconstruction_chunk_stats,
    checkpoint_from_source, read_legacy_reference, resolve_evaluation_source_dir,
)
from evaluation.metrics import (
    ChunkedMetricAccumulator, MetricCollector, external_cost,
    partition_diagnostics, point_metrics, quadrant_metrics, regional_behavior,
    shape_metrics, state_coverage,
)
from evaluation.partition import compute_partition
from evaluation.qc import build_qc_report
from evaluation.training import TrainingRecorder


def _args(root):
    return SimpleNamespace(
        evaluation_config=None, evaluation_root=str(root),
        evaluation_experiment_id="UNIT",
        evaluation_resume=False, evaluation_source_dir=None,
        data_path="synthetic.csv", data="custom", model="Tiny",
        loss="mae", loss_alpha=0.0, loss_beta=0.0, loss_gamma=0.5,
        random_seed=7, seq_len=4, pred_len=4, batch_size=2,
        learning_rate=1e-3, train_epochs=2, patience=1,
    )


def test_non_overwrite():
    with tempfile.TemporaryDirectory() as directory:
        args = _args(directory)
        run = EvaluationRun(args, "synthetic setting/with unsafe chars")
        assert run.run_dir.exists()
        assert run.run_id == "synthetic_setting_with_unsafe_chars"
        refused = False
        try:
            EvaluationRun(args, "synthetic setting/with unsafe chars")
        except FileExistsError:
            refused = True
        assert refused, "existing run directory must be refused by default"


def test_default_retention_is_metrics_only_and_opt_in_is_explicit():
    config = load_evaluation_config()
    test_config = config["test_evaluation"]
    assert not test_config["save_predictions"]
    assert not test_config["save_scaled_predictions"]
    assert not test_config["save_partition_artifacts"]
    assert not test_config["save_compatibility_npz"]

    retain_path = Path(__file__).resolve().parents[1] / "evaluation" / "configs" / "retain_arrays.json"
    retain = load_evaluation_config(retain_path)["test_evaluation"]
    assert retain["save_predictions"]
    assert retain["save_partition_artifacts"]
    assert retain["save_compatibility_npz"]


def test_catchable_failure_cleans_only_temporary_arrays():
    class FailingEvaluator(UnifiedEvaluator):
        def _run_evaluation(self):
            raise RuntimeError("deliberate failure after inference")

    with tempfile.TemporaryDirectory() as directory:
        working = Path(directory) / "working_arrays"
        working.mkdir()
        temporary_paths = [working / "y_pred_scaled.npy", working / "y_true_scaled.npy"]
        for path in temporary_paths:
            path.write_bytes(b"temporary")
        unexpected = working / "resume_note.txt"
        unexpected.write_text("keep", encoding="utf-8")
        evaluator = FailingEvaluator.__new__(FailingEvaluator)
        evaluator._temporary_array_paths = temporary_paths
        try:
            evaluator.run_evaluation()
        except RuntimeError:
            pass
        else:
            raise AssertionError("deliberate evaluator failure must propagate")
        assert all(not path.exists() for path in temporary_paths)
        assert unexpected.read_text(encoding="utf-8") == "keep"


def test_partition_and_empty_quadrant():
    true = np.asarray([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    pred = true - 1.0
    partition = compute_partition(pred, true, 0.3, 0.8, 0, 0)
    counts = [int((partition["quadrant"] == code).sum()) for code in range(4)]
    assert sum(counts) == true.size
    collector = MetricCollector({
        "experiment_id": "UNIT", "run_id": "partition", "dataset": "synthetic",
        "model": "Tiny", "model_family": "Tiny", "loss": "mae", "mode": None,
        "q": 0.5, "seed": 7, "input_length": 4, "output_length": 4,
    })
    quadrant_metrics(collector, pred, true, partition)
    empty_rows = [row for row in collector.rows if row["status"] == "empty_quadrant"]
    assert empty_rows and all(row["value"] is None for row in empty_rows)


def test_failure_is_recorded():
    with tempfile.TemporaryDirectory() as directory:
        run = EvaluationRun(_args(directory), "failure_setting")
        run.set_stage("quality_control")
        run.mark_failure(ValueError("dataset window alignment failed"))
        payload = json.loads((run.run_dir / "failure.json").read_text(encoding="utf-8"))
        assert payload["status"] == "data_error"
        assert payload["failure_stage"] == "quality_control"
        assert (run.run_dir / "failures_run_level.csv").exists()


def test_default_training_record_has_no_intermediate_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        run = EvaluationRun(_args(directory), "training_setting")
        recorder = TrainingRecorder(run)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        assert recorder.maybe_save_diagnostic(model, optimizer, 1, 1) is None
        assert not (run.run_dir / "checkpoints").exists()
        recorder.record_epoch(
            epoch=1, global_step=2, optimizer_step=2, train_objective=1.0,
            validation_mse_or_objective=0.9, test_mse_or_objective=0.8,
            learning_rate=1e-3, epoch_time_seconds=0.1,
            samples_per_second=20.0, peak_allocated_bytes=0,
            peak_reserved_bytes=0, gradient_norm_preclip=None,
            gradient_norm_postclip=None, clip_event_count=0,
            clipped=False, nan_or_inf=False, diagnostic_checkpoint=None,
        )
        assert (run.run_dir / "train_trace.csv").exists()


def test_setting_resolves_checkpoint_and_legacy_result_from_source_dir():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "TQNet" / "TSout_main"
        setting = "mae_a0.0_b0.0_g0.0_synthetic_setting"
        checkpoint = source / "checkpoints" / setting / "checkpoint.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"read-only checkpoint fixture")
        result_sum = source / "result_sum_1245.txt"
        result_sum.write_text(
            f"2026-09-02 00:00  \n{setting}  \nmse:0.250000, mae:0.400000\n",
            encoding="utf-8",
        )

        resolved = resolve_evaluation_source_dir("TSout_main", cwd=root)
        assert resolved == source.resolve()
        assert checkpoint_from_source(resolved, setting) == checkpoint.resolve()
        reference = read_legacy_reference(resolved, setting)
        assert reference["mse"] == 0.25 and reference["mae"] == 0.4
        assert reference["source"] == str(result_sum.resolve())

        (source / "result_sum_conflict.txt").write_text(
            f"{setting}  \nmse:0.260000, mae:0.400000\n", encoding="utf-8"
        )
        refused = False
        try:
            read_legacy_reference(resolved, setting)
        except RuntimeError:
            refused = True
        assert refused, "conflicting legacy references must fail instead of being guessed"


def _metric_key(row):
    return tuple(row.get(name) for name in (
        "metric", "scope", "region", "direction", "channel", "horizon_step",
        "threshold_rule", "cost_ratio", "cost_matrix_id",
    ))


def test_chunked_metrics_match_full_metrics():
    rng = np.random.default_rng(17)
    true = rng.normal(size=(35, 8, 3)).astype(np.float64)
    pred = true + rng.normal(scale=0.4, size=true.shape)
    train_values = rng.normal(size=(101, 3)).astype(np.float64)
    channel_names = ["a", "b", "c"]
    metric_config = {
        "per_channel": True, "per_horizon_step": True,
        "save_per_sample_candidates": False,
        "state_high_quantile": 0.9, "state_low_quantile": 0.1,
    }
    external_config = {
        "enabled": True, "threshold_quantiles": [0.8], "cost_ratios": [2.0],
        "surplus_unit_cost": 1.0, "cost_matrix_id": "unit",
    }
    modules = {
        "point_metrics", "quadrant_metrics", "regional_behavior",
        "shape_metrics", "state_coverage", "partition_diagnostics",
        "external_cost",
    }
    identity = {
        "experiment_id": "UNIT", "run_id": "full", "dataset": "synthetic",
        "model": "Tiny", "model_family": "Tiny", "loss": "mae",
        "mode": None, "q": 0.8, "seed": 17, "input_length": 8,
        "output_length": 8,
    }
    full_collector = MetricCollector(identity)
    full_partition = compute_partition(pred, true, 0.3, 0.8, 0, 0)
    point_metrics(full_collector, pred, true, channel_names, metric_config)
    quadrant_metrics(full_collector, pred, true, full_partition)
    regional_behavior(full_collector, pred, true, full_partition)
    shape_metrics(full_collector, pred, true, channel_names)
    state_coverage(full_collector, pred, true, train_values, channel_names, metric_config)
    partition_diagnostics(full_collector, full_partition)
    external_cost(full_collector, pred, true, train_values, channel_names, external_config)

    quantiles = {
        0.9: np.quantile(train_values, 0.9, axis=0),
        0.1: np.quantile(train_values, 0.1, axis=0),
        0.8: np.quantile(train_values, 0.8, axis=0),
    }
    chunk_collector = MetricCollector({**identity, "run_id": "chunk"})
    accumulator = ChunkedMetricAccumulator(
        chunk_collector, channel_names, 8, 35, modules, metric_config,
        external_config, quantiles, training_count=len(train_values),
        margin_sample_cap=1000000,
    )
    for sample_slice in iter_sample_slices(len(true), 16):
        partition = compute_partition(pred[sample_slice], true[sample_slice], 0.3, 0.8, 0, 0)
        accumulator.update(
            pred[sample_slice], true[sample_slice], partition,
            sample_offset=sample_slice.start,
        )
    accumulator.finalize()

    full_rows = {_metric_key(row): row for row in full_collector.rows}
    chunk_rows = {_metric_key(row): row for row in chunk_collector.rows}
    assert full_rows.keys() == chunk_rows.keys()
    for key in full_rows:
        left, right = full_rows[key], chunk_rows[key]
        assert left["valid_count"] == right["valid_count"], key
        if left["value"] is None:
            assert right["value"] is None, key
        else:
            assert np.isclose(left["value"], right["value"], rtol=1e-6, atol=1e-8), (key, left["value"], right["value"])


def test_solar_regular_index_is_warning_not_error():
    config = load_evaluation_config()
    rows = []
    for index in range(3):
        start = 100 + index
        rows.append({
            "sample_id": f"test:{index:08d}", "target_start_index": start,
            "target_end_index": start + 4, "split_start_index": 90,
            "split_end_index": 120, "forecast_start_timestamp": None,
            "forecast_start_position": start, "forecast_end_position": start + 3,
            "time_axis_type": "regular_index",
        })
    shapes = {name: [3, 4, 2] for name in (
        "aligned_trend", "above_mask", "under_mask", "quadrant", "boundary_margin",
    )}
    report = build_qc_report(
        scaled_shape=(3, 4, 2), original_shape=(3, 4, 2),
        sample_ids=[row["sample_id"] for row in rows], window_rows=rows,
        channel_names=["a", "b"], partition_shapes=shapes,
        quadrant_counts=[6, 6, 6, 6], config=config,
        mode_report={"valid": True}, inverse_roundtrip_max_abs=0.0,
        window_reconstruction_max_abs=0.0,
        window_reconstruction_violation_count=0,
        window_reconstruction_max_allowed_abs=2 * np.finfo(np.float32).eps,
        nonfinite_values=0,
        legacy_reference={"mse": 1.0, "mae": 1.0},
        scaled_mse=1.0, scaled_mae=1.0,
    )
    timestamp = next(item for item in report["checks"] if item["name"] == "timestamps_available")
    assert report["status"] == "pass"
    assert timestamp["status"] == "warning"
    assert timestamp["details"]["fallback"] == "regular_index"


def test_traffic_shape_never_reaches_full_fft():
    samples, horizon, channels, chunk = 3413, 96, 862, 32
    slices = list(iter_sample_slices(samples, chunk))
    maximum = max(value.stop - value.start for value in slices)
    assert maximum == chunk
    assert maximum * horizon * channels == 2648064
    assert maximum * horizon * channels < samples * horizon * channels / 100


def test_window_reconstruction_uses_float32_rounding_bound():
    dataset = SimpleNamespace(
        seq_len=2, pred_len=2,
        data_y=np.asarray([[0.0], [1.0], [16.123456789], [-23.987654321]]),
    )
    rounded = dataset.data_y[2:4].astype(np.float32)[None, :, :]
    maximum, violations, maximum_allowed = _window_reconstruction_chunk_stats(
        dataset, rounded, "M", sample_offset=0,
    )
    assert maximum > 1e-7
    assert maximum <= maximum_allowed
    assert violations == 0
    shifted = rounded.copy()
    shifted[0, 0, 0] += 0.01
    _, violations, _ = _window_reconstruction_chunk_stats(
        dataset, shifted, "M", sample_offset=0,
    )
    assert violations == 1


class TinyExp:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.device = torch.device("cpu")
        self.args = SimpleNamespace(
            loss="mae", loss_alpha=0.0, loss_beta=0.0, loss_gamma=0.5,
        )

    def _evaluation_forward_batch(self, batch, enable_grad=False):
        x, y = batch
        return self.model(x), y


def test_gradient_diagnostic_is_read_only():
    exp = TinyExp()
    before = [value.detach().clone() for value in exp.model.parameters()]
    x = torch.asarray([[[1.0], [2.0], [3.0], [4.0]], [[2.0], [1.0], [0.0], [-1.0]]])
    y = 0.5 * x + 0.2
    loader = [(x, y)]
    with tempfile.TemporaryDirectory() as directory:
        result = run_gradient_diagnostics(
            exp, loader, Path(directory) / "gradient.json",
            {"fixed_batch_index": 0, "layer_group_profile": None},
            {"ema_alpha": 0.3},
        )
        assert result["parameters_unchanged"]
        assert not result["optimizer_step_called"]
        assert all(torch.equal(old, current.detach()) for old, current in zip(before, exp.model.parameters()))


def main():
    test_non_overwrite()
    test_default_retention_is_metrics_only_and_opt_in_is_explicit()
    test_catchable_failure_cleans_only_temporary_arrays()
    test_partition_and_empty_quadrant()
    test_failure_is_recorded()
    test_default_training_record_has_no_intermediate_checkpoint()
    test_setting_resolves_checkpoint_and_legacy_result_from_source_dir()
    test_chunked_metrics_match_full_metrics()
    test_solar_regular_index_is_warning_not_error()
    test_traffic_shape_never_reaches_full_fft()
    test_window_reconstruction_uses_float32_rounding_bound()
    test_gradient_diagnostic_is_read_only()
    print("evaluation_system_synthetic_tests=PASS")


if __name__ == "__main__":
    main()
