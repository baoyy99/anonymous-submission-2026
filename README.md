# TAQ: Code for Double-Blind Peer Review

This repository accompanies the paper **TAQ: Time-Series Forecasting with Four Controllable Modes via Trend-Based Asymmetric Quantile Loss**. TAQ combines a peak-aligned exponential moving average with region-dependent asymmetric error weights to control forecasting behavior across above- and below-trend regions.
[View the TAQ framework diagram (PDF).](TAQ_framework.pdf)

The review package includes the core TAQ implementation, forecasting backbones, data loaders, training code, regional metrics, and experiment scripts. Current reproduction limitations are documented below. Author identities are withheld for double-blind review.

## Repository Structure

```text
.
├── LICENSE-REVIEW.txt       # Review terms for covered original material
├── requirements.txt
├── TQNet/
│   ├── run.py              # Training and testing entry point
│   ├── Loss_Metric/        # TAQ, comparison losses, and regional metrics
│   ├── models/             # Forecasting backbones
│   ├── layers/             # Shared model components
│   ├── data_provider/      # Dataset loading and preprocessing
│   ├── exp/                # Training and testing procedures
│   ├── evaluation/configs/ # Unified evaluation configurations
│   ├── tests/              # Evaluation test definitions
│   ├── requirements.txt
│   └── LICENSE             # Retained Apache-2.0 license text
└── scripts/
    ├── TQNet/              # Backbone and loss experiment scripts
    ├── OpenLTM/            # Scripts requiring the separate OpenLTM integration
    └── TimeFilter/          # Scripts requiring the separate TimeFilter integration
```

## Environment Setup

Run commands from the repository root containing `TQNet/` and `scripts/`.

```bash
conda create -n taq python=3.11
conda activate taq
python -m pip install -r requirements.txt
```


## Data Preparation

Datasets are not bundled. Acquisition instructions for common forecasting benchmarks are available through the [Time-Series Library](https://github.com/thuml/Time-Series-Library). Obtain datasets from their original sources and follow their respective access and license terms.

The scripts expect these files under `dataset/` at the repository root:

| Subdirectory | Files |
| --- | --- |
| `ETT-small/` | `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv` |
| `traffic/` | `traffic.csv` |
| `weather/` | `weather.csv` |
| `Solar/` | `solar_AL.txt` |
| `electricity/` | `electricity.csv` |
| `exchange_rate/` | `exchange_rate.csv` |

Preserve the original data format and column order. The Solar loader expects its text format rather than a CSV conversion. Custom locations can be supplied through `--root_path` and `--data_path`.

## Core TAQ Implementation

The implementation is in [`TQNet/Loss_Metric/loss_factory.py`](TQNet/Loss_Metric/loss_factory.py):

- `EMA` and `PALL_EMA`: trend extraction and peak alignment.
- `trend_guided_reweight`: quadrant masks and asymmetric weights.
- `PALL_EMA_TQA_MAELoss`: the TAQ objective, selected with `--loss taq`.
- `get_loss_function`: loss construction from command-line arguments.

`--loss_alpha` and `--loss_beta` select the above- and below-trend weighting directions using values `0` or `1`. `--loss_gamma` is the quantile parameter `q` in `[0.5, 1]`. 



## Review Access and Planned Public Release

The covered original material is provided for peer review under [`LICENSE-REVIEW.txt`](LICENSE-REVIEW.txt). Authorized review participants may inspect, execute, and locally modify it to evaluate the submission and verify experimental claims. Uses outside that permission require separate authorization unless another applicable license already permits them.

Third-party materials and previously granted rights remain governed by their respective licenses. The Apache-2.0 text retained at [`TQNet/LICENSE`](TQNet/LICENSE) is not replaced by the review license.

A public release of the code and experiment configurations, including the OpenLTM and TimeFilter integrations and a detailed reproduction guide, is planned upon acceptance. Public datasets and pretrained models will be referenced through their original sources, subject to their respective licenses. Any future release will carry its own applicable license.

Code-related questions during review should be raised through the venue's review system to preserve anonymity.
