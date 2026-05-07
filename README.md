# TAQ
 
## Getting Started

### Installation
1. Create a new Conda environment.
   ```bash
   conda create -n taq python=3.11
   conda activate taq
   ```

2. Install Core Dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Train and Evaluate
We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
bash ./scripts/TQNet.sh
```
