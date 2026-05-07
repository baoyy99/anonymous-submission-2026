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
### Data Preparation
To accommodate anonymity requirements, please refer to the commonly used benchmark library [TSlib](https://github.com/thuml/Time-Series-Library) for the dataset acquisition path. 
Create a separate folder named ```./dataset``` and place all the CSV files in this directory. 
**Note**: Place the CSV files directly into this directory, such as "./dataset/ETTh1.csv"

### Train and Evaluate
We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
bash ./scripts/TQNet.sh
```
