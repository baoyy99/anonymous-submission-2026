# Code Repository (For Double-Blind Review Only)
 
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

### Train and Evaluate
We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
bash scripts/TQNet/TQNet_B.sh
```


## Project Overview
This repository provides an anonymized implementation of TAQ for peer review, including the core loss function and the training and evaluation code needed to reproduce the main experimental results. A public release of the code and experiment configurations is planned upon acceptance. Public datasets and pretrained models will be referenced through their original sources, subject to their respective licenses.

## Repository Structure
```
├── dataset/            # data
├── TQNet/              # TQNet
├── scripts/            # Experimental running scripts 
```

## Contents to Be Added (After Acceptance)
1. TAQ implementation in OpenLTM and TimeFilter.
2. Detailed reproduction guide and parameter specifications.

## Support During Review
- Provide detailed responses to any code-related questions from reviewers.
- Provide code snippets for specific modules upon reviewer request (without disclosing the overall core logic).
- Provide additional experimental results as needed.

## Reproducibility Statement
The complete code will be uploaded to a public GitHub repository within 7 working days upon acceptance of the paper, and the final version of the paper will be updated with the corresponding Git link. 
