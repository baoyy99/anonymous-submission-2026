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
Create a separate folder named ```./dataset2026``` and place all the CSV files in this directory. 
**Note**: Place the CSV files directly into this directory, such as "./dataset2026/ETTh1.csv"

### Train and Evaluate
We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
bash ./scripts/TQNet.sh
```


## Project Overview
This repository contains the code implementation for the paper "TAQ". To adhere to double-blind review principles and protect intellectual property, we adopt a **phased open-source strategy**:
- **Review Phase**: Provide all non-core code to ensure reviewers can understand the project structure of our method.
- **Acceptance Phase**: Fully open-source all code, data, and models to guarantee experimental reproducibility.

## Repository Structure
```
├── dataset2026/        # data
├── 10-TQNet-master/    # TQNet Adaptation
├── 12-tslib-asymmetry/ # TSLib Adaptation
├── scripts/            # Experimental running scripts 
```

## Contents to Be Added (After Acceptance)
1. TAQ implementation in CrossLinear and TimeFilter.
2. Detailed reproduction guide and parameter specifications.

## Support During Review
- Provide detailed responses to any code-related questions from reviewers.
- Provide code snippets for specific modules upon reviewer request (without disclosing the overall core logic).
- Provide additional experimental results as needed.

## Reproducibility Statement
The complete code will be uploaded to a public GitHub repository within 7 working days upon acceptance of the paper, and the final version of the paper will be updated with the corresponding Git link. All experimental results reported in this paper can be fully reproduced using the complete code provided after acceptance.
