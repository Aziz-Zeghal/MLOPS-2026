# Data Drift Detection with Eurybia and CIFAR-10

This folder contains resources and code for detecting data drift in tabular and image datasets using Eurybia and Alibi Detect.


## Contents

- Jupyter notebook for data drift detection
- Conda environment file
- Example datasets and reports
- Output folder for generated reports

## How to Use

1. **Set up the environment**
   - Create the conda environment:
     ```bash
     conda env create -f conda_env.yaml
     conda activate mlops-drift-env
     ```
2. **Open the notebook**
   - Launch JupyterLab or Jupyter Notebook:
     ```bash
     jupyter lab
     # or
     jupyter notebook
     ```
   - Open `2025_Drift_model_and_retraining.ipynb` and follow the instructions in the notebook.

## Main Libraries Used
- [Eurybia](https://eurybia.readthedocs.io/en/latest/overview.html): For tabular data drift detection.
- [Alibi Detect](https://github.com/SeldonIO/alibi-detect): For image data drift detection.
- TensorFlow, scikit-learn, pandas, matplotlib, seaborn.
