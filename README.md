# ROD (RNAs in Otologic Diseases) - Machine Learning Validation

This repository contains the source code for the machine learning validation pipeline of the ROD (RNAs in Otologic Diseases) database. The code evaluates the structural consistency and machine-readability of the curated human miRNA-otologic disease associations.

## 📌 Project Overview
The pipeline extracts sequence features (e.g., k-mer frequencies, GC content) from pre-miRNAs and one-hot encodes disease terms to build a joint feature representation. It trains and evaluates three traditional machine learning classifiers (Logistic Regression, Random Forest, and XGBoost) to distinguish between positive associations and unlabelled pairs. 

The dataset is temporally split based on publication year to simulate a real-world knowledge discovery scenario.

## 🛠 Dependencies & Environment
The scripts have been tested on a standard workstation. For optimal performance, we recommend:
- CPU: Multi-core processor
- RAM: 32 GB 
- OS: Windows 11

All analyses were performed in a Python environment on a Windows workstation. Please ensure the following packages are installed:
* Python 3.12.4
* pandas == 2.3.3 
* scikit-learn == 1.8.0 
* xgboost == 3.2.0 
* matplotlib == 3.10.8
* numpy == 2.3.5

You can install the dependencies using:
`pip install pandas==2.3.3 scikit-learn==1.8.0 xgboost==3.2.0 matplotlib numpy`

## 📂 File Description
* `data_split.py`: Splits the initial dataset into training (≤ 2024) and independent test (≥ 2025) sets. It performs simple random sampling (1:1 ratio) to construct the negative sample pool, with a fixed random seed (`random_state=42`) for reproducibility.
* `feature_extraction.py`: Processes the raw sequences and disease IDs. It extracts 64 3-mer frequency combinations, basic sequence attributes, and concatenates them with one-hot encoded MeSH/DO disease features.
* `model_training.py`: The core training script. It runs 5-fold cross-validation on the training set and evaluates generalization on the independent test set. It automatically selects the best model and generates performance visualization plots (ROC, PR curves, Feature Importance, and PCA).

## 🚀 Usage Instructions
To reproduce the findings reported in the paper, please run the scripts in the following order:

**Step 1: Data Splitting**
Place the raw data file (`dataset.xlsx`) in the root directory and run:
`python data_split.py`
*(This will generate P_train.tsv, P_test.tsv, N_train.tsv, and N_test.tsv)*

**Step 2: Feature Extraction**
`python feature_extraction.py`
*(This will process the splits and generate the final machine-readable `train_dataset.tsv` and `test_dataset.tsv`)*

**Step 3: Model Training and Evaluation**
`python model_training.py`
*(This will output the performance metrics summary and generate all TIFF figures in the `figure3_outputs_repro` folder)*

## 🌐 Data Availability
The complete ROD database and its web platform can be freely accessed at:
https://www.n-bimlab.com/ROD/ or http://47.83.134.12/ROD/.

## 📝 License
This code is provided for research and educational purposes to support computational reproducibility.