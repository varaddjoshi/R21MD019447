# R21MD019447

This project contains machine learning training scripts for label-based classification with support for both individual label training and K-fold cross-validation with multiple seeds.

## Important Note

**Dataset Privacy**: Due to the sensitive nature of the dataset and privacy concerns, we cannot share the dataset used in this project. To run these scripts, you will need to have your own dataset and modify the dataset paths accordingly in the training scripts.

## Project Structure

### Files in Root Directory

- **`requirements.txt`**: Contains the list of all required Python libraries needed to run the project
- **`setup.sh`**: Bash script to create the environment and install all required libraries
- **`LICENSE`**: Project license file

### Training_for_each_label/ Folder

This folder contains scripts for training models on individual labels:

- **`run.sh`**: Bash script that runs `training.py` for each label. The labels are defined within the script and can be modified according to your dataset's label names
- **`training.py`**: Training code for training models on each individual label of the dataset. You can modify hyperparameters in this file according to your needs and requirements
- 
### KFoldCV+Seeds/ Folder

This folder contains scripts for K-fold cross-validation training on the most important labels:

- **`run.sh`**: Bash script that runs `training_k_cv_seeds.py` for the 13 most important labels. The labels are defined within the script and can be modified according to your dataset's label names
- **`training_k_cv_seeds.py`**: Training code for K-fold cross-validation training on the 13 most important labels of the dataset. You can modify hyperparameters in this file according to your needs and requirements

## Model Configurations
### Training_for_each_label

| Setting | Value / Location |
|--------|------------------|
| **Model architecture** | `BertForSequenceClassification` (`bert-base-uncased`) |
| **Tokenizer** | `BertTokenizer.from_pretrained('bert-base-uncased')` |
| **num_labels** | `1` (single logit → `BCEWithLogitsLoss`) |
| **Max token length (per chunk)** | ` 512` |
| **Sliding window stride** | ` 256` |
| **Epochs** | `15` |
| **Learning rate** | `1e-5` |
| **Optimizer** | `AdamW` |
| **Loss function** | `BCEWithLogitsLoss` with `pos_weight` computed from class balance |
| **Effective batch size** | `1` example per optimizer step (chunked texts averaged into one logit) |
| **Train/val/test split** | `80% / 10% / 10%` (stratified, `random_state=42`) |
| **Metric for saving best model** | Validation **F1**|

### KFoldCV+Seeds

| Setting | Value / Location |
|--------|------------------|
| **Model architecture** | `BertForSequenceClassification` (`bert-base-uncased`) |
| **Tokenizer** | `BertTokenizer.from_pretrained('bert-base-uncased')` |
| **num_labels** | `1` (single logit → `BCEWithLogitsLoss`) |
| **Max token length (per chunk)** | `512` |
| **Sliding window stride** | `256` |
| **Epochs** | `15` |
| **Learning rate** | `1e-5` |
| **Optimizer** | `AdamW(model.parameters(), lr=LR)` |
| **Loss function** | `BCEWithLogitsLoss` with `pos_weight` computed per fold using `compute_class_weight` |
| **Effective batch size** | `1` example per optimization step |
| **Hold-out test split** | `10%` of full dataset (`train_test_split`, stratified, `random_state=42`) |
| **Cross-validation** | `5-fold StratifiedKFold` (`N_SPLITS = 5`) |
| **Multiple seeds** | `seeds = [7, 46, 99, 123, 2024]` — full CV repeated for each seed |
| **Splitting strategy** | 1) Hold out test set once. 2) Run 5-fold CV on remaining data for each seed. |
| **Metric for saving best model (per fold)** | Validation **F1 score** |
| **Global best model selection** | Script tracks best F1 across **all seeds and folds**; stores in `best_model_info` |

### Additional Notes

- **Chunking**: Long text is tokenized and split into overlapping 512-token chunks with stride 256. Model logits for each chunk are averaged to produce a final prediction.
- **Class balancing**: `pos_weight` is dynamically computed for each fold from class distribution.
- **Reproducibility**: For each seed iteration, both NumPy and PyTorch seeds are set. StratifiedKFold also uses that seed.
- **Performance consideration**: Training runs one example at a time (no DataLoader batching), which is slower but ensures custom chunk handling.

## How to Run

### 1. Environment Setup

First, you must run the setup script to create the environment and install all required libraries:

```bash
./setup.sh
```

### 2. Training for All Labels

If you want to run training for all labels:

1. Navigate to the `Training_for_each_label` folder:
   ```bash
   cd Training_for_each_label
   ```

2. Modify the `PATH` variable in the `training.py` file to point to your dataset location

3. Run the bash script:
   ```bash
   ./run.sh
   ```

### 3. K-Fold Cross-Validation Training

If you want to run K-fold cross-validation training on the 13 most important labels:

1. Navigate to the `KFoldCV+Seeds` folder:
   ```bash
   cd KFoldCV+Seeds
   ```

2. Modify the `PATH` variable in the `training_k_cv_seeds.py` file to point to your dataset location

3. Run the bash script:
   ```bash
   ./run.sh
   ```

## Customization

- **Dataset Path**: Modify the `PATH` variable in both `training.py` and `training_k_cv_seeds.py` files to point to your dataset location before running
- **Labels**: Modify the label lists in the respective `run.sh` scripts to match your dataset's label names
- **Hyperparameters**: Adjust hyperparameters in `training.py` and `training_k_cv_seeds.py` files according to your requirements
