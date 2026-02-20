# DeeperNull

Library and workflows for integrating spatiotemporal covariates (e.g. time of day, time of year, birth/home location) into polygenic scores. Also includes workflows for training and evaluation of different classes of null models with various covariate features and interpretation of null models to identify important covariate effects and interactions. The following describes how to fit models, compute SHAP values, and the various UK Biobank workflows used. Additional details are provided in the docstrings of the underlying files.

[Preprint](https://www.medrxiv.org/content/10.1101/2025.11.26.25341090)

## Training Models with fit_model.py

The `deeper_null/fit_model.py` script is used to train DeepNull style models on data. It supports various model types:

- Scikit-learn linear and penalized linear models
- XGBoost models  
- PyTorch neural network models

### Requirements

1. A whitespace delimited covariate file with a header row (used as model input)
2. A whitespace delimited phenotype file with sample IDs and phenotype values
3. A model configuration JSON file describing the model to fit
4. An output directory for results

### Key Command Line Arguments

- `--covar_file` / `-c`: Path to covariate file
- `--pheno_file` / `-p`: Path to phenotype file(s)
- `--model_config` / `-m`: Path to model configuration JSON file
- `--out_dir` / `-o`: Path to output directory (default: current directory)
- `--save_models`: Save models to output directory (currently XGBoost only)
- `--sample_id_col` / `-s`: Column name for sample IDs (default: 'IID')
- `--n_folds` / `-n`: Number of cross-validation folds (default: 5)
- `--train_samples`: File containing training sample IDs
- `--pred_samples`: File(s) containing sample IDs for prediction (not training)
- `--train_one_fold`: Train only one fold (useful for model evaluation)

### Output Files

The script creates:
- `model_config.json`: Model configuration with number of folds
- `ho_preds.csv`: Holdout predictions for training samples (from cross-validation)
- `ens_preds.csv`: Ensemble predictions for non-training samples (if provided)
- `ho_scores.json`: Performance metrics for holdout predictions
- Scatter plots and joint plots visualizing predictions vs. true values

### Example Usage

```bash
python fit_model.py \
    --covar_file ../data/dev/covariates.tsv \
    --pheno_file ../data/dev/phenotype_0_5.tsv \
    --model_config ../data/dev/model_config.json \
    --out_dir ../../output \
    --train_samples ../data/dev/train_samples.txt \
    --pred_samples ../data/dev/val_samples.txt ../data/dev/test_samples.txt
```

## Getting Shapley Values with get_shapley_values.py

The `deeper_null/get_shapley_values.py` script computes Shapley values and first-order Shapley Interaction Index (SII) values for trained models. Shapley values are local explanations calculated for each individual, helping identify important covariate effects and interactions.

### Command Line Arguments

- `--model_files` / `-m`: Path(s) to one or more model save files (required)
- `--covar_file` / `-c`: Path to covariate file (required)
- `--pred_samples` / `-p`: File with sample IDs to compute Shapley values for (optional; uses all samples if not provided)
- `--model_type` / `-t`: Model type - 'linear', 'xgb', or 'nn' (required; currently only 'xgb' supported)
- `--out_dir` / `-o`: Directory to save output JSON files (default: current directory)
- `--sample_id_col`: Column name for sample IDs (default: 'IID')
- `--classification`: Flag for classification models (default: False)

### Output Files

The script generates two JSON files:

1. **shapley_individual_values.json**: Individual-level Shapley values and SII values
   - Keys: model names (from save file names), 'feature_names'
   - Second level: individual IDs
   - Third level: 'Shapley' or '1-SII' arrays

2. **shapley_agg_values.json**: Aggregated Shapley values across all individuals and models
   - Keys: 'Shapley', '1-SII', 'feature_names'
   - Aggregation methods: 'mean', 'median', 'std' (of absolute values)

### Example Usage

```bash
python get_shapley_values.py \
    --model_files model_fold_0.json model_fold_1.json \
    --covar_file ../data/covariates.tsv \
    --pred_samples ../data/test_samples.txt \
    --model_type xgb \
    --out_dir ../../shap_output
```

## UK Biobank RAP Workflows (ukb_rap_workflows)

The `ukb_rap_workflows` directory contains workflow scripts for running analyses on the UK Biobank Research Analysis Platform (RAP). Each subdirectory serves a specific purpose:

### GWAS Workflows

- **GWAS_plink**: PLINK2 GWAS workflow
- **GWAS_plotting**: Create visualization plots for GWAS results
- **PRS_PRScs_GWAS_manhattan_local**: Create Manhattan plots and scatter plots for GWAS run as part of PRScs. Also creates tables of changes in GWAS hits when adding null with additional covariates.

### PRS (Polygenic Risk Score) Workflows

- **PRS_PRScs**: Launch PRScs PRS workflow with optional null model integration
- **PRS_basil**: BASIL PRS workflow launcher with optional null model integration
- **PRS_score_preds**: Score PRS predictions and create evaluation plots
- **PRS_eval_local**: Local evaluation of PRS results including plotting scores, paired comparisons with bootstrap confidence intervals, and score table generation

### DeepNull Model Workflows

- **fit_dn_model**: Launch DeeperNull model fitting workflows on UK Biobank RAP
- **dn_eval_local**: Local evaluation of DeepNull models including score plotting, score table generation, and binary classification evaluation
- **dn_shap**: Compute Shapley values for DeepNull models on UK Biobank data
- **dn_shap_eval_local**: Evaluate and visualize Shapley values including SII bar plots and aggregated Shapley value analysis

### Comparison and Preprocessing

- **compare_null_and_prs_improvements**: Compare performance improvements between null models and PRS methods. Plot PGS vs. null model improvements with additional covariates over baseline. Plot improvement of PGS over null model alone.
- **geno_prepro**: Genotype data preprocessing workflows for UK Biobank

### Resources

- **resources**: Docker image resources including PLINK2 binary and PRSice-2 executable


## Dependencies

Some workflows install this library before running a script. To avoid having packages install then, we do not have a requirements.txt file. The dependencies for the training and Shapley value scripts are provided below. Dependencies for the all workflows can be found in their associated Dockerfiles. The docker images created by the Makefiles can also be used to run these workflows.

### Training and Shapley value dependencies

- matplotlib
- numpy
- pandas
- pytorch-lightning
- scikit-learn
- scipy
- seaborn
- shapiq
- torch>=2.0
- torchmetrics
- tqdm
- xgboost

