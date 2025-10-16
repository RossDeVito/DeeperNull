import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("-v", "--val-preds", required=True)
	parser.add_argument("-t", "--test-preds", required=True)
	parser.add_argument("-p", "--pheno-file", required=True)
	parser.add_argument("-o", "--out-dir", required=True)
	parser.add_argument(
		"-c", "--case-control",
		action="store_true",
		help="If included, the phenotype is case-control and the ground truth is binary."
	)
	parser.add_argument(
		"-b", "--bootstrap-iters",
		type=int,
		default=1000,
		help="Number of bootstrap iterations to estimate confidence intervals."
	)

	return parser.parse_args()

def score_preds(y_true, y_pred, case_control):
	if case_control:
		average_precision = metrics.average_precision_score(y_true, y_pred)
		roc_auc = metrics.roc_auc_score(y_true, y_pred)
		return {
			'average_precision': average_precision,
			'roc_auc': roc_auc
		}
	else:
		return {
			'mse': metrics.mean_squared_error(y_true, y_pred),
			'r2': metrics.r2_score(y_true, y_pred),
			'mae': metrics.mean_absolute_error(y_true, y_pred),
			'mape': metrics.mean_absolute_percentage_error(y_true, y_pred),
			'pearson_r': stats.pearsonr(y_true, y_pred).statistic,
			'spearman_r': stats.spearmanr(y_true, y_pred).statistic
		}


if __name__ == '__main__':
	args = parse_args()

	val_preds = pd.read_csv(args.val_preds)
	test_preds = pd.read_csv(args.test_preds)
	pheno = pd.read_csv(args.pheno_file, sep='\s+')
	pheno_col = pheno.columns.difference(['IID']).values[0]
	pheno = pheno.rename(columns={pheno_col: 'true'})

	val_preds = val_preds.merge(pheno, on='IID', how='inner')
	test_preds = test_preds.merge(pheno, on='IID', how='inner')

	case_control = args.case_control
	val_preds['true'] = val_preds['true'].astype(int if case_control else float)
	test_preds['true'] = test_preds['true'].astype(int if case_control else float)

	val_scores = score_preds(val_preds['true'], val_preds['pred'], case_control)
	test_scores = score_preds(test_preds['true'], test_preds['pred'], case_control)

	# Bootstrap scores
	bootstrap_scores = []
	for _ in range(args.bootstrap_iters):
		boot_sample = test_preds.sample(frac=1, replace=True)
		score = score_preds(boot_sample['true'], boot_sample['pred'], case_control)
		bootstrap_scores.append(score)

	# Transform bootstrap scores to dict of lists
	bootstrap_scores_dict = {
		k: [score[k] for score in bootstrap_scores] for k in bootstrap_scores[0]
	}
	bootstrap_scores = bootstrap_scores_dict

	# Save scores and bootstrap results
	scores = {
		'val': val_scores,
		'test': test_scores,
		'bootstrap_scores': bootstrap_scores
	}

	save_fname = f'scores_w_bootstrap_{args.bootstrap_iters}.json'
	with open(os.path.join(args.out_dir, save_fname), 'w') as f:
		json.dump(scores, f, indent=4)
