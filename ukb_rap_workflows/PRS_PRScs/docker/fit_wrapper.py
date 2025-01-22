"""Fit wrapper model with PRScs scores.

Args:
	-p, --pheno-file: Path to phenotype file.
	-c, --covar-file: Path to covariate file. If not provided, no covariates
		will be used.
	-s, --score-file: Path to plink2 score file using PRScs output.
	-v, --val-iids: Path to validation set IIDs file.
	-t, --test-iids: Path to test set IIDs file.
	-o, --out-dir: Path to output directory.
"""

import argparse
import os

import pandas as pd
from sklearn.linear_model import LinearRegression


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument("-p", "--pheno-file", required=True)
	parser.add_argument("-c", "--covar-file", required=False)
	parser.add_argument("-s", "--score-file", required=True)
	parser.add_argument("-v", "--val-iids", required=True)
	parser.add_argument("-t", "--test-iids", required=True)
	parser.add_argument("-o", "--out-dir", required=True)

	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()

	# Load scores
	scores_df = pd.read_csv(
		args.score_file,
		usecols=['IID', 'SCORE1_SUM'],
		sep='\s+'
	)

	# Load phenotype
	pheno_df = pd.read_csv(
		args.pheno_file,
		sep='\s+'
	)

	# Load covariates
	if args.covar_file is not None:
		covar_df = pd.read_csv(
			args.covar_file,
			sep='\s+'
		)
	else:
		covar_df = None

	# Load sample sets
	val_split = pd.read_csv(
		args.val_iids,
		sep='\s+',
		header=None
	).values.flatten().tolist()

	test_split = pd.read_csv(
		args.test_iids,
		sep='\s+',
		header=None
	).values.flatten().tolist()

	# Join data
	if covar_df is not None:
		data_df = covar_df.merge(scores_df, on='IID', how='inner')
	else:
		data_df = scores_df
	data_df = data_df.merge(pheno_df, on='IID', how='inner')

	# Get phenotype and feature column names
	pheno_name = [c for c in list(pheno_df.columns) if c != 'IID']
	assert len(pheno_name) == 1
	pheno_name = pheno_name[0]

	col_names = list(data_df.columns)
	feat_cols = [c for c in col_names if c not in ['IID', pheno_name]]
	label_col = [pheno_name]

	# Get training and test data, then free other memory
	training_data = data_df[data_df['IID'].isin(val_split)]
	IID_train = training_data['IID'].values.flatten()	# type: ignore
	X_train = training_data[feat_cols]
	y_train = training_data[label_col]

	test_data = data_df[data_df['IID'].isin(test_split)]
	IID_test = test_data['IID'].values.flatten()	# type: ignore
	X_test = test_data[feat_cols]

	del data_df, scores_df, pheno_df, covar_df

	# Fit linear regression
	fit_model = LinearRegression().fit(
		X_train,
		y_train
	)

	# Predict
	val_preds = fit_model.predict(X_train).flatten()
	test_preds = fit_model.predict(X_test).flatten()

	# Save predictions
	pd.DataFrame(
		{'IID': IID_train, 'pred': val_preds}
	).to_csv(
		os.path.join(args.out_dir, 'val_preds.csv'),
		index=False
	)

	test_pred_df = pd.DataFrame(
		{'IID': IID_test, 'pred': test_preds}
	).to_csv(
		os.path.join(args.out_dir, 'test_preds.csv'),
		index=False
	)
