"""Rescale coordinates and optionally add null prediction to covariate TSV file.

Outputs to a new TSV called 'prepro_covar.tsv'.

CL arguments:

* --covar-file: Path to the base covariate TSV file.
* -rc, --rescale-coords: Standardizes coordinates by linearly transforming them
	to have a mean of 0 and a standard deviation of 1 if flag is present.
* -rp, --rescale-preds: Standardizes predictions by linearly transforming
	them to have a mean of 0 and a standard deviation of 1 if flag is present.
* -rt, --rescale-time: Standardizes continuous time fields using linear
	transformation. 'time_of_day' and 'day_of_year' are standardized, while
	'month_of_year' is output as as integer unless --string-month flag is 
	present.

Optional args that if given, will be used to merge null model predictions 
	with the base covariates:

* --train-pred-file: Path to the null model prediction file for the
	training set.
* --test-val-pred-file: Path to the null model prediction file for the
	validation/test set.
* --string-month: If flag is present, 'month_of_year' is output as a string
	instead of an integer.
"""

import argparse
from pprint import pprint

import pandas as pd


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--covar-file',
		required=True,
		help='Path to the base covariate TSV file.'
	)
	parser.add_argument(
		'--train-pred-file',
		required=False,
		help='Path to the null model prediction file for the training set.'
	)
	parser.add_argument(
		'--test-val-pred-file',
		required=False,
		help='Path to the null model prediction file for the validation/test set.'
	)
	parser.add_argument(
		'-rc', '--rescale-coords',
		action='store_true',
		help='Standardizes coordinates if flag is present.'
	)
	parser.add_argument(
		'-rp', '--rescale-preds',
		action='store_true',
		help='Standardizes predictions ("pred" column) if flag is present.'
	)
	parser.add_argument(
		'-rt', '--rescale-time',
		action='store_true',
		help='Standardizes continuous time fields using linear transformation. '
		'"time_of_day" and "day_of_year" are standardized, while "month_of_year" '
		'is left as an integer.'
	)
	parser.add_argument(
		'--string-month',
		action='store_true',
		help='If flag is present, "month_of_year" is output as a string instead '
		'of an integer.'
	)
	return parser.parse_args()


if __name__ == '__main__':

	args = parse_args()
	pprint(vars(args))

	# Load base covariates file
	covar_df = pd.read_csv(
		args.covar_file,
		sep='\s+',
	)

	print(covar_df.columns, flush=True)

	# Rescale all columns with 'coord' in the name if flag is present
	if args.rescale_coords:
		for col in covar_df.columns:
			if 'coord' in col:
				covar_df[col] = (
					covar_df[col] - covar_df[col].mean()
				) / covar_df[col].std()

	# Load null model predictions if both are given. If only one is given,
	# raise an error.
	if args.train_pred_file is not None and args.test_val_pred_file is None:
		raise ValueError(
			'Both --train-pred-file and --test-val-pred-file must be given '
			'if either is given.'
		)
	elif args.train_pred_file is None and args.test_val_pred_file is not None:
		raise ValueError(
			'Both --train-pred-file and --test-val-pred-file must be given '
			'if either is given.'
		)
	elif args.train_pred_file is not None and args.test_val_pred_file is not None:
		train_pred_df = pd.read_csv(args.train_pred_file)
		test_val_pred_df = pd.read_csv(args.test_val_pred_file)[['IID', 'pred']]

		# Merge null model predictions with base covariates and save
		covar_df = covar_df.merge(
			pd.concat([train_pred_df, test_val_pred_df]),
			on='IID',
			how='inner',
		)

	# Standardizes predictions
	if args.rescale_preds:
		if 'pred' in covar_df.columns:
			covar_df['pred'] = (
				covar_df['pred'] - covar_df['pred'].mean()
			) / covar_df['pred'].std()
		else:
			print('No "pred" column found in covariate file.')

	# Standardize time fields
	if args.rescale_time:
		for time_field in ['time_of_day', 'day_of_year']:
			if time_field in covar_df.columns:
				covar_df[time_field] = (
					covar_df[time_field] - covar_df[time_field].mean()
				) / covar_df[time_field].std()
			else:
				print(f'No "{time_field}" column found in covariate file.')

	if args.string_month and 'month_of_year' in covar_df.columns:
			covar_df['month_of_year'] = covar_df['month_of_year'].apply(lambda x: 'm' + str(x))
	elif 'month_of_year' in covar_df.columns:
		covar_df['month_of_year'] = covar_df['month_of_year'].astype(int)
	else:
		print('No "month_of_year" column found in covariate file.')

	print(covar_df.iloc[0])

	covar_df.to_csv(
		'prepro_covar.tsv',
		sep='\t',
		index=False
	)
