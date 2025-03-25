"""Get and save Shapley and first-order Shapley Interaction Index (SII) values
for the k models in the null model.

Shapley values are local, meaning they are calculated for each individual.
Output is saved as a JSON file. The first level key is the model, named by
the save file name with the extension removed. There is also a first level key
'feature_names' with the feature names in an order corresponding to the Shapley
value indices in the output for each individual. The second level key is the
individual ID. The value is a dictionary with the following keys:
- 'Shapley': Shapley values. Vector
- '1-SII': First-order Shapley Interaction Index values. Matrix

Command line arguments:

	-m, --model_files: Path(s) to one or more model save files to get
		interactions for.
	-c, --covar_file: Path to covariate file.
	-p, --pred_samples: Path to file containing sample IDs to compute
		Shapley values for. If not provided, Shapley values are computed
		for all samples in the covariate file.
	-t, --model_type: Type of models. Options are 'linear', 'xgb', and 'nn'.
		Default is 'xgb'. NOTE: Only 'xgb' supported
	-o, --out_dir: Directory to save output JSON file to. File will be named
		'shapley_values.json'. Default is '.'.
	--sample_id_col: Name of column in covariate file that contains
		sample IDs. Default is 'IID'.
	--classification: Whether the model is a classification model. Default is
		False.

"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import shapiq
from tqdm import tqdm
import xgboost as xgb


def parse_args():
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument(
		'-m', '--model_files',
		nargs='+',
		required=True,
		help='Path(s) to one or more model save files to get interactions for.'
	)
	parser.add_argument(
		'-c', '--covar_file',
		required=True,
		help='Path to covariate file.'
	)
	parser.add_argument(
		'-p', '--pred_samples',
		help='Path to file containing sample IDs to compute Shapley values for. '
			 'If not provided, Shapley values are computed for all samples in the covariate file.'
	)
	parser.add_argument(
		'-t', '--model_type',
		required=True,
		choices=['linear', 'xgb', 'nn'],
		help='Type of models. Options are "linear", "xgb", and "nn". Default is "xgb". '
			 'NOTE: Only "xgb" supported'
	)
	parser.add_argument(
		'-o', '--out_dir',
		default='.',
		help='Directory to save output JSON file to. File will be named "shapley_values.json".'
			' Default is ".".'
	)
	parser.add_argument(
		'--sample_id_col',
		default='IID',
		help='Name of column in covariate file that contains sample IDs.'
	)
	parser.add_argument(
		'--classification',
		action='store_true',
		help='Whether the model is a classification model. Default is False.'
	)

	return parser.parse_args()


def load_covar_data(
	covar_file,
	sample_id_col,
):
	"""Load covariate data from file.

	Args:
		covar_file (str): Path to covariate file.
		sample_id_col (str): Name of column in covariate file that contains sample IDs.
	"""

	covar_df = pd.read_csv(covar_file, sep='\s+')
	covar_df = covar_df.set_index(sample_id_col)

	return covar_df


def load_model(model_file, model_type, classification):
	"""Load model from file.

	Args:
		model_file (str): Path to model file.
		model_type (str): Type of model. Options are 'linear', 'xgb', and 'nn'.
		classification (bool): Whether the model is a classification model.

	Returns:
		model: Loaded model.
	"""

	if model_type == 'xgb':
		if classification:
			model = xgb.XGBClassifier()
		else:
			model = xgb.XGBRegressor()
		model.load_model(model_file)
	else:
		raise ValueError('Only "xgb" models are supported currently.')

	return model


if __name__ == '__main__':

	args = parse_args()

	# Create output directory
	os.makedirs(args.out_dir, exist_ok=True)

	# Load sample file if provided
	if args.pred_samples:
		with open(args.pred_samples, 'r') as f:
			pred_samples = [l.strip() for l in f.readlines()]
	else:
		pred_samples = None

	# Load and subset covariate data
	covar_data = load_covar_data(
		args.covar_file,
		args.sample_id_col,
	)

	if pred_samples:
		covar_data = covar_data.loc[pred_samples]
	else:
		pred_samples = covar_data.index

	# Compute Shapley values for each model
	n_models = len(args.model_files)
	shapley_output = {
		'feature_names': covar_data.columns.tolist(),
	}

	for model_file in tqdm(
		args.model_files,
		desc='Computing Shapley values',
		total=n_models,
	):
		# Load and convert model
		model = load_model(
			model_file,
			args.model_type,
			args.classification,
		)
		model = shapiq.explainer.tree.conversion.xgboost.convert_xgboost_booster(
			model.get_booster()
		)

		# Create explainer and compute Shapley values
		explainer = shapiq.TreeExplainer(
			model=model,
			index="k-SII",
			max_order=2
		)

		interaction_values = explainer.explain_X(
			covar_data.values,
			n_jobs=-1,
		)

		# Add Shapley values to output dictionary
		model_fname = os.path.splitext(os.path.basename(model_file))[0]

		shapley_output[model_fname] = {
			'Shapley': dict(),
			'1-SII': dict(),
		}

		for i, sample_id in enumerate(pred_samples):
			shapley_output[model_fname]['Shapley'][
				sample_id
			] = interaction_values[i].get_n_order_values(1).tolist()
			shapley_output[model_fname]['1-SII'][
				sample_id
			] = interaction_values[i].get_n_order_values(2).tolist()

	# Save output
	with open(f'{args.out_dir}/shapley_values.json', 'w') as f:
		json.dump(shapley_output, f)