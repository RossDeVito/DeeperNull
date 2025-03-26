"""Get and save Shapley and first-order Shapley Interaction Index (SII) values
for the k models in the null model.

Shapley values are local, meaning they are calculated for each individual.

There are two output JSON files. One, "shapley_individual_values.json", contains
the Shapley values and SII values for each individual for each model. The other,
"shapley_agg_values.json", contains the aggregated Shapley values and SII values
for each model across all people and models.

For 'shapley_individual_values.json':

	- The first level key is the model, named by the save file name with the
	  extension removed. There is also a first level key 'feature_names' with
	  the feature names in an order corresponding to the Shapley value indices
	  in the output for each individual.
	- The second level key is the individual ID
	- The third level key is 'Shapley' or '1-SII', which contains the Shapley
	  values or SII values for the individual, respectively.

For 'shapley_agg_values.json':

	- The first level key is the value type, either 'Shapley' or '1-SII'. Also
		containing a first level key 'feature_names' with the feature names in
		an order corresponding to the Shapley value indices in the output.
	- The second level key is the aggregation method, either 'mean', 'median',
	  or 'std'.
	- The value is then a list for Shapley values or a 2-D list for SII values.

Command line arguments:

	-m, --model_files: Path(s) to one or more model save files to get
		interactions for.
	-c, --covar_file: Path to covariate file.
	-p, --pred_samples: Path to file containing sample IDs to compute
		Shapley values for. If not provided, Shapley values are computed
		for all samples in the covariate file.
	-t, --model_type: Type of models. Options are 'linear', 'xgb', and 'nn'.
		Default is 'xgb'. NOTE: Only 'xgb' supported
	-o, --out_dir: Directory to save output JSON files to. Default is '.'.
	--sample_id_col: Name of column in covariate file that contains
		sample IDs. Default is 'IID'.
	--classification: Whether the model is a classification model. Default is
		False.

"""

import argparse
import json
import os
import pickle
from pprint import pprint

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
		help='Directory to save output JSON file to. Default is ".".'
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
	pprint(vars(args))

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
		# Get intersection of sample IDs in covariate data and pred_samples
		pred_samples = set(pred_samples)
		covar_samples = set(covar_data.index)
		pred_samples = list(pred_samples.intersection(covar_samples))
	else:
		pred_samples = covar_data.index

	covar_data = covar_data.loc[pred_samples]

	print(f'Computing Shapley values for {len(covar_data)} samples')


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

	# Save individual level Shapley values
	with open(f'{args.out_dir}/shapley_individual_values.json', 'w') as f:
		json.dump(shapley_output, f)
	
	# Aggregate Shapley values
	agg_output = {
		'feature_names': shapley_output['feature_names'],
		'Shapley': {},
		'1-SII': {}
	}

	for value_type in ['Shapley', '1-SII']:
		all_values = []
		for model in shapley_output:
			if model == 'feature_names':
				continue
			model_values = shapley_output[model][value_type]

			for sample_id in model_values:
				all_values.append(model_values[sample_id])

		all_values = np.array(all_values)

		# Compute mean, median, and std for each feature or feature interaction
		agg_output[value_type]['mean'] = np.mean(all_values, axis=0).tolist()
		agg_output[value_type]['median'] = np.median(all_values, axis=0).tolist()
		agg_output[value_type]['std'] = np.std(all_values, axis=0).tolist()

	# Save aggregated Shapley values
	with open(f'{args.out_dir}/shapley_agg_values.json', 'w') as f:
		json.dump(agg_output, f)
