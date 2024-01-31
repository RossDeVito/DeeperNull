"""Fit DeepNull style model to data.

Model is either an XGBoost model or a Pytorch Lightning 2.0 style model.

The script has the following requirements to run:

	1. A whitespace delimited covariate file with a header row. This will
		be used for the input to the model.
	2. A whitespace delimited phenotype file with a header row. Either should
		have one column of sample IDs and one column of phenotype values, or
		use the --pheno_name argument to specify the name of the phenotype.
	3. A model configuration file. This is a JSON file that contains a
		description of the model to be fit. See the documentation [TODO].
	4. A directory to write the output to. If the directory does not exist,
		it will be created. The output will be:
			- A file called 'model_config.json' that contains the model
				configuration used with the number of folds added.
			- A file preds.csv that contains the predictions of the model
				on both samples in the training set and those specified
				to still be predicted for. For the training set predictions
				are from when the sample was part of the fold held out of
				training. For the non-training set, predictions will be an
				average of the n models predictions. The columns will be:
					- The value of sample_id_col (below) for the sample ID
					- 'pred' for the predicted value
					- 'train' which is True if the sample was in the training
						set and False if it was not.
	5. (Optional) Files that identify samples to be in the training set and
		non-training set files that predictions should be made for. If neither
		is provided all samples will be used for training. Files defining
		the non-training set can only be used when a file defining the
		training set is also provided. The files should be text files with
		one sample ID per line.

Script has the following command line arguments:

	--covar_file: Path to covariate file.
	--pheno_file: Path to phenotype file.
	--model_config: Path to model configuration JSON file.
	--out_dir: Path to output directory. If it does not exist, will 
		be created. Default is current working directory.
	--sample_id_col: Name of column in covariate file that contains
		sample IDs. Default is 'IID'.
	--pheno_sample_id_col: Name of column in phenotype file that
		contains sample IDs. Defaults to value of sample_id_col.
	--binary_pheno: Whether phenotype is binary. Default is False.
	--n_folds: Number of folds to use for cross validation. Default is 5.
	--train_samples: Path to file containing sample IDs to use for training
		(Optional, see above).
	--pred_samples: Path or paths to file(s) containing sample IDs to 
		predict for, but not use for training (Optional, see above). Sample
		IDs in all files will be combined. If a sample ID is in both the
		training and prediction set, it will be used for training and the
		prediction will be the prediction from the fold it was held out of,
		not the ensemble average.

Example usage:

	python fit_model.py \
		--covar_file ../data/dev/covariates.tsv \
		--pheno_file ../data/dev/phenotype_0_5.tsv \
		--model_config ../data/dev/model_config.json \
		--out_dir ../.. \
		--train_samples ../data/dev/train_samples.txt \
		--pred_samples ../data/dev/val_samples.txt ../data/dev/test_samples.txt

		-c ../data/dev/covariates.tsv -p ../data/dev/phenotype_0_5.tsv -m ../data/dev/model_config.json -o ../.. --train_samples ../data/dev/train_samples.txt --pred_samples ../data/dev/val_samples.txt ../data/dev/test_samples.txt
		
"""
import argparse
import os
import json
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm
import xgboost as xgb

from deeper_null.xgb_models import XGB_MODEL_TYPES, create_xgb_model



def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--covar_file', required=True,
					help='Path to covariate file.')
	parser.add_argument('-p', '--pheno_file', required=True,
					help='Path to phenotype file.')
	parser.add_argument('-m', '--model_config', required=True,
					help='Path to model configuration JSON file.')
	parser.add_argument('-o', '--out_dir', default='.',
					help='Path to output directory. If it does not exist, '
						'will be created.')
	parser.add_argument('-s', '--sample_id_col', default='IID',
					help='Name of column in covariate file that contains '
						'sample IDs.')
	parser.add_argument('--pheno_sample_id_col', default=None,
					help='Name of column in phenotype file that contains '
						'sample IDs.')
	parser.add_argument('-b', '--binary_pheno', action='store_true',
					help='Whether phenotype is binary.')
	parser.add_argument('-n', '--n_folds', default=5, type=int,
					help='Number of folds to use for cross validation.')
	parser.add_argument('--train_samples', default=None,
					help='Path to file containing sample IDs to use for '
						'training (Optional, see above).')
	parser.add_argument('--pred_samples', default=None, nargs='+',
						help='Path or paths to file(s) containing sample IDs '
							'to predict for, but not use for training '
							'(Optional, see above).')
	return parser.parse_args()


def load_sample_files(train_samp_file, pred_samp_files):
	"""Load files containing sample IDs that specify which samples to use
	for training and which to predict for.

	Args:
		train_samp_file: Path to file containing sample IDs to use for
			training. If None, all samples will be used for training.
		pred_samp_files: List of paths to files containing sample IDs to
			predict for, but not use for training.

	Returns:
		train_samp_ids: Set of sample IDs to use for training or None if
			train_samp_file is None and all samples should be used.
		pred_samp_ids: Set of sample IDs to predict for, but not use for
			training. None if pred_samp_files is None.

	Raises:
		ValueError: If pred_samp_files is not None and train_samp_file is None.
	"""
	if pred_samp_files is not None and train_samp_file is None:
		raise ValueError('Cannot specify prediction samples without '
							'specifying training samples.')

	if train_samp_file is None:
		train_samp_ids = None
	else:
		with open(train_samp_file, 'r') as f:
			train_samp_ids = set([l.strip() for l in f.readlines()])
	if pred_samp_files is None:
		pred_samp_ids = None
	else:
		pred_samp_ids = set()
		for f in pred_samp_files:
			with open(f, 'r') as f:
				pred_samp_ids.update([l.strip() for l in f.readlines()])
	return train_samp_ids, pred_samp_ids


def load_covar_pheno_data(
	covar_file,
	pheno_file,
	sample_id_col,
	pheno_sample_id_col=None,
	training_samples=None,
	prediction_samples=None,
):
	"""Load covariate and phenotype data.
	
	Returns two pairs of input output dataframes. The first pair is the
	training data the will used with n-fold cross validation. The second
	pair is the data that will be used for prediction with the ensemble
	of n models.

	When both training_samples and prediction_samples, all loaded data
	will be used for training and the dataframes for the prediction
	pair will be None. When just training_samples is provided, the
	dataframes for the prediction pair will also be None.

	Raises error if prediction_samples is provided without training_samples.
	"""

	if prediction_samples is not None and training_samples is None:
		raise ValueError('Cannot specify prediction samples without '
							'specifying training samples.')

	if pheno_sample_id_col is None:
		pheno_sample_id_col = sample_id_col

	# Load covariate data.
	covar_df = pd.read_csv(covar_file, sep='\s+')
	covar_df = covar_df.set_index(sample_id_col)

	# Load phenotype data.
	pheno_df = pd.read_csv(pheno_file, sep='\s+')
	pheno_df = pheno_df.set_index(pheno_sample_id_col)

	# Subset to just samples in both covariate and phenotype files
	samp_ids = list(set(covar_df.index).intersection(pheno_df.index))
	covar_df = covar_df.loc[samp_ids]
	pheno_df = pheno_df.loc[samp_ids]

	if training_samples is None:
		# Use all samples for training.
		return (covar_df, pheno_df), (None, None)
	else:
		# Subset to just training samples.
		train_samp_ids = list(set(training_samples).intersection(samp_ids))
		train_covar_df = covar_df.loc[train_samp_ids]
		train_pheno_df = pheno_df.loc[train_samp_ids]

		if prediction_samples is None:
			return (train_covar_df, train_pheno_df), (None, None)
		else:
			# Subset to just prediction samples.
			pred_samp_ids = list(set(prediction_samples).intersection(samp_ids))
			pred_covar_df = covar_df.loc[pred_samp_ids]
			pred_pheno_df = pheno_df.loc[pred_samp_ids]

			return (train_covar_df, train_pheno_df), (pred_covar_df, pred_pheno_df)
		

def create_model(model_config):
	if model_config['model_type'].lower() in XGB_MODEL_TYPES:
		return create_xgb_model(model_config)
	else:
		raise ValueError('Unknown model type: {}'.format(model_config['model_type']))
	

def score_and_plot_regression(y_true, y_pred, out_dir, plot_prefix=''):
	"""Compute regression metrics and plot pred v true scatter.

	Args:
		y_true: True values.
		y_pred: Predicted values.
		out_dir: Path to output directory.
		plot_prefix: Prefix to add to plot file names.

	Returns:
		scores: Dictionary of scores.
	"""
	scores = dict()
	scores['r2'] = metrics.r2_score(y_true, y_pred)
	scores['mse'] = metrics.mean_squared_error(y_true, y_pred)
	scores['mae'] = metrics.mean_absolute_error(y_true, y_pred)

	# Plot pred v true scatter
	fig, ax = plt.subplots()
	ax.scatter(y_true, y_pred, marker='.', alpha=0.5) # type: ignore
	ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')  # Add dashed line
	ax.set_xlabel('True')
	ax.set_ylabel('Predicted')
	ax.set_title('Pred v True Phenotype')
	ax.set_aspect('equal')
	plt.savefig(os.path.join(out_dir, f'{plot_prefix}_scatter.png'), dpi=300)
	plt.close()

	# Also plot seaborn jointplot with distribution of each
	# variable on the axes and the scatter in the middle
	g = sns.jointplot(x=y_true, y=y_pred, kind='scatter', joint_kws={'marker': '.', 'alpha': 0.5})
	g.ax_joint.set_aspect('equal', adjustable='box')

	# Get the overall min/max of the data
	min_val = min(y_true.min(), y_pred.min())
	max_val = max(y_true.max(), y_pred.max())

	# Set the limits
	g.ax_joint.set_xlim(min_val, max_val)
	g.ax_joint.set_ylim(min_val, max_val)

	# Add dashed line for x=y
	g.ax_joint.plot([min_val, max_val], [min_val, max_val], 'k--')

	g.ax_joint.set_xlabel('True')
	g.ax_joint.set_ylabel('Predicted')
	g.ax_marg_x.set_title('Pred v True Phenotype')
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, f'{plot_prefix}_jointplot.png'), dpi=300)
	plt.close()

	return scores


def score_and_plot_binary(y_true, y_pred, out_dir, plot_prefix):
	"""Compute binary classification metrics and plot confusion matrix
	and PR curve.

	Args:
		y_true: True values.
		y_pred: Predicted values.
		out_dir: Path to output directory.
		plot_prefix: Prefix to add to plot file names.

	Returns:
		scores: Dictionary of scores.
	"""
	scores = dict()
	scores['auc'] = metrics.roc_auc_score(y_true, y_pred)
	scores['acc'] = metrics.accuracy_score(y_true, y_pred)
	scores['f1'] = metrics.f1_score(y_true, y_pred)

		# Plot confusion matrix
	fig, ax = plt.subplots()
	cm = metrics.confusion_matrix(y_true, y_pred)
	sns.heatmap(cm, annot=True, ax=ax)
	ax.set_xlabel('Predicted')
	ax.set_ylabel('True')
	ax.set_title('Confusion Matrix')
	plt.savefig(os.path.join(out_dir, f'{plot_prefix}_confusion_matrix.png'))
	plt.close()

	# Plot PR curve
	fig, ax = plt.subplots()
	pr_curve = metrics.precision_recall_curve(y_true, y_pred)
	ax.plot(pr_curve[0], pr_curve[1])
	ax.set_xlabel('Recall')
	ax.set_ylabel('Precision')
	ax.set_title('PR Curve')
	plt.savefig(os.path.join(out_dir, f'{plot_prefix}_pr_curve.png'))
	plt.close()

	return scores


if __name__ == '__main__':
	args = parse_args()

	# Create output directory
	os.makedirs(args.out_dir, exist_ok=True)

	# Read in model configuration then save with n_folds added.
	with open(args.model_config, 'r') as f:
		model_config = json.load(f)
	model_config['n_folds'] = args.n_folds
	with open(os.path.join(args.out_dir, 'model_config.json'), 'w') as f:
		json.dump(model_config, f, indent=4)

	# Load sample files.
	train_samp_ids, pred_samp_ids = load_sample_files(
		args.train_samples, args.pred_samples
	)

	# Load covariate and phenotype data.
	train_Xy, pred_Xy = load_covar_pheno_data(
		args.covar_file,
		args.pheno_file,
		args.sample_id_col,
		args.pheno_sample_id_col,
		train_samp_ids,
		pred_samp_ids,
	)

	# Create n_folds folds using index of train_Xy[0] for the sample IDs.
	shuffled_index = train_Xy[0].index.to_list()
	np.random.shuffle(shuffled_index)
	folds = np.array_split(shuffled_index, args.n_folds)

	# Fit and make predictions with n-fold cross validation.
	print('Fitting model with {} folds.'.format(args.n_folds))
	train_ho_preds = dict()
	ensemble_preds = defaultdict(lambda: [])

	for i in tqdm(range(args.n_folds), desc="Training folds", total=args.n_folds):
		# Get training and holdout samples.
		train_samp_ids = folds[:i] + folds[i+1:]
		train_samp_ids = [samp_id for fold in train_samp_ids for samp_id in fold]
		ho_samp_ids = folds[i]

		# Make input and output dataframes for training and holdout samples.
		train_X = train_Xy[0].loc[train_samp_ids]
		train_y = train_Xy[1].loc[train_samp_ids]
		ho_X = train_Xy[0].loc[ho_samp_ids]
		ho_y = train_Xy[1].loc[ho_samp_ids]

		# Create model
		## TODO
		model = create_model(model_config)

		# Fit model
		model.fit(train_X, train_y)

		# Make predictions on holdout samples
		ho_preds = model.predict(ho_X)
		train_ho_preds.update(dict(zip(ho_X.index, ho_preds)))

		# Make predictions on prediction samples and add to ensemble values
		# so that they can be averaged later
		if pred_Xy[0] is not None:
			pred_preds = model.predict(pred_Xy[0])

			for samp_id, pred in zip(pred_Xy[0].index, pred_preds):
				ensemble_preds[samp_id].append(pred)

	# Save predictions on holdout samples
	ho_preds = pd.DataFrame.from_dict(train_ho_preds, orient='index')
	ho_preds.columns = ['pred']
	ho_preds.index.name = args.sample_id_col
	ho_preds.reset_index().to_csv(
		os.path.join(args.out_dir, 'ho_preds.csv'), index=False
	)
	
	# Join with holdout phenotype values and name columns 'true'
	ho_preds = ho_preds.join(train_Xy[1])
	ho_preds = ho_preds.rename(columns={train_Xy[1].columns[0]: 'true'})

	# Calculate metrics
	if args.binary_pheno:
		scores = score_and_plot_binary(
			ho_preds['true'],
			ho_preds['pred'],
			args.out_dir,
			'ho'
		)
	else:
		scores = score_and_plot_regression(
			ho_preds['true'],
			ho_preds['pred'],
			args.out_dir,
			'ho'
		)

	# Save scores
	with open(os.path.join(args.out_dir, 'ho_scores.json'), 'w') as f:
		json.dump(scores, f, indent=4)

	# Save ensemble predictions and standard deviation and save as one csv
	if pred_Xy[0] is not None:
		ens_pred_dev = {
			sid: (np.mean(preds), np.std(preds)) for sid, preds in ensemble_preds.items()
		}
		ens_pred_dev = pd.DataFrame.from_dict(ens_pred_dev, orient='index')
		ens_pred_dev.columns = ['pred', 'std']
		ens_pred_dev.index.name = args.sample_id_col
		ens_pred_dev.reset_index().to_csv(
			os.path.join(args.out_dir, 'ens_preds.csv'), index=False
		)


	
