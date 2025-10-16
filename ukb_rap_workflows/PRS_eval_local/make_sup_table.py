import os
import re

import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Patch


# Set figure DPI for saving figs
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 1200


if __name__ == '__main__':

	metric = 'r2'
	ci = 95
	scores_dir = 'scores'
	plot_BASIL = False
	
	# Load scores for test all, wb, and nwb sets
	test_scores = pd.read_csv(f'{scores_dir}/test_scores.csv')
	test_bs_scores = pd.read_csv(f'{scores_dir}/test_boot_scores.csv')

	# Cast test_bs_scores[metric] to list from literal
	# Parse bootstrap string-lists into numpy arrays for all metrics and compute 95% CIs
	metrics = ['mse', 'r2', 'mae', 'mape', 'pearson_r', 'spearman_r']

	def _to_array(x):
		if isinstance(x, (list, np.ndarray)):
			return np.array(x, dtype=float)
		if pd.isna(x):
			return np.array([], dtype=float)
		s = str(x).strip()
		s = s.strip('[]')
		if s == '':
			return np.array([], dtype=float)
		parts = [p.strip() for p in s.split(',') if p.strip() != '']
		return np.array([float(p) for p in parts], dtype=float)

	# convert each metric column in test_bs_scores to numpy arrays
	for m in metrics:
		if m in test_bs_scores.columns:
			test_bs_scores[m] = test_bs_scores[m].apply(_to_array)

	# compute CI bounds for each metric and attach as new columns in test_bs_scores
	alpha = 100 - ci
	lower_q = alpha / 2
	upper_q = 100 - (alpha / 2)

	for m in metrics:
		if m in test_bs_scores.columns:
			test_bs_scores[f'{m}_lower'] = test_bs_scores[m].apply(
				lambda arr: np.percentile(arr, lower_q) if getattr(arr, 'size', 0) > 0 else np.nan
			)
			test_bs_scores[f'{m}_upper'] = test_bs_scores[m].apply(
				lambda arr: np.percentile(arr, upper_q) if getattr(arr, 'size', 0) > 0 else np.nan
			)

	# merge all metric CI columns into test_scores in one go
	ci_cols = ['model_desc', 'pheno'] + \
		[s for m in metrics for s in (f'{m}_lower', f'{m}_upper') if f'{m}_lower' in test_bs_scores.columns]
	test_scores = test_scores.merge(
		test_bs_scores[ci_cols],
		on=['model_desc', 'pheno'],
		how='left'
	)

	# keep backward-compatible 'lower'/'upper' for the selected metric
	if metric in metrics:
		test_scores['lower'] = test_scores.get(f'{metric}_lower')
		test_scores['upper'] = test_scores.get(f'{metric}_upper')

	# Compute upper and lower bounds for bootstrap confidence intervals
	alpha = 100 - ci
	lower_q = alpha / 2
	upper_q = 100 - (alpha / 2)

	# Compute confidence bounds
	test_bs_scores['lower'] = test_bs_scores[metric].apply(
		lambda x: np.percentile(x, lower_q)
	)
	test_bs_scores['upper'] = test_bs_scores[metric].apply(
		lambda x: np.percentile(x, upper_q)
	)

	# Join lower and upper bounds with test_scores
	test_scores = test_scores.merge(
		test_bs_scores[['model_desc', 'pheno', 'lower', 'upper']],
		on=['model_desc', 'pheno'],
		how='left'
	)

	sns.set_style('whitegrid')

	# Define metric_display based on metric
	if metric == 'r2':
		metric_display = 'R^2'
	elif metric == 'mse':
		metric_display = 'MSE'
	elif metric == 'mae':
		metric_display = 'MAE'
	else:
		metric_display = metric.upper()

	# Compute upper and lower error ranges for plotting
	test_scores['err_low'] = test_scores[metric] - test_scores['lower']
	test_scores['err_high'] = test_scores['upper'] - test_scores[metric]

	# Create FacetGrid without plotting yet
	pheno_order = sorted(test_scores['pheno'].unique(), key=lambda x: x.lower())

	def extract_covar_set(desc):
		"""
		Returns the covariate-set label used for grouping/positioning on x-axis.
		Special-case: if the XGB null includes PCs, we return the unpaired label
		'age, sex, locations, times, PCs for null'.
		Otherwise, we return the 'lin: ... PCs' covariate set.
		"""
		# Base covariate set from the linear part
		m = re.search(r'lin:\s*(.+?),\s*PCs', desc)
		base = m.group(1) if m else None

		# If the null clause explicitly contains PCs, treat it as the special unpaired bucket
		if 'XGB null:' in desc:
			null_clause = desc.split('XGB null:')[1]
			if re.search(r'PCs\b', null_clause):
				return 'age, sex, locations, times, PCs for null'

		return base  # e.g., 'age, sex', 'age, sex, locations', 'age, sex, times', 'age, sex, locations, times'

	def has_xgb_null(desc):
		return 'XGB null' in desc

	test_scores['covar_set'] = test_scores['model_desc'].apply(extract_covar_set)
	test_scores['with_null']  = test_scores['model_desc'].apply(has_xgb_null)