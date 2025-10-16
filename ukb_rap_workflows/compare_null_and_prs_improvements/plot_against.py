import re

import numpy as np
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


if __name__ == '__main__':
	
	mpl.rcParams['savefig.dpi'] = 1200

	dn_scores_dir = '../dn_eval_local/scores'
	prs_scores_dir = '../PRS_eval_local/scores'

	# Load null model and PRS scores
	dn_scores = pd.read_csv(f'{dn_scores_dir}/scores.csv')
	prs_scores = pd.read_csv(f'{prs_scores_dir}/test_scores.csv')

	# Filter to just BASIL with all time and location covariates & baseline
	dn_scores = dn_scores[
		(
			(dn_scores['covar_set'] == 'age_sex_all_coords_time')
			| (dn_scores['covar_set'] == 'age_sex')
		)
		& (
			(dn_scores['model_type'] == 'xgb_3')
			| (dn_scores['model_type'] == 'lin_reg_1')
		)
	]
	prs_scores = prs_scores[
		(prs_scores['model_desc'] == 'BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)')
		| (prs_scores['model_desc'] == 'BASIL (lin: age, sex, PCs)')
	]

	# For deep null scores, get improvement over baseline in terms of R^2
	# and percentage improvement

	# For DN
	dn_baselines = dn_scores[
		(dn_scores['covar_set'] == 'age_sex')
		& (dn_scores['model_type'] == 'lin_reg_1')
	]
	dn_scores = dn_scores[
		(dn_scores['covar_set'] != 'age_sex')
		& (dn_scores['model_type'] == 'xgb_3')
	]
	dn_scores = dn_scores.merge(
		dn_baselines[['pheno', 'r2']],
		on='pheno',
		suffixes=('', '_baseline'),
		how='left'
	).assign(
		r2_improvement=lambda x: x['r2'] - x['r2_baseline'],
		r2_improvement_pct=lambda x: 100 * (x['r2'] - x['r2_baseline']) / x['r2_baseline']
	)

	# For PRS
	prs_baselines = prs_scores[
		prs_scores['model_desc'] == 'BASIL (lin: age, sex, PCs)'
	]
	prs_scores = prs_scores[
		prs_scores['model_desc'] != 'BASIL (lin: age, sex, PCs)'
	].merge(
		prs_baselines[['pheno', 'r2']],
		on='pheno',
		suffixes=('', '_baseline'),
		how='left'
	).assign(
		r2_improvement=lambda x: x['r2'] - x['r2_baseline'],
		r2_improvement_pct=lambda x: 100 * (x['r2'] - x['r2_baseline']) / x['r2_baseline']
	)

	# Create one dataframe
	scores = pd.merge(
		dn_scores[['pheno', 'r2_improvement', 'r2_improvement_pct']],
		prs_scores[['pheno', 'r2_improvement', 'r2_improvement_pct']],
		on='pheno',
		suffixes=('_null', '_prs')
	)

	# plot scatter of r^2 improvement
	fig, ax = plt.subplots()

	use_log_scale = True
	use_percent_improvement = False

	# Choose the data columns
	if use_percent_improvement:
		x_col = 'r2_improvement_pct_null'
		y_col = 'r2_improvement_pct_prs'
	else:
		x_col = 'r2_improvement_null'
		y_col = 'r2_improvement_prs'


	# Plot the scatter
	sns.scatterplot(
		data=scores,
		x=x_col,
		y=y_col,
		hue='pheno',
		palette='tab20',
		ax=ax
	)
	sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

	# Equal aspect ratio
	ax.set_aspect('equal', adjustable='box')

	# Optional: log scale
	if use_log_scale:
		ax.set_xscale('log')
		ax.set_yscale('log')

	# Get combined range for diagonal line
	x_vals = scores[x_col]
	y_vals = scores[y_col]
	min_val = min(x_vals.min(), y_vals.min())
	max_val = max(x_vals.max(), y_vals.max())

	# For log scale, avoid <= 0
	if use_log_scale:
		min_val = max(min_val, 1e-10)

	# Add diagonal line
	ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='gray')

	plt.xlabel(x_col)
	plt.ylabel(y_col)
	plt.title('Scatter of R² Improvement')

	plt.tight_layout()
	plt.show()

	# print those where null improvement is greater than PRS improvement
	print("Phenotypes where null model improvement is greater than PRS improvement:")
	for _, row in scores.iterrows():
		if row[x_col] > row[y_col]:
			print(f"{row['pheno']}: Null = {row[x_col]}, PRS = {row[y_col]}")

	# Get correlation and p-value
	corr, p_value = ss.pearsonr(scores[x_col], scores[y_col])
	print(f"Pearson correlation: {corr:.4f}, p-value: {p_value:.4e}")