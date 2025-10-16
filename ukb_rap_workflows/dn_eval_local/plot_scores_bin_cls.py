import os
import json

import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns


def run_and_plot_rank_tests(scores_df, block_col, group_col, metric):
		# Run omnibus test
		pivot_df = scores_df.pivot_table(
			index=block_col,
			columns=group_col,
			values=metric
		)

		# Apply Friedman test
		stat, p_value = ss.friedmanchisquare(
			*pivot_df.values.T,
			nan_policy='omit',
		)

		print(f"\tp-value: {p_value}")

		# Post-hoc
		test_results = sp.posthoc_nemenyi_friedman(
			scores_df,
			melted=True,
			block_col=block_col,
			block_id_col=block_col,
			group_col=group_col,
			y_col=metric
		)

		# Critical difference diagram
		avg_rank = scores_df.groupby(block_col)[metric].rank(
			pct=True, ascending=True
		).groupby(scores_df[group_col]).mean()

		plt.title(f'Critical difference diagram of average {metric} ranks')
		sp.critical_difference_diagram(avg_rank, test_results)
		plt.show()


if __name__ == '__main__':

	scores_dir = 'scores_bin_cls'
	exclude_w_pc = False
	
	# Load scores for test all, wb, and nwb sets
	scores_df = pd.read_csv(f'{scores_dir}/scores.csv')

	scores_df = scores_df.dropna()

	# Optionally filter out PC inclusion
	if exclude_w_pc:
		scores_df = scores_df[scores_df.pc == False]

	# Add columns for comparison across all phenotypes
	scores_df['pheno-covar'] = scores_df.pheno + '-' + scores_df.covar_set
	scores_df['pheno-model'] = scores_df.pheno + '-' + scores_df.model_type

	# Plotting
	sns.set_style('whitegrid')

	# Options
	metric = 'avg_prec'
	# metric = 'f1'
	# metric = 'mae'

	# By model
	block_col = 'pheno'
	group_col = 'covar_set'

	run_and_plot_rank_tests(scores_df, block_col, group_col, metric)

	# Plot barplot for each phenotype using seaborn
	
	sns.catplot(
		data=scores_df,
		y='covar_set',
		x=metric,
		row='pheno',
		hue='covar_set',
		kind='bar',
		height=4,
		aspect=1
	)
	plt.show()

	# Plot all PR curves
	with open(f'{scores_dir}/pr_curves.json', 'r') as f:
		scores = json.load(f)

	# Plot PR curve for each phenotype (first level key) as subplots.
	# There will be one line per covariate set (second level key).
	# Plots are go horizontal.

	fig, axes = plt.subplots(
		nrows=1,
		ncols=len(scores),
		figsize=(8, 6 * len(scores)),
	)
	fig.subplots_adjust(hspace=0.5)

	for i, (pheno, covar_sets) in enumerate(scores.items()):
		# Get the axes for this subplot
		ax = axes[i]

		# Plot each covariate set
		for covar_set, data in covar_sets.items():
			ax.plot(
				data['recall'],
				data['precision'],
				label=covar_set,
				alpha=0.7,
			)

		ax.set_title(f'PR curve for {pheno}')
		ax.set_xlabel('Recall')
		ax.set_ylabel('Precision')
		ax.legend()
		ax.grid(True)
		ax.set_ylim(0, 1)
		ax.set_xlim(0, 1)

	# Save the figure
	fig.savefig(f'{scores_dir}/pr_curves.png', bbox_inches='tight')
	# Show the figure
	plt.show()

