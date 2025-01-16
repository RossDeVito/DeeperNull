import os

import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

	scores_dir = 'scores'
	
	# Load scores for test all, wb, and nwb sets
	test_scores = pd.read_csv(f'{scores_dir}/test_scores.csv')

	sns.set_style('whitegrid')

	# Plot test R^2 scores
	g = sns.catplot(
		data=test_scores,
		col='pheno',
		x='model_desc',
		hue='model_desc',
		y='r2',
		kind='bar',
		margin_titles=True,
		dodge=False,
		height=2.5,
		legend=True,
		legend_out=True,
		sharey=False,	# type: ignore
		col_wrap=4,
	)

	g.set_titles(col_template="{col_name}")
	# g.set_titles(row_template="{row_name}")

	# Remove x-axis tick labels from each subplot
	for ax in g.axes.flat:
		ax.set_xticklabels([])

	# # Rotate x tick labels
	# for ax in g.axes.flat:
	# 	for label in ax.get_xticklabels():
	# 		label.set_rotation(35)
	# 		label.set_ha('right')

	plt.suptitle('Test R^2 scores')
	plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)

	# g.add_legend()

	plt.show()
	plt.close()

	# # Plot test pearson_r
	# g = sns.catplot(
	# 	data=test_scores,
	# 	col='pheno',
	# 	x='model_desc',
	# 	hue='model_desc',
	# 	y='pearson_r',
	# 	kind='bar',
	# 	margin_titles=True,
	# 	dodge=False,
	# 	height=2.5,
	# 	legend=True,
	# 	legend_out=True,
	# 	sharey=False,	# type: ignore
	# 	col_wrap=4,
	# )

	# g.set_titles(col_template="{col_name}")

	# # Remove x-axis tick labels from each subplot
	# for ax in g.axes.flat:
	# 	ax.set_xticklabels([])

	# plt.suptitle('Test Pearson correlation')
	# plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.25)

	# plt.show()
	# plt.close()


	# comparisons
	metric = 'r2'

	pivot_df = test_scores.pivot_table(
		index='pheno',
		columns='model_desc',
		values=metric
	)

	# Apply Friedman test
	stat, p_value = ss.friedmanchisquare(*pivot_df.values.T)
	print(f"\tp-value: {p_value}")

	# Post-hoc
	test_results = sp.posthoc_nemenyi_friedman(
		test_scores,
		melted=True,
		block_col='pheno',
		block_id_col='pheno',
		group_col='model_desc',
		y_col=metric,
	)

	# plot
	avg_rank = test_scores.groupby('pheno')[metric].rank(
		pct=True, ascending=(metric == 'r2')
	).groupby(test_scores['model_desc']).mean()

	plt.title(f'Critical difference diagram of average R^2 ranks')
	sp.critical_difference_diagram(
		avg_rank,
		test_results,
		color_palette={
			"BASIL (lin: age, sex, PCs)": '#1f77b4',
			"BASIL (lin: age, sex, locations, PCs)": '#ff7f0e',
			"BASIL (lin: age, sex, PCs; XGB null: age, sex)": '#2ca02c',
			"BASIL (lin: age, sex, locations, PCs; XGB null: age, sex, locations)": '#d62728',
		}
	)
	plt.tight_layout()
	plt.show()