import os

import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

	scores_dir = 'scores'

	# Load scores
	scores_df = pd.read_csv(f'{scores_dir}/mt_scores.csv')

	# Create 'desc' column that is a combination of 'model_type' and 'covar_set'
	scores_df['desc'] = scores_df['model_type'] + ' | ' + scores_df['covar_set']

	# Plot scores
	metric = 'r2'

	sns.set_style('whitegrid')
	g = sns.catplot(
		data=scores_df,
		col='pheno',
		x='desc',
		hue='desc',
		y=metric,
		kind='bar',
		margin_titles=True,
		dodge=False,
		height=2.5,
		legend=True,
		legend_out=True,
		sharey=False,	# type: ignore
		col_wrap=4,
		palette='Paired',
	)

	g.set_xticklabels(rotation=90)

	plt.show()
	plt.close()

	# Comparisons
	pivot_df = scores_df.pivot_table(
		index='pheno',
		columns='desc',
		values=metric,
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
		block_col='pheno',
		block_id_col='pheno',
		group_col='desc',
		y_col=metric,
	)

	# Plot
	avg_rank = scores_df.groupby('pheno')[metric].rank(
		pct=True, ascending=('r2' in metric)
	).groupby(scores_df['desc']).mean()

	plt.title(f'Critical difference diagram of average {metric} ranks')
	sp.critical_difference_diagram(avg_rank, test_results)
	plt.show()
	