import os

import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator


desc_to_color = {
	"BASIL (lin: age, sex, PCs)": "black",
	"PRScs (lin: age, sex, PCs)": "#666666",  # slightly lighter than black
	"BASIL (lin: age, sex, locations, PCs)": sns.color_palette("tab20", 18)[0],
	"PRScs (lin: age, sex, locations, PCs)": sns.color_palette("tab20", 18)[1],
	"BASIL (lin: age, sex, times, PCs)": sns.color_palette("tab20", 18)[2],
	"PRScs (lin: age, sex, times, PCs)": sns.color_palette("tab20", 18)[3],
	"BASIL (lin: age, sex, locations, times, PCs)": sns.color_palette("tab20", 18)[4],
	"PRScs (lin: age, sex, locations, times, PCs)": sns.color_palette("tab20", 18)[5],
	"BASIL (lin: age, sex, PCs; XGB null: age, sex)": sns.color_palette("tab20", 18)[10],
	"PRScs (lin: age, sex, PCs; XGB null: age, sex)": sns.color_palette("tab20", 18)[11],
	"BASIL (lin: age, sex, locations, PCs; XGB null: age, sex, locations)": sns.color_palette("tab20", 18)[8],
	"PRScs (lin: age, sex, locations, PCs; XGB null: age, sex, locations)": sns.color_palette("tab20", 18)[9],
	"BASIL (lin: age, sex, times, PCs; XGB null: age, sex, times)": sns.color_palette("tab20", 18)[6],
	"PRScs (lin: age, sex, times, PCs; XGB null: age, sex, times)": sns.color_palette("tab20", 18)[7],
	"BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)": sns.color_palette("tab20", 18)[16],
	"PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)": sns.color_palette("tab20", 18)[17],
	"BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)": sns.color_palette("tab20", 18)[12],
	"PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)": sns.color_palette("tab20", 18)[13],
}


def plot_pr_curves(df, desc_to_color):
	phenos = df['pheno'].unique()
	num_phenos = len(phenos)

	cols = 4
	rows = int(np.ceil(num_phenos / cols))

	fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
	axes = axes.flatten()

	handles_labels = {}

	for i, pheno in enumerate(phenos):
		ax = axes[i]

		pr_data = df[df['pheno'] == pheno]

		for _, row in pr_data.iterrows():
			model_desc = row['model_desc']

			# Safely parse strings to lists if needed
			precision = row['pr_precision']
			recall = row['pr_recall']

			if isinstance(precision, str):
				precision = ast.literal_eval(precision)
			if isinstance(recall, str):
				recall = ast.literal_eval(recall)

			line, = ax.step(
				recall,
				precision,
				where='post',
				label=model_desc,
				color=desc_to_color.get(model_desc, 'grey'),
				alpha=0.8,  # Adjust transparency
				marker=None,  # Add markers to the line
				linewidth=2,  # Adjust line width
			)

			if model_desc not in handles_labels:
				handles_labels[model_desc] = line

		ax.set_title(pheno)
		ax.set_xlabel('Recall')
		ax.set_ylabel('Precision')
		ax.set_xlim([0, 1])
		ax.set_ylim([0, 1])
		ax.grid(which='major', linestyle='-', linewidth=0.5, alpha=0.7)
		ax.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.3)

		ax.xaxis.set_minor_locator(AutoMinorLocator())
		ax.yaxis.set_minor_locator(AutoMinorLocator())

		ax.set_aspect('equal', adjustable='box')

	# Hide unused axes
	for j in range(i + 1, len(axes)):
		fig.delaxes(axes[j])

	# Single unified legend to the right
	plt.tight_layout(rect=[0, 0, 0.85, 0.97])
	fig.legend(
		handles_labels.values(), handles_labels.keys(),
		loc='center left', bbox_to_anchor=(0.68, 0.5),
		fontsize='small', ncol=1
	)
	fig.suptitle('Precision-Recall Curves by Phenotype', fontsize=16)
	
	plt.show()


if __name__ == '__main__':

	metric = 'average_precision'
	# metric = 'roc_auc'
	ci = 95
	scores_dir = 'scores'
	
	# Load scores for test all, wb, and nwb sets
	test_scores = pd.read_csv(f'{scores_dir}/test_scores_bin_cls.csv')
	test_bs_scores = pd.read_csv(f'{scores_dir}/test_boot_scores_bin_cls.csv')

	# Cast test_bs_scores[metric] to list from literal
	test_bs_scores[metric] = test_bs_scores[metric].apply(
		lambda x: np.array([float(i) for i in x.strip('[]').split(',')])
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
		metric_display = metric.replace('_', ' ').title()

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

	g = sns.FacetGrid(
		data=test_scores,
		col="pheno",
		col_order=pheno_order,
		margin_titles=True,
		col_wrap=4,
		sharey=False,
		height=2.5
	)

	for ax, pheno in zip(g.axes.flat, pheno_order):
		group = test_scores[test_scores['pheno'] == pheno]
		
		x = range(len(group))
		y = group[metric].values
		err_low = group['err_low'].values
		err_high = group['err_high'].values
		model_labels = group['model_desc'].values
		colors = [desc_to_color[desc] for desc in model_labels]

		bars = ax.bar(x, y, color=colors, tick_label=[''] * len(x))
		ax.errorbar(
			x, y,
			yerr=[err_low, err_high],
			fmt='none',
			ecolor='black',
			capsize=3,
			linewidth=1
		)

		ax.set_title(pheno)

	# Global adjustments
	g.figure.subplots_adjust(
		top=0.92,
		bottom=0.1,
		hspace=0.3,
		wspace=0.25
	)
	g.figure.suptitle(f"Test {metric_display} scores with {ci}% CI")

	plt.show()

	# Find where performance was outside 95% CI of standard (lin: age, sex, PCs)

	def get_outside_standard_CI(df, method_str, metric):
		"""For each pheno, get standard approach's upper 95% CI bound and
		compare other models against it."""

		baseline_df = df[df['model_desc'] == f'{method_str} (lin: age, sex, PCs)']

		assert len(baseline_df) == len(df.pheno.unique()), \
			"Expected one baseline model per phenotype."
		
		pheno_to_upper_ci = baseline_df.set_index('pheno')['upper'].to_dict()

		df.loc[:, 'baseline_upper_ci'] = df['pheno'].map(pheno_to_upper_ci)

		df = df[df[metric] > df['baseline_upper_ci']]

		print(f"{len(df['pheno'].value_counts())} phenotypes with {method_str} models outside 95% CI of standard approach:")
		print(df['pheno'].value_counts())

		df_null = df[df['model_desc'].str.contains('null')]
		df_non_null = df[~df['model_desc'].str.contains('null')]

		print(f"\n{len(df_null['model_desc'].value_counts())} {method_str} models outside 95% CI of standard approach with null:")
		print(df_null['model_desc'].value_counts())
		print(df_null['pheno'].value_counts())
		print(f"\n{len(df_non_null['model_desc'].value_counts())} {method_str} models outside 95% CI of standard approach without null:")
		print(df_non_null['model_desc'].value_counts())
		print(df_non_null['pheno'].value_counts())

		# Print where just adding null without changing covariates
		# results in outside CI
		null_only = df_null[
			df_null['model_desc'] == f'{method_str} (lin: age, sex, PCs; XGB null: age, sex)'
		]
		if not null_only.empty:
			print(f"\n{method_str} models outside 95% CI of standard approach with null only:")
			print(null_only[['pheno', 'model_desc', metric]])
		else:
			print(f"\nNo {method_str} models outside 95% CI of standard approach with null only.")


		return df
	
	outside_ci_df = get_outside_standard_CI(test_scores, 'BASIL', metric)