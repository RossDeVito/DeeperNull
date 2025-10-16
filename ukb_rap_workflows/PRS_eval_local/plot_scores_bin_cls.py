import ast
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

	metric = 'average_precision' # or 'roc_auc'
	scores_dir = 'scores'
	
	# Load scores for test all, wb, and nwb sets
	test_scores = pd.read_csv(f'{scores_dir}/test_scores_bin_cls.csv')

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

	# Plot test R^2 scores
	g = sns.catplot(
		data=test_scores,
		col='pheno',
		x='model_desc',
		hue='model_desc',
		y=metric,
		kind='bar',
		margin_titles=True,
		dodge=False,
		height=2.5,
		legend=True,
		legend_out=True,
		sharey=False,	# type: ignore
		col_wrap=4,
		palette=desc_to_color,
	)

	g.set_titles(col_template="{col_name}")
	# g.set_titles(row_template="{row_name}")

	# Remove x-axis tick labels from each subplot
	for ax in g.axes.flat:
		ax.set_xticklabels([])

	# Add minor y-axis grid lines
	for ax in g.axes.flat:
		ax.set_axisbelow(True)
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		ax.grid(which='minor', axis='y', linestyle='-', alpha=0.3)

	# # Rotate x tick labels
	# for ax in g.axes.flat:
	# 	for label in ax.get_xticklabels():
	# 		label.set_rotation(35)
	# 		label.set_ha('right')

	plt.suptitle(f"Test {metric_display} scores")
	plt.subplots_adjust(
		top=0.92,
		bottom=0.1,
		hspace=0.3,
		wspace=0.25
	)

	# g.add_legend()

	plt.show()
	plt.close()

	# comparisons
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
		pct=True,
		ascending=True
	).groupby(test_scores['model_desc']).mean()

	plt.title(f'Critical difference diagram of average {metric_display} ranks')
	sp.critical_difference_diagram(
		avg_rank,
		test_results,
		color_palette=desc_to_color,
	)
	plt.tight_layout()
	plt.show()

	basil_scores = test_scores[test_scores["model_desc"].str.contains("BASIL")]
	pivot_basil = basil_scores.pivot_table(
		index="pheno",
		columns="model_desc",
		values=metric
	)
	stat_basil, p_basil = ss.friedmanchisquare(*pivot_basil.values.T)
	test_results_basil = sp.posthoc_nemenyi_friedman(
		basil_scores,
		melted=True,
		block_col="pheno",
		block_id_col="pheno",
		group_col="model_desc",
		y_col=metric,
	)

	# Plot PR curves
	plot_pr_curves(test_scores, desc_to_color)


	# prscs_scores = test_scores[test_scores["model_desc"].str.contains("PRScs")]
	# pivot_prscs = prscs_scores.pivot_table(
	# 	index="pheno",
	# 	columns="model_desc",
	# 	values=metric
	# )
	# stat_prscs, p_prscs = ss.friedmanchisquare(*pivot_prscs.values.T)
	# test_results_prscs = sp.posthoc_nemenyi_friedman(
	# 	prscs_scores,
	# 	melted=True,
	# 	block_col="pheno",
	# 	block_id_col="pheno",
	# 	group_col="model_desc",
	# 	y_col=metric,
	# )

	# # Create a new color palette for PRScs models
	# prscs_to_basil_color = {
	# 	"PRScs (lin: age, sex, PCs)": desc_to_color["BASIL (lin: age, sex, PCs)"],
	# 	"PRScs (lin: age, sex, locations, PCs)": desc_to_color["BASIL (lin: age, sex, locations, PCs)"],
	# 	"PRScs (lin: age, sex, times, PCs)": desc_to_color["BASIL (lin: age, sex, times, PCs)"],
	# 	"PRScs (lin: age, sex, locations, times, PCs)": desc_to_color["BASIL (lin: age, sex, locations, times, PCs)"],
	# 	"PRScs (lin: age, sex, PCs; XGB null: age, sex)": desc_to_color["BASIL (lin: age, sex, PCs; XGB null: age, sex)"],
	# 	"PRScs (lin: age, sex, locations, PCs; XGB null: age, sex, locations)": desc_to_color["BASIL (lin: age, sex, locations, PCs; XGB null: age, sex, locations)"],
	# 	"PRScs (lin: age, sex, times, PCs; XGB null: age, sex, times)": desc_to_color["BASIL (lin: age, sex, times, PCs; XGB null: age, sex, times)"],
	# 	"PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)": desc_to_color["BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)"],
	# 	"PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)": 
	# 		desc_to_color["BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)"],
	# }

	# # Remove separate plt.show() calls and create subplots
	# fig, (ax_basil, ax_prscs) = plt.subplots(2, 1, figsize=(10, 8))

	# plt.sca(ax_basil)
	# ax_basil.set_title("BASIL", fontsize=14)
	# avg_rank_basil = basil_scores.groupby("pheno")[metric].rank(
	# 	pct=True, ascending=(metric == "r2")
	# ).groupby(basil_scores["model_desc"]).mean()
	# sp.critical_difference_diagram(
	# 	avg_rank_basil,
	# 	test_results_basil,
	# 	color_palette=desc_to_color
	# )

	# plt.sca(ax_prscs)
	# ax_prscs.set_title("PRScs", fontsize=14)
	# avg_rank_prscs = prscs_scores.groupby("pheno")[metric].rank(
	# 	pct=True, ascending=(metric == "r2")
	# ).groupby(prscs_scores["model_desc"]).mean()
	# sp.critical_difference_diagram(
	# 	avg_rank_prscs,
	# 	test_results_prscs,
	# 	color_palette=prscs_to_basil_color
	# )

	# plt.suptitle("Critical Difference Diagrams", fontsize=14)
	# plt.tight_layout(rect=[0, 0, 1, 0.96])
	# plt.show()

	# Additional analysis:
	# 1) Identify best performer per phenotype for BASIL and PRScs
	basil_grp = basil_scores.groupby("pheno")
	basil_best = basil_grp.apply(lambda df: df.loc[df[metric].idxmax()])
	# prscs_grp = prscs_scores.groupby("pheno")
	# prscs_best = prscs_grp.apply(lambda df: df.loc[df[metric].idxmax()])

	# Print best performer counts
	print("\nBASIL best counts:\n", basil_best["model_desc"].value_counts())
	# print("\nPRScs best counts:\n", prscs_best["model_desc"].value_counts())

	# Print best performer per phenotype
	print("\nBASIL best by phenotype:")
	for pheno, row in basil_best.iterrows():
		print(f"  {pheno}: {row['model_desc']}")
	# print("\nPRScs best by phenotype:")
	# for pheno, row in prscs_best.iterrows():
	# 	print(f"  {pheno}: {row['model_desc']}")

	# # 2) Compute difference in R^2 for 'age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time'
	# #    vs other sets for BASIL and PRScs separately
	# basil_ref_desc = "BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)"
	# prscs_ref_desc = "PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)"
	# basil_compare_descs = [
	# 	"BASIL (lin: age, sex, PCs)",
	# 	"BASIL (lin: age, sex, locations, times, PCs)",
	# 	"BASIL (lin: age, sex, PCs; XGB null: age, sex)",
	# 	"BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)",
	# ]
	# prscs_compare_descs = [
	# 	"PRScs (lin: age, sex, PCs)",
	# 	"PRScs (lin: age, sex, locations, times, PCs)",
	# 	"PRScs (lin: age, sex, PCs; XGB null: age, sex)",
	# 	"PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)",
	# ]

	