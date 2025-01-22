import os

import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

# Map each model_desc to an index in tab20 so BASIL & PRScs share paired colors
desc_to_color = {
	"BASIL (lin: age, sex, PCs)": sns.color_palette("tab20", 18)[0],
	"PRScs (lin: age, sex, PCs)": sns.color_palette("tab20", 18)[1],
	"BASIL (lin: age, sex, locations, PCs)": sns.color_palette("tab20", 18)[2],
	"PRScs (lin: age, sex, locations, PCs)": sns.color_palette("tab20", 18)[3],
	"BASIL (lin: age, sex, times, PCs)": sns.color_palette("tab20", 18)[4],
	"PRScs (lin: age, sex, times, PCs)": sns.color_palette("tab20", 18)[5],
	"BASIL (lin: age, sex, locations, times, PCs)": sns.color_palette("tab20", 18)[6],
	"PRScs (lin: age, sex, locations, times, PCs)": sns.color_palette("tab20", 18)[7],
	"BASIL (lin: age, sex, PCs; XGB null: age, sex)": sns.color_palette("tab20", 18)[8],
	"PRScs (lin: age, sex, PCs; XGB null: age, sex)": sns.color_palette("tab20", 18)[9],
	"BASIL (lin: age, sex, locations, PCs; XGB null: age, sex, locations)": sns.color_palette("tab20", 18)[10],
	"PRScs (lin: age, sex, locations, PCs; XGB null: age, sex, locations)": sns.color_palette("tab20", 18)[11],
	"BASIL (lin: age, sex, times, PCs; XGB null: age, sex, times)": sns.color_palette("tab20", 18)[12],
	"PRScs (lin: age, sex, times, PCs; XGB null: age, sex, times)": sns.color_palette("tab20", 18)[13],
	"BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)": sns.color_palette("tab20", 18)[14],
	"PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)": sns.color_palette("tab20", 18)[15],
	"BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)": sns.color_palette("tab20", 18)[16],
	"PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)": sns.color_palette("tab20", 18)[17],
	
}

if __name__ == '__main__':

	metric = 'r2'
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
		ax.yaxis.set_minor_locator(AutoMinorLocator())
		ax.grid(which='minor', axis='y', linestyle='-', alpha=0.3)

	# # Rotate x tick labels
	# for ax in g.axes.flat:
	# 	for label in ax.get_xticklabels():
	# 		label.set_rotation(35)
	# 		label.set_ha('right')

	plt.suptitle('Test R^2 scores')
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
		pct=True, ascending=(metric == 'r2')
	).groupby(test_scores['model_desc']).mean()

	plt.title(f'Critical difference diagram of average R^2 ranks')
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

	prscs_scores = test_scores[test_scores["model_desc"].str.contains("PRScs")]
	pivot_prscs = prscs_scores.pivot_table(
		index="pheno",
		columns="model_desc",
		values=metric
	)
	stat_prscs, p_prscs = ss.friedmanchisquare(*pivot_prscs.values.T)
	test_results_prscs = sp.posthoc_nemenyi_friedman(
		prscs_scores,
		melted=True,
		block_col="pheno",
		block_id_col="pheno",
		group_col="model_desc",
		y_col=metric,
	)

	# Create a new color palette for PRScs models
	prscs_to_basil_color = {
		"PRScs (lin: age, sex, PCs)": desc_to_color["BASIL (lin: age, sex, PCs)"],
		"PRScs (lin: age, sex, locations, PCs)": desc_to_color["BASIL (lin: age, sex, locations, PCs)"],
		"PRScs (lin: age, sex, times, PCs)": desc_to_color["BASIL (lin: age, sex, times, PCs)"],
		"PRScs (lin: age, sex, locations, times, PCs)": desc_to_color["BASIL (lin: age, sex, locations, times, PCs)"],
		"PRScs (lin: age, sex, PCs; XGB null: age, sex)": desc_to_color["BASIL (lin: age, sex, PCs; XGB null: age, sex)"],
		"PRScs (lin: age, sex, locations, PCs; XGB null: age, sex, locations)": desc_to_color["BASIL (lin: age, sex, locations, PCs; XGB null: age, sex, locations)"],
		"PRScs (lin: age, sex, times, PCs; XGB null: age, sex, times)": desc_to_color["BASIL (lin: age, sex, times, PCs; XGB null: age, sex, times)"],
		"PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)": desc_to_color["BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)"],
		"PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)": 
			desc_to_color["BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)"],
	}

	# Remove separate plt.show() calls and create subplots
	fig, (ax_basil, ax_prscs) = plt.subplots(2, 1, figsize=(10, 8))

	plt.sca(ax_basil)
	ax_basil.set_title("BASIL", fontsize=14)
	avg_rank_basil = basil_scores.groupby("pheno")[metric].rank(
		pct=True, ascending=(metric == "r2")
	).groupby(basil_scores["model_desc"]).mean()
	sp.critical_difference_diagram(
		avg_rank_basil,
		test_results_basil,
		color_palette=desc_to_color
	)

	plt.sca(ax_prscs)
	ax_prscs.set_title("PRScs", fontsize=14)
	avg_rank_prscs = prscs_scores.groupby("pheno")[metric].rank(
		pct=True, ascending=(metric == "r2")
	).groupby(prscs_scores["model_desc"]).mean()
	sp.critical_difference_diagram(
		avg_rank_prscs,
		test_results_prscs,
		color_palette=prscs_to_basil_color
	)

	plt.suptitle("Critical Difference Diagrams", fontsize=14)
	plt.tight_layout(rect=[0, 0, 1, 0.96])
	plt.show()