import os

import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

# Map each model_desc to an index in tab20 so BASIL & PRScs share paired colors
# desc_to_color = {
# 	"BASIL (lin: age, sex, PCs)": sns.color_palette("tab20", 18)[0],
# 	"PRScs (lin: age, sex, PCs)": sns.color_palette("tab20", 18)[1],
# 	"BASIL (lin: age, sex, locations, PCs)": sns.color_palette("tab20", 18)[2],
# 	"PRScs (lin: age, sex, locations, PCs)": sns.color_palette("tab20", 18)[3],
# 	"BASIL (lin: age, sex, times, PCs)": sns.color_palette("tab20", 18)[4],
# 	"PRScs (lin: age, sex, times, PCs)": sns.color_palette("tab20", 18)[5],
# 	"BASIL (lin: age, sex, locations, times, PCs)": sns.color_palette("tab20", 18)[6],
# 	"PRScs (lin: age, sex, locations, times, PCs)": sns.color_palette("tab20", 18)[7],
# 	"BASIL (lin: age, sex, PCs; XGB null: age, sex)": sns.color_palette("tab20", 18)[8],
# 	"PRScs (lin: age, sex, PCs; XGB null: age, sex)": sns.color_palette("tab20", 18)[9],
# 	"BASIL (lin: age, sex, locations, PCs; XGB null: age, sex, locations)": sns.color_palette("tab20", 18)[10],
# 	"PRScs (lin: age, sex, locations, PCs; XGB null: age, sex, locations)": sns.color_palette("tab20", 18)[11],
# 	"BASIL (lin: age, sex, times, PCs; XGB null: age, sex, times)": sns.color_palette("tab20", 18)[12],
# 	"PRScs (lin: age, sex, times, PCs; XGB null: age, sex, times)": sns.color_palette("tab20", 18)[13],
# 	"BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)": sns.color_palette("tab20", 18)[14],
# 	"PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)": sns.color_palette("tab20", 18)[15],
# 	"BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)": sns.color_palette("tab20", 18)[16],
# 	"PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)": sns.color_palette("tab20", 18)[17],
# }

# Like above, but (lin: age, sex, PCs) should use black and a slightly lighter black for PRScs
# Otherwise colors shift down by 1
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

if __name__ == '__main__':

	metric = 'r2'  # or 'r2'
	scores_dir = 'scores'
	
	# Load scores for test all, wb, and nwb sets
	test_scores = pd.read_csv(f'{scores_dir}/test_scores.csv')

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
		ascending=(metric == 'r2')
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

	# Additional analysis:
	# 1) Identify best performer per phenotype for BASIL and PRScs
	basil_grp = basil_scores.groupby("pheno")
	basil_best = basil_grp.apply(lambda df: df.loc[df[metric].idxmax()])
	prscs_grp = prscs_scores.groupby("pheno")
	prscs_best = prscs_grp.apply(lambda df: df.loc[df[metric].idxmax()])

	# Print best performer counts
	print("\nBASIL best counts:\n", basil_best["model_desc"].value_counts())
	print("\nPRScs best counts:\n", prscs_best["model_desc"].value_counts())

	# Print best performer per phenotype
	print("\nBASIL best by phenotype:")
	for pheno, row in basil_best.iterrows():
		print(f"  {pheno}: {row['model_desc']}")
	print("\nPRScs best by phenotype:")
	for pheno, row in prscs_best.iterrows():
		print(f"  {pheno}: {row['model_desc']}")

	# 2) Compute difference in R^2 for 'age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time'
	#    vs other sets for BASIL and PRScs separately
	basil_ref_desc = "BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)"
	prscs_ref_desc = "PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)"
	basil_compare_descs = [
		"BASIL (lin: age, sex, PCs)",
		"BASIL (lin: age, sex, locations, times, PCs)",
		"BASIL (lin: age, sex, PCs; XGB null: age, sex)",
		"BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)",
	]
	prscs_compare_descs = [
		"PRScs (lin: age, sex, PCs)",
		"PRScs (lin: age, sex, locations, times, PCs)",
		"PRScs (lin: age, sex, PCs; XGB null: age, sex)",
		"PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)",
	]

	def compute_diff_stats(df, ref_desc, compare_descs):
		ref = df[df["model_desc"] == ref_desc].set_index("pheno")[metric]
		for desc in compare_descs:
			compare = df[df["model_desc"] == desc].set_index("pheno")[metric]
			common = ref.index.intersection(compare.index)
			if metric == 'r2':
				improvements = 100.0 * (ref.loc[common] - compare.loc[common]) / compare.loc[common]
			else:
				improvements = 100.0 * (compare.loc[common] - ref.loc[common]) / ref.loc[common]
			print(
				f"\n{ref_desc} vs {desc} ({metric}): "
				f"mean improvement={improvements.mean():.2f}%, std={improvements.std():.2f}%"
			)

	compute_diff_stats(basil_scores, basil_ref_desc, basil_compare_descs)
	compute_diff_stats(prscs_scores, prscs_ref_desc, prscs_compare_descs)

	# Collect improvements for plotting
	improvements_data = []

	def gather_improvements(df, ref_desc, compare_descs, method):
		ref = df[df["model_desc"] == ref_desc].set_index("pheno")[metric]
		for desc in compare_descs:
			compare = df[df["model_desc"] == desc].set_index("pheno")[metric]
			common = ref.index.intersection(compare.index)
			if metric == 'r2':
				improvements = 100.0 * (ref.loc[common] - compare.loc[common]) / compare.loc[common]
			else:
				improvements = 100.0 * (compare.loc[common] - ref.loc[common]) / ref.loc[common]
			for pheno in common:
				improvements_data.append({
					"method": method,
					"ref_desc": ref_desc,
					"desc": desc,
					"pheno": pheno,
					"percent improvement": improvements.loc[pheno],
				})

	gather_improvements(basil_scores, basil_ref_desc, basil_compare_descs, "BASIL")
	gather_improvements(prscs_scores, prscs_ref_desc, prscs_compare_descs, "PRScs")

	improvements_df = pd.DataFrame(improvements_data)

	# Add covar desc (the part in parentheses in desc) as column
	improvements_df["covar_desc"] = improvements_df["desc"].str.extract(r"\((.*?)\)")

	# Format covar_desc for better readability before plotting
	def format_label(text):
		words = text.split()
		lines = []
		current_line = []
		current_length = 0
		
		for word in words:
			if current_length + len(word) > 20:  # line length threshold
				lines.append(' '.join(current_line))
				current_line = [word]
				current_length = len(word)
			else:
				current_line.append(word)
				current_length += len(word) + 1  # +1 for space
		
		if current_line:
			lines.append(' '.join(current_line))
		
		return '\n'.join(lines)

	improvements_df['covar_desc'] = improvements_df['covar_desc'].apply(format_label)

	# # Create box plot of improvements
	# # plt.figure(figsize=(8, 8))

	# g = sns.catplot(
	# 	data=improvements_df,
	# 	x="covar_desc",
	# 	y="percent improvement",
	# 	col='method',
	# 	hue="desc",
	# 	kind="box",
	# 	palette=desc_to_color,
	# 	sharey=True,
	# 	height=8,
	# 	aspect=0.8,
	# 	showfliers=False,
	# 	legend=False,
	# )

	# # Add minor gridlines
	# for ax_ in g.axes.flat:
	# 	ax_.yaxis.set_minor_locator(AutoMinorLocator())
	# 	ax_.grid(which='minor', axis='y', linestyle='-', alpha=0.3)

	# # Add title
	# plt.suptitle(f"Percent Improvement in {metric_display} when using all covs. and an XGB null model w/o PCs", fontsize=14)

	# plt.tight_layout(rect=[0.01, 0, 1, 0.96])
	# plt.show()

	# # Bar plot of improvements by method
	# ax = sns.catplot(
	# 	data=improvements_df,
	# 	x="covar_desc",
	# 	y="percent improvement",
	# 	col='method',
	# 	hue="desc",
	# 	kind="bar",
	# 	estimator='median', 
	# 	palette=desc_to_color,
	# 	sharey=True,
	# 	height=6,
	# 	aspect=1,
	# 	errorbar=('ci', 95),
	# 	legend=False,
	# )

	# # Add minor gridlines
	# for c_ax in ax.axes.flat:
	# 	c_ax.yaxis.set_minor_locator(AutoMinorLocator())
	# 	c_ax.grid(which='minor', axis='y', linestyle='-', alpha=0.3)
	
	# # do ax.bar_label() for two subplots of ax
	# for c_ax in ax.axes.flat:
	# 	for bar in c_ax.containers:
	# 		# c_ax.bar_label(bar, fmt='%.2f', label_type='edge')
	# 		# Offset label to the right to avoid ci line
	# 		c_ax.bar_label(bar, fmt='           %.2f', label_type='edge')

	# # Add title
	# plt.suptitle(f"Median percent improvement in {metric_display} (ci=95%)", fontsize=14)

	# plt.tight_layout(rect=[0.01, 0, 1.0, 0.96])
	# plt.show()