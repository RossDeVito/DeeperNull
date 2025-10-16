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

	metric = 'r2'
	ci = 95
	scores_dir = 'scores'
	only_BASIL = False
	
	# Load scores for test all, wb, and nwb sets
	test_scores = pd.read_csv(f'{scores_dir}/test_scores.csv')
	test_bs_scores = pd.read_csv(f'{scores_dir}/test_boot_scores.csv')

	# Cast test_bs_scores[metric] to list from literal
	test_bs_scores[metric] = test_bs_scores[metric].apply(
		lambda x: np.array([float(i) for i in x.strip('[]').split(',')])
	)

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

	if only_BASIL:
		test_scores = test_scores[test_scores['model_desc'].str.contains('BASIL')]

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

	# Create separate figure for legend
	fig_legend, ax_legend = plt.subplots(figsize=(4, len(desc_to_color) * 0.3))

	# Hide axes
	ax_legend.axis('off')

	# Create legend handles
	handles = [
		Patch(color=color, label=desc)
		for desc, color in desc_to_color.items()
	]

	# Add the legend to the separate figure
	ax_legend.legend(
		handles=handles,
		loc='center left',
		frameon=False,
		title='Model',
		fontsize='small',
		title_fontsize='medium'
	)

	plt.show()

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
				f"mean={improvements.mean():.2f}%, "
				f"median={improvements.median():.2f}%, "
				f"std={improvements.std():.2f}%"
			)

	compute_diff_stats(basil_scores, basil_ref_desc, basil_compare_descs)
	compute_diff_stats(prscs_scores, prscs_ref_desc, prscs_compare_descs)

	
	# Compare null vs non-null models given the same covariates
	basil_scores_paired = basil_scores[
		~basil_scores["model_desc"].str.contains("XGB null: age, sex, locations, times, PCs")
	]
	basil_scores_paired['covar_set'] = basil_scores_paired[
		"model_desc"
	].str.extract(
		r'lin: (.+?), PCs'
	)
	basil_scores_paired['has_null'] = basil_scores_paired[
		"model_desc"
	].str.contains("XGB null")

	# Filter to only pairs where both null and non-null versions exist
	paired = (
		basil_scores_paired
		.dropna(subset=['covar_set'])  # drop if covar_set couldn't be parsed
		.groupby(['pheno', 'covar_set'])
		.filter(lambda df: df['has_null'].nunique() == 2)
	)

	# Pivot so we can compare side-by-side
	pivot = paired.pivot_table(
		index=['pheno', 'covar_set'],
		columns='has_null',
		values=metric
	).rename(columns={False: 'no_null', True: 'with_null'})

	# Compute improvement (e.g., R^2 increase)
	pivot['improvement'] = (pivot['with_null'] - pivot['no_null'])
	print("\nBASIL no improvement with null models:")
	print(pivot[pivot['improvement'] <= 0])

	# Now for PRScs
	prscs_scores_paired = prscs_scores[
		~prscs_scores["model_desc"].str.contains("XGB null: age, sex, locations, times, PCs")
	]
	prscs_scores_paired['covar_set'] = prscs_scores_paired[
		"model_desc"
	].str.extract(
		r'lin: (.+?), PCs'
	)
	prscs_scores_paired['has_null'] = prscs_scores_paired[
		"model_desc"
	].str.contains("XGB null")
	
	# Filter to only pairs where both null and non-null versions exist
	paired_prscs = (
		prscs_scores_paired
		.dropna(subset=['covar_set'])  # drop if covar_set couldn't be parsed
		.groupby(['pheno', 'covar_set'])
		.filter(lambda df: df['has_null'].nunique() == 2)
	)
	# Pivot so we can compare side-by-side
	pivot_prscs = paired_prscs.pivot_table(
		index=['pheno', 'covar_set'],
		columns='has_null',
		values=metric
	).rename(columns={False: 'no_null', True: 'with_null'})
	# Compute improvement (e.g., R^2 increase)
	pivot_prscs['improvement'] = (pivot_prscs['with_null'] - pivot_prscs['no_null'])
	print("\nPRScs no improvement with null models:")
	print(pivot_prscs[pivot_prscs['improvement'] <= 0])


	# Compare BASIL vs PRScs for the same covariates
	def extract_full_covar(desc):
		match = re.search(r'\((.*?)\)', desc)
		return match.group(1) if match else None

	basil_scores['covar_set'] = basil_scores['model_desc'].apply(extract_full_covar)
	prscs_scores['covar_set'] = prscs_scores['model_desc'].apply(extract_full_covar)

	# Filter to just models with covar_set extracted
	basil_filtered = basil_scores.dropna(subset=['covar_set']).set_index(['pheno', 'covar_set'])[[metric]].rename(columns={metric: 'basil_score'})
	prscs_filtered = prscs_scores.dropna(subset=['covar_set']).set_index(['pheno', 'covar_set'])[[metric]].rename(columns={metric: 'prscs_score'})

	# Join on pheno + covariate set
	joined = basil_filtered.join(prscs_filtered, how='inner')

	# Find where PRScs outperformed BASIL
	outperformed = joined[joined['prscs_score'] > joined['basil_score']].copy()
	outperformed['percent_improvement'] = 100 * (outperformed['prscs_score'] - outperformed['basil_score']) / joined['basil_score']

	# Show results
	print("\nPRScs outperformed BASIL for these phenotype + covariate set combos:")
	print(outperformed.sort_values('percent_improvement', ascending=False))


	# Find where performance was outside 95% CI of standard (lin: age, sex,
	# PCs) approach for BASIL and PRSCS

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

	basil_outside_ci = get_outside_standard_CI(basil_scores, 'BASIL', metric)
	prscs_outside_ci = get_outside_standard_CI(prscs_scores, 'PRScs', metric)

