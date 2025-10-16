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
	plot_BASIL = True
	
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

	if plot_BASIL:
		test_scores = test_scores[test_scores['model_desc'].str.contains('BASIL')]
	else:
		test_scores = test_scores[~test_scores['model_desc'].str.contains('BASIL')]

	test_scores['pheno'] = test_scores.pheno.str.split('_').apply(
		lambda x: ' '.join(x[:-1])
	)

	preferred_covar_order = [
		'age, sex',
		'age, sex, locations',
		'age, sex, times',
		'age, sex, locations, times',
		'age, sex, locations, times, PCs for null',
	]

	present_sets = [c for c in test_scores['covar_set'].dropna().unique()]
	# Keep preferred order, then append any extras we didn’t anticipate
	covar_order = [c for c in preferred_covar_order if c in present_sets] + \
				[c for c in sorted(present_sets) if c not in preferred_covar_order]

	# Map covariate set -> color (reuse your palette choices; pick a distinct color for the new bucket)
	covar_to_color = {
		'age, sex': "#928F8F",
		'age, sex, locations': "#0072B2",
		'age, sex, times': "#EDA200",
		'age, sex, locations, times': "#FF68BB",
		'age, sex, locations, times, PCs for null': "#AE3FC1",
	}

	SPECIAL_NO_NULL_SOURCE = {
		'age, sex, locations, times, PCs for null': 'age, sex, locations, times'
	}

	# Fallback for any unexpected sets
	for c in covar_order:
		if c not in covar_to_color:
			covar_to_color[c] = sns.color_palette("tab20", 20)[hash(c) % 20]

	# Create FacetGrid
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

	# Plot paired bars per covariate set
	group_width = 0.8
	bar_width = group_width / 2.2
	x_positions = np.arange(len(covar_order))

	err_bar_color = "#313030"

	for ax, pheno in zip(g.axes.flat, pheno_order):
		sub = test_scores[test_scores['pheno'] == pheno]

		xs_no  = x_positions - bar_width/2
		xs_yes = x_positions + bar_width/2

		for i, cov in enumerate(covar_order):
			col = covar_to_color[cov]

			if cov in SPECIAL_NO_NULL_SOURCE:
				# Left bar (no XGB) should mirror the no-XGB value from the mapped source bucket
				source_cov = SPECIAL_NO_NULL_SOURCE[cov]
				row_no_src = sub[(sub['covar_set'] == source_cov) & (~sub['with_null'])]
				if not row_no_src.empty:
					y  = float(row_no_src[metric]); el = float(row_no_src['err_low']); eh = float(row_no_src['err_high'])
					# Draw with the SPECIAL bucket's color, but values from source bucket
					ax.bar(xs_no[i], y, width=bar_width, facecolor='none', hatch='////',
						edgecolor=col, linewidth=1.0)
					ax.errorbar(xs_no[i], y, yerr=[[el], [eh]], fmt='none',
								ecolor=err_bar_color, capsize=3, linewidth=1.0)

				# Right bar (with XGB) comes from the special bucket itself
				row_yes = sub[(sub['covar_set'] == cov) & (sub['with_null'])]
				if not row_yes.empty:
					y  = float(row_yes[metric]); el = float(row_yes['err_low']); eh = float(row_yes['err_high'])
					ax.bar(xs_yes[i], y, width=bar_width, color=col, edgecolor=col, linewidth=1.0)
					ax.errorbar(xs_yes[i], y, yerr=[[el], [eh]], fmt='none',
								ecolor=err_bar_color, capsize=3, linewidth=1.0)
				continue

			# Normal paired buckets: left = no XGB null (outline), right = with XGB null (filled)
			row_no  = sub[(sub['covar_set'] == cov) & (~sub['with_null'])]
			row_yes = sub[(sub['covar_set'] == cov) & ( sub['with_null'])]

			if not row_no.empty:
				y  = float(row_no[metric]); el = float(row_no['err_low']); eh = float(row_no['err_high'])
				ax.bar(xs_no[i], y, width=bar_width, facecolor='none', hatch='////',
					edgecolor=col, linewidth=1.0)
				ax.errorbar(xs_no[i], y, yerr=[[el], [eh]], fmt='none',
							ecolor=err_bar_color, capsize=3, linewidth=1.0)

			if not row_yes.empty:
				y  = float(row_yes[metric]); el = float(row_yes['err_low']); eh = float(row_yes['err_high'])
				ax.bar(xs_yes[i], y, width=bar_width, color=col, edgecolor=col, linewidth=1.0)
				ax.errorbar(xs_yes[i], y, yerr=[[el], [eh]], fmt='none',
							ecolor=err_bar_color, capsize=3, linewidth=1.0)
				
		# cosmetics
		ax.set_title(pheno)
		ax.set_xlim(-0.5, len(covar_order)-0.5)
		ax.set_xticks(x_positions)
		ax.set_xticklabels(
			[
				'age+sex' if c == 'age, sex'
				else c.replace('age, sex, ', '')
				for c in covar_order
			],
			rotation=30, ha='right'
		)
		ax.xaxis.grid(False)
		ax.yaxis.grid(False)

	# Global adjustments
	g.figure.subplots_adjust(top=0.92, bottom=0.13, hspace=0.3, wspace=0.25)
	g.figure.suptitle(f"Test {metric_display} scores with {ci}% CI")

	# Legend (colors = covariate sets; fill style = with/without XGB null)
	
	fig_legend, ax_legend = plt.subplots(figsize=(5.7, max(2.7, len(covar_order)*0.28)))
	ax_legend.axis('off')

	color_handles = [Patch(facecolor=covar_to_color[c], edgecolor=covar_to_color[c], label=c) for c in covar_order]
	style_handles = [
		Patch(facecolor='white', edgecolor='black', hatch='////', linewidth=1.0, label='No XGB null'),
		Patch(facecolor='black', edgecolor='black', label='With XGB null'),
	]

	leg1 = ax_legend.legend(handles=color_handles, title='Covariate set', loc='center left', bbox_to_anchor=(0.0, 0.5), frameon=False)
	ax_legend.add_artist(leg1)
	ax_legend.legend(handles=style_handles, title='Bar style', loc='center left', bbox_to_anchor=(0.58, 0.5), frameon=False)

	plt.show()

