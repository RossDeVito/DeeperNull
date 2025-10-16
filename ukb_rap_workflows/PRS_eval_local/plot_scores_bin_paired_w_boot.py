import os
import re
import ast

import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Patch

# ---------- CONFIG ----------
# Metric/IO are kept for binary classification files
metric = 'average_precision'   # or 'roc_auc'
ci = 95
scores_dir = 'scores'
plot_BASIL = True             # False => plot PRScs; True => plot BASIL
# ----------------------------

sns.set_style('whitegrid')
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 1200

def _metric_display(m):
	if m.lower() == 'r2':
		return 'R^2'
	elif m.lower() == 'mse':
		return 'MSE'
	elif m.lower() == 'mae':
		return 'MAE'
	else:
		return m.replace('_', ' ').upper()

def extract_covar_set(desc: str):
	"""
	Parse `model_desc` strings like:
	 1) 'PRScs (lin: age, sex, PCs)'
	 2) 'PRScs (lin: age, sex, locations, PCs; XGB null: age, sex, locations)'
	We return the linear covariate subset (without 'PCs') to use as the x-axis bucket.
	Special case: if the 'XGB null:' clause contains 'PCs', use a special unpaired bucket.
	"""
	# Base covariate set from the linear part (strip trailing ", PCs")
	m = re.search(r'lin:\s*(.+?)(?:,\s*PCs|\))', desc)
	base = m.group(1).strip() if m else None  # e.g., 'age, sex', 'age, sex, locations', 'age, sex, times', ...

	# If the null clause has PCs, treat as special unpaired bucket
	if 'XGB null:' in desc:
		null_clause = desc.split('XGB null:')[1]
		if re.search(r'\bPCs\b', null_clause):
			return 'age, sex, locations, times, PCs for null'

	return base

def has_xgb_null(desc: str) -> bool:
	return 'XGB null' in desc

def main():
	# Load scores for test set and bootstrap
	test_scores = pd.read_csv(f'{scores_dir}/test_scores_bin_cls.csv')
	test_bs_scores = pd.read_csv(f'{scores_dir}/test_boot_scores_bin_cls.csv')

	# Cast bootstrap column to arrays
	test_bs_scores[metric] = test_bs_scores[metric].apply(
		lambda x: np.array([float(i) for i in x.strip('[]').split(',')])
	)

	# Compute CI bounds
	alpha = 100 - ci
	lower_q = alpha / 2
	upper_q = 100 - (alpha / 2)

	test_bs_scores['lower'] = test_bs_scores[metric].apply(
		lambda x: np.percentile(x, lower_q)
	)
	test_bs_scores['upper'] = test_bs_scores[metric].apply(
		lambda x: np.percentile(x, upper_q)
	)

	# Merge CI bounds back
	test_scores = test_scores.merge(
		test_bs_scores[['model_desc', 'pheno', 'lower', 'upper']],
		on=['model_desc', 'pheno'],
		how='left'
	)

	# Compute error ranges
	test_scores['err_low'] = test_scores[metric] - test_scores['lower']
	test_scores['err_high'] = test_scores['upper'] - test_scores[metric]

	# Parse model_desc -> covariate set + with/without XGB null
	test_scores['covar_set'] = test_scores['model_desc'].apply(extract_covar_set)
	test_scores['with_null']  = test_scores['model_desc'].apply(has_xgb_null)

	# Optional: select BASIL vs PRScs (paired layout assumes one algorithm at a time)
	if plot_BASIL:
		test_scores = test_scores[test_scores['model_desc'].str.contains('BASIL', na=False)]
	else:
		test_scores = test_scores[test_scores['model_desc'].str.contains('PRScs', na=False)]

	# Safety: drop rows without parsed covariate set
	test_scores = test_scores.dropna(subset=['covar_set']).copy()

	# Order phenotypes alphabetically (case-insensitive)
	pheno_order = sorted(test_scores['pheno'].unique(), key=lambda x: str(x).lower())

	# Preferred x-axis order (plus special unpaired bucket)
	preferred_covar_order = [
		'age, sex',
		'age, sex, locations',
		'age, sex, times',
		'age, sex, locations, times',
		'age, sex, locations, times, PCs for null',   # special unpaired
	]

	present_sets = list(test_scores['covar_set'].dropna().unique())
	covar_order = [c for c in preferred_covar_order if c in present_sets] + \
	              [c for c in sorted(present_sets) if c not in preferred_covar_order]

	# Color by covariate set
	covar_to_color = {
		'age, sex': "#928F8F",
		'age, sex, locations': "#0072B2",
		'age, sex, times': "#EDA200",
		'age, sex, locations, times': "#FF68BB",
		'age, sex, locations, times, PCs for null': "#AE3FC1",
	}
	for c in covar_order:
		if c not in covar_to_color:
			# deterministic fallback
			covar_to_color[c] = sns.color_palette("tab20", 20)[hash(c) % 20]

	# >>> NEW: Define where to pull the "no XGB" bar for special/unpaired buckets
	SPECIAL_NO_NULL_SOURCE = {
		'age, sex, locations, times, PCs for null': 'age, sex, locations, times'
	}

	# --------- Plotting ----------
	g = sns.FacetGrid(
		data=test_scores,
		col="pheno",
		col_order=pheno_order,
		margin_titles=True,
		col_wrap=4,
		sharey=False,
		height=2.5
	)

	group_width = 0.8
	bar_width = group_width / 2.2
	x_positions = np.arange(len(covar_order))
	err_bar_color = "#313030"

	for ax, pheno in zip(g.axes.flat, pheno_order):
		sub = test_scores[test_scores['pheno'] == pheno]

		xs_no  = x_positions - bar_width/2   # left bar = no XGB null (outline hatch)
		xs_yes = x_positions + bar_width/2   # right bar = with XGB null (filled)

		for i, cov in enumerate(covar_order):
			col = covar_to_color[cov]

			# >>> CHANGED: handle special bucket by borrowing the no-null value from a source bucket
			if cov in SPECIAL_NO_NULL_SOURCE:
				source_cov = SPECIAL_NO_NULL_SOURCE[cov]

				# Left bar (no XGB) uses the no-null row from the source bucket
				row_no_src = sub[(sub['covar_set'] == source_cov) & (~sub['with_null'])]
				if not row_no_src.empty:
					y  = float(row_no_src[metric]); el = float(row_no_src['err_low']); eh = float(row_no_src['err_high'])
					ax.bar(xs_no[i], y, width=bar_width, facecolor='none', hatch='////',
					       edgecolor=col, linewidth=1.0)
					ax.errorbar(xs_no[i], y, yerr=[[el], [eh]], fmt='none',
					            ecolor=err_bar_color, capsize=3, linewidth=1.0)

				# Right bar (with XGB) uses the special bucket itself
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
				ax.bar(xs_no[i], y, width=bar_width, facecolor='none', hatch='////', edgecolor=col, linewidth=1.0)
				ax.errorbar(xs_no[i], y, yerr=[[el], [eh]], fmt='none', ecolor=err_bar_color, capsize=3, linewidth=1.0)

			if not row_yes.empty:
				y  = float(row_yes[metric]); el = float(row_yes['err_low']); eh = float(row_yes['err_high'])
				ax.bar(xs_yes[i], y, width=bar_width, color=col, edgecolor=col, linewidth=1.0)
				ax.errorbar(xs_yes[i], y, yerr=[[el], [eh]], fmt='none', ecolor=err_bar_color, capsize=3, linewidth=1.0)

		# Cosmetics
		ax.set_title(pheno)
		ax.set_xlim(-0.5, len(covar_order) - 0.5)
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

	g.figure.subplots_adjust(top=0.90, bottom=0.13, hspace=0.32, wspace=0.25)
	g.figure.suptitle(f"Test {_metric_display(metric)} with {ci}% CI")

	# Legend (colors = covariate sets; fill style = with/without XGB null)
	fig_legend, ax_legend = plt.subplots(figsize=(5.7, max(2.7, len(covar_order) * 0.28)))
	ax_legend.axis('off')

	color_handles = [Patch(facecolor=covar_to_color[c], edgecolor=covar_to_color[c], label=c) for c in covar_order]
	style_handles = [
		Patch(facecolor='white', edgecolor='black', hatch='////', linewidth=1.0, label='No XGB null'),
		Patch(facecolor='black', edgecolor='black', label='With XGB null'),
	]

	leg1 = ax_legend.legend(handles=color_handles, title='Covariate set', loc='center left',
	                        bbox_to_anchor=(0.0, 0.5), frameon=False)
	ax_legend.add_artist(leg1)
	ax_legend.legend(handles=style_handles, title='Bar style', loc='center left',
	                 bbox_to_anchor=(0.58, 0.5), frameon=False)

	plt.show()

if __name__ == '__main__':
	main()
