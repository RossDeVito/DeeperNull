import os

import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import colorsys


N_FEAT_COVAR_SET = {
	'age_sex': 2,
	'age_sex_pc': 22,
	'age_sex_birth_coords': 4,
	'age_sex_birth_coords_pc': 24,
	'age_sex_home_coords': 4,
	'age_sex_home_coords_pc': 24,
	'age_sex_all_coords': 6,
	'age_sex_all_coords_pc': 26,
	'age_sex_tod': 3,
	'age_sex_tod_pc': 23,
	'age_sex_toy': 4,
	'age_sex_toy_pc': 24,
	'age_sex_time': 5,
	'age_sex_time_pc': 25,
	'age_sex_all_coords_time': 9,
	'age_sex_all_coords_time_pc': 29,
}

N_SAMP_PHENO = {
	'FEV1_3063': 231888,
	'FVC_3062': 231888,
	'HDL_cholesterol_30760': 221748,
	'IGF1_30770': 240945,
	'LDL_direct_30780': 241783,
	'SHBG_30830': 219697,
	'adjusted_telomere_ratio_22191': 246014,
	'alanine_aminotransferase_30620': 242140,
	# 'arterial_stiffness_index_21021': 83283,
	'aspartate_aminotransferase_30650': 241346,
	'body_fat_percentage_23099': 249522,
	'c-reactive_protein_30710': 241711,
	'creatinine_30700': 242109,
	'diastolic_blood_pressure_4079': 237332,
	# 'fluid_intelligence_score_20016': 82482,
	'glucose_30740': 221605,
	'glycated_haemoglobin_30750': 242250,
	'grip_strength': 253489,
	'haemoglobin_concentration_30020': 246547,
	# 'hearing_SRT': 81820,
	'heel_bone_mineral_density_3148': 146774,
	'mean_corpuscular_volume_30040': 246546,
	'mean_time_to_identify_matches_20023': 252407,
	'number_of_incorrect_matches_399': 253930,
	'platelet_count_30080': 246548,
	'red_blood_cell_count_30010': 246548,
	'sleep_duration_1160': 252648,
	'standing_height_50': 253488,
	'systolic_blood_pressure_4080': 237327,
	'testosterone_30850': 219602,
	'triglycerides_30870': 242029,
	'urate_30880': 241911,
	'vitamin_d_30890': 231672,
	'white_blood_cell_count_30000': 246544,
}

MODEL_DISPLAY_NAME_MAP = {
	'lin_reg_1': 'Linear Regression',
	'lasso_1': 'LASSO',
	'ridge_2': 'Ridge Regression',
	'deepnull_orig_1': 'DeepNull (Original)',
	'deepnull_eswp_1': 'DeepNull (ESWP)',
	'deepnull_eswp_sm_1': 'DeepNull (ESWP, Small)',
	'xgb_1': 'XGBoost (100 estimators)',
	'xgb_2': 'XGBoost (1000 estimators)',
	'xgb_3': 'XGBoost (2500 estimators)',
}


def run_and_plot_rank_tests(
		scores_df,
		block_col,
		group_col,
		metric,
		palette=None
	):
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
			pct=True, ascending=('r2' in metric)
		).groupby(scores_df[group_col]).mean()

		plt.title(f'Critical difference diagram of average {metric} ranks')
		sp.critical_difference_diagram(
			avg_rank,
			test_results,
			color_palette=palette
		)
		plt.show()


def print_percent_improvement(
		scores_df,
		block_col,
		group_col,
		metric='r2'
	):
	"""Compute and print percent improvement over baseline."""

	baseline_val = {
		'model_type': 'Linear Regression',
		'covar_set': 'age_sex'
	}[group_col]

	baseline_df = scores_df[
		(scores_df[group_col] == baseline_val)
	][['pheno', block_col, metric]].rename(
		columns={metric: 'baseline_metric'}
	)

	improvement_df = scores_df.copy()
	improvement_df = improvement_df.merge(
		baseline_df,
		on=['pheno', block_col],
	)
	improvement_df['improvement'] = improvement_df[metric] - improvement_df['baseline_metric']
	improvement_df['percent_change'] = (
		improvement_df['improvement'] / improvement_df['baseline_metric']
	) * 100

	print(
		improvement_df.groupby(group_col)['percent_change'].median().sort_values(
			ascending=False
		)
	)


if __name__ == '__main__':

	import matplotlib as mpl
	mpl.rcParams['savefig.dpi'] = 1200

	scores_dir = 'scores'
	exclude_w_pc = False
	exclude_dnes1 = True
	
	# Load scores for test all, wb, and nwb sets
	scores_df = pd.read_csv(f'{scores_dir}/scores.csv')

	scores_df = scores_df.dropna()

	# Optionally filter out PC inclusion
	if exclude_w_pc:
		scores_df = scores_df[scores_df.pc == False]

	# Filter out deepnull_es_1
	if exclude_dnes1:
		scores_df = scores_df[~(scores_df.model_type == 'deepnull_es_1')]

	# Add columns for comparison across all phenotypes
	scores_df['pheno-covar'] = scores_df.pheno + '-' + scores_df.covar_set
	scores_df['pheno-model'] = scores_df.pheno + '-' + scores_df.model_type

	# Adjust R^2 scores
	scores_df['adj_r2'] = scores_df.apply(
		lambda x : 1 - (
			(
				(1 - x['r2']) * (N_SAMP_PHENO[x['pheno']] - 1)
			) / (
				N_SAMP_PHENO[x['pheno']] 
				- N_FEAT_COVAR_SET[x['covar_set']] 
				- 1
			)
		),
		axis=1
	)

	# Rename model_type for display
	scores_df['model_type'] = scores_df['model_type'].map(MODEL_DISPLAY_NAME_MAP)

	# Plotting
	sns.set_style('whitegrid')

	# Options
	metric = 'r2'
	# metric = 'mae'

	# By model
	block_col = 'pheno-covar'
	group_col = 'model_type'

	# palette = [
	# 	"#1f4b99", "#4169e1", "#7b68ee",   # Blue/Purple shades
	# 	"#228b22", "#3cb371", "#6b8e23",   # Green shades
	# 	"#b22222", "#ff6347", "#ff8c00"    # Red/Orange shades
	# ]

	# palette = [
	# 	"#000000", "#000000", "#000000",   # Black for baseline
	# 	"#8B4513", "#8B4513", "#8B4513",   # Green shades
	# 	"#6A0DAD", "#6A0DAD", "#6A0DAD"    # Red/Orange shades
	# ]

	palette = [
		"#4F4E4E", "#4F4E4E", "#4F4E4E",   # Black for baseline
		"#954535", "#954535", "#954535",   # Brown nn
		"#6109A0", "#6109A0", "#6109A0"    # purple xgb
	]

	palette = zip(MODEL_DISPLAY_NAME_MAP.values(), palette)
	palette = dict(palette)

	run_and_plot_rank_tests(
		scores_df,
		block_col,
		group_col,
		metric,
		palette=palette,
	)

	# Compute relative improvement
	print("Percent improvement over baseline in R^2:")
	print_percent_improvement(
		scores_df,
		block_col,
		group_col,
		metric='r2'
	)
	print("Percent improvement over baseline in MAE:")
	print_percent_improvement(
		scores_df,
		block_col,
		group_col,
		metric='mae'
	)

	print("Best for pheno-covar count by model in terms of R^2:")
	best_r2 = scores_df.groupby('pheno-covar')['r2'].max().reset_index()
	scores_w_best_r2 = scores_df.merge(
		best_r2[['pheno-covar', 'r2']],
		on=['pheno-covar', 'r2'],
		how='inner'
	)
	print(
		scores_w_best_r2.groupby('model_type')['pheno-covar'].count().sort_values(
			ascending=False
		)
	)

	print("Best for pheno count by model in terms of R^2:")
	best_r2 = scores_df.groupby('pheno')['r2'].max().reset_index()
	scores_w_best_r2 = scores_df.merge(
		best_r2[['pheno', 'r2']],
		on=['pheno', 'r2'],
		how='inner'
	)
	print(
		scores_w_best_r2.groupby('model_type')['pheno'].count().sort_values(
			ascending=False
		)
	)

	print("Best for pheno-covar count by model in terms of MAE:")
	best_mae = scores_df.groupby('pheno-covar')['mae'].min().reset_index()
	scores_w_best_mae = scores_df.merge(
		best_mae[['pheno-covar', 'mae']],
		on=['pheno-covar', 'mae'],
		how='inner'
	)
	print(
		scores_w_best_mae.groupby('model_type')['pheno-covar'].count().sort_values(
			ascending=False
		)
	)
	print("Best for pheno count by model in terms of MAE:")
	best_mae = scores_df.groupby('pheno')['mae'].min().reset_index()
	scores_w_best_mae = scores_df.merge(
		best_mae[['pheno', 'mae']],
		on=['pheno', 'mae'],
		how='inner'
	)
	print(
		scores_w_best_mae.groupby('model_type')['pheno'].count().sort_values(
			ascending=False
		)
	)

	print("Best for pheno-covar count by model in terms of MSE:")
	best_mse = scores_df.groupby('pheno-covar')['mse'].min().reset_index()
	scores_w_best_mse = scores_df.merge(
		best_mse[['pheno-covar', 'mse']],
		on=['pheno-covar', 'mse'],
		how='inner'
	)
	print(
		scores_w_best_mse.groupby('model_type')['pheno-covar'].count().sort_values(
			ascending=False
		)
	)
	print("Best for pheno count by model in terms of MSE:")
	best_mse = scores_df.groupby('pheno')['mse'].min().reset_index()
	scores_w_best_mse = scores_df.merge(
		best_mse[['pheno', 'mse']],
		on=['pheno', 'mse'],
		how='inner'
	)
	print(
		scores_w_best_mse.groupby('model_type')['pheno'].count().sort_values(
			ascending=False
		)
	)

	print("Best for pheno-covar count by model in terms of MAPE:")
	best_mape = scores_df.groupby('pheno-covar')['mape'].min().reset_index()
	scores_w_best_mape = scores_df.merge(
		best_mape[['pheno-covar', 'mape']],
		on=['pheno-covar', 'mape'],
		how='inner'
	)
	print(
		scores_w_best_mape.groupby('model_type')['pheno-covar'].count().sort_values(
			ascending=False
		)
	)
	print("Best for pheno count by model in terms of MAPE:")
	best_mape = scores_df.groupby('pheno')['mape'].min().reset_index()
	scores_w_best_mape = scores_df.merge(
		best_mape[['pheno', 'mape']],
		on=['pheno', 'mape'],
		how='inner'
	)
	print(
		scores_w_best_mape.groupby('model_type')['pheno'].count().sort_values(
			ascending=False
		)
	)


	# By covar set
	block_col = 'pheno-model'
	group_col = 'covar_set'

	run_and_plot_rank_tests(
		scores_df,
		block_col,
		group_col,
		metric,
		# palette=dict(zip(N_FEAT_COVAR_SET.keys(), sns.color_palette("Paired", len(N_FEAT_COVAR_SET)))),	
	)

	print("Percent improvement over baseline in R^2 by covar set:")
	print_percent_improvement(
		scores_df,
		block_col,
		group_col,
		metric='r2'
	)
	print("Percent improvement over baseline in MAE by covar set:")
	print_percent_improvement(
		scores_df,
		block_col,
		group_col,
		metric='mae'
	)

	print("Best for pheno-model count by covar set in terms of R^2:")
	best_r2 = scores_df.groupby('pheno-model')['r2'].max().reset_index()
	scores_w_best_r2 = scores_df.merge(
		best_r2[['pheno-model', 'r2']],
		on=['pheno-model', 'r2'],
		how='inner'
	)
	print(
		scores_w_best_r2.groupby('covar_set')['pheno-model'].count().sort_values(
			ascending=False
		)
	)

	print("Best for pheno count by covar set in terms of R^2:")
	best_r2 = scores_df.groupby('pheno')['r2'].max().reset_index()
	scores_w_best_r2 = scores_df.merge(
		best_r2[['pheno', 'r2']],
		on=['pheno', 'r2'],
		how='inner'
	)
	print(
		scores_w_best_r2.groupby('covar_set')['pheno'].count().sort_values(
			ascending=False
		)
	)

	print("Best for pheno-model count by covar set in terms of MAE:")
	best_mae = scores_df.groupby('pheno-model')['mae'].min().reset_index()
	scores_w_best_mae = scores_df.merge(
		best_mae[['pheno-model', 'mae']],
		on=['pheno-model', 'mae'],
		how='inner'
	)
	print(
		scores_w_best_mae.groupby('covar_set')['pheno-model'].count().sort_values(
			ascending=False
		)
	)

	print("Best for pheno count by covar set in terms of MAE:")
	best_mae = scores_df.groupby('pheno')['mae'].min().reset_index()
	scores_w_best_mae = scores_df.merge(
		best_mae[['pheno', 'mae']],
		on=['pheno', 'mae'],
		how='inner'
	)
	print(
		scores_w_best_mae.groupby('covar_set')['pheno'].count().sort_values(
			ascending=False
		)
	)

	
	# Plot improvement relative to baseline for xgb3 models
	
	metric = 'r2'
	# metric = 'mae'

	# Filter for xgb3 models
	# xgb3_scores_df = scores_df[
	# 	scores_df.model_type == 'XGBoost (2500 estimators)'
	# ]

	# Compute improvement relative to baseline within each phenotype
	baseline_df = scores_df[
		(scores_df['model_type'] == 'Linear Regression')
		& (scores_df['covar_set'] == 'age_sex')
	][['pheno', metric]].rename(columns={metric: 'baseline_metric'})

	# Merge baseline metrics back into xgb3_scores_df
	merged_df = scores_df.merge(baseline_df, on='pheno')

	# Calculate improvement and percentage improvement
	if metric == 'r2':  # higher is better
		merged_df['improvement'] = merged_df[metric] - merged_df['baseline_metric']
		merged_df['percent_improvement'] = (
			merged_df['improvement'] / merged_df['baseline_metric']
		) * 100
	elif metric == 'mae':  # lower is better
		merged_df['improvement'] = merged_df['baseline_metric'] - merged_df[metric]
		merged_df['percent_improvement'] = (
			merged_df['improvement'] / merged_df['baseline_metric']
		) * 100

	# subset to best per phenotype-covar set
	merged_df = merged_df.sort_values(
		['pheno','covar_set','r2'], ascending=[True, True, False]
	).drop_duplicates(['pheno','covar_set'])

	# Define categorical ordering explicitly
	covar_cats = [
		'age_sex', 'age_sex_pc',
		'age_sex_birth_coords', 'age_sex_birth_coords_pc',
		'age_sex_home_coords', 'age_sex_home_coords_pc',
		'age_sex_all_coords', 'age_sex_all_coords_pc',
		'age_sex_tod', 'age_sex_tod_pc',
		'age_sex_toy', 'age_sex_toy_pc',
		'age_sex_time', 'age_sex_time_pc',
		'age_sex_all_coords_time', 'age_sex_all_coords_time_pc',
	]

	merged_df['covar_set'] = pd.Categorical(
		merged_df['covar_set'],
		categories=covar_cats,
		ordered=True
	)

	merged_df['pheno'] = merged_df['pheno'].str.split('_').apply(
		lambda x: ' '.join(x[:-1]).title()
	)
	merged_df.loc[merged_df['pheno'] == 'Fev1', 'pheno'] = 'FEV1'
	merged_df.loc[merged_df['pheno'] == 'Fvc', 'pheno'] = 'FVC'
	merged_df.loc[merged_df['pheno'] == 'Hdl Cholesterol', 'pheno'] = 'HDL Cholesterol'
	merged_df.loc[merged_df['pheno'] == 'Ldl Direct', 'pheno'] = 'LDL Cholesterol'

	merged_df['covar_set_wo_pc'] = merged_df['covar_set'].str.replace('_pc', '')
	
	# If 'pc' isn't already in your df, derive it from covar_set
	if 'pc' not in merged_df.columns:
		merged_df['pc'] = merged_df['covar_set'].str.contains('_pc', regex=False)

	merged_df['PCs'] = np.where(merged_df['pc'], 'included', 'excluded')

	# Create new version of covar_set without _pc with display names
	covar_name_map = {
		'age_sex': "age, sex",
		'age_sex_birth_coords': 'age, sex, birth location',
		'age_sex_home_coords': 'age, sex, home location',
		'age_sex_all_coords': 'age, sex, birth & home locations',
		'age_sex_tod': 'age, sex, time of day',
		'age_sex_toy': 'age, sex, time of year',
		'age_sex_time': 'age, sex, time of day & year',
		'age_sex_all_coords_time': 'age, sex, both locations, both times',
	}

	merged_df['Covariate Set'] = merged_df['covar_set_wo_pc'].map(covar_name_map)

	# Build base categories (without _pc), preserving order
	covar_base_cats = []
	for c in covar_cats:
		base = c.replace('_pc', '')
		if base not in covar_base_cats:
			covar_base_cats.append(base)

	base_palette = {
		# baseline
		"age, sex": "#474545",                 # dark gray

		# locations (cool family, each distinct)
		"age, sex, birth location": "#0072B2", # blue
		"age, sex, home location":  "#56B4E9", # light blue
		"age, sex, birth & home locations": "#009E73",  # teal/green

		# time (warm family, each distinct)
		"age, sex, time of day":   "#EDA200",  # orange
		"age, sex, time of year":  "#DDCF10",  # yellow
		"age, sex, time of day & year": "#D52B00",  # vermillion / red-orange

		# both (stands apart but not clashing)
		"age, sex, both locations, both times": "#FF68BB",  # magenta
	}

	# Keep legend ordered by your grouped reading
	hue_order = [
		"age, sex",
		"age, sex, birth location",
		"age, sex, home location",
		"age, sex, birth & home locations",
		"age, sex, time of day",
		"age, sex, time of year",
		"age, sex, time of day & year",
		"age, sex, both locations, both times",
	]

	merged_df['pheno'] = pd.Categorical(
		merged_df['pheno'],
		categories=sorted(merged_df['pheno'].unique()),
		ordered=True
	)

	# Remove outline (axes spines) for subsequent plots
	mpl.rcParams['axes.linewidth'] = 0.0

	# Plot: color encodes covariate family, marker style encodes PC vs no-PC
	ax = sns.scatterplot(
		data=merged_df,
		x='improvement',
		y='pheno',
		hue='Covariate Set',
		hue_order=hue_order,
		palette=base_palette,
		style='PCs',
		style_order=['excluded', 'included'],
		# markers={'no_pc': '|', 'pc': 'x'},
		# markers={'no_pc': 'x', 'pc': '+'},
		markers={'excluded': '1', 'included': '|'},
		s=100,
		linewidth=2,
		alpha=0.9,
	)

	# remove vertical gridlines
	ax.xaxis.grid(False)

	handles, labels = ax.get_legend_handles_labels()
	ax.legend(
		handles=handles,
		labels=labels,
		title="Covariates",
		bbox_to_anchor=(1.05, 0.5), # right side, centered vertically
		loc="center left"
	)
	plt.tight_layout(rect=[0, 0, 0.85, 1])
	plt.show()

	# Save results to CSV that will become a supplementary table
	# Add 'Covariate Set' and 'PCs' columns to scores_df
	scores_df['covar_set_wo_pc'] = scores_df['covar_set'].str.replace('_pc', '')
	scores_df['Covariate Set'] = scores_df['covar_set_wo_pc'].map(covar_name_map)
	scores_df['PCs'] = np.where(scores_df['pc'], 'included', 'excluded')

	out_df = scores_df[[
		'pheno', 'Covariate Set', 'PCs',
		'model_type', 'r2', 'mse', 'mae', 'mape',
		'r2_male', 'r2_female',
		'mse_male', 'mse_female',
		'mae_male', 'mae_female',
		'mape_male', 'mape_female',
	]]

	out_df.to_csv('null_model_scores.csv', index=False)
