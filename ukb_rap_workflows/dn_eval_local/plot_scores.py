import os

import numpy as np
import pandas as pd
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns


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


def run_and_plot_rank_tests(scores_df, block_col, group_col, metric):
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
			y_col=metric,
		)

		# Critical difference diagram
		avg_rank = scores_df.groupby(block_col)[metric].rank(
			pct=True, ascending=('r2' in metric)
		).groupby(scores_df[group_col]).mean()

		plt.title(f'Critical difference diagram of average {metric} ranks')
		sp.critical_difference_diagram(avg_rank, test_results)
		plt.show()


if __name__ == '__main__':

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

	# Plotting
	sns.set_style('whitegrid')

	# Options
	# metric = 'r2'
	metric = 'adj_r2'
	# metric = 'mae'

	# By model
	block_col = 'pheno-covar'
	group_col = 'model_type'

	run_and_plot_rank_tests(scores_df, block_col, group_col, metric)

	# By covar set
	block_col = 'pheno-model'
	group_col = 'covar_set'

	run_and_plot_rank_tests(scores_df, block_col, group_col, metric)
