"""Make and save table of PRS performance metrics for all models.


"""

import os
import json
from collections import defaultdict

import dxpy
import pandas as pd


def get_dxlink_from_path(path_to_link):
	"""Get dxlink from path."""
	print(f'Finding data object for {path_to_link}', flush=True)

	data_objs = list(dxpy.find_data_objects(
		name=path_to_link.split('/')[-1],
		folder='/'.join(path_to_link.split('/')[:-1]),
		project=dxpy.PROJECT_CONTEXT_ID
	))

	if len(data_objs) == 0:
		return None
	elif len(data_objs) > 1:
		raise ValueError(f'Multiple data objects found for {path_to_link}')
	else:
		return dxpy.dxlink(data_objs[0]['id'])


def get_scores_dict(scores_json_path):
	"""Return scores.json in storage as a dictionary.
	
	Returns None if scores.json does not exist.
	"""
	# Download scores JSON to temp dir
	scores_link = get_dxlink_from_path(scores_json_path)

	if scores_link is None:
		return None
	
	dxpy.download_dxfile(
		scores_link,
		scores_fname_local
	)

	# Load scores
	with open(scores_fname_local, 'r') as f:
		scores = json.load(f)

	return scores


if __name__ == '__main__':

	# Options
	phenos = [
		"standing_height_50",
		"body_fat_percentage_23099",
		"platelet_count_30080",
		"glycated_haemoglobin_30750",
		"vitamin_d_30890",
		"diastolic_blood_pressure_4079",
		"systolic_blood_pressure_4080",
		"FEV1_3063",
		"FVC_3062",
		"HDL_cholesterol_30760",
		"LDL_direct_30780",
		"triglycerides_30870",
		"c-reactive_protein_30710",
		"creatinine_30700",
		"alanine_aminotransferase_30620",
		"aspartate_aminotransferase_30650",
		"grip_strength",
		"heel_bone_mineral_density_3148",
		"mean_time_to_identify_matches_20023",
		"fluid_intelligence_score_20016",
		"number_of_incorrect_matches_399",
		"arterial_stiffness_index_21021",
		"hearing_SRT",
		"sleep_duration_1160",
		"adjusted_telomere_ratio_22191",
		"white_blood_cell_count_30000",
		"red_blood_cell_count_30010",
		"haemoglobin_concentration_30020",
		"mean_corpuscular_volume_30040",
		"glucose_30740",
		"urate_30880",
		"testosterone_30850",
		"IGF1_30770",
		"SHBG_30830",
	]

	model_types = [
		'basil_age_sex_pc_lasso_25',
		'basil_age_sex_all_coords_pc_lasso_25',
		'basil_age_sex_pc_null_xgb_3_age_sex_lasso_25',
		'basil_age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords_lasso_25'
	]

	model_dirs = {
		'basil_age_sex_pc_lasso_25': '/rdevito/deep_null/output/PRS_BASIL/{:}/age_sex_pc_lasso_25',
		'basil_age_sex_all_coords_pc_lasso_25': '/rdevito/deep_null/output/PRS_BASIL/{:}/age_sex_all_coords_pc_lasso_25',
		'basil_age_sex_pc_null_xgb_3_age_sex_lasso_25': '/rdevito/deep_null/output/PRS_BASIL/{:}/age_sex_pc_null_xgb_3_age_sex_lasso_25',
		'basil_age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords_lasso_25': '/rdevito/deep_null/output/PRS_BASIL/{:}/age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords_lasso_25',
	}

	model_class_meta = {
		'prsice': {
			'model_desc': 'PRSice-2',
		},
		'basil_age_sex_pc_lasso_25': {
			'model_desc': 'BASIL (lin: age, sex, PCs)',
		},
		'basil_age_sex_all_coords_pc_lasso_25': {
			'model_desc': 'BASIL (lin: age, sex, locations, PCs)',
		},
		'basil_age_sex_pc_null_xgb_3_age_sex_lasso_25': {
			'model_desc': 'BASIL (lin: age, sex, PCs; XGB null: age, sex)',
		},
		'basil_age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords_lasso_25': {
			'model_desc': 'BASIL (lin: age, sex, locations, PCs; XGB null: age, sex, locations)',
		}, 
	}

	save_dir = 'scores'
	temp_dir = 'tmp'

	# Make temp dir and scores dir locally if they don't exist
	if not os.path.exists(temp_dir):
		os.makedirs(temp_dir)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	scores_fname_local = f'{temp_dir}/scores.json'

	# Get scores
	val_scores = []
	test_scores = []

	for model_type in model_types:
		print(f'\nGetting scores for {model_type} models...', flush=True)

		# Get scores for each phenotype
		for pheno in phenos:
			print(f'\t{pheno}', flush=True)

			# Get directory
			if model_type.startswith('basil_'):
				model_out_dir = model_dirs[model_type].format(pheno)
			else:
				model_out_dir = model_dirs[model_type]

			# Get scores for models trained on all samples
			all_scores = get_scores_dict(
				f'{model_out_dir}/scores.json'
			)

			if all_scores is not None:
				val_scores.append({
					'model_desc': model_class_meta[model_type]['model_desc'],
					'pheno': pheno,
					**all_scores['val']
				})
				test_scores.append({
					'model_desc': model_class_meta[model_type]['model_desc'],
					'pheno': pheno,
					**all_scores['test']
				})
			else:
				print(f'\t\tNo scores found for {pheno} {model_type}')
			
	# Make dataframes
	val_scores_df = pd.DataFrame(val_scores)
	test_scores_df = pd.DataFrame(test_scores)

	# Save
	val_scores_df.to_csv(f'{save_dir}/val_scores.csv', index=False)
	test_scores_df.to_csv(f'{save_dir}/test_scores.csv', index=False)
