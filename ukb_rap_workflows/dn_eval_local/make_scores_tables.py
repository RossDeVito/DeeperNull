"""Make and save table of null model performance metrics for all models.

Also tracks what still has not been done.
"""

import itertools
import os
import json
from collections import defaultdict

import dxpy
import pandas as pd
from tqdm import tqdm


def get_dxlink_from_path(path_to_link):
	"""Get dxlink from path."""
	# print(f'Finding data object for {path_to_link}', flush=True)

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
	include_with_pc = True
	version = 'V4'

	dn_out_dir = '/rdevito/deep_null/dn_output'

	save_dir = 'scores'
	temp_dir = 'tmp'

	covar_sets = [
		"age_sex",
		"age_sex_birth_coords",
		"age_sex_home_coords",
		"age_sex_all_coords",
		"age_sex_tod",
		"age_sex_toy",
		"age_sex_time",
		"age_sex_all_coords_time",
	]

	model_types = [
		"lin_reg_1",
		"lasso_1",
		"ridge_2",
		"deepnull_orig_1",
		"deepnull_es_1",
		"deepnull_eswp_1",
		"deepnull_eswp_sm_1",
		"xgb_1",
		"xgb_2",
		"xgb_3",
	]

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
		"number_of_incorrect_matches_399",
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

	# Removed phenos for less than 100,000 samples
		# "arterial_stiffness_index_21021",
		# "fluid_intelligence_score_20016",
		# "hearing_SRT",

	# Make temp dir and scores dir locally if they don't exist
	if not os.path.exists(temp_dir):
		os.makedirs(temp_dir)
	else:
		# Clear temp dir if existed
		for f in os.listdir(temp_dir):
			os.remove(f'{temp_dir}/{f}')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	scores_fname_local = f'{temp_dir}/scores.json'

	# Get scores
	loaded_scores = []
	not_done = []

	if include_with_pc:
		covar_sets += [c + '_pc' for c in covar_sets]

	all_combinations = list(
		itertools.product(phenos, model_types, covar_sets)
	)

	pbar = tqdm(
		total=len(all_combinations),
		desc="Overall progress",
		unit="combination"
	)

	for pheno, model_type, covar_set in all_combinations:
		# Update the progress bar’s description each iteration:
		pbar.set_description_str(f"{pheno} | {model_type} | {covar_set}")

		scores_dict = get_scores_dict(
			f"{dn_out_dir}/{version}/{pheno}/{covar_set}/{model_type}/ho_scores.json"
		)

		if scores_dict is not None:
			loaded_scores.append({
				'pheno': pheno,
				'covar_set': covar_set,
				'model_type': model_type,
				'pc': covar_set.endswith('_pc'),
				**scores_dict
			})
		else:
			not_done.append({
				'pheno': pheno,
				'covar_set': covar_set,
				'model_type': model_type,
			})

		# Show how many have been loaded vs. missing so far
		pbar.set_postfix(
			loaded=len(loaded_scores),
			missing=len(not_done)
		)
		
		pbar.update(1)

	pbar.close()

	# Make dataframes
	scores_df = pd.DataFrame(loaded_scores)
	not_done_df = pd.DataFrame(not_done)

	# Save
	scores_df.to_csv(f'{save_dir}/scores.csv', index=False)
	not_done_df.to_csv(f'{save_dir}/not_done.csv', index=False)
