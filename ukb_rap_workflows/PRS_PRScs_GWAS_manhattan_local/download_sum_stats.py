import os
import dxpy
import pandas as pd

from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

import matplotlib.pyplot as plt
from geneview import manhattanplot, qqplot


def get_dxlink_from_path(path_to_link):
	"""
	Returns a DNAnexus link object for the file at path_to_link, or None if not found.
	Raises ValueError if multiple objects are found.
	"""
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
	

def get_ss_dict(scores_json_path, local_json_path):
	"""
	Download summary stats to local, parse it, then remove the local file.

	Returns:
	  - scores dict if valid white-space separated file
	  - None if file is not found or not white-space separated
	"""

	# 1) Get dxlink
	dxlink = get_dxlink_from_path(scores_json_path)
	if dxlink is None:
		# scores.json not found in DNAnexus
		return None

	# 2) Download to local path
	dxpy.download_dxfile(dxlink, local_json_path)
	
	return True
	

def retrieve_summary_stats(
	gwas_res_dir,
	pheno,
	covar_set,
	save_dir,
):
	"""
	Download scores.json for a single-task model and phenotype.
	"""

	sum_stats = get_ss_dict(
		f"{gwas_res_dir.format(pheno)}/gwas_plink2.{pheno}.glm.linear",
		f"{save_dir}/ss_{pheno}_{covar_set}.json"
	)

	return (covar_set, sum_stats)


if __name__ == "__main__":
	
	# Options

	phenos = [
		"standing_height_50",				
		# "body_fat_percentage_23099",
		# "platelet_count_30080",				
		# "glycated_haemoglobin_30750",
		# "vitamin_d_30890",					
		# "diastolic_blood_pressure_4079",
		"systolic_blood_pressure_4080",
		# "FEV1_3063",
		# "FVC_3062",
		# "HDL_cholesterol_30760",
		# "LDL_direct_30780",					
		# "triglycerides_30870",				
		# "c-reactive_protein_30710",
		# "creatinine_30700",
		# "alanine_aminotransferase_30620",
		# "aspartate_aminotransferase_30650"
	]

	save_dir = "sum_stats"

	covar_sets = [
		'age_sex_pc',
		# 'age_sex_all_coords_pc',
		# 'age_sex_time_pc',
		# 'age_sex_all_coords_time_pc',
		# 'age_sex_pc_null_xgb_3_age_sex',
		# 'age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords',
		# 'age_sex_time_pc_null_xgb_3_age_sex_time',
		'age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time',
		# 'age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc',
	]

	covar_set_res_dirs = {
		"age_sex_pc": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_pc",
		"age_sex_all_coords_pc": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_all_coords_pc",
		"age_sex_time_pc": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_time_pc",
		"age_sex_all_coords_time_pc": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_all_coords_time_pc",
		"age_sex_pc_null_xgb_3_age_sex": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_pc_null_xgb_3_age_sex",
		"age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords",
		"age_sex_time_pc_null_xgb_3_age_sex_time": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_time_pc_null_xgb_3_age_sex_time",
		"age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time",
		"age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc",
	}

	covar_set_descs = {
		"age_sex_pc": "age, sex, PCs",
		"age_sex_all_coords_pc": "age, sex, home & birth coords., PCs",
		"age_sex_time_pc": "age, sex, time of day & year, PCs",
		"age_sex_all_coords_time_pc": "age, sex, home & birth coords., time of day & year, PCs",
		"age_sex_pc_null_xgb_3_age_sex": "age, sex, PCs, Null(age, sex)",
		"age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords": "age, sex, home & birth coords., PCs, Null(age, sex, home & birth coords.)",
		"age_sex_time_pc_null_xgb_3_age_sex_time": "age, sex, time of day & year, PCs, Null(age, sex, time of day & year)",
		"age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time": "age, sex, home & birth coords., time of day & year, PCs, Null(age, sex, home & birth coords., time of day & year)",
		"age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc": "age, sex, home & birth coords., time of day & year, PCs, Null(age, sex, home & birth coords., time of day & year, PCs)",
	}

	# Ensure local directories exist

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Retrieve results in parallel
	with tqdm_joblib(
		tqdm(desc="Retrieving results", total=len(covar_sets) * len(phenos))
	) as progress_bar:
		results = Parallel(n_jobs=-1)(
			delayed(retrieve_summary_stats)(
				covar_set_res_dirs[covar_set],
				pheno,
				covar_set,
				save_dir,
			)
			for covar_set in covar_sets
			for pheno in phenos
		)

	# Check for missing results
	missing_results = [r for r in results if r[1] is None]
	print(f"Number of missing results: {len(missing_results)}")
	print(f"Missing results: {missing_results}")