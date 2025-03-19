import os
import json
import dxpy
import pandas as pd

from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


def get_dxlink_from_path(path_to_link):
	"""
	Returns a DNAnexus link object for the file at path_to_link, or None if not found.
	Raises ValueError if multiple objects are found.
	"""
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
	

def get_scores_dict(scores_json_path, local_json_path):
	"""
	Download scores.json to local_json_path, parse it, then remove the local file.

	Returns:
	  - scores dict if valid JSON
	  - None if file is not found or not valid JSON
	"""

	# 1) Get dxlink
	dxlink = get_dxlink_from_path(scores_json_path)
	if dxlink is None:
		# scores.json not found in DNAnexus
		return None

	# 2) Download to local path
	dxpy.download_dxfile(dxlink, local_json_path)

	# 3) Attempt to parse JSON; remove the file afterward no matter what
	try:
		with open(local_json_path, 'r') as f:
			try:
				scores = json.load(f)
			except json.JSONDecodeError as e:
				print(f"Invalid JSON in {local_json_path}: {e}")
				scores = None
	finally:
		# Remove the local file so we don't clutter up the tmp directory
		if os.path.exists(local_json_path):
			os.remove(local_json_path)

	return scores


def retrieve_single_task_score(
	model_type,
	pheno,
	covar_set,
	dn_out_dir,
	version_dir,
	temp_dir,
):
	"""
	Download scores.json for a single-task model and phenotype.
	"""

	scores_dict = get_scores_dict(
		f"{dn_out_dir}/{version_dir}/{pheno}/{covar_set}/{model_type}/ho_scores.json",
		f"{temp_dir}/{model_type}_{pheno}_{covar_set}_scores.json"
	)

	if scores_dict is None:
		return {
			"model_type": model_type,
			"pheno": pheno,
			"covar_set": covar_set,
			"missing": True,
		}
	else:
		return {
			"model_type": model_type,
			"pheno": pheno,
			"covar_set": covar_set,
			"missing": False,
			"multi-task": False,
			**scores_dict
		}
	

def retrieve_multi_task_score(
	model_type,
	phenos_dir,
	covar_set,
	dn_out_dir,
	version_dir,
	temp_dir,
):
	"""
	Download scores.json for a multi-task model.

	phenos_dir is the directory whose name is a list of all phenotype tasks.
	"""

	scores_dict = get_scores_dict(
		f"{dn_out_dir}/{version_dir}/{phenos_dir}/{covar_set}/{model_type}/ho_scores.json",
		f"{temp_dir}/{model_type}_{covar_set}_scores.json"
	)

	if scores_dict is None:
		return {
			"model_type": model_type,
			"pheno": phenos_dir,
			"covar_set": covar_set,
			"missing": True,
		}
	else:
		return {
			"model_type": model_type,
			"covar_set": covar_set,
			"missing": False,
			"multi-task": True,
			"scores": scores_dict,
		}



if __name__ == "__main__":

	# Options
	dn_st_out_dir = '/rdevito/deep_null/dn_output'
	st_version = 'V4'

	mt_out_pheno_dir = "['standing_height_50', 'body_fat_percentage_23099', 'platelet_count_30080', 'glycated_haemoglobin_30750', 'vitamin_d_30890', 'diastolic_blood_pressure_4079', 'systolic_blood_pressure_4080', 'FEV1_3063', 'FVC_3062', 'HDL_cholesterol_30760', 'LDL_direct_30780', 'triglycerides_30870', 'c-reactive_protein_30710', 'creatinine_30700', 'alanine_aminotransferase_30620', 'aspartate_aminotransferase_30650']"
	mt_version = 'V_dev_mt_1'

	save_dir = 'scores'
	temp_dir = 'tmp'

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
	]

	covar_sets_st = [
		'age_sex',
		'age_sex_all_coords_time',
	]

	model_types_st = [
		'xgb_3',
		'deepnull_eswp_sm_1',
		'deepnull_eswp_1',
	]

	covar_sets_mt = [
		'age_sex_all_coords_time',
	]

	model_types_mt = [
		'dev_1',
		'dev_2',
		'dev_3',
		'dev_4',
		'dev_5',
		'dev_5a',
		'dev_5b',
		'dev_5c',
	]

	# Ensure local directories exist
	if not os.path.exists(temp_dir):
		os.makedirs(temp_dir)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Load single-task 
	num_tasks = len(model_types_st) * len(phenos) * len(covar_sets_st)

	# Parallel retrieval with a progress bar
	# Wrap joblib's Parallel with tqdm_joblib so we see live progress
	with tqdm_joblib(tqdm(desc="Retrieving single-task scores...", total=num_tasks)) as _:
		results = Parallel(n_jobs=-1)(
			delayed(retrieve_single_task_score)(
				model_type,
				pheno,
				covar_set,
				dn_st_out_dir,
				st_version,
				temp_dir,
			)
			for model_type in model_types_st
			for pheno in phenos
			for covar_set in covar_sets_st
		)

	# Create dataframe from results
	df_st = pd.DataFrame(results)

	# Save items with 'missing' == True to a separate file
	df_st_missing = df_st[df_st['missing'] == True]
	df_st_missing.to_csv(f"{save_dir}/mt_single_task_missing.csv", index=False)

	# Remove missing items from main dataframe
	df_st = df_st[df_st['missing'] == False]
	
	# Drop 'missing' column
	df_st = df_st.drop(columns=['missing'])


	# Load multi-task
	num_tasks = len(model_types_mt) * len(covar_sets_mt)

	# Parallel retrieval with a progress bar
	with tqdm_joblib(tqdm(desc="Retrieving multi-task scores...", total=num_tasks)) as _:
		results = Parallel(n_jobs=-1)(
			delayed(retrieve_multi_task_score)(
				model_type,
				mt_out_pheno_dir,
				covar_set,
				dn_st_out_dir,
				mt_version,
				temp_dir,
			)
			for model_type in model_types_mt
			for covar_set in covar_sets_mt
		)
	
	# Check for missing, save to separate file and remove from main dataframe
	mt_missing = [r for r in results if r['missing'] == True]
	df_mt_missing = pd.DataFrame(mt_missing).to_csv(
		f"{save_dir}/mt_multi_task_missing.csv",
		index=False
	)
	results = [r for r in results if r['missing'] == False]

	# Unpack scores for each pheno using keys of "scores" dict, then make df
	mt_scores = []
	for r in results:
		for pheno, scores in r['scores'].items():
			mt_scores.append({
				"model_type": r['model_type'],
				"pheno": pheno,
				"covar_set": r['covar_set'],
				"multi-task": True,
				**scores
			})

	df_mt = pd.DataFrame(mt_scores)

	# Join single and multi-task dataframes
	df = pd.concat([df_st, df_mt])

	# Save
	df.to_csv(f"{save_dir}/mt_scores.csv", index=False)
