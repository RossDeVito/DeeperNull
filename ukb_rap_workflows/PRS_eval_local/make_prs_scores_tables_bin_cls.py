#!/usr/bin/env python

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

	dxlink = get_dxlink_from_path(scores_json_path)
	if dxlink is None:
		return None

	dxpy.download_dxfile(dxlink, local_json_path)

	try:
		with open(local_json_path, 'r') as f:
			try:
				scores = json.load(f)
			except json.JSONDecodeError as e:
				print(f"Invalid JSON in {local_json_path}: {e}")
				scores = None
	finally:
		if os.path.exists(local_json_path):
			os.remove(local_json_path)

	return scores


def retrieve_scores_for_pheno(model_type, pheno,
							  model_dirs, model_class_meta,
							  temp_dir):
	"""
	1) Builds the path to the scores.json in DNAnexus
	2) Downloads it into a unique local file
	3) Parses and returns (val_dict, test_dict, was_missing, model_type, pheno)
	"""
	# Construct model output directory
	if model_type.startswith('basil_') or model_type.startswith('prscs_'):
		model_out_dir = model_dirs[model_type].format(pheno)
	else:
		raise ValueError(f'Invalid model type: {model_type}')

	# Unique local JSON path so we don't overwrite in parallel
	local_json_path = os.path.join(
		temp_dir, f"scores_{model_type}_{pheno}.json"
	)

	all_scores = get_scores_dict(
		scores_json_path=f"{model_out_dir}/scores.json",
		local_json_path=local_json_path
	)

	if all_scores is None:
		# Not found or invalid
		return (None, None, True, model_type, pheno)

	# If successfully parsed
	val_dict = {
		"model_desc": model_class_meta[model_type]["model_desc"],
		"pheno": pheno,
		**all_scores["val"]
	}
	test_dict = {
		"model_desc": model_class_meta[model_type]["model_desc"],
		"pheno": pheno,
		**all_scores["test"]
	}
	return (val_dict, test_dict, False, model_type, pheno)


if __name__ == "__main__":

	phenos = [
		"asthma_42015",
		"depression_20438",
		"diabetes_2443",
	]

	model_types = [
		"basil_age_sex_pc_lasso_25",
		"basil_age_sex_all_coords_pc_lasso_25",
		"basil_age_sex_time_pc_lasso_25",
		"basil_age_sex_all_coords_time_pc_lasso_25",
		"basil_age_sex_pc_null_xgb_3_age_sex_lasso_25",
		"basil_age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords_lasso_25",
		"basil_age_sex_time_pc_null_xgb_3_age_sex_time_lasso_25",
		"basil_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_lasso_25",
		"basil_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc_lasso_25",
		"prscs_age_sex_pc",
		"prscs_age_sex_all_coords_pc",
		"prscs_age_sex_time_pc",
		"prscs_age_sex_all_coords_time_pc",
		"prscs_age_sex_pc_null_xgb_3_age_sex",
		"prscs_age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords",
		"prscs_age_sex_time_pc_null_xgb_3_age_sex_time",
		"prscs_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time",
		"prscs_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc",
	]

	model_dirs = {
		"basil_age_sex_pc_lasso_25": "/rdevito/deep_null/output/PRS_BASIL/{:}/age_sex_pc_lasso_25",
		"basil_age_sex_all_coords_pc_lasso_25": "/rdevito/deep_null/output/PRS_BASIL/{:}/age_sex_all_coords_pc_lasso_25",
		"basil_age_sex_time_pc_lasso_25": "/rdevito/deep_null/output/PRS_BASIL/{:}/age_sex_time_pc_lasso_25",
		"basil_age_sex_all_coords_time_pc_lasso_25": "/rdevito/deep_null/output/PRS_BASIL/{:}/age_sex_all_coords_time_pc_lasso_25",
		"basil_age_sex_pc_null_xgb_3_age_sex_lasso_25": "/rdevito/deep_null/output/PRS_BASIL/{:}/age_sex_pc_null_xgb_3_age_sex_lasso_25",
		"basil_age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords_lasso_25": "/rdevito/deep_null/output/PRS_BASIL/{:}/age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords_lasso_25",
		"basil_age_sex_time_pc_null_xgb_3_age_sex_time_lasso_25": "/rdevito/deep_null/output/PRS_BASIL/{:}/age_sex_time_pc_null_xgb_3_age_sex_time_lasso_25",
		"basil_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_lasso_25": "/rdevito/deep_null/output/PRS_BASIL/{:}/age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_lasso_25",
		"basil_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc_lasso_25": "/rdevito/deep_null/output/PRS_BASIL/{:}/age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc_lasso_25",
		"prscs_age_sex_pc": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_pc",
		"prscs_age_sex_all_coords_pc": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_all_coords_pc",
		"prscs_age_sex_time_pc": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_time_pc",
		"prscs_age_sex_all_coords_time_pc": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_all_coords_time_pc",
		"prscs_age_sex_pc_null_xgb_3_age_sex": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_pc_null_xgb_3_age_sex",
		"prscs_age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords",
		"prscs_age_sex_time_pc_null_xgb_3_age_sex_time": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_time_pc_null_xgb_3_age_sex_time",
		"prscs_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time",
		"prscs_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc": "/rdevito/deep_null/output/PRS_PRScs/{:}/age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc",
	}

	model_class_meta = {
		"basil_age_sex_pc_lasso_25": {
			"model_desc": "BASIL (lin: age, sex, PCs)",
		},
		"basil_age_sex_all_coords_pc_lasso_25": {
			"model_desc": "BASIL (lin: age, sex, locations, PCs)",
		},
		"basil_age_sex_time_pc_lasso_25": {
			"model_desc": "BASIL (lin: age, sex, times, PCs)",
		},
		"basil_age_sex_all_coords_time_pc_lasso_25": {
			"model_desc": "BASIL (lin: age, sex, locations, times, PCs)",
		},
		"basil_age_sex_pc_null_xgb_3_age_sex_lasso_25": {
			"model_desc": "BASIL (lin: age, sex, PCs; XGB null: age, sex)",
		},
		"basil_age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords_lasso_25": {
			"model_desc": "BASIL (lin: age, sex, locations, PCs; XGB null: age, sex, locations)",
		},
		"basil_age_sex_time_pc_null_xgb_3_age_sex_time_lasso_25": {
			"model_desc": "BASIL (lin: age, sex, times, PCs; XGB null: age, sex, times)",
		},
		"basil_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_lasso_25": {
			"model_desc": "BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)",
		},
		"basil_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc_lasso_25": {
			"model_desc": "BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)",
		},
		"prscs_age_sex_pc": {
			"model_desc": "PRScs (lin: age, sex, PCs)",
		},
		"prscs_age_sex_all_coords_pc": {
			"model_desc": "PRScs (lin: age, sex, locations, PCs)",
		},
		"prscs_age_sex_time_pc": {
			"model_desc": "PRScs (lin: age, sex, times, PCs)",
		},
		"prscs_age_sex_all_coords_time_pc": {
			"model_desc": "PRScs (lin: age, sex, locations, times, PCs)",
		},
		"prscs_age_sex_pc_null_xgb_3_age_sex": {
			"model_desc": "PRScs (lin: age, sex, PCs; XGB null: age, sex)",
		},
		"prscs_age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords": {
			"model_desc": "PRScs (lin: age, sex, locations, PCs; XGB null: age, sex, locations)",
		},
		"prscs_age_sex_time_pc_null_xgb_3_age_sex_time": {
			"model_desc": "PRScs (lin: age, sex, times, PCs; XGB null: age, sex, times)",
		},
		"prscs_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time": {
			"model_desc": "PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)",
		},
		"prscs_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc": {
			"model_desc": "PRScs (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times, PCs)",
		},
	}

	save_dir = "scores"
	temp_dir = "tmp"

	# Ensure local directories exist
	if not os.path.exists(temp_dir):
		os.makedirs(temp_dir)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# We expect len(model_types)*len(phenos) total tasks
	num_tasks = len(model_types) * len(phenos)

	# Parallel retrieval with a progress bar
	# Wrap joblib's Parallel with tqdm_joblib so we see live progress
	with tqdm_joblib(tqdm(desc="Retrieving scores...", total=num_tasks)) as _:
		results = Parallel(n_jobs=-1)(
			delayed(retrieve_scores_for_pheno)(
				model_type,
				pheno,
				model_dirs,
				model_class_meta,
				temp_dir
			)
			for model_type in model_types
			for pheno in phenos
		)

	# results is a list of (val_dict, test_dict, was_missing, model_type, pheno)

	val_scores = []
	test_scores = []
	missing_info = []

	for val_dict, test_dict, was_missing, model_type, pheno in results:
		if was_missing:
			missing_info.append({
				"model_type": model_type,
				"pheno": pheno,
			})
		else:
			val_scores.append(val_dict)
			test_scores.append(test_dict)

	# Convert to DataFrames
	val_scores_df = pd.DataFrame(val_scores)
	test_scores_df = pd.DataFrame(test_scores)
	missing_df = pd.DataFrame(missing_info)

	# Save all outputs
	val_scores_df.to_csv(os.path.join(save_dir, "val_scores_bin_cls.csv"), index=False)
	test_scores_df.to_csv(os.path.join(save_dir, "test_scores_bin_cls.csv"), index=False)
	missing_df.to_csv(os.path.join(save_dir, "missing_scores_bin_cls.csv"), index=False)

	print(f"\nAll done! Saved:\n  - {save_dir}/val_scores.csv\n  - {save_dir}/test_scores.csv\n  - {save_dir}/missing_scores.csv")
