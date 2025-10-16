"""Make and save table of null model performance metrics for all models and
PR curve info.

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
	version = 'V4_w_save_bin_cls'

	dn_out_dir = '/rdevito/deep_null/dn_output'

	save_dir = 'scores_bin_cls'
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
		"xgb_3",
	]

	phenos = [
		"asthma_42015",
		"depression_20438",
		"diabetes_2443",
	]

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
	pr_curves = defaultdict(dict)
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

		# Pop 'recall' and 'precision' from scores_dict
		# Save seperately as JSON
		if scores_dict is not None:
			pr_data = scores_dict.pop('pr_curve')
			pr_curves[pheno][covar_set] = {
				'recall': pr_data['recall'],
				'precision': pr_data['precision'],
			}

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

	# Save PR curves
	pr_curves_fname = f'{save_dir}/pr_curves.json'
	with open(pr_curves_fname, 'w') as f:
		json.dump(pr_curves, f)
