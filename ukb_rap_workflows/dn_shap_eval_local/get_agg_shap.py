"""Download aggregated SHAP values from the cloud to the local machine."""

import os
import json
from itertools import product

import dxpy
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
	

if __name__ == '__main__':
	
	# Options
	# version = 'V4_w_save'
	version = 'V4_w_save_bin_cls'
	dn_out_dir = '/rdevito/deep_null/dn_output'

	tmp_dir = 'tmp'
	save_dir = 'agg_shap'

	covar_set = 'age_sex_all_coords_time'
	# model_type = 'xgb_3'
	model_type = 'xgb_3_bin_cls'

	# phenos = [
	# 	"standing_height_50",
	# 	"body_fat_percentage_23099",
	# 	"platelet_count_30080",
	# 	"glycated_haemoglobin_30750",
	# 	"vitamin_d_30890",
	# 	"diastolic_blood_pressure_4079",
	# 	"systolic_blood_pressure_4080",
	# 	"FEV1_3063",
	# 	"FVC_3062",
	# 	"HDL_cholesterol_30760",
	# 	"LDL_direct_30780",
	# 	"triglycerides_30870",
	# 	"c-reactive_protein_30710",
	# 	"creatinine_30700",
	# 	"alanine_aminotransferase_30620",
	# 	"aspartate_aminotransferase_30650",
	# ]

	phenos = [
		"asthma_42015",
		"depression_20438",
		"diabetes_2443",
	]

	# Make save dir if it doesn't exist
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Download and save aggregated SHAP values
	for pheno in tqdm(phenos):

		shap_link = get_dxlink_from_path(
			f'{dn_out_dir}/{version}/{pheno}/{covar_set}/{model_type}/shapley_agg_values.json'
		)

		if shap_link is None:
			print(f'No SHAP values found for {pheno}')
			continue

		dxpy.download_dxfile(
			shap_link,
			f'{save_dir}/{pheno}_agg_shap.json'
		)
