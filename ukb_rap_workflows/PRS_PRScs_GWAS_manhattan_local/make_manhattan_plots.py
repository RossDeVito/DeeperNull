import os
import pandas as pd

from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

import matplotlib.pyplot as plt
from geneview import manhattanplot, qqplot


if __name__ == "__main__":
	
	# Options

	pheno = [
		"standing_height_50",				# 0
		"body_fat_percentage_23099",		# 1
		"platelet_count_30080",				# 2
		"glycated_haemoglobin_30750",		# 3
		"vitamin_d_30890",					# 4
		"diastolic_blood_pressure_4079",	# 5
		"systolic_blood_pressure_4080",		# 6
		"FEV1_3063",						# 7
		"FVC_3062",							# 8
		"HDL_cholesterol_30760",			# 9
		"LDL_direct_30780",					# 10
		"triglycerides_30870",				# 11
		"c-reactive_protein_30710",			# 12
		"creatinine_30700",					# 13
		"alanine_aminotransferase_30620",	# 14
		"aspartate_aminotransferase_30650"	# 15
	][4]

	ss_dir = "sum_stats"
	manhattan_out_dir = "manhattan_plots"

	covar_sets = [
		'age_sex_pc',
		'age_sex_all_coords_pc',
		'age_sex_time_pc',
		'age_sex_all_coords_time_pc',
		'age_sex_pc_null_xgb_3_age_sex',
		'age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords',
		'age_sex_time_pc_null_xgb_3_age_sex_time',
		'age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time',
		'age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc',
	]

	side_by_side_covs = [
		'age_sex_pc',
		'age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time'
	]

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

	# Load summary stats
	def load_summary_stats(cov_set):
		ss_df = pd.read_csv(
			f"{ss_dir}/ss_{pheno}_{cov_set}.json",
			sep='\s+'
		)
		return cov_set, ss_df

	with tqdm_joblib(tqdm(desc="Loading summary stats...", total=len(covar_sets))):
		results = Parallel(n_jobs=-1)(
			delayed(load_summary_stats)(cov_set) for cov_set in covar_sets
		)

	ss_dict = {cov_set: ss_df for cov_set, ss_df in results}

	# Set 0 p-vals to next lowest
	for covar_set, ss_df in ss_dict.items():
		low_fill_val = ss_df[ss_df.P != 0].P.min()
		ss_df.loc[ss_df.P == 0, 'P'] = low_fill_val
		ss_dict[covar_set] = ss_df

	# Plot Manhattan and QQ plots
	if not os.path.exists(manhattan_out_dir):
		os.makedirs(manhattan_out_dir)
	if not os.path.exists(f"{manhattan_out_dir}/{pheno}"):
		os.makedirs(f"{manhattan_out_dir}/{pheno}")

	for covar_set, ss_df in ss_dict.items():
		# Plot Manhattan
		ax = manhattanplot(
			data=ss_df,
			pv="P",
		)
		plt.title(f"{pheno} Manhattan Plot ({covar_set_descs[covar_set]})")
		plt.savefig(
			f"{manhattan_out_dir}/{pheno}/manhattan_{covar_set}.png",
			dpi=400
		)
		plt.close()

		# # Plot QQ
		# ax = qqplot(
		# 	data=ss_df["P"]
		# )
		# plt.title(f"{pheno} QQ Plot ({covar_set_descs[covar_set]})")
		# plt.savefig(
		# 	f"{manhattan_out_dir}/{pheno}/qq_{covar_set}.png",
		# 	dpi=400
		# )
		# plt.close()

	# Plot side-by-side Manhattan plots sharing y-axis and adding 
	# y-axis lines
	if not os.path.exists(f"{manhattan_out_dir}/{pheno}/side_by_side"):
		os.makedirs(f"{manhattan_out_dir}/{pheno}/side_by_side")

	ss_dict_side_by_side = {cov_set: ss_dict[cov_set] for cov_set in side_by_side_covs}

	fig, axs = plt.subplots(
		nrows=1, ncols=len(ss_dict_side_by_side), figsize=(20, 8)
	)

	for i, (covar_set, ss_df) in enumerate(ss_dict_side_by_side.items()):

		# Plot Manhattan
		ax = manhattanplot(
			data=ss_df,
			pv="P",
			ax=axs[i],
		)
		ax.set_title(f"{pheno} Manhattan Plot ({covar_set_descs[covar_set]})")
		ax.set_xlabel("Chromosome")
		ax.set_ylabel("-log10(P)")
		ax.grid(True)

	plt.show()







