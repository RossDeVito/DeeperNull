"""Create table with changes in GWAS hits with and without null
and additional covariates.
"""

import os
import pandas as pd
import numpy as np


WINDOW_SIZE = 500000  # 500kb clumping window overlap

PHENO_ALIAS = {
	"standing_height_50": "Standing height",
	"body_fat_percentage_23099": "Body fat percentage",
	"platelet_count_30080": "Platelet count",
	"glycated_haemoglobin_30750": "Glycated haemoglobin",
	"vitamin_d_30890": "Vitamin D",
	"diastolic_blood_pressure_4079": "Diastolic blood pressure",
	"systolic_blood_pressure_4080": "Systolic blood pressure",
	"FEV1_3063": "FEV1",
	"FVC_3062": "FVC",
	"HDL_cholesterol_30760": "HDL cholesterol",
	"LDL_direct_30780": "LDL cholesterol",
	"triglycerides_30870": "Triglycerides",
	"c-reactive_protein_30710": "C-reactive protein",
	"creatinine_30700": "Creatinine",
	"alanine_aminotransferase_30620": "Alanine aminotransferase",
	"aspartate_aminotransferase_30650": "Aspartate aminotransferase",
}


def count_unique_matches(df_ref, df_target, window_bp=500000):
	"""
	Performs greedy 1-to-1 matching to ensure the intersection count
	never exceeds the total count of either set.
	"""
	if df_ref.empty or df_target.empty:
		return 0, len(df_ref), len(df_target)

	# 1. Prepare data
	# We need to track which target indices have already been matched
	matched_target_indices = set()
	matches_found = 0

	# 2. Iterate through Reference clumps
	# (Optional: Sort by P-value here if you want the 'strongest' signal to claim the match first)
	# df_ref = df_ref.sort_values('P') 
	
	for _, row_ref in df_ref.iterrows():
		chr_ref = row_ref['#CHROM']
		bp_ref = row_ref['POS']
		
		# Find candidates on the same chromosome that haven't been used yet
		# AND are within the window
		candidates = df_target[
			(df_target['#CHROM'] == chr_ref) & 
			(~df_target.index.isin(matched_target_indices)) &
			((df_target['POS'] - bp_ref).abs() <= window_bp)
		]
		
		if not candidates.empty:
			# Greedy step: Pick the physically closest match
			candidates['dist'] = (candidates['POS'] - bp_ref).abs()
			best_match_idx = candidates['dist'].idxmin()
			
			# Mark this target as 'taken'
			matched_target_indices.add(best_match_idx)
			matches_found += 1

	# 3. Calculate the buckets
	n_both = matches_found
	n_only_ref = len(df_ref) - matches_found
	n_only_target = len(df_target) - matches_found

	return n_both, n_only_ref, n_only_target


if __name__ == "__main__":

	# Options
	p_val_thresh = 5e-8

	phenos = [
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
	]

	ss_dir = "sum_stats"
	clumped_ss_dir = "clumped_sum_stats"

	# Covariate sets for comparison
	standard_cov = "age_sex_pc"
	modified_cov = "age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time"


	# Load data and get counts
	rows = []

	for pheno in phenos:
		print(f"Processing {pheno}")
		ss_standard = pd.read_csv(
			f"{ss_dir}/ss_{pheno}_{standard_cov}.json",
			sep='\s+'
		)
		ss_modified = pd.read_csv(
			f"{ss_dir}/ss_{pheno}_{modified_cov}.json",
			sep='\s+'
		)
		assert len(ss_standard) == len(ss_modified)
		print(f"  Total SNPs: {len(ss_standard)}\n")

		# Replace 0 p-values with minimum non-zero p-value across both sets
		min_nonzero_p = min(
			ss_standard.loc[ss_standard['P'] > 0, 'P'].min(),
			ss_modified.loc[ss_modified['P'] > 0, 'P'].min()
		)
		ss_standard.loc[ss_standard['P'] == 0, 'P'] = min_nonzero_p
		ss_modified.loc[ss_modified['P'] == 0, 'P'] = min_nonzero_p

		# Drop loci with a nan p-value
		ss_standard = ss_standard.dropna(subset=['P'])
		ss_modified = ss_modified.dropna(subset=['P'])

		# Determine significant SNPs
		ss_standard['significant'] = ss_standard['P'] < p_val_thresh
		ss_modified['significant'] = ss_modified['P'] < p_val_thresh

		# Merge on SNP ID
		merged_ss = pd.merge(
			ss_standard[['ID', 'P', 'significant']],
			ss_modified[['ID', 'P', 'significant']],
			how='inner',
			on='ID',
			suffixes=('_standard', '_modified')
		)

		# Check no rows with nan values
		nan_rows = merged_ss[merged_ss.isna().any(axis=1)]
		assert len(nan_rows) == 0, f"NaN values found in merged data for {pheno}:\n{nan_rows}"

		# Get info
		data = {"Phenotype": PHENO_ALIAS[pheno]}

		data['n_signif_var_std'] = merged_ss['significant_standard'].sum()
		data['n_signif_var_null_st'] = merged_ss['significant_modified'].sum()
		data['n_signif_var_both'] = ((merged_ss['significant_standard']) & (merged_ss['significant_modified'])).sum()
		data['n_signif_var_only_std'] = ((merged_ss['significant_standard']) & (~merged_ss['significant_modified'])).sum()
		data['n_signif_var_only_null_st'] = ((~merged_ss['significant_standard']) & (merged_ss['significant_modified'])).sum()
		data['n_signif_var_neither'] = ((~merged_ss['significant_standard']) & (~merged_ss['significant_modified'])).sum()

		# Now do with clumped data
		clumped_ss_standard = pd.read_csv(
			f"{clumped_ss_dir}/{pheno}_{standard_cov}.clumps",
			sep='\s+'
		)
		clumped_ss_modified = pd.read_csv(
			f"{clumped_ss_dir}/{pheno}_{modified_cov}.clumps",
			sep='\s+'
		)

		# Check overlaps
		n_both, n_only_std, n_only_mod = count_unique_matches(
			clumped_ss_standard, 
			clumped_ss_modified, 
			WINDOW_SIZE
		)

		# Fill table
		# Total counts
		data['n_clumps_std'] = len(clumped_ss_standard)
		data['n_clumps_mod'] = len(clumped_ss_modified)
		
		# The Matched Intersection
		data['n_clumps_both'] = n_both
		
		# The Leftovers
		data['n_clumps_only_std'] = n_only_std
		data['n_clumps_only_mod'] = n_only_mod

		rows.append(data)


	# Create dataframe and save
	df = pd.DataFrame(rows)
	print(df)
	df.to_csv(
		"gwas_changes_table/gwas_significance_change_table.csv",
		index=False
	)
	
