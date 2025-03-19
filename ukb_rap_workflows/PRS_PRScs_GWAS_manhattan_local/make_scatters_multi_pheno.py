import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Options
p_val_thresh = 5e-8

phenos = [
	# "standing_height_50",				# 0
	# "body_fat_percentage_23099",		# 1
	# "platelet_count_30080",				# 2
	# "glycated_haemoglobin_30750",		# 3
	# "vitamin_d_30890",					# 4
	# "diastolic_blood_pressure_4079",	# 5
	# "systolic_blood_pressure_4080",		# 6
	# "FEV1_3063",						# 7
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
x_axis_cov = "age_sex_pc"
y_axis_cov = "age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time"

# Load summary stats
def load_summary_stats(pheno, cov_set):
	ss_df = pd.read_csv(
		f"{ss_dir}/ss_{pheno}_{cov_set}.json",
		sep='\s+'
	)
	return ss_df

# Generate scatter plots
sns.set_style("whitegrid")
fig, axes = plt.subplots(len(phenos), 1, figsize=(8, 4 * len(phenos)))

for i, pheno in enumerate(tqdm(phenos, desc="Processing phenotypes")):
	x_axis_ss = load_summary_stats(pheno, x_axis_cov)
	y_axis_ss = load_summary_stats(pheno, y_axis_cov)

	# Merge dataframes
	merged_ss = pd.merge(
		x_axis_ss, y_axis_ss, on='ID', suffixes=('_x', '_y')
	)
	
	# Remove loci with p-values above threshold in both sets
	merged_ss = merged_ss[(merged_ss.P_x < p_val_thresh) & (merged_ss.P_y < p_val_thresh)]
	
	# Convert to -log10(p-value)
	merged_ss['-log10(P_x)'] = -np.log10(merged_ss['P_x'])
	merged_ss['-log10(P_y)'] = -np.log10(merged_ss['P_y'])
	
	# Scatter plot
	ax = axes[i] if len(phenos) > 1 else axes
	sns.scatterplot(
		data=merged_ss, x='-log10(P_x)', y='-log10(P_y)', s=15, alpha=0.5, ax=ax
	)
	
	# x=y reference line
	max_val = max(merged_ss['-log10(P_x)'].max(), merged_ss['-log10(P_y)'].max())
	ax.plot([0, max_val], [0, max_val], 'k--')
	
	ax.set_xlabel(f"-log10(p) {x_axis_cov}")
	ax.set_ylabel(f"-log10(p) {y_axis_cov}")
	ax.set_title(f"{pheno} GWAS Comparison")

plt.tight_layout()
plt.show()
