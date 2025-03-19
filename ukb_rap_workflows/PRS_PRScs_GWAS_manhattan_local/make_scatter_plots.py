import os
import numpy as np
import pandas as pd

from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

import matplotlib.pyplot as plt
import seaborn as sns
from geneview import manhattanplot, qqplot


if __name__ == "__main__":
	
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
	scatter_out_dir = "scatter_plots"

	x_axis_cov_sets = [
		'age_sex_pc',
		# 'age_sex_all_coords_time_pc',
		# 'age_sex_pc_null_xgb_3_age_sex',
		# 'age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time',
	]

	y_axis_cov_sets = [
		# 'age_sex_all_coords_pc',
		# 'age_sex_time_pc',
		# 'age_sex_all_coords_time_pc',
		# 'age_sex_pc_null_xgb_3_age_sex',
		# 'age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords',
		# 'age_sex_time_pc_null_xgb_3_age_sex_time',
		'age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time',
		# 'age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc',
	]

	# covar_sets = [
	# 	'age_sex_pc',
	# 	# 'age_sex_all_coords_pc',
	# 	# 'age_sex_time_pc',
	# 	# 'age_sex_all_coords_time_pc',
	# 	# 'age_sex_pc_null_xgb_3_age_sex',
	# 	# 'age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords',
	# 	# 'age_sex_time_pc_null_xgb_3_age_sex_time',
	# 	'age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time',
	# 	# 'age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc',
	# ]

	covar_sets = list(set(x_axis_cov_sets + y_axis_cov_sets))

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

	for pheno in phenos:
		with tqdm_joblib(tqdm(desc="Loading summary stats...", total=len(covar_sets))):
			results = Parallel(n_jobs=-1)(
				delayed(load_summary_stats)(cov_set) for cov_set in covar_sets
			)

		# Load summary stats w/o parallel
		results = []
		for cov_set in tqdm(covar_sets, desc="Loading summary stats"):
			ss_df = pd.read_csv(
				f"{ss_dir}/ss_{pheno}_{cov_set}.json",
				sep='\s+'
			)
			results.append((cov_set, ss_df))

		ss_dict = {cov_set: ss_df for cov_set, ss_df in results}

		# Set 0 p-vals to next lowest
		for covar_set, ss_df in ss_dict.items():
			low_fill_val = ss_df[ss_df.P != 0].P.min()
			ss_df.loc[ss_df.P == 0, 'P'] = low_fill_val
			ss_dict[covar_set] = ss_df

		# Make out directory
		if not os.path.exists(scatter_out_dir):
			os.makedirs(scatter_out_dir)
		if not os.path.exists(f"{scatter_out_dir}/{pheno}"):
			os.makedirs(f"{scatter_out_dir}/{pheno}")

		# Function to insert newlines into spaces after a certain number of characters
		def insert_newlines(text, max_length):
			words = text.split()
			lines = []
			current_line = []

			for word in words:
				if sum(len(w) for w in current_line) + len(current_line) + len(word) > max_length:
					lines.append(' '.join(current_line))
					current_line = [word]
				else:
					current_line.append(word)

			if current_line:
				lines.append(' '.join(current_line))

			return '\n'.join(lines)

		# Define color palette
		palette = {
			'more significant for x-axis': 'blue',
			'more significant for y-axis': 'orange',
			'only significant for x-axis': 'green',
			'only significant for y-axis': 'red',
			'same': 'gray'
		}

		hue_order = [
			'only significant for x-axis',
			'only significant for y-axis',
			'more significant for x-axis',
			'more significant for y-axis',
			'same'
		]

		# Plot grid
		sns.set_style("whitegrid")

		# Set the size of the final overall figure
		# figsize = (9, 10)
		figsize = (10, 10)

		fig, axes = plt.subplots(
			len(y_axis_cov_sets),
			len(x_axis_cov_sets),
			figsize=figsize,
			sharex=True,
			sharey=True
		)

		max_title_length_y = 30  # Maximum number of characters before inserting a newline
		max_title_length_x = 20  # Maximum number of characters before inserting a newline

		# Initialize handles and labels for the legend
		handles, labels = None, None

		# Initialize a set to track which categories appear
		all_rel_sign_cats = set()

		for i, y_axis_cov in enumerate(y_axis_cov_sets):
			for j, x_axis_cov in enumerate(x_axis_cov_sets):
				ax = axes[i, j]

				x_axis_desc = covar_set_descs[x_axis_cov]
				y_axis_desc = covar_set_descs[y_axis_cov]

				x_axis_ss = ss_dict[x_axis_cov]
				y_axis_ss = ss_dict[y_axis_cov]

				# Merge dataframes
				merged_ss = pd.merge(
					x_axis_ss,
					y_axis_ss,
					on='ID',
					suffixes=('_x', '_y')
				)

				# Remove loci with p-values above threshold in both sets
				merged_ss = merged_ss[
					(merged_ss.P_x < p_val_thresh) &
					(merged_ss.P_y < p_val_thresh)
				]

				# Create new 'Relative Significance' column
				merged_ss['Relative Significance'] = 'same'

				# More significant for x-axis
				more_significant_x = merged_ss[
					(merged_ss.P_x < merged_ss.P_y)
				]
				merged_ss.loc[more_significant_x.index, 'Relative Significance'] = 'more significant for x-axis'

				# More significant for y-axis
				more_significant_y = merged_ss[
					(merged_ss.P_x > merged_ss.P_y)
				]
				merged_ss.loc[more_significant_y.index, 'Relative Significance'] = 'more significant for y-axis'

				# Only significant for x-axis
				only_significant_x = merged_ss[
					(merged_ss.P_x < p_val_thresh) &
					(merged_ss.P_y >= p_val_thresh)
				]
				merged_ss.loc[only_significant_x.index, 'Relative Significance'] = 'only significant for x-axis'

				# Only significant for y-axis
				only_significant_y = merged_ss[
					(merged_ss.P_x >= p_val_thresh) &
					(merged_ss.P_y < p_val_thresh)
				]
				merged_ss.loc[only_significant_y.index, 'Relative Significance'] = 'only significant for y-axis'

				# Print counts of only significant points
				num_only_significant_x = len(only_significant_x)
				num_only_significant_y = len(only_significant_y)
				if num_only_significant_x > 0:
					print(f"{num_only_significant_x} points are only significant for x-axis ({x_axis_desc}) vs y-axis ({y_axis_desc})")
				if num_only_significant_y > 0:
					print(f"{num_only_significant_y} points are only significant for y-axis ({y_axis_desc}) vs x-axis ({x_axis_desc})")

				# Create -log10 p-values
				merged_ss['-log10(P_x)'] = -1 * np.log10(merged_ss['P_x'])
				merged_ss['-log10(P_y)'] = -1 * np.log10(merged_ss['P_y'])

				# Update the set of categories encountered
				all_rel_sign_cats.update(merged_ss['Relative Significance'].unique())

				# Scatter plot
				scatter = sns.scatterplot(
					data=merged_ss,
					x='-log10(P_x)',
					y='-log10(P_y)',
					hue='Relative Significance',
					palette=palette,
					hue_order=hue_order,
					s=15,
					alpha=0.5,
					edgecolor=None,
					ax=ax
				)

				# Get handles and labels for the legend from the first plot
				if handles is None and labels is None:
					handles, labels = scatter.get_legend_handles_labels()

				# Disable legend for individual plots
				ax.legend_.remove()

				# Correct x=y line
				max_val = max(merged_ss['-log10(P_x)'].max(), merged_ss['-log10(P_y)'].max())
				ax.plot([0, max_val], [0, max_val], 'k--')

				# Make the plot square
				ax.set_aspect('equal', adjustable='box')

				# Label rows and columns
				if j == 0:
					ax.set_ylabel(
						insert_newlines(y_axis_desc, max_title_length_y),
						rotation='horizontal',
						ha='center',
						va='center',
						labelpad=90
					)
				if i == len(y_axis_cov_sets) - 1:
					ax.set_xlabel(insert_newlines(x_axis_desc, max_title_length_x))

		# After all subplots are created, filter out categories not used
		label_to_handle = dict(zip(labels, handles))
		final_labels = [l for l in hue_order if l in all_rel_sign_cats]
		final_handles = [label_to_handle[l] for l in final_labels]

		# Create a single legend outside the plots
		fig.legend(final_handles, final_labels, loc='center right', ncol=1)

		# Add title
		fig.suptitle(f"{pheno} GWAS -log10(p-value) Comparison", fontsize=16)

		# Adjust layout
		plt.tight_layout(rect=[0, 0.01, 0.8, 0.95])

		plt.savefig(f"{scatter_out_dir}/{pheno}/{pheno}_scatter_plots.png")

		# plt.show()
		plt.close()









