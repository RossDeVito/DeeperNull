import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Options
p_val_thresh = 5e-8

phenos = [
	"standing_height_50",				# 0
	"body_fat_percentage_23099",		# 1
	# "platelet_count_30080",				# 2
	# "glycated_haemoglobin_30750",		# 3
	"vitamin_d_30890",					# 4
	# "diastolic_blood_pressure_4079",	# 5
	"systolic_blood_pressure_4080",		# 6
	# "FEV1_3063",						# 7
	"FVC_3062",						# 8
	# "HDL_cholesterol_30760",			# 9
	# "LDL_direct_30780",					# 10
	# "triglycerides_30870",			# 11
	# "c-reactive_protein_30710",			# 12
	# "creatinine_30700",				# 13
	# "alanine_aminotransferase_30620",	# 14
	# "aspartate_aminotransferase_30650"# 15
]

ss_dir = "sum_stats"

# Covariate sets for comparison
x_cov = "age_sex_pc"
y_cov = "age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time"

covar_set_descs = {
	"age_sex_pc": "age, sex, PCs",
	"age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time":
		"age, sex, home & birth coords., time of day & year, PCs, Null(age, sex, home & birth coords., time of day & year)"
}

# Define color palette and order for the relative significance categories
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

# Function to insert newlines into text after a maximum length
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

sns.set_style("whitegrid")

# Arrange subplots in a grid: here we use 3 columns (adjust ncols as needed)
n_phenos = len(phenos)
ncols = 5
nrows = (n_phenos + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

# Initialize variables to capture handles and labels from the first subplot
handles, labels = None, None

for idx, pheno in enumerate(tqdm(phenos, desc="Processing phenotypes")):
	# Load summary stats for the two covariate sets
	ss_x = pd.read_csv(f"{ss_dir}/ss_{pheno}_{x_cov}.json", sep='\s+')
	ss_y = pd.read_csv(f"{ss_dir}/ss_{pheno}_{y_cov}.json", sep='\s+')
	
	# Replace 0 p-values with the lowest nonzero value for each covariate set
	for ss_df in [ss_x, ss_y]:
		low_fill_val = ss_df.loc[ss_df.P != 0, 'P'].min()
		ss_df.loc[ss_df.P == 0, 'P'] = low_fill_val

	# Merge on 'ID'
	merged_ss = pd.merge(ss_x, ss_y, on='ID', suffixes=('_x', '_y'))
	
	# Filter to only include loci significant in both sets
	merged_ss = merged_ss[(merged_ss.P_x < p_val_thresh) & (merged_ss.P_y < p_val_thresh)]
	
	# Create a new 'Relative Significance' column
	merged_ss['Relative Significance'] = 'same'
	merged_ss.loc[merged_ss.P_x < merged_ss.P_y, 'Relative Significance'] = 'more significant for x-axis'
	merged_ss.loc[merged_ss.P_x > merged_ss.P_y, 'Relative Significance'] = 'more significant for y-axis'
	merged_ss.loc[(merged_ss.P_x < p_val_thresh) & (merged_ss.P_y >= p_val_thresh), 'Relative Significance'] = 'only significant for x-axis'
	merged_ss.loc[(merged_ss.P_x >= p_val_thresh) & (merged_ss.P_y < p_val_thresh), 'Relative Significance'] = 'only significant for y-axis'
	
	# Compute -log10 p-values
	merged_ss['-log10(P_x)'] = -np.log10(merged_ss.P_x)
	merged_ss['-log10(P_y)'] = -np.log10(merged_ss.P_y)
	
	# Determine subplot axis
	ax = axes[idx // ncols, idx % ncols]
	
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
	
	# For the first subplot, capture legend handles and labels
	if handles is None and labels is None:
		handles, labels = scatter.get_legend_handles_labels()
		# Remove the extra legend title entry if it exists
		if 'Relative Significance' in labels:
			idx_title = labels.index('Relative Significance')
			del labels[idx_title]
			del handles[idx_title]
	
	# Remove individual legend from the current subplot
	leg = ax.get_legend()
	if leg is not None:
		leg.remove()
	
	# Plot the diagonal line (x = y)
	max_val = max(merged_ss['-log10(P_x)'].max(), merged_ss['-log10(P_y)'].max())
	ax.plot([0, max_val], [0, max_val], 'k--')
	
	ax.set_aspect('equal', adjustable='box')
	ax.set_title(pheno)
	ax.set_xlabel(insert_newlines(covar_set_descs[x_cov], 20))
	# Set y-axis label only for plots in the leftmost column
	if idx % ncols == 0:
		ax.set_ylabel(insert_newlines(covar_set_descs[y_cov], 30))
	else:
		ax.set_ylabel('')

# Remove any extra subplots
for idx in range(n_phenos, nrows * ncols):
	fig.delaxes(axes[idx // ncols, idx % ncols])

# Create a single legend outside the subplots using handles from the first subplot
fig.legend(handles, labels, loc='center right', ncol=1)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()
