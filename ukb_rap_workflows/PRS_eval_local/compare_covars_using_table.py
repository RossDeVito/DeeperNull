""" Loads supplemental table, then checks if using all covariates
(with and without PCs) is ever worse than the best option in terms
of being outside the 95% confidence interval.
"""

import os
import pandas as pd


if __name__ == '__main__':

	# Options
	# metric = 'r2'
	metric = 'mae'

	# Load supplemental table
	supp_table_path = os.path.join(
		'sup_table', 'sup_table_results_long.csv'
	)
	sup_df = pd.read_csv(supp_table_path)

	# Filter
	sup_df = sup_df[sup_df['metric'] == metric]
	basil_df = sup_df[sup_df['model_type'] == 'BASIL']
	prscs_df = sup_df[sup_df['model_type'] == 'PRScs']

	basil_wo_pc_df = basil_df[
		(basil_df['covar_set'] == 'age, sex, locations, times')
		& (basil_df['uses_null'] == True)
	].reset_index(drop=True)
	basil_pc_df = basil_df[
		(basil_df['covar_set'] == 'age, sex, locations, times, PCs for null')
		& (basil_df['uses_null'] == True)
	].reset_index(drop=True)

	prscs_wo_pc_df = prscs_df[
		(prscs_df['covar_set'] == 'age, sex, locations, times')
		& (prscs_df['uses_null'] == True)
	].reset_index(drop=True)
	prscs_pc_df = prscs_df[
		(prscs_df['covar_set'] == 'age, sex, locations, times, PCs for null')
		& (prscs_df['uses_null'] == True)
	].reset_index(drop=True)

	# Get best score rows per phenotype

	if metric in ['r2']:
		best_basil_scores = basil_df.loc[
			basil_df.groupby('pheno')['value'].idxmax()
		].reset_index(drop=True)
		best_prscs_scores = prscs_df.loc[
			prscs_df.groupby('pheno')['value'].idxmax()
		].reset_index(drop=True)
	else:
		best_basil_scores = basil_df.loc[
			basil_df.groupby('pheno')['value'].idxmin()
		].reset_index(drop=True)
		best_prscs_scores = prscs_df.loc[
			prscs_df.groupby('pheno')['value'].idxmin()
		].reset_index(drop=True)

	# Check if any corresponding values for a phenotype are outside the CI
	# of the best score (only need to check one direction since all are max)

	print('BASIL w/o PCs outside CI:')
	for i, row in basil_wo_pc_df.iterrows():
		pheno = row['pheno']
		val = row['value']
		best_row = best_basil_scores[best_basil_scores['pheno'] == pheno]
		if metric in ['r2']:
			lower_ci = best_row['lower'].values[0]
			if val < lower_ci:
				print(f'  {pheno}: {val} < {lower_ci}')
		else:
			lower_ci = best_row['upper'].values[0]
			if val > lower_ci:
				print(f'  {pheno}: {val} > {lower_ci}')

	print('BASIL with PCs outside CI:')
	for i, row in basil_pc_df.iterrows():
		pheno = row['pheno']
		val = row['value']
		best_row = best_basil_scores[best_basil_scores['pheno'] == pheno]
		if metric in ['r2']:
			lower_ci = best_row['lower'].values[0]
			if val < lower_ci:
				print(f'  {pheno}: {val} < {lower_ci}')
		else:
			lower_ci = best_row['upper'].values[0]
			if val > lower_ci:
				print(f'  {pheno}: {val} > {lower_ci}')

	print('PRScs w/o PCs outside CI:')
	for i, row in prscs_wo_pc_df.iterrows():
		pheno = row['pheno']
		val = row['value']
		best_row = best_prscs_scores[best_prscs_scores['pheno'] == pheno]
		if metric in ['r2']:
			lower_ci = best_row['lower'].values[0]
			if val < lower_ci:
				print(f'  {pheno}: {val} < {lower_ci}')
		else:
			lower_ci = best_row['upper'].values[0]
			if val > lower_ci:
				print(f'  {pheno}: {val} > {lower_ci}')

	print('PRScs with PCs outside CI:')
	for i, row in prscs_pc_df.iterrows():
		pheno = row['pheno']
		val = row['value']
		best_row = best_prscs_scores[best_prscs_scores['pheno'] == pheno]
		if metric in ['r2']:
			lower_ci = best_row['lower'].values[0]
			if val < lower_ci:
				print(f'  {pheno}: {val} < {lower_ci}')
		else:
			lower_ci = best_row['upper'].values[0]
			if val > lower_ci:
				print(f'  {pheno}: {val} > {lower_ci}')




