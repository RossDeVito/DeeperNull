"""Evaluate results of paired Fisher-Pitman tests.

Implementation of test not included, but based on:
https://github.com/ybestgen/BootCIRealData/blob/main/Fisher_Pitman_paired_test.py
"""

import os

import pandas as pd


if __name__ == "__main__":

	paired_res_dir = 'fisher_pitman_paired'

	basil_reg_fname = 'basil_reg_n50000.csv'
	basil_cls_fname = 'basil_cls_n50000.csv'
	prscs_reg_fname = 'prscs_reg_n50000.csv'

	# Load results
	basil_reg_res = pd.read_csv(os.path.join(paired_res_dir, basil_reg_fname))
	basil_cls_res = pd.read_csv(os.path.join(paired_res_dir, basil_cls_fname))
	prscs_reg_res = pd.read_csv(os.path.join(paired_res_dir, prscs_reg_fname))

	# Merge results into one table
	res_df = pd.concat([
		basil_reg_res.assign(model='BASIL regression'),
		basil_cls_res.assign(model='BASIL classification'),
		prscs_reg_res.assign(model='PRS-CS regression'),
	], ignore_index=True)

	# Correct 'p-val' col by total rows (Bonferroni correction)
	res_df['p-val_bonf_corrected'] = res_df['p-val'] * len(res_df)

	# Print some summary stats
	print(
		"Percentage of comparisons with nominal p < 0.05:",
		(res_df['p-val'] < 0.05).mean()
	)
	print(
		"Percentage of comparisons with Bonferroni-corrected p < 0.05:",
		(res_df['p-val_bonf_corrected'] < 0.05).mean()
	)

	# Print for each 'model' cases where not significant with correction
	pd.set_option('display.max_colwidth', None)
	for model in res_df['model'].unique():
		print(f"Comparisons for {model} with Bonferroni-corrected p >= 0.05:")
		print(res_df[
				(res_df['model'] == model)
				& (res_df['p-val_bonf_corrected'] >= 0.05)
			][
				['pheno', 'covars', 'p-val']
			].reset_index(drop=True)
		)
