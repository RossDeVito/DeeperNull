"""Plot improvement of full PGS model over null model for continuous phenotypes."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


PHENO_ALIAS = {
	"standing_height_50": "Standing height",
	"body_fat_percentage_23099": "Body fat percentage",
	"platelet_count_30080": "Platelet count",
	"glycated_haemoglobin_30750": "Glycated haemoglobin",
	"vitamin_d_30890": "Vitamin D",
	"diastolic_blood_pressure_4079": "Diastolic BP",
	"systolic_blood_pressure_4080": "Systolic BP",
	"FEV1_3063": "FEV1",
	"FVC_3062": "FVC",
	"HDL_cholesterol_30760": "HDL-C",
	"LDL_direct_30780": "LDL-C",
	"triglycerides_30870": "Triglycerides",
	"c-reactive_protein_30710": "CRP",
	"creatinine_30700": "Creatinine",
	"alanine_aminotransferase_30620": "ALT",
	"aspartate_aminotransferase_30650": "AST",
}


def add_r2_row(pheno, desc, score, list_of_dicts):
	list_of_dicts.append({
		'pheno': PHENO_ALIAS[pheno],
		'desc': desc,
		'R^2': score,
	})
	return list_of_dicts

if __name__ == "__main__":

	mpl.rcParams['savefig.dpi'] = 1200

	dn_scores_dir = '../dn_eval_local/scores'
	prs_scores_dir = '../PRS_eval_local/scores'

	# Load null model and PRS scores
	dn_scores = pd.read_csv(f'{dn_scores_dir}/scores.csv')
	prs_scores = pd.read_csv(f'{prs_scores_dir}/test_scores.csv')

	# Make table of scores of both types
	rows = []

	for pheno in PHENO_ALIAS.keys():

		# Get standard linear null model
		lin_null_r2 = dn_scores[
			(dn_scores['pheno'] == pheno)
			& (dn_scores['covar_set'] == 'age_sex')
			& (dn_scores['model_type'] == 'lin_reg_1')
		]['r2'].values[0]

		rows = add_r2_row(pheno, "Linear age & sex null", lin_null_r2, rows)

		# Get XGB3 null with all time and location covariates
		xgb_null_spatiotemporal_r2 = dn_scores[
			(dn_scores['pheno'] == pheno)
			& (dn_scores['covar_set'] == 'age_sex_all_coords_time')
			& (dn_scores['model_type'] == 'xgb_3')
		]['r2'].values[0]

		rows = add_r2_row(
			pheno, 
			"XGB null w/ spatiotemporal covariates", 
			xgb_null_spatiotemporal_r2, 
			rows
		)

		# Get standard PRS model
		standard_prs_r2 = prs_scores[
			(prs_scores['pheno'] == pheno)
			& (prs_scores['model_desc'] == 'BASIL (lin: age, sex, PCs)')
		]['r2'].values[0]

		rows = add_r2_row(
			pheno,
			"Standard BASIL PGS model",
			standard_prs_r2,
			rows
		)

		# Get PRS with XGB null with all time and location covariates
		prs_xgb_null_spatiotemporal_r2 = prs_scores[
			(prs_scores['pheno'] == pheno)
			& (prs_scores['model_desc'] == 'BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)')
		]['r2'].values[0]

		rows = add_r2_row(
			pheno,
			"BASIL PGS w/ XGB null & spatiotemporal covariates",
			prs_xgb_null_spatiotemporal_r2,
			rows
		)

	scores_df = pd.DataFrame(rows)

	# Seaborn 4x4 barplot of R^2 values

	g = sns.catplot(
		data=scores_df,
		x='pheno',
		y='R^2',
		hue='desc',
		kind='bar',
		height=6,
		aspect=1.5,
		palette='Paired',
		# legend_out=True,
		legend='full',
	)
	sns.move_legend(
		g,
		"upper right",
		bbox_to_anchor=(1, 0.8),
		title='Model Type',
	)
	g.set_xticklabels(rotation=45, horizontalalignment='right')
	g.set_axis_labels("Phenotype", "R²")
	plt.title("Comparison of PGS and Null Model R² by Phenotype")
	plt.tight_layout()
	plt.show()