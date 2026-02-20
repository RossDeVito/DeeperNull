"""Plot improvement of full PGS model over null model for binary
classification phenotypes."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

PHENO_ALIAS = {
	"asthma_42015": "Asthma",
	"depression_20438": "Depression",
	"diabetes_2443": "Diabetes",
}


def add_avg_prec_row(pheno, desc, score, list_of_dicts):
	list_of_dicts.append({
		'pheno': PHENO_ALIAS[pheno],
		'desc': desc,
		'Average Precision': score,
	})
	return list_of_dicts


if __name__ == "__main__":

	mpl.rcParams['savefig.dpi'] = 1200

	dn_scores_dir = '../dn_eval_local/scores_bin_cls'
	prs_scores_dir = '../PRS_eval_local/scores'

	# Load null model and PRS scores
	dn_scores = pd.read_csv(f'{dn_scores_dir}/scores.csv')
	prs_scores = pd.read_csv(f'{prs_scores_dir}/test_scores_bin_cls.csv')

	# Make table of scores of both types
	rows = []

	for pheno in PHENO_ALIAS.keys():

		# Get XGB3 null with all time and location covariates
		xgb_null_spatiotemporal_ap = dn_scores[
			(dn_scores['pheno'] == pheno)
			& (dn_scores['covar_set'] == 'age_sex_all_coords_time')
			& (dn_scores['model_type'] == 'xgb_3')
		]['avg_prec'].values[0]

		rows = add_avg_prec_row(
			pheno,
			"XGB null w/ spatiotemporal covariates",
			xgb_null_spatiotemporal_ap,
			rows
		)

		# Get standard PRS model
		standard_prs_ap = prs_scores[
			(prs_scores['pheno'] == pheno)
			& (prs_scores['model_desc'] == 'BASIL (lin: age, sex, PCs)')
		]['average_precision'].values[0]

		rows = add_avg_prec_row(
			pheno,
			"Standard BASIL PGS model",
			standard_prs_ap,
			rows
		)

		# Get PRS with XGB null with all time and location covariates
		prs_xgb_null_spatiotemporal_ap = prs_scores[
			(prs_scores['pheno'] == pheno)
			& (prs_scores['model_desc'] == 'BASIL (lin: age, sex, locations, times, PCs; XGB null: age, sex, locations, times)')
		]['average_precision'].values[0]

		rows = add_avg_prec_row(
			pheno,
			"BASIL PGS w/ XGB null & spatiotemporal covariates",
			prs_xgb_null_spatiotemporal_ap,
			rows
		)

	score_table = pd.DataFrame(rows)

	# Plot
	
	# Use Paired palette, skipping the first color
	g = sns.catplot(
		data=score_table,
		x='pheno',
		y='Average Precision',
		hue='desc',
		kind='bar',
		height=6,
		aspect=1.5,
		palette=sns.color_palette("Paired")[1:],
		legend='full',
	)
	sns.move_legend(
		g,
		"upper right",
		bbox_to_anchor=(1, 0.8),
		title='Model Type',
	)
	g.set_xticklabels(rotation=45, horizontalalignment='right')
	g.set_axis_labels("Phenotype", "Average Precision")
	plt.title("Comparison of PGS and Null Model Average Precision by Phenotype")
	plt.tight_layout()
	plt.show()