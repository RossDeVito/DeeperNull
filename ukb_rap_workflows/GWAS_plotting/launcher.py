"""Launch Manhattan and QQ plot workflow."""

import dxpy


WORKFLOW_ID = 'workflow-Gqq21QQJv7B2yF681p8P6581'
DEFAULT_INSTANCE = 'mem1_ssd1_v2_x8'

# Options
PHENOS = [
	# "standing_height_50",
	# "body_fat_percentage_23099",
	"platelet_count_30080",
	"glycated_haemoglobin_30750",
	# "vitamin_d_30890",
	"diastolic_blood_pressure_4079",
	"systolic_blood_pressure_4080",
	"FEV1_3063",
	"FVC_3062",
	"HDL_cholesterol_30760",
	"LDL_direct_30780",
	"triglycerides_30870",
	"c-reactive_protein_30710",
	"creatinine_30700",
	"alanine_aminotransferase_30620",
	"aspartate_aminotransferase_30650",
]

COVAR_SETS = [
	# 'age_sex_pc',
	# 'age_sex_pc_null_xgb_3_age_sex',
	# 'age_sex_all_coords_pc',
	'age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords',
]

# Output directory
OUTPUT_DIR = "/rdevito/deep_null/output/GWAS_plink"
SS_FNAME = "gwas_plink2.{:}.glm.linear"	# .format(PHENO)


def get_dxlink_from_path(path_to_link):
	"""Get dxlink from path."""
	print(f'Finding data object for {path_to_link}', flush=True)
	return dxpy.dxlink(
		list(dxpy.find_data_objects(
			name=path_to_link.split('/')[-1],
			folder='/'.join(path_to_link.split('/')[:-1]),
			project=dxpy.PROJECT_CONTEXT_ID
		))[0]['id']
	)


if __name__ == '__main__':

	# Get workflow
	workflow = dxpy.dxworkflow.DXWorkflow(dxid=WORKFLOW_ID)

	for pheno in PHENOS:
		for covar_set in COVAR_SETS:

			# Get data link to summary stats
			ss_link = get_dxlink_from_path(
				f"{OUTPUT_DIR}/{pheno}/{covar_set}/{SS_FNAME.format(pheno)}"
			)

			# Set up workflow input
			prefix = 'stage-common.'
			workflow_input = {
				f'{prefix}summary_stats': ss_link,
				f'{prefix}phenotype': pheno,
			}

			# Run workflow
			analysis = workflow.run(
				workflow_input,
				folder=f"{OUTPUT_DIR}/{pheno}/{covar_set}",
				name=f'plot_gwas_manhattan_qq_{pheno}_{covar_set}',
				instance_type=DEFAULT_INSTANCE,
				priority='high',
				ignore_reuse=True
			)
			print(f"Started plot manhattan & qq for {pheno} {covar_set} ({analysis.get_id()})\n")