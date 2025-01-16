"""Launch genotype preprocessing step 1 workflow.

See README.md for more details.
"""

import dxpy


WORKFLOW_ID = 'workflow-Gqk1FG8Jv7B25Y1xGG9KyK84'
DEFAULT_INSTANCE = 'mem1_ssd1_v2_x72'

# BGEN and filtered variants files
BGEN_FILE = "/Bulk/Imputation/UKB imputation from genotype/ukb22828_c{:}_b0_v3.bgen"
SAMPLE_FILE = "/Bulk/Imputation/UKB imputation from genotype/ukb22828_c{:}_b0_v3.sample"
FILTERED_VARS_FILE = "/rdevito/nonlin_prs/data/geno_data/filtered_SNPs/common_init_qc_chr{:}.txt"

# White British sample file w/ initial QC
FILTERED_SAMP_FILE = "/rdevito/nonlin_prs/data/sample_data/passed_init_qc_white_british.csv"

# Output directory
OUTPUT_DIR = "/rdevito/deep_null/data/geno/prepro_intermediate"


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

	chromosomes = list(range(1, 23))

	for chr in chromosomes:

		# Get data links for inputs
		bgen_link = get_dxlink_from_path(BGEN_FILE.format(chr))
		sample_link = get_dxlink_from_path(SAMPLE_FILE.format(chr))
		filtered_vars_link = get_dxlink_from_path(FILTERED_VARS_FILE.format(chr))
		filtered_samp_link = get_dxlink_from_path(FILTERED_SAMP_FILE)

		# Set up workflow input
		prefix = 'stage-common.'
		workflow_input = {
			f'{prefix}chromosome_number': chr,
			f'{prefix}bgen_file': bgen_link,
			f'{prefix}sample_file': sample_link,
			f'{prefix}filtered_vars_file': filtered_vars_link,
			f'{prefix}filtered_samples_file': filtered_samp_link,
		}

		# Run workflow
		analysis = workflow.run(
			workflow_input,
			folder=OUTPUT_DIR,
			name=f'geno_prepro_step1_chr{chr}',
			instance_type=DEFAULT_INSTANCE,
			priority='high',
			ignore_reuse=True
		)
		print(f"Started geno prepro step 1 chr {chr} ({analysis.get_id()})\n")