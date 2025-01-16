"""Launch genotype preprocessing step 3 workflow.

See README.md for more details.
"""

import dxpy


WORKFLOW_ID = 'workflow-GqkQG5jJv7BP8bBbJb3K29Yf'
DEFAULT_INSTANCE = 'mem2_ssd2_v2_x64'

BGEN_FILE = '/rdevito/deep_null/data/geno/prepro_intermediate/allchr_step2.bgen'
SAMPLE_FILE = '/rdevito/deep_null/data/geno/prepro_intermediate/chr1_step1.sample'

OUTPUT_DIR = "/rdevito/deep_null/data/geno/QCed_common"


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

	# Create links to files
	bgen_link = get_dxlink_from_path(BGEN_FILE)
	sample_link = get_dxlink_from_path(SAMPLE_FILE)

	# Set up workflow input
	prefix = 'stage-common.'
	workflow_input = {
		f'{prefix}bgen_file': bgen_link,
		f'{prefix}sample_file': sample_link,
	}

	# Run workflow
	analysis = workflow.run(
		workflow_input,
		folder=OUTPUT_DIR,
		name=f'geno_prepro_step3',
		instance_type=DEFAULT_INSTANCE,
		priority='high',
		ignore_reuse=True
	)
	print(f"Started geno prepro step 3 ({analysis.get_id()})\n")