"""Launch genotype preprocessing step 2 workflow.

See README.md for more details.
"""

import dxpy


WORKFLOW_ID = 'workflow-Gqk1pv8Jv7B5FBJf1Vk8p8gX'
DEFAULT_INSTANCE = 'mem2_ssd2_v2_x64'

BGEN_FILE_TEMPLATE = '/rdevito/deep_null/data/geno/prepro_intermediate/chr{:}_step1.bgen'

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

	# Create list of BGEN files
	bgen_links = [
		get_dxlink_from_path(BGEN_FILE_TEMPLATE.format(chr)) for chr in chromosomes
	]

	# Set up workflow input
	prefix = 'stage-common.'
	workflow_input = {
		f'{prefix}bgen_files': bgen_links,
	}

	# Run workflow
	analysis = workflow.run(
		workflow_input,
		folder=OUTPUT_DIR,
		name=f'geno_prepro_step2',
		instance_type=DEFAULT_INSTANCE,
		priority='high',
		ignore_reuse=True
	)
	print(f"Started geno prepro step 2 ({analysis.get_id()})\n")