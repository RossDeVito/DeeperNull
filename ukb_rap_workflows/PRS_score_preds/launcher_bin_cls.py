"""Launch PRS scoring and plotting workflow.

Required args:

* -m, --model-type: Model type. One of "prsice" or "basil_{save_subdir_name}"
* -p, --pheno-name: Phenotype name
"""

import argparse

import dxpy


WORKFLOW_ID = 'workflow-J05PP9jJv7BPB0zxvZ3F2G7P'
DEFAULT_INSTANCE = 'mem1_ssd1_v2_x2'

PHENO_DIR = '/rdevito/nonlin_prs/data/pheno_data/pheno'


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-m', '--model-type',
		required=True,
		help='Model type'
	)
	parser.add_argument(
		'-p', '--pheno-name',
		required=True,
		help='Phenotype name'
	)
	return parser.parse_args()


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


def launch_workflow(
	model_dir,
	pheno_file,
	instance_type=DEFAULT_INSTANCE,
	name='score_prs_preds'
):
	"""Launch PRS scoring and plotting workflow."""
	
	# Get links
	val_pred_link = get_dxlink_from_path(f'{model_dir}/val_preds.csv')
	test_pred_link = get_dxlink_from_path(f'{model_dir}/test_preds.csv')
	pheno_link = get_dxlink_from_path(pheno_file)

	# Set up workflow input
	prefix = 'stage-common.'
	workflow_input = {
		f'{prefix}val_preds': val_pred_link,
		f'{prefix}test_preds': test_pred_link,
		f'{prefix}pheno_file': pheno_link,
	}

	# Get workflow
	workflow = dxpy.dxworkflow.DXWorkflow(dxid=WORKFLOW_ID)

	# Run workflow
	analysis = workflow.run(
		workflow_input,
		folder=model_dir,
		name=name,
		instance_type=instance_type,
		ignore_reuse=True
	)
	print("Started analysis %s (%s)\n"%(analysis.get_id(), name))

	return analysis


if __name__ == '__main__':

	args = parse_args()

	# If first part of args.model_type is 'basil'
	if args.model_type.startswith('basil_'):
		model_out_dir = f'/rdevito/deep_null/output/PRS_BASIL/{args.pheno_name}'
		save_subdir_name = args.model_type.replace('basil_', '')
		model_dir = f'{model_out_dir}/{save_subdir_name}'

	elif args.model_type.startswith('prscs'):
		model_out_dir = f'/rdevito/deep_null/output/PRS_PRScs/{args.pheno_name}'
		save_subdir_name = args.model_type.replace('prscs_', '')
		model_dir = f'{model_out_dir}/{save_subdir_name}'

	else:
		raise ValueError(f'Invalid model type: {args.model_type}')
	
	# Set pheno and split file paths
	pheno_file = f'{PHENO_DIR}/{args.pheno_name}.pheno'

	# Launch workflow
	name = f'score_prs_preds_bin_cls_{args.pheno_name}_{args.model_type}'

	launch_workflow(
		model_dir,
		pheno_file,
		instance_type=DEFAULT_INSTANCE,
		name=name
	)
