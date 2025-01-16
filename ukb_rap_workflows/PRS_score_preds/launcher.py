"""Launch PRS scoring and plotting workflow.

Required args:

* -m, --model-type: Model type. One of "prsice" or "basil_{save_subdir_name}"
* -p, --pheno-name: Phenotype name
"""

import argparse

import dxpy


WORKFLOW_ID = 'workflow-Gv04Gp8Jv7B7Gx1G7B0J3G74'
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

	# Set model output dir and model dir
	if args.model_type == 'prsice':
		raise NotImplementedError('PRSice2 not implemented yet')
		model_out_dir = '/rdevito/nonlin_prs/sum_stats_prs/PRSice2/prsice2_output'
		model_dir = f'{model_out_dir}/{args.pheno_name}'
		if args.wb:
			model_dir = f'{model_dir}_wb'
	# If first part of args.model_type is 'basil'
	elif args.model_type.startswith('basil_'):
		model_out_dir = f'/rdevito/deep_null/output/PRS_BASIL/{args.pheno_name}'
		save_subdir_name = args.model_type.replace('basil_', '')
		model_dir = f'{model_out_dir}/{save_subdir_name}'

	# If first part of args.model_type is 'automl'
	elif args.model_type.startswith('automl'):
		raise NotImplementedError('AutoML not implemented yet')
		model_out_dir = '/rdevito/nonlin_prs/automl_prs/output'
		model_desc = '_'.join(args.model_type.split('_')[1:])
		model_dir = f'{model_out_dir}/{args.pheno_name}'
		if args.wb:
			model_dir = f'{model_dir}_wb'
		model_dir = f'{model_dir}_{model_desc}'
	else:
		raise ValueError(f'Invalid model type: {args.model_type}')
	
	# Set pheno and split file paths
	pheno_file = f'{PHENO_DIR}/{args.pheno_name}.pheno'

	# Launch workflow
	name = f'score_prs_preds_{args.pheno_name}_{args.model_type}'

	launch_workflow(
		model_dir,
		pheno_file,
		instance_type=DEFAULT_INSTANCE,
		name=name
	)
