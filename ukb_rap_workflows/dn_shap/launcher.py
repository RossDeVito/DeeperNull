"""Script to get DeepNull Shapley values on UKB.

Required args:

* -d, --desc (str): Description of the analysis to be used in job name.
	Typically phenotype name.
* -c, --covar_set (str): Name of the covariate set to use as input.
	Corresponds to a file in covar_dir.
* -m, --model_files (str or list of str): Path(s) to one or more model save
	files to get Shapley values and 1st-order Shapley Interaction Index for.
* -s, --save_dir (str): Directory to save output JSON file to. File will be
	named "shapley_values.json".

Optional args:

* -p, --pred_samples (str): Filename of file containing sample IDs to compute
	Shapley values for. File should be in samp_dir. If not provided, default is
	'test_iids.txt'.
* -t, --model_type (str): Type of models. Options are "linear", "xgb", and "nn".
	Default is "xgb". NOTE: Only "xgb" supported.
* --classification (bool): Whether the model is a classification model. Default
	is False.
* --covar-dir: Directory containing the covariate files. Default:
	'/rdevito/deep_null/data/covar'
* --samp_dir (str): Path to storage directory containing sample files.
	Default: '/rdevito/deep_null/data/sample'

"""

import argparse
import dxpy
import sys


WORKFLOW_ID = "workflow-GzQxgJjJv7BBqPyQ6BX64PFb"
DEFAULT_INSTANCE = 'mem1_ssd1_v2_x16'


def parse_args():
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument(
		'-d', '--desc',
		required=True,
		help='Description of the analysis to be used in job name. Typically '
			'phenotype name.'
	)
	parser.add_argument(
		'-c', '--covar_set',
		required=True,
		help='Name of the covariate set to use as input. Corresponds to a file '
			'in covar_dir.'
	)
	parser.add_argument(
		'-m', '--model_files',
		nargs='+',
		required=True,
		help='Path(s) to one or more model save files to get Shapley values '
			'and 1st-order Shapley Interaction Index for.'
	)
	parser.add_argument(
		'-s', '--save_dir',
		required=True,
		default='.',
		help='Directory to save output JSON file to. File will be named '
			'"shapley_values.json".'
	)
	parser.add_argument(
		'-p', '--pred_samples',
		default='test_iids.txt',
		help='Filename of file containing sample IDs to compute Shapley values '
			 'for. File should be in samp_dir. If not provided, default is "test_iids.txt".'
	)
	parser.add_argument(
		'-t', '--model_type',
		default='xgb',
		choices=['linear', 'xgb', 'nn'],
		help='Type of models. Options are "linear", "xgb", and "nn". Default is '
			'"xgb". NOTE: Only "xgb" supported.'
	)
	parser.add_argument(
		'--classification',
		action='store_true',
		help='Whether the model is a classification model. Default is False.'
	)
	parser.add_argument(
		'--covar-dir',
		default='/rdevito/deep_null/data/covar',
		help='Directory containing the covariate files.'
	)
	parser.add_argument(
		'--samp_dir',
		default='/rdevito/deep_null/data/sample',
		help='Path to storage directory containing sample files.'
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


if __name__ == '__main__':
	args = parse_args()
	
	# Get workflow
	workflow = dxpy.dxworkflow.DXWorkflow(dxid=WORKFLOW_ID)

	# Get data links for inputs
	model_links = [
		get_dxlink_from_path(model_file) for model_file in args.model_files
	]
	covar_link = get_dxlink_from_path(f'{args.covar_dir}/{args.covar_set}.tsv')
	pred_samp_link = get_dxlink_from_path(f'{args.samp_dir}/{args.pred_samples}')

	# Run workflow
	prefix='stage-common.'
	workflow_input = {
		f"{prefix}model_files": model_links,
		f"{prefix}covar_file": covar_link,
		f"{prefix}pred_samples": pred_samp_link,
		f"{prefix}model_type": str(args.model_type),
		f"{prefix}classification": args.classification,
	}
	analysis = workflow.run(
		workflow_input,
		folder=args.save_dir,
		name=f'SHAP_{args.desc}',
		instance_type=DEFAULT_INSTANCE,
		priority='high',
		ignore_reuse=True
	)