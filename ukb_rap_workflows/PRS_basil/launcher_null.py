"""Launch BASIL PRS workflow.

Uses PGEN input genotype file.

Required args:

* -p, --pheno-name: Phenotype name. Must be the file name of a 
	phenotype file in --pheno-dir without the '.pheno' extension.
* -m, --model-type: One of 'lasso', 'ridge', 'elastic_net_0_1',
	'elastic_net_0_5', or 'elastic_net_0_9'. Sets the alpha parameter
	to 1, 1e-3, 0.1, 0.5, or 0.9, respectively.
* --covar-set: Name of covariate set file in covar_dir w/o .tsv extension.
* --null-covar-set: Name of covariate set used for null model.
* --null-model: Name of the null model used for predictions.

Optional args:

* -n, --n-iter: Number of iterations to train the model. Default: 50 unless
	--dev is provided, in which case it is 2.
* --pheno-dir: Directory containing the phenotype files. Default:
	'/rdevito/nonlin_prs/data/pheno_data/pheno'
* --geno-dir: Directory containing the genotype files. Default: 
	'/rdevito/nonlin_prs/data/geno_data/qced_common/pgen'
* --splits-dir: Directory containing train/val/test splits in
	the form of list of sample IDs. Default: 
	'/rdevito/deep_null/data/sample'
* --covar-dir: Directory containing the covariate files. Default:
	'/rdevito/deep_null/data/covar'
* --null-dir: Directory containing the null model predictions. Default:
	'/rdevito/deep_null/dn_output/V4'
* --output-dir: Directory in which a folder will be created to store
	the output of the PRS workflow. Default: 
	'/rdevito/deep_null/output/PRS_BASIL'. Final
	output directory will be of the form: 
	{output_dir}/{pheno_name}/{covar_set}_null_{null_model}_{null_covar_set}_{model_type}_{n_iter}
"""

import argparse
import json

import dxpy

WORKFLOW_ID = 'workflow-Gv3K038Jv7B982XQjF2JXbjp'
# DEFAULT_INSTANCE = 'mem2_ssd1_v2_x64'
DEFAULT_INSTANCE = 'mem3_ssd1_v2_x32'


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


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-p', '--pheno-name',
		required=True, 
		help='Phenotype name. Must be the file name of a phenotype file in '
			'--pheno-dir without the \'.pheno\' extension.'
	)
	parser.add_argument(
		'--covar-set',
		required=True,
		help='Name of covariate set file in covar_dir w/o .tsv extension.'
	)
	parser.add_argument(
		'--null-covar-set',
		required=True,
		help='Name of covariate set used for null model.'
	)
	parser.add_argument(
		'--null-model',
		required=True,
		help='Name of the null model used for predictions.'
	)
	parser.add_argument(
		'-m', '--model-type',
		required=True,
		choices=[
			'lasso', 'ridge',
			'elastic_net_0_1', 'elastic_net_0_5', 'elastic_net_0_9'
		],
		help='One of \'lasso\', \'ridge\', or \'elastic_net\'. Sets the alpha '
			'parameter to 1, 0, or 0.5, respectively.'
	)
	parser.add_argument(
		'-n', '--n-iter',
		type=int,
		default=50,
		help='Number of iterations to train the model. Default: 50 unless '
			'--dev is provided, in which case it is 2.'
	)
	parser.add_argument(
		'--pheno-dir',
		default='/rdevito/nonlin_prs/data/pheno_data/pheno',
		help='Directory containing the phenotype files.'
	)
	parser.add_argument(
		'--geno-dir',
		default='/rdevito/nonlin_prs/data/geno_data/qced_common/pgen',
		help='Directory containing the genotype files.'
	)
	parser.add_argument(
		'--splits-dir',
		default='/rdevito/deep_null/data/sample',
		help='Directory containing train/val/test splits in the form of list '
			'of sample IDs.'
	)
	parser.add_argument(
		'--covar-dir',
		default='/rdevito/deep_null/data/covar',
		help='Directory containing the covariate files.'
	)
	parser.add_argument(
		'--null-dir',
		default='/rdevito/deep_null/dn_output/V4',
		help='Directory containing the null model predictions.'
	)
	parser.add_argument(
		'--output-dir',
		default='/rdevito/deep_null/output/PRS_BASIL',
		help='Directory in which a folder will be created to store the output '
			'of the PRS workflow. Final output directory will be of the form: '
			'{output_dir}/{pheno_name}/{covar_set}_null_{null_model}_{null_covar_set}_{model_type}_{n_iter}'
	)
	
	return parser.parse_args()


def launch_basil_workflow(
	geno_prefix,
	pheno_file,
	pheno_name,
	covar_file,
	null_train_pred_file,
	null_test_val_pred_file,
	train_samp_file,
	val_samp_file,
	test_samp_file,
	alpha,
	n_iter,
	output_dir,
	workflow_id=WORKFLOW_ID,
	instance_type=DEFAULT_INSTANCE,
	name='prs_basil_null'
):
	"""Launch BASIL PRS workflow on UKB RAP.
	
	Args:
		geno_prefix (str): Prefix of the PGEN genotype file.
		pheno_file (str): Path to the phenotype file.
		pheno_name (str): Name of the phenotype.
		covar_file (str): Path to the covariate file.
		null_train_pred_file: Path to the null model prediction file for the
			training set.
		null_test_val_pred_file: Path to the null model prediction file for the
			validation and test sets.
		train_samp_file (str): Path to the training sample IDs file.
		val_samp_file (str): Path to the validation sample IDs file.
		test_samp_file (str): Path to the test sample IDs file.
		alpha (float): Alpha parameter for the model.
		n_iter (int): Number of iterations to train the model.
		output_dir (str): Path to the output directory.
		workflow_id (str): ID of the workflow to launch. Default: WORKFLOW_ID.
		instance_type (str): Instance type to use for the workflow. Default:
			DEFAULT_INSTANCE.
		name (str): Name of the workflow. Default: 'prs_basil'.
	"""

	# Get workflow
	workflow = dxpy.dxworkflow.DXWorkflow(dxid=workflow_id)

	# Get data links for inputs
	geno_pgen_link = get_dxlink_from_path(f'{geno_prefix}.pgen')
	geno_psam_link = get_dxlink_from_path(f'{geno_prefix}.psam')
	geno_pvar_link = get_dxlink_from_path(f'{geno_prefix}.pvar')
	pheno_link = get_dxlink_from_path(pheno_file)
	covar_link = get_dxlink_from_path(covar_file)
	null_train_pred_link = get_dxlink_from_path(null_train_pred_file)
	null_test_val_pred_link = get_dxlink_from_path(null_test_val_pred_file)
	train_samp_link = get_dxlink_from_path(train_samp_file)
	val_samp_link = get_dxlink_from_path(val_samp_file)
	test_samp_link = get_dxlink_from_path(test_samp_file)

	# Set workflow input
	prefix = 'stage-common.'
	workflow_input = {
		f'{prefix}geno_pgen': geno_pgen_link,
		f'{prefix}geno_psam': geno_psam_link,
		f'{prefix}geno_pvar': geno_pvar_link,
		f'{prefix}pheno_file': pheno_link,
		f'{prefix}pheno_name': pheno_name,
		f'{prefix}covar_file': covar_link,
		f'{prefix}null_train_pred_file': null_train_pred_link,
		f'{prefix}null_testval_pred_file': null_test_val_pred_link,
		f'{prefix}train_samples': train_samp_link,
		f'{prefix}val_samples': val_samp_link,
		f'{prefix}test_samples': test_samp_link,
		f'{prefix}alpha': alpha,
		f'{prefix}n_iter': n_iter,
	}

	# Run workflow
	analysis = workflow.run(
		workflow_input,
		folder=output_dir,
		name=name,
		instance_type=instance_type,
		priority='high',
		ignore_reuse=True
	)
	print("Started analysis %s (%s)\n"%(analysis.get_id(), name))

	return analysis
	

if __name__ == '__main__':

	# Parse args
	args = parse_args()
	print(f'Phenotype: {args.pheno_name}')
	print(f'Model type: {args.model_type}')

	# Set alpha and number of iterations
	if args.model_type == 'lasso':
		alpha = 1.0
	elif args.model_type == 'ridge':
		alpha = 1e-3
	elif args.model_type == 'elastic_net_0_1':
		alpha = 0.1
	elif args.model_type == 'elastic_net_0_5':
		alpha = 0.5
	elif args.model_type == 'elastic_net_0_9':
		alpha = 0.9
	else:
		raise ValueError(
			"Invalid model type. Must be one of 'lasso', 'ridge', "
			"'elastic_net_0_1', 'elastic_net_0_5', or 'elastic_net_0_9'."
		)
	
	n_iter = args.n_iter

	# Get the covariate set to use
	covar_set = args.covar_set
	print(f'Using covariate set {covar_set}.')

	# Set paths to null model predictions
	null_pred_dir = f'{args.null_dir}/{args.pheno_name}/{args.null_covar_set}/{args.null_model}'
	null_train_pred_file = f'{null_pred_dir}/ho_preds.csv'
	null_test_val_pred_file = f'{null_pred_dir}/ens_preds.csv'

	# Set sample split ID paths
	train_samp_fname = f'{args.splits_dir}/train_iids.txt'
	val_samp_fname = f'{args.splits_dir}/val_iids.txt'
	test_samp_fname = f'{args.splits_dir}/test_iids.txt'

	print(f'Train sample IDs file: {train_samp_fname}')
	print(f'Val sample IDs file: {val_samp_fname}')
	print(f'Test sample IDs file: {test_samp_fname}')

	# Set the output directory
	desc = f'{args.covar_set}_null_{args.null_model}_{args.null_covar_set}_{args.model_type}_{n_iter}'
	output_dir = f'{args.output_dir}/{args.pheno_name}/{desc}'
	print(f'Output directory: {output_dir}')

	# Launch the workflow
	job_name = f'prs_basil_{args.pheno_name}_{desc}'
	print(f'Launching BASIL workflow with name: {job_name}')

	launch_basil_workflow(
		geno_prefix=f'{args.geno_dir}/allchr_wbqc',
		# geno_prefix=f'{args.geno_dir}/allchr_allqc_dev',	# For testing
		pheno_file=f'{args.pheno_dir}/{args.pheno_name}.pheno',
		pheno_name=args.pheno_name,
		covar_file=f'{args.covar_dir}/{covar_set}.tsv',
		null_train_pred_file=null_train_pred_file,
		null_test_val_pred_file=null_test_val_pred_file,
		train_samp_file=train_samp_fname,
		val_samp_file=val_samp_fname,
		test_samp_file=test_samp_fname,
		alpha=alpha,
		n_iter=n_iter,
		output_dir=output_dir,
		name=job_name
	)