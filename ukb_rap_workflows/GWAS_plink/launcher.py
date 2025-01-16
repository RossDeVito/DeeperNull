"""Launch plink2 GWAS workflow on the UKB RAP.

Uses PGEN input genotype file.

Example usage:
```
python launcher.py -p platelet_count_30080 --covar-set age_sex_pc
```

Required args:

* -p, --pheno-name: Name of the phenotype to use for the GWAS.
* --covar-set: Name of covariate set file in covar_dir w/o .tsv extension.

Optional args:

* --pheno-dir: Directory containing the phenotype files. Default:
	'/rdevito/nonlin_prs/data/pheno_data/pheno'
* --geno-dir: Directory containing the genotype files. Default: 
	'/rdevito/nonlin_prs/data/geno_data/qced_common/pgen'
* --splits-dir: Directory containing train/val/test splits in
	the form of list of sample IDs. Default: 
	'/rdevito/deep_null/data/sample'
* --covar-dir: Directory containing the covariate files. Default:
	'/rdevito/deep_null/data/covar'
* --output-dir: Directory in which a folder will be created to store
	the output of the GWAS workflow. Default: 
	'/rdevito/deep_null/output/GWAS_plink'. Final output directory
	will be of the form: {output_dir}/{pheno_name}_glm[_wb][_dev]
"""

import argparse
from pprint import pprint

import dxpy


WORKFLOW_ID = 'workflow-Gv46zkjJv7BF965QqpFFgy8Z'
DEFAULT_INSTANCE = 'mem2_ssd1_v2_x64'


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-p', '--pheno-name',
		required=True,
		help='Name of the phenotype to use for the GWAS.'
	)
	parser.add_argument(
		'--covar-set',
		required=True,
		help='Name of covariate set file in covar_dir w/o .tsv extension.'
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
		help='Directory containing train/val/test splits in the form of '
			 'list of sample IDs.'
	)
	parser.add_argument(
		'--covar-dir',
		default='/rdevito/deep_null/data/covar',
		help='Directory containing the covariate files.'
	)
	parser.add_argument(
		'--output-dir',
		default='/rdevito/deep_null/output/GWAS_plink',
		help='Directory in which a folder will be created to store the output '
			 'of the GWAS workflow. Final output directory will be of the form: '
			 '{output_dir}/{pheno_name}_glm[_wb][_dev]'
	)
	return parser.parse_args()


def launch_gwas_workflow(
	geno_file,
	covar_file,
	pheno_file,
	split_file,
	output_dir,
	workflow_id=WORKFLOW_ID,
	instance_type=DEFAULT_INSTANCE,
	name='gwas_plink2'
):
	"""Launch the GWAS workflow on the UKB RAP.
	
	Args:
		geno_file: Path to the PGEN genotype file in UKB RAP storage.
			The filename should exclude the .pgen/.psam/.pvar extensions.
		covar_file: Path to the covariate file in UKB RAP storage.
		pheno_file: Path to the phenotype file in UKB RAP storage.
		split_file: Path to the train split file in UKB RAP storage.
		output_dir: Path to the output directory in UKB RAP storage.
		workflow_id: ID of the plink2 GWAS workflow on the UKB RAP.
			Defaults to WORKFLOW_ID constant.
		instance_type: Instance type to use for the workflow. Defaults
			to DEFAULT_INSTANCE constant.
		name: Name of the job. Defaults to 'gwas_plink2'.
	"""

	# Get workflow
	workflow = dxpy.dxworkflow.DXWorkflow(dxid=workflow_id)

	# Get data links for inputs
	geno_pgen_link = dxpy.dxlink(
		list(dxpy.find_data_objects(
			name=geno_file.split('/')[-1] + '.pgen',
			folder='/'.join(geno_file.split('/')[:-1]),
			project=dxpy.PROJECT_CONTEXT_ID
		))[0]['id']
	)
	geno_psam_link = dxpy.dxlink(
		list(dxpy.find_data_objects(
			name=geno_file.split('/')[-1] + '.psam',
			folder='/'.join(geno_file.split('/')[:-1]),
			project=dxpy.PROJECT_CONTEXT_ID
		))[0]['id']
	)
	geno_pvar_link = dxpy.dxlink(
		list(dxpy.find_data_objects(
			name=geno_file.split('/')[-1] + '.pvar',
			folder='/'.join(geno_file.split('/')[:-1]),
			project=dxpy.PROJECT_CONTEXT_ID
		))[0]['id']
	)
	covar_link = dxpy.dxlink(
		list(dxpy.find_data_objects(
			name=covar_file.split('/')[-1],
			folder='/'.join(covar_file.split('/')[:-1]),
			project=dxpy.PROJECT_CONTEXT_ID
		))[0]['id']
	)
	pheno_link = dxpy.dxlink(
		list(dxpy.find_data_objects(
			name=pheno_file.split('/')[-1],
			folder='/'.join(pheno_file.split('/')[:-1]),
			project=dxpy.PROJECT_CONTEXT_ID
		))[0]['id']
	)
	split_link = dxpy.dxlink(
		list(dxpy.find_data_objects(
			name=split_file.split('/')[-1],
			folder='/'.join(split_file.split('/')[:-1]),
			project=dxpy.PROJECT_CONTEXT_ID
		))[0]['id']
	)

	# Set up workflow input
	prefix = 'stage-common.'
	workflow_input = {
		f'{prefix}geno_pgen_file': geno_pgen_link,
		f'{prefix}geno_psam_file': geno_psam_link,
		f'{prefix}geno_pvar_file': geno_pvar_link,
		f'{prefix}covar_file': covar_link,
		f'{prefix}pheno_file': pheno_link,
		f'{prefix}split_file': split_link
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
	pprint(vars(args))
	print(f'\nPhenotype: {args.pheno_name}')
	
	# Get the covariate file to use
	covar_set = args.covar_set
	print(f'Using covariate set {covar_set}.')

	# Set the output directory
	output_dir = f'{args.output_dir}/{args.pheno_name}/{covar_set}'
	print(f'Output directory: {output_dir}')

	# Launch the GWAS workflow
	job_name = f'gwas_plink_{args.pheno_name}_{covar_set}'

	print(f'Launching GWAS workflow with name: {job_name}')
	launch_gwas_workflow(
		geno_file=f'{args.geno_dir}/allchr_wbqc',
		# geno_file=f'{args.geno_dir}/allchr_allqc_dev',	# For testing
		covar_file=f'{args.covar_dir}/{covar_set}.tsv',
		pheno_file=f'{args.pheno_dir}/{args.pheno_name}.pheno',
		split_file=f'{args.splits_dir}/train_iids.txt',
		output_dir=output_dir,
		name=job_name
	)