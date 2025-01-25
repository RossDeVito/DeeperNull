"""Launch PRScs PRS workflow on the UKB RAP.

Required args:

* -p, --pheno-name: Name of the phenotype to use for the GWAS.
* --covar-set: Name of covariate set file in covar_dir w/o .tsv extension.

Optional args:

* --null-covar-set: Name of the covariate set file for the null model. If
	null-covar-set is provided, a null model will be used and --null-model
	must also be provided.
* --null-model: Name of the null model.
* --pheno-dir: Directory containing the phenotype files. Default:
	'/rdevito/nonlin_prs/data/pheno_data/pheno'
* --geno-dir: Directory containing the plink1 BED genotype files with rsID.
	Default: '/rdevito/nonlin_prs/data/geno_data/qced_common/bed_rsid'
* --splits-dir: Directory containing train/val/test splits in
	the form of list of sample IDs. Default: 
	'/rdevito/deep_null/data/sample'
* --covar-dir: Directory containing the covariate files. Default:
	'/rdevito/deep_null/data/covar'
* --output-dir: Directory in which a folder will be created to store
	the output of the GWAS workflow. Default: 
	'/rdevito/deep_null/output/PRS_PRScs'. Final output directory
	will be of the form: {output_dir}/{pheno_name}/{covar_set}
* --geno-fname: Filename without extention of BED files. Default: 'allchr_wbqc'
* --null-dir: Directory containing the null model predictions. Default:
	'/rdevito/deep_null/dn_output/V4'
"""

import argparse
from pprint import pprint

import dxpy


WORKFLOW_ID = 'workflow-Gy52yP8Jv7B9bBP64FP81Qf3'
DEFAULT_INSTANCE = 'mem1_ssd1_v2_x72'

N_SAMP_PHENO = {
	'FEV1_3063': 231888,
	'FVC_3062': 231888,
	'HDL_cholesterol_30760': 221748,
	'IGF1_30770': 240945,
	'LDL_direct_30780': 241783,
	'SHBG_30830': 219697,
	'adjusted_telomere_ratio_22191': 246014,
	'alanine_aminotransferase_30620': 242140,
	# 'arterial_stiffness_index_21021': 83283,
	'aspartate_aminotransferase_30650': 241346,
	'body_fat_percentage_23099': 249522,
	'c-reactive_protein_30710': 241711,
	'creatinine_30700': 242109,
	'diastolic_blood_pressure_4079': 237332,
	# 'fluid_intelligence_score_20016': 82482,
	'glucose_30740': 221605,
	'glycated_haemoglobin_30750': 242250,
	'grip_strength': 253489,
	'haemoglobin_concentration_30020': 246547,
	# 'hearing_SRT': 81820,
	'heel_bone_mineral_density_3148': 146774,
	'mean_corpuscular_volume_30040': 246546,
	'mean_time_to_identify_matches_20023': 252407,
	'number_of_incorrect_matches_399': 253930,
	'platelet_count_30080': 246548,
	'red_blood_cell_count_30010': 246548,
	'sleep_duration_1160': 252648,
	'standing_height_50': 253488,
	'systolic_blood_pressure_4080': 237327,
	'testosterone_30850': 219602,
	'triglycerides_30870': 242029,
	'urate_30880': 241911,
	'vitamin_d_30890': 231672,
	'white_blood_cell_count_30000': 246544,
}


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
		default='/rdevito/nonlin_prs/data/geno_data/qced_common/bed_rsid',
		help='Directory containing the plink1 BED genotype files with rsIDs.'
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
		default='/rdevito/deep_null/output/PRS_PRScs',
		help='Directory in which a folder will be created to store the output '
			 'of the GWAS workflow. Final output directory will be of the form: '
			 '{output_dir}/{pheno_name}/{covar_set}'
	)
	parser.add_argument(
		'--geno-fname',
		default='allchr_wbqc',
		help="Filename without extention of BED files. Default: 'allchr_wbqc'"
	)
	parser.add_argument(
		'--null-covar-set',
		help='Name of the covariate set file for the null model.',
		default=None
	)
	parser.add_argument(
		'--null-model',
		help='Name of the null model.',
		default=None
	)
	parser.add_argument(
		'--null-dir',
		default='/rdevito/deep_null/dn_output/V4',
		help='Directory containing the null model predictions.'
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


def get_ld_file_links():
	ld_file_dir = '/rdevito/deep_null/data/geno/PRScs_LD_info/ldblk_1kg_eur'
	lf_fname_template = 'ldblk_1kg_chr{:}.hdf5'
	snp_info_fname = 'snpinfo_1kg_hm3'

	# Create file paths
	ld_file_names = [snp_info_fname]

	for chr_n in range(1, 23):
		ld_file_names.append(
			lf_fname_template.format(chr_n)
		)

	# Get file links
	file_links = []

	for f_path in ld_file_names:
		file_links.append(
			dxpy.dxlink(
				list(dxpy.find_data_objects(
					name=f_path,
					folder=ld_file_dir,
					project=dxpy.PROJECT_CONTEXT_ID
				))[0]['id']
			)
		)

	return file_links


def launch_gwas_workflow(
	geno_file,
	covar_file,
	pheno_file,
	train_split_file,
	val_split_file,
	test_split_file,
	output_dir,
	n_samp,
	null_train_pred_file=None,
	null_test_val_pred_file=None,
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
		train_split_file: Path to the file containing the training sample IDs.
		val_split_file: Path to the file containing the validation sample IDs.
		test_split_file: Path to the file containing the test sample IDs.
		output_dir: Path to the output directory in UKB RAP storage.
		n_samp: Number of GWAS samples.
		workflow_id: ID of the plink2 GWAS workflow on the UKB RAP.
			Defaults to WORKFLOW_ID constant.
		instance_type: Instance type to use for the workflow. Defaults
			to DEFAULT_INSTANCE constant.
		name: Name of the job. Defaults to 'gwas_plink2'.
	"""

	# Get workflow
	workflow = dxpy.dxworkflow.DXWorkflow(dxid=workflow_id)

	# Get data links for inputs
	geno_bed_link = get_dxlink_from_path(f'{geno_file}.bed')
	geno_bim_link = get_dxlink_from_path(f'{geno_file}.bim')
	geno_fam_link = get_dxlink_from_path(f'{geno_file}.fam')

	covar_link = get_dxlink_from_path(covar_file)
	pheno_link = get_dxlink_from_path(pheno_file)
	train_sample_link = get_dxlink_from_path(train_split_file)
	val_sample_link = get_dxlink_from_path(val_split_file)
	test_sample_link = get_dxlink_from_path(test_split_file)

	# Set up workflow input
	prefix = 'stage-common.'

	if null_train_pred_file is not None and null_test_val_pred_file is not None:
		workflow_input = {
			f'{prefix}geno_bed': geno_bed_link,
			f'{prefix}geno_bim': geno_bim_link,
			f'{prefix}geno_fam': geno_fam_link,
			f'{prefix}covar_file': covar_link,
			f'{prefix}pheno_file': pheno_link,
			f'{prefix}train_samples': train_sample_link,
			f'{prefix}val_samples': val_sample_link,
			f'{prefix}test_samples': test_sample_link,
			f'{prefix}ld_files': get_ld_file_links(),
			f'{prefix}sample_size': n_samp,
			f'{prefix}null_train_pred_file': get_dxlink_from_path(null_train_pred_file),
			f'{prefix}null_testval_pred_file': get_dxlink_from_path(null_test_val_pred_file),
		}
	else:
		workflow_input = {
			f'{prefix}geno_bed': geno_bed_link,
			f'{prefix}geno_bim': geno_bim_link,
			f'{prefix}geno_fam': geno_fam_link,
			f'{prefix}covar_file': covar_link,
			f'{prefix}pheno_file': pheno_link,
			f'{prefix}train_samples': train_sample_link,
			f'{prefix}val_samples': val_sample_link,
			f'{prefix}test_samples': test_sample_link,
			f'{prefix}ld_files': get_ld_file_links(),
			f'{prefix}sample_size': n_samp,
		}

	# Run workflow
	analysis = workflow.run(
		workflow_input,
		folder=output_dir,
		name=name,
		instance_type=instance_type,
		priority='low',
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

	# If using null model
	if args.null_covar_set is not None:
		# Check that null-model is also provided
		if args.null_model is None:
			raise ValueError('If null-covar-set is provided, null-model must also be provided.')
		using_null_model = True
		print(f'Using null model {args.null_model} with covariate set {args.null_covar_set}.')
	else:
		if args.null_model is not None:
			raise ValueError('If null-model is provided, null-covar-set must also be provided.')
		using_null_model = False

	# Get the number of samples
	n_samp = N_SAMP_PHENO[args.pheno_name]
	print(f'num. samples: {n_samp}')

	# Set the output directory
	if using_null_model:
		desc = f'{covar_set}_null_{args.null_model}_{args.null_covar_set}'
	else:
		desc = covar_set
	output_dir = f'{args.output_dir}/{args.pheno_name}/{desc}'
	print(f'Output directory: {output_dir}')

	# Launch the PRS workflow
	job_name = f'prs_prscs_{args.pheno_name}_{desc}'

	print(f'Launching PRS workflow with name: {job_name}')
	
	if using_null_model:
		# Set paths to null model predictions
		null_pred_dir = f'{args.null_dir}/{args.pheno_name}/{args.null_covar_set}/{args.null_model}'
		null_train_pred_file = f'{null_pred_dir}/ho_preds.csv'
		null_test_val_pred_file = f'{null_pred_dir}/ens_preds.csv'

		launch_gwas_workflow(
			geno_file=f'{args.geno_dir}/{args.geno_fname}',
			covar_file=f'{args.covar_dir}/{covar_set}.tsv',
			pheno_file=f'{args.pheno_dir}/{args.pheno_name}.pheno',
			train_split_file=f'{args.splits_dir}/train_iids.txt',
			val_split_file=f'{args.splits_dir}/val_iids.txt',
			test_split_file=f'{args.splits_dir}/test_iids.txt',
			output_dir=output_dir,
			n_samp=n_samp,
			name=job_name,
			null_train_pred_file=null_train_pred_file,
			null_test_val_pred_file=null_test_val_pred_file,
		)
	else:
		launch_gwas_workflow(
			geno_file=f'{args.geno_dir}/{args.geno_fname}',
			covar_file=f'{args.covar_dir}/{covar_set}.tsv',
			pheno_file=f'{args.pheno_dir}/{args.pheno_name}.pheno',
			train_split_file=f'{args.splits_dir}/train_iids.txt',
			val_split_file=f'{args.splits_dir}/val_iids.txt',
			test_split_file=f'{args.splits_dir}/test_iids.txt',
			output_dir=output_dir,
			n_samp=n_samp,
			name=job_name,
		)