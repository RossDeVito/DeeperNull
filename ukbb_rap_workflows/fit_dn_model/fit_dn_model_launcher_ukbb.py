"""Script to launch UKBB DeeperNull model fitting workflow.

Required args:
parser.add_argument(
	'-d', '--model_desc', type=str, required=True,
	help='Name used when saving the model.'
)
parser.add_argument(
	'-j', '--model_config', type=str, required=True,
	help='Local path to model configuration file.'
)
parser.add_argument(
	'-c', '--covar_set', type=str, required=True,
	help='Name of the covariate set to use as input. Corresponds to a file in covar_dir.'
)
parser.add_argument(
	'-p', '--pheno', type=str, required=True,
	help='Name of the phenotype to use as input. Corresponds to a file in pheno_dir.'
)
parser.add_argument(
	'--pat', type=str, required=True,
	help='Text file containing GitHub personal access token.'
)

* -d, --model-desc (str): Name used when saving the model.
* -j, --model-config (str): Local path to model configuration file.
* -c, --covar_set (str): Name of the covariate set to use as input.
	Corresponds to a file in covar_dir.
* -p, --pheno (str): Name of the phenotype to use as input. Corresponds
	to a file in pheno_dir.
* --pat (str): Text file containing GitHub personal access token.

Optional args:

* -s, --save-dir (str): Path to directory where output files will be
	saved in storage. Default: '/rdevito/deep_null/dn_output'
* -v, --out-version_dir (str): All output files will be saved in a
	subdirectory of out-dir with this name. If None (default), output
	saved in out-dir.
* -i, --instance-type (str): Instance type to use for running the
	workflow. Default: 'mem1_ssd1_x16' if no GPU flag, or 
	'mem1_ssd1_gpu_x16' if GPU flag.
* -g, --gpu (str): Flag to use GPU for training Pytorch models.
	Default: False
* --covar_dir (str): Path to storage directory containing covariate
	files. Default: '/rdevito/deep_null/data/covar'
* --pheno_dir (str): Path to storage directory containing phenotype
	files. Default: '/rdevito/nonlin_prs/data/pheno_data/pheno'
* --samp_dir (str): Path to storage directory containing sample files.
	Default: '/rdevito/deep_null/data/sample'
* --train_samp_fname (str): Name of the training sample file in samp_dir.
	Default: 'train_iids.txt'
* --pred_samp_fnames (str): Name(s) of the prediction sample file(s) in
	samp_dir. Default: ['val_iids.txt', 'test_iids.txt']
"""

import argparse
import dxpy
import sys


CONFIG_UPLOAD_DIR = '/rdevito/deep_null/model_configs'
CPU_WORKFLOW_ID = 'workflow-Gg8QXP0Jv7BJY20GjkpPv85P'	# Need workflow ID to launch
GPU_WORKFLOW_ID = 'workflow-xxxx'

DEFAULT_CPU_INSTANCE = 'mem1_ssd1_v2_x16'


def upload_model_config(file_path, upload_dir=CONFIG_UPLOAD_DIR):
	"""Upload a file to DNA Nexus and return its file ID.

	Args:
		file_path (str): Local path to model config JSON file to upload.

	Returns:
		file : dxlink
			{"$dnanexus_link": file_id}
	"""
	sys.stderr.write("Uploading model config JSON...\n")
	dxfile = dxpy.upload_local_file(
			file_path, folder=upload_dir, parents=True
	)
	print(dxpy.dxlink(dxfile))
	return dxpy.dxlink(dxfile)


def parse_args():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		'-d', '--model-desc', type=str, required=True,
		help='Name used when saving the model.'
	)
	parser.add_argument(
		'-j', '--model-config', type=str, required=True,
		help='Local path to model configuration file.'
	)
	parser.add_argument(
		'-c', '--covar-set', type=str, required=True,
		help='Name of the covariate set to use as input. Corresponds to a file in covar_dir.'
	)
	parser.add_argument(
		'-p', '--pheno', type=str, required=True,
		help='Name of the phenotype to use as input. Corresponds to a file in pheno_dir.'
	)
	parser.add_argument(
		'--pat', type=str, required=True,
		help='Text file containing GitHub personal access token.'
	)
	parser.add_argument(
		'-s', '--save-dir', type=str,
		default='/rdevito/deep_null/dn_output',
		help='Path to directory where output files will be saved in storage.'
	)
	parser.add_argument(
		'-v', '--out-version-dir', type=str, 
		default=None,
		help='All output files will be saved in a subdirectory of out-dir '
			'with this name. If None (default), output saved in out-dir.'
	)
	parser.add_argument(
		'-i', '--instance-type', type=str,
		help='Instance type to use for running the workflow.'
	)
	parser.add_argument(
		'-g', '--gpu', action='store_true',
		help='Flag to use GPU for training Pytorch models.'
	)
	parser.add_argument(
		'--covar-dir', type=str,
		default='/rdevito/deep_null/data/covar',
		help='Path to storage directory containing covariate files.'
	)
	parser.add_argument(
		'--pheno-dir', type=str, 
		default='/rdevito/nonlin_prs/data/pheno_data/pheno',
		help='Path to storage directory containing phenotype files.'
	)
	parser.add_argument(
		'--samp-dir', type=str,
		default='/rdevito/deep_null/data/sample',
		help='Path to storage directory containing sample files.'
	)
	parser.add_argument(
		'--train-samp-fname', type=str,
		default='train_iids.txt',
		help='Name of the training sample file in samp_dir.'
	)
	parser.add_argument(
		'--pred-samp-fnames', type=str, nargs='+',
		default=['val_iids.txt', 'test_iids.txt'],
		help='Name(s) of the prediction sample file(s) in samp_dir.'
	)
	return parser.parse_args()


def launch_fit(
	covar_file,
	pheno_file,
	model_config_link,
	train_samp_file,
	pred_samp_files,
	save_dir,
	pat,
	instance_type,
	workflow_id,
	name='fit_dn_model'
):
	"""Launch DeeperNull model fitting workflow.

	Calls a WDL workflow that runs the following bash commands and uploads
	the output to storage:

	```
	# Clone DeeperNull repo
	git clone https://${pat}@github.com/RossDeVito/DeeperNull.git

	# Install DeeperNull
	cd DeeperNull
	pip3 install .
	cd ..

	# Run DeeperNull model fitting
	python3 DeeperNull/deeper_null/fit_model.py \
		--covar_file ${covar_file} \
		--pheno_file ${pheno_file} \
		--model_config ${model_config_link} \
		--train_samples ${train_samp_file} \
		--pred_samples ${*pred_samp_files}
	```

	Args:
		covar_file (str): Path to covariate file in storage.
		pheno_file (str): Path to phenotype file in storage.
		model_config_link (str): Link to model configuration file in storage.
		train_samp_file (str): Path to training sample file in storage.
		pred_samp_files (list of str): Path(s) to prediction sample files
			in storage.
		save_dir (str): Path to directory where output files will be saved
			in storage.
		pat (str): GitHub personal access token.
		instance_type (str): Instance type to use for running the workflow.
		workflow_id (str): ID of the workflow to run.
	"""
	sys.stderr.write(
		f"Launching DeeperNull model fitting workflow {name}...\n"
	)

	# Get workflow
	workflow = dxpy.dxworkflow.DXWorkflow(dxid=workflow_id)

	# Get data links for inputs
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

	train_samp_link = dxpy.dxlink(
		list(dxpy.find_data_objects(
			name=train_samp_file.split('/')[-1],
			folder='/'.join(train_samp_file.split('/')[:-1]),
			project=dxpy.PROJECT_CONTEXT_ID
		))[0]['id']
	)

	pred_samp_links = [
		dxpy.dxlink(
			list(dxpy.find_data_objects(
				name=fname.split('/')[-1],
				folder='/'.join(fname.split('/')[:-1]),
				project=dxpy.PROJECT_CONTEXT_ID
			))[0]['id']
		) for fname in pred_samp_files
	]
	
	# Run workflow
	prefix='stage-common.'
	workflow_input = {
		f"{prefix}covar_file": covar_link,
		f"{prefix}pheno_file": pheno_link,
		f"{prefix}model_config": model_config_link,
		f"{prefix}train_samp_file": train_samp_link,
		f"{prefix}pred_samp_files": pred_samp_links,
		f"{prefix}pat": pat
	}
	analysis = workflow.run(
		workflow_input,
		folder=save_dir,
		name=name,
		instance_type=instance_type,
	)

	sys.stderr.write("Started analysis %s (%s)\n"%(analysis.get_id(), name))
	return analysis


if __name__ == '__main__':
	"""Example usage:
	
	python fit_dn_model_launcher_ukbb.py \
		-d lin_reg_1 \
		-j model_configs/lin_reg_1.json \
		-c age_sex_birth_coords \
		-p standing_height_50 \
		--pat ../../../pat_dn_read.txt
	"""

	# Parse command line arguments
	args = parse_args()

	# Read personal access token from file
	with open(args.pat, 'r') as f:
		pat = f.read().strip()
	print(pat)

	# Upload model configuration file to storage
	model_config_link = upload_model_config(args.model_config)

	# Launch DeeperNull model fitting workflow
	covar_file = f'{args.covar_dir}/{args.covar_set}.tsv'
	pheno_file = f'{args.pheno_dir}/{args.pheno}.pheno'
	
	if args.out_version_dir:
		save_dir = f'{args.save_dir}/{args.out_version_dir}/{args.pheno}/{args.covar_set}/{args.model_desc}'
	else:
		save_dir = f'{args.save_dir}/{args.pheno}/{args.covar_set}/{args.model_desc}'

	train_samp_file = f'{args.samp_dir}/{args.train_samp_fname}'
	pred_samp_files = [f'{args.samp_dir}/{fname}' for fname in args.pred_samp_fnames]

	# Set instance type and workflow ID (CPU or GPU version)
	if args.gpu:
		if args.instance_type is None:
			instance_type = 'mem1_ssd1_gpu_x16'
		else:
			instance_type = args.instance_type
		workflow_id = GPU_WORKFLOW_ID
	else:
		if args.instance_type is None:
			instance_type = DEFAULT_CPU_INSTANCE
		else:
			instance_type = args.instance_type
		workflow_id = CPU_WORKFLOW_ID

	# Launch workflow
	workflow = launch_fit(
		covar_file=covar_file,
		pheno_file=pheno_file,
		model_config_link=model_config_link,
		train_samp_file=train_samp_file,
		pred_samp_files=pred_samp_files,
		save_dir=save_dir,
		pat=pat,
		instance_type=instance_type,
		workflow_id=workflow_id,
		name=f'{args.model_desc}_{args.pheno}_{args.covar_set}'
	)

	covar_file = f'/rdevito/deep_null/data/covar/age_sex.tsv'
	project = 'project-GG25fB8Jv7B928vqK7k6vYY6'
	fname = 'age_sex.tsv'
	folder = '/rdevito/deep_null/data/covar'
	f = dxpy.find_data_objects(
		name=fname,
		folder=folder,
		project=project
	)
	r = list(f)[0]

