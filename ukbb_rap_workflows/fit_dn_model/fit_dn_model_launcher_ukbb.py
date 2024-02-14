"""Script to launch UKBB DeeperNull model fitting workflow.

Required args:

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
import time


CONFIG_UPLOAD_DIR = '/rdevito/deep_null/model_configs'


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
		'-g', '--gpu', action='store_true',
		help='Flag to use GPU for training Pytorch models.'
	)
	parser.add_argument(
		'--covar_dir', type=str,
		default='/rdevito/deep_null/data/covar',
		help='Path to storage directory containing covariate files.'
	)
	parser.add_argument(
		'--pheno_dir', type=str, 
		default='/rdevito/nonlin_prs/data/pheno_data/pheno',
		help='Path to storage directory containing phenotype files.'
	)
	parser.add_argument(
		'--samp_dir', type=str,
		default='/rdevito/deep_null/data/sample',
		help='Path to storage directory containing sample files.'
	)
	parser.add_argument(
		'--train_samp_fname', type=str,
		default='train_iids.txt',
		help='Name of the training sample file in samp_dir.'
	)
	parser.add_argument(
		'--pred_samp_fnames', type=str, nargs='+',
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
	output_dir,
	upload_root,
	save_dir,
	pat
):
	"""Launch DeeperNull model fitting workflow.

	Calls a WDL workflow that runs the following bash commands:

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
		--out_dir ${output_dir} \
		--train_samples ${train_samp_file} \
		--pred_samples ${*pred_samp_files}

	# Upload output files to storage
	dx upload ${upload_root} -r \
    	--destination ${save_dir}
	```

	Args:
		covar_file (str): Path to covariate file in storage.
		pheno_file (str): Path to phenotype file in storage.
		model_config_link (str): Link to model configuration file in storage.
		train_samp_file (str): Path to training sample file in storage.
		pred_samp_files (list of str): Path(s) to prediction sample files
			in storage.
		output_dir (str): Path to directory where output files will be
			saved in storage.
		upload_root (str): Files will be uploaded to storage recursively
			from this directory.
		save_dir (str): Path to directory where output files will be saved
			in storage.
		pat (str): GitHub personal access token.
	"""
	sys.stderr.write("Launching DeeperNull model fitting workflow...\n")
	


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
		output_dir = f'./{args.out_version_dir}/{args.pheno}/{args.covar_set}/{args.model_desc}'
		upload_root = f'./{args.out_version_dir}'
	else:
		output_dir = f'./{args.pheno}/{args.covar_set}/{args.model_desc}'
		upload_root = f'./{args.pheno}'

	train_samp_file = f'{args.samp_dir}/{args.train_samp_fname}'
	pred_samp_files = [f'{args.samp_dir}/{fname}' for fname in args.pred_samp_fnames]

	launch_fit(
		covar_file,
		pheno_file,
		model_config_link,
		train_samp_file,
		pred_samp_files,
		output_dir,
		upload_root,
		args.save_dir,
		pat
	)	# Need workflow ID to launch

