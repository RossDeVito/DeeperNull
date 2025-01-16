# Used to launch the DeeperNull model fitting workflow on UKBB RAP

## fit_dn_model_launcher_ukbb.py

Script to launch UKBB DeeperNull model fitting workflow.

Required args:

* -d, --model-desc (str): Name used when saving the model.
* -j, --model-config (str): Local path to model configuration file.
* -c, --covar_set (str): Name of the covariate set to use as input.
	Corresponds to a file in covar_dir.
* -p, --pheno (str): Name of the phenotype to use as input. Corresponds
	to a file in pheno_dir.

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