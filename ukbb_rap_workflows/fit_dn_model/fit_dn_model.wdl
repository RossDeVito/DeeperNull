version 0.1

workflow fit_dn_model {
	input {
		File covar_file
		File pheno_file
		File model_config
		File train_samp_file
		Array[File] pred_samp_files
		String out_dir	# Should these paths to directories be File?
		String upload_dir
		String save_dir
		String pat
	}

	call fit_dn_model_task {
		input:
			covar_file=covar_file,
			pheno_file=pheno_file,
			model_config=model_config,
			train_samp_file=train_samp_file,
			pred_samp_files=pred_samp_files,
			out_dir=out_dir,
			upload_dir=upload_dir,
			save_dir=save_dir,
			pat=pat
	}

	output {
    	# Want equivalent of dx upload ${upload_root} -r \
    	#						--destination ${save_dir}
	}

	meta {
		description: "Fit deep null style model on CPU"
	}
}

task fit_dn_model_task {
	input {
		File covar_file
		File pheno_file
		File model_config
		File train_samp_file
		Array[File] pred_samp_files
		String out_dir
		String upload_dir
		String save_dir
		String pat
	}

	command <<<
		# Clone DeeperNull repo
		git clone https://~{pat}@github.com/RossDeVito/DeeperNull.git

		# Install DeeperNull
		cd DeeperNull
		pip3 install .
		cd ..

		PRED_SAMP_FILES=(~{sep=' ' pred_samp_files}) # Load array into bash variable

		# Run DeeperNull model fitting
		python3 DeeperNull/deeper_null/fit_model.py \
			--covar_file ~{covar_file} \
			--pheno_file ~{pheno_file} \
			--model_config ~{model_config} \
			--out_dir ~{out_dir} \
			--train_samples ~{train_samp_file} \
			--pred_samples ${PRED_SAMP_FILES}
	>>>

	runtime {
		container: "gcr.io/ucsd-medicine-cast/fit_dn_model:latest"
		cpu: 32
		# can I specify an instance type?
	}

	output {
		# Want equivalent of dx upload ${upload_root} -r \
		#						--destination ${save_dir}
	}
}