version 1.0

workflow fit_dn_model {
	input {
		File covar_file
		Array[File] pheno_files
		File model_config
		File train_samp_file
		File val_samp_file
		File test_samp_file
	}

	call fit_dn_mt_model_task {
		input:
			covar_file=covar_file,
			pheno_files=pheno_files,
			model_config=model_config,
			train_samp_file=train_samp_file,
			val_samp_file=val_samp_file,
			test_samp_file=test_samp_file,
	}

	output {
		Array[File] ens_preds = fit_dn_mt_model_task.ens_preds
		Array[File] ho_jointplot = fit_dn_mt_model_task.ho_jointplot
		Array[File] ho_preds = fit_dn_mt_model_task.ho_preds
		Array[File] ho_scatter = fit_dn_mt_model_task.ho_scatter
		File ho_scores = fit_dn_mt_model_task.ho_scores
		File fit_model_config = fit_dn_mt_model_task.fit_model_config
	}

	meta {
		description: "Fit deep null style model on CPU"
	}
}

task fit_dn_mt_model_task {
	input {
		File covar_file
		Array[File] pheno_files
		File model_config
		File train_samp_file
		File val_samp_file
		File test_samp_file
	}

	command <<<
		# Clone DeeperNull repo
		git clone https://github.com/RossDeVito/DeeperNull.git

		# Install DeeperNull
		cd DeeperNull
		pip3 install . --no-deps
		cd ..

		pip3 list

		# Turn phenp_files into a string with space separated values
		PHENO_FILE_STR=$(IFS=" "; echo "${pheno_files[*]}")

		# Run DeeperNull model fitting
		python3 DeeperNull/deeper_null/fit_model.py \
			--covar_file ~{covar_file} \
			--pheno_files ${PHENO_FILE_STR} \
			--model_config ~{model_config} \
			--train_samples ~{train_samp_file} \
			--pred_samples ~{val_samp_file} ~{test_samp_file}
	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/fit_dn_mt_model:latest"
	}

	output {
		Array[File] ens_preds = glob("ens_preds*.csv")
		Array[File] ho_jointplot = glob("ho_*jointplot.png")
		Array[File] ho_preds = glob("ho_preds*.csv")
		Array[File] ho_scatter = glob("ho_*scatter.png")
		File ho_scores = "ho_scores.json"
		File fit_model_config = "model_config.json"
	}
}