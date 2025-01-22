version 1.0

workflow prs_basil {
	input {
		File geno_pgen
		File geno_psam
		File geno_pvar
		File pheno_file
		String pheno_name
		File covar_file
		File train_samples
		File val_samples
		File test_samples
		Float alpha
		Int n_iter
	}

	call prs_basil_task {
		input:
			geno_pgen = geno_pgen,
			geno_psam = geno_psam,
			geno_pvar = geno_pvar,
			pheno_file = pheno_file,
			pheno_name = pheno_name,
			covar_file = covar_file,
			train_samples = train_samples,
			val_samples = val_samples,
			test_samples = test_samples,
			alpha = alpha,
			n_iter = n_iter
	}

	output {
		File val_preds = prs_basil_task.val_preds
		File test_preds = prs_basil_task.test_preds
		File runtime_json = prs_basil_task.runtime_json
		File included_features = prs_basil_task.included_features
	}

	meta {
		description: "Run BASIL PRS"
	}
}

task prs_basil_task {
	input {
		File geno_pgen
		File geno_psam
		File geno_pvar
		File pheno_file
		String pheno_name
		File covar_file
		File train_samples
		File val_samples
		File test_samples
		Float alpha
		Int n_iter
	}

	command <<<
		# Preprocess covariate file
		python3 /home/prepro_covar.py \
			--covar-file ~{covar_file} \
			--rescale-coords \
			--rescale-time

		# Get common prefix for PGEN files
		PGEN_DIR=$(dirname ~{geno_pgen})
		PGEN_FNAME=$(basename ~{geno_pgen} .pgen)
		PGEN_PREFIX=${PGEN_DIR}/${PGEN_FNAME}

		PSAM_DIR=$(dirname ~{geno_psam})
		PSAM_FNAME=$(basename ~{geno_psam} .psam)
		PSAM_PREFIX=${PSAM_DIR}/${PSAM_FNAME}

		PVAR_DIR=$(dirname ~{geno_pvar})
		PVAR_FNAME=$(basename ~{geno_pvar} .pvar)
		PVAR_PREFIX=${PVAR_DIR}/${PVAR_FNAME}

		echo "PGEN_PREFIX: $PGEN_PREFIX"
		echo "PSAM_PREFIX: $PSAM_PREFIX"
		echo "PVAR_PREFIX: $PVAR_PREFIX"

		# Assert that all files have the same prefix
		if [ "$PGEN_PREFIX" != "$PSAM_PREFIX" ] || [ "$PGEN_PREFIX" != "$PVAR_PREFIX" ]; then
			echo "PGEN, PSAM, and PVAR files must have the same prefix"
			exit 1
		fi

		Rscript /home/run_basil.R \
			--pheno_file ~{pheno_file} \
			--pheno_name ~{pheno_name} \
			--covar_file prepro_covar.tsv \
			--geno_file $PGEN_PREFIX \
			--train_samples ~{train_samples} \
			--val_samples ~{val_samples} \
			--test_samples ~{test_samples} \
			--alpha ~{alpha} \
			--n_iter ~{n_iter}
	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/geonull_prs_basil:latest"
	}

	output {
		File val_preds = "val_preds.csv"
		File test_preds = "test_preds.csv"
		File runtime_json = "runtime.json"
		File included_features = "included_features.csv"
	}
}