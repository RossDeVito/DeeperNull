version 1.0

workflow gwas_plink2 {
	input {
		File geno_pgen_file
		File geno_psam_file
		File geno_pvar_file
		File covar_file
		File pheno_file
		File split_file
	}

	call gwas_plink2_task {
		input:
			geno_pgen_file = geno_pgen_file,
			geno_psam_file = geno_psam_file,
			geno_pvar_file = geno_pvar_file,
			covar_file = covar_file,
			pheno_file = pheno_file,
			split_file = split_file
	}

	output {
		Array[File] glm_linear_output = gwas_plink2_task.glm_linear_output
		File runtime_json = gwas_plink2_task.runtime_json
	}

	meta {
		description: "Run GWAS using plink2"
	}
}

task gwas_plink2_task {
	input {
		File geno_pgen_file
		File geno_psam_file
		File geno_pvar_file
		File covar_file
		File pheno_file
		File split_file
	}

	command <<<
	echo "Running GWAS using plink2 with no null model"

	# Preprocess covariate file
	python3 /usr/local/prepro_covar.py \
		--covar-file ~{covar_file} \
		--rescale-coords \
		--rescale-time \
		--string-month

	# Time GWAS
	START_TIME=$(date +%s)

	plink2 --glm 'hide-covar' \
		--pgen ~{geno_pgen_file} \
		--pvar ~{geno_pvar_file} \
		--psam ~{geno_psam_file} \
		--keep ~{split_file} \
		--covar prepro_covar.tsv \
		--pheno ~{pheno_file} \
		--vif 1000000 \
		--out gwas_plink2

	# Save runtime as JSON as 'runtime_seconds' key
	END_TIME=$(date +%s)
	ELAPSED_TIME=$((END_TIME-START_TIME))
	echo "{\"runtime_seconds\": $ELAPSED_TIME}" > runtime.json
	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/geonull_gwas_plink:latest"
	}

	output {
		Array[File] glm_linear_output = glob("gwas_plink2*")
		File runtime_json = "runtime.json"
	}
}