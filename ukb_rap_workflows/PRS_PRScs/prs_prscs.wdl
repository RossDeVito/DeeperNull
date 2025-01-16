version 1.0

workflow prs_prscs {
	input {
		File geno_bed
		File geno_bim
		File geno_fam
		File covar_file
		File pheno_file
		File split_file
		Array[File] ld_files
		Int sample_size
	}

	call gwas_plink2_task {
		input:
			geno_bed = geno_bed,
			geno_bim = geno_bim,
			geno_fam = geno_fam,
			covar_file = covar_file,
			pheno_file = pheno_file,
			split_file = split_file
	}

	call prs_prscs_task {
		input:
			summary_stats = gwas_plink2_task.summary_stats,
			ld_files = ld_files,
			geno_bed = geno_bed,
			geno_bim = geno_bim,
			geno_fam = geno_fam,
			sample_size = sample_size
	}

	call score_with_plink {
		input:
			geno_bed = geno_bed,
			geno_bim = geno_bim,
			geno_fam = geno_fam,
			weights = prs_prscs_task.prscs_output
	}

	output {
		Array[File] gwas_output = gwas_plink2_task.glm_linear_output
		File gwas_runtime_json = gwas_plink2_task.runtime_json
		File prscs_output = prs_prscs_task.prscs_output
		File scores = score_with_plink.scores
	}

	meta {
		description: "Run prscs PRS using BED files for GWAS"
	}
}

task gwas_plink2_task {
	input {
		File geno_bed
		File geno_bim
		File geno_fam
		File covar_file
		File pheno_file
		File split_file
	}

	command <<<
		echo "Running GWAS using plink2 with BED files"

		# Preprocess covariate file
		python3 /usr/local/prepro_covar.py \
		--covar-file ~{covar_file} \
		--rescale-coords \
		--rescale-time \
		--string-month

		# Time GWAS
		START_TIME=$(date +%s)

		plink2 --glm 'hide-covar' \
			--bed ~{geno_bed} \
			--bim ~{geno_bim} \
			--fam ~{geno_fam} \
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
		docker: "gcr.io/ucsd-medicine-cast/geonull_prs_prscs:latest"
	}

	output {
		File summary_stats = glob("gwas_plink2.*.glm.linear")[0]
		Array[File] glm_linear_output = glob("gwas_plink2*")
		File runtime_json = "runtime.json"
	}
}

task prs_prscs_task {
	input {
		File summary_stats
		Array[File] ld_files
		File geno_bed
		File geno_bim
		File geno_fam
		Int sample_size
	}

	command <<<
		# Create a directory for LD files
		mkdir -p ld_files_1kg

		# Move LD files into the directory
		for file in ~{sep=' ' ld_files}; do
			cp $file ld_files_1kg/
		done

		# Check for common prefix for BED files
		BED_DIR=$(dirname ~{geno_bed})
		BED_FNAME=$(basename ~{geno_bed} .bed)
		BED_PREFIX=${BED_DIR}/${BED_FNAME}

		BIM_DIR=$(dirname ~{geno_bim})
		BIM_FNAME=$(basename ~{geno_bim} .bim)
		BIM_PREFIX=${BIM_DIR}/${BIM_FNAME}

		FAM_DIR=$(dirname ~{geno_fam})
		FAM_FNAME=$(basename ~{geno_fam} .fam)
		FAM_PREFIX=${FAM_DIR}/${FAM_FNAME}

		echo "BED_PREFIX: $BED_PREFIX"
		echo "BIM_PREFIX: $BIM_PREFIX"
		echo "FAM_PREFIX: $FAM_PREFIX"

		# Assert that all files have the same prefix
		if [ "$BED_PREFIX" != "$BIM_PREFIX" ] || [ "$BED_PREFIX" != "$FAM_PREFIX" ]; then
			echo "BED, BIM, and FAM files must have the same prefix"
			exit 1
		fi

		# Reformat summary_stats to: SNP, A1, A2, BETA, P
		awk 'BEGIN {OFS="\t"} NR == 1 {print "SNP", "A1", "A2", "BETA", "P"; next} \
			{print $3, $4, $5, $12, $15}' ~{summary_stats} > reformatted_summary_stats.txt

		# Run PRScs
		mkdir prscs_output

		python3 ${PRSCS_DIR}/PRScs.py \
			--ref_dir=ld_files_1kg \
			--bim_prefix=${BED_PREFIX} \
			--sst_file=reformatted_summary_stats.txt \
			--n_gwas=~{sample_size} \
			--out_dir='prscs_output/output'

		# Stack PRScs outputs into a single file
		cat prscs_output/output*.txt > combined_prscs_output.txt
	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/geonull_prs_prscs:latest"
	}

	output {
		# PRScs output files
		File prscs_output = "combined_prscs_output.txt"
	}
}

task score_with_plink {
	input {
		File geno_bed
		File geno_bim
		File geno_fam
		File weights
	}

	command <<<
		# Run PLINK2 scoring
		plink2 \
			--bed ~{geno_bed} \
			--bim ~{geno_bim} \
			--fam ~{geno_fam} \
			--score ~{weights} 2 4 6 \
			--out plink2_score
	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/geonull_prs_prscs:latest"
	}

	output {
		File scores = glob("*.sscore")[0]
	}
}
