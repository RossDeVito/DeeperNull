version 1.0

workflow prs_prscs {
	input {
		File geno_bed
		File geno_bim
		File geno_fam
		File covar_file
		File pheno_file
		File train_samples
		File val_samples
		File test_samples
		Array[File] ld_files
		Int sample_size
		File? null_train_pred_file
		File? null_testval_pred_file
	}

	call gwas_plink2_task {
		input:
			geno_bed = geno_bed,
			geno_bim = geno_bim,
			geno_fam = geno_fam,
			covar_file = covar_file,
			pheno_file = pheno_file,
			split_file = train_samples,
			null_train_pred_file = null_train_pred_file,
			null_testval_pred_file = null_testval_pred_file
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

	call fit_wrapper_model {
		input:
			covar_file = covar_file,
			pheno_file = pheno_file,
			score_file = score_with_plink.scores,
			val_samples = val_samples,
			test_samples = test_samples,
			null_train_pred_file = null_train_pred_file,
			null_testval_pred_file = null_testval_pred_file
	}

	call score_preds {
		input:
			val_preds = fit_wrapper_model.val_preds,
			test_preds = fit_wrapper_model.test_preds,
			pheno_file = pheno_file
	}

	output {
		File val_preds = fit_wrapper_model.val_preds
		File test_preds = fit_wrapper_model.test_preds
		File scores_json = score_preds.scores_json
		Array[File] plots = score_preds.plots
		File gwas_summary_stats = gwas_plink2_task.summary_stats
	}

	meta {
		description: "Run prscs PRS using BED files for GWAS, then scores predictions"
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
		File? null_train_pred_file
		File? null_testval_pred_file
	}

	command <<<
		echo "Running GWAS using plink2 with BED files"

		# Preprocess covariate file
		python3 /usr/local/prepro_covar.py \
			--covar-file ~{covar_file} \
			--rescale-coords \
			--rescale-time \
			--string-month \
			~{if defined(null_train_pred_file) then "--train-pred-file " + null_train_pred_file else ""} \
			~{if defined(null_testval_pred_file) then "--test-val-pred-file " + null_testval_pred_file else ""} \
			~{if defined(null_train_pred_file) then "--rescale-preds" else ""}

		# Run GWAS using plink2
		plink2 --glm 'hide-covar' \
			--bed ~{geno_bed} \
			--bim ~{geno_bim} \
			--fam ~{geno_fam} \
			--keep ~{split_file} \
			--covar prepro_covar.tsv \
			--pheno ~{pheno_file} \
			--vif 1000000 \
			--1 \
			--out gwas_plink2

	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/geonull_prs_prscs:latest"
	}

	output {
		File summary_stats = glob("gwas_plink2.*.glm.linear")[0]
		Array[File] glm_linear_output = glob("gwas_plink2*")
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
			--score ~{weights} 2 4 6 ignore-dup-ids cols=+scoresums \
			--out plink2_score
	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/geonull_prs_prscs:latest"
	}

	output {
		File scores = glob("*.sscore")[0]
	}
}

task fit_wrapper_model {
	input {
		File covar_file
		File pheno_file
		File score_file
		File val_samples
		File test_samples
		File? null_train_pred_file
		File? null_testval_pred_file
	}

	command <<<
		# Preprocess covariate file
		python3 /usr/local/prepro_covar.py \
			--covar-file ~{covar_file} \
			--rescale-coords \
			--rescale-time \
			--one-hot-month \
			~{if defined(null_train_pred_file) then "--train-pred-file " + null_train_pred_file else ""} \
			~{if defined(null_testval_pred_file) then "--test-val-pred-file " + null_testval_pred_file else ""} \
			~{if defined(null_train_pred_file) then "--rescale-preds" else ""}

		# Fit wrapper model
		python3 /usr/local/fit_wrapper.py \
			--pheno-file ~{pheno_file} \
			--covar-file prepro_covar.tsv \
			--score-file ~{score_file} \
			--val-iids ~{val_samples} \
			--test-iids ~{test_samples} \
			--binary \
			--out-dir $(pwd)

	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/geonull_prs_prscs:latest"
	}

	output {
		File val_preds = "val_preds.csv"
		File test_preds = "test_preds.csv"
	}
}

task score_preds {
	input {
		File val_preds
		File test_preds
		File pheno_file
	}

	command <<<
		CURRENT_DIR=$(pwd)

		python3 /home/score_preds.py \
			--val-preds ~{val_preds} \
			--test-preds ~{test_preds} \
			--pheno-file ~{pheno_file} \
			--out-dir $CURRENT_DIR \
			--case-control
	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/geonull_prs_score_preds:latest"
	}

	output {
		File scores_json = "scores.json"
		Array[File] plots = glob("*.png")
	}
}