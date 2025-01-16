version 1.0

workflow geno_prepro_step_3 {
	input {
		File bgen_file
		File sample_file
	}

	call step_3_tasks {
		input:
			bgen_file = bgen_file,
			sample_file = sample_file,
	}

	output {
		Array[File] qced_geno_files = step_3_tasks.qced_geno_files
	}

	meta {
		description: "Remove samples and vars w/ missing >20%. Remove variants then samples w/ missing >2%. Save as PGEN and BGEN."
	}
}

task step_3_tasks {
	input {
		File bgen_file
		File sample_file
	}

	command <<<
		# Coarse filter and convert to PGEN
		plink2 \
			--bgen ~{bgen_file} ref-first \
			--sample ~{sample_file} \
			--geno 0.2 \
			--mind 0.2 \
			--make-pgen \
			--out allchr_step3a

		# Remove genotypes with > 2% missing rate
		plink2 \
			--pfile allchr_step3a \
			--geno 0.02 \
			--make-pgen \
			--out allchr_step3b

		# Remove samples with > 2% missing rate
		plink2 \
			--pfile allchr_step3b \
			--mind 0.02 \
			--make-pgen \
			--out allchr_qced

		# Also create BGEN files
		plink2 \
			--pfile allchr_qced \
			--export 'bgen-1.2' 'ref-first' \
			--out allchr_qced
	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/geonull_geno_prepro:latest"
	}

	output {
		Array[File] qced_geno_files = glob("allchr_qced*")
	}
}