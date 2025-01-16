version 1.0

workflow geno_prepro_step_1 {
	input {
		Int chromosome_number
		File bgen_file
		File sample_file
		File filtered_vars_file
		File filtered_samples_file
	}

	call prelim_filter {
		input:
			chromosome_number = chromosome_number,
			bgen_file = bgen_file,
			sample_file = sample_file,
			filtered_vars_file = filtered_vars_file,
			filtered_samples = filtered_samples_file
	}

	output {
		File bgen_out = prelim_filter.bgen_out
		File sample_out = prelim_filter.sample_out
	}

	meta {
		description: "Keep White British sample that pass initial QC. Keep variants that pass initial QC. Filter out multi-allelic variants."
	}
}

task prelim_filter {
	input {
		Int chromosome_number
		File bgen_file
		File sample_file
		File filtered_vars_file
		File filtered_samples
	}

	command <<<
		# Make 2-column version of sample IDs to please plink
		awk '{print $1 "\t" $1}' ~{filtered_samples} > temp_sample_info.txt
		head -n 4 temp_sample_info.txt

		# Subset to samples and variants that passed initial QC and
		# filter out multiallelic variants
		plink2 \
			--bgen ~{bgen_file} ref-first \
			--sample ~{sample_file} \
			--extract ~{filtered_vars_file} \
			--keep temp_sample_info.txt \
			--max-alleles 2 \
			--export bgen-1.2 ref-first \
			--out chr~{chromosome_number}_step1
	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/geonull_geno_prepro:latest"
	}

	output {
		File bgen_out = glob("chr*_step1.bgen")[0]
		File sample_out = glob("chr*_step1.sample")[0]
	}
}