version 1.0

workflow geno_prepro_step_2 {
	input {
		Array[File] bgen_files
	}

	call combine_bgen_files {
		input:
			bgen_files = bgen_files
	}

	output {
		File bgen_out = combine_bgen_files.bgen_out
	}

	meta {
		description: "Combine per-chromosome BGEN files into one BGEN file."	}
}

task combine_bgen_files {
	input {
		Array[File] bgen_files
	}

	command <<<
		# Create array of bgen files
		BGEN_FILES=(~{sep=" " bgen_files})

		# Create the cat-bgen command for all genotypes
		CMD="cat-bgen"

		# Add each bgen file to the command
		for FILE in "${BGEN_FILES[@]}"; do
			CMD="$CMD -g $FILE"
		done

		# Specify the output file
		CMD+=' -og allchr_step2.bgen -clobber'

		echo $CMD

		# Run the command
		$CMD
	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/geonull_geno_prepro:latest"
	}

	output {
		File bgen_out = "allchr_step2.bgen"
	}
}