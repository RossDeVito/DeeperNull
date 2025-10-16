version 1.0

workflow prs_score_preds {
	input {
		File val_preds
		File test_preds
		File pheno_file
	}

	call score_preds {
		input:
			val_preds = val_preds,
			test_preds = test_preds,
			pheno_file = pheno_file
	}

	output {
		Array[File] scores_json = score_preds.scores_json
	}

	meta {
		description: "Score a model's predictions n times for bootstrap."
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

		python3 /home/score_preds_bootstrap.py \
			--val-preds ~{val_preds} \
			--test-preds ~{test_preds} \
			--pheno-file ~{pheno_file} \
			--out-dir $CURRENT_DIR \
			--bootstrap-iters 100000
	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/geonull_prs_score_preds:latest"
	}

	output {
		Array[File] scores_json = glob("scores_w_bootstrap*.json")
	}
}