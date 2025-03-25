version 1.0

workflow run_shap {
	input {
		Array[File] model_files
		File covar_file
		File pred_samples
		String model_type
		Boolean classification
	}

	call run_shap_task {
		input:
			model_files = model_files,
			covar_file = covar_file,
			pred_samples = pred_samples,
			model_type = model_type,
			classification = classification
	}

	output {
		File shap_values = run_shap_task.shapley_values
	}

	meta {
		description: "Get Shapley values for DeeperNull model"
	}
}

task run_shap_task {
	input {
		Array[File] model_files
		File covar_file
		File pred_samples
		String model_type
		Boolean classification
	}

	command <<<
		pip3 list

		# Clone DeeperNull repo
		git clone https://github.com/RossDeVito/DeeperNull.git

		# Install DeeperNull
		cd DeeperNull
		pip3 install .
		cd ..

		# If classification, set classification flag
		if [ ~{classification} == "true" ]; then
			classification_flag="--classification"
		else
			classification_flag=""
		fi

		MODEL_PATH_ARGS=""
		for model_file in ~{sep=' ' model_files}; do
			MODEL_PATH_ARGS="$MODEL_PATH_ARGS $model_file"
		done

		echo "Model files: ${MODEL_PATH_ARGS}"

		# Run DeeperNull Shapley values script
		python3 DeeperNull/deeper_null/get_shapley_values.py \
			--model_files ${MODEL_PATH_ARGS} \
			--covar_file ~{covar_file} \
			--pred_samples ~{pred_samples} \
			--model_type ~{model_type} \
			$classification_flag
		
	>>>

	runtime {
		docker: "gcr.io/ucsd-medicine-cast/dn_shap:latest"
	}

	output {
		File shapley_values = "shapley_values.json"
	}
}