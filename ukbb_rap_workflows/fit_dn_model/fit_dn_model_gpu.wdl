version 1.0

workflow fit_dn_model_gpu {
    input {
        File covar_file
        File pheno_file
        File model_config
        File train_samp_file
        Array[File] pred_samp_files
        String pat
    }

    call fit_dn_model_gpu_task {
        input:
            covar_file=covar_file,
            pheno_file=pheno_file,
            model_config=model_config,
            train_samp_file=train_samp_file,
            pred_samp_files=pred_samp_files,
            pat=pat
    }

    output {
        File ens_preds = fit_dn_model_gpu_task.ens_preds
        File ho_jointplot = fit_dn_model_gpu_task.ho_jointplot
        File ho_jointplot_male = fit_dn_model_gpu_task.ho_jointplot_male
        File ho_jointplot_female = fit_dn_model_gpu_task.ho_jointplot_female
        File ho_preds = fit_dn_model_gpu_task.ho_preds
        File ho_scatter = fit_dn_model_gpu_task.ho_scatter
        File ho_scores = fit_dn_model_gpu_task.ho_scores
        File fit_model_config = fit_dn_model_gpu_task.fit_model_config
    }

    meta {
        description: "Fit deep null style model on GPU"
    }
}

task fit_dn_model_gpu_task {
    input {
        File covar_file
        File pheno_file
        File model_config
        File train_samp_file
        Array[File] pred_samp_files
        String pat
    }

    command <<<
        # Clone DeeperNull repo
        git clone https://~{pat}@github.com/RossDeVito/DeeperNull.git

        # Install DeeperNull
        cd DeeperNull
        pip3 install --no-deps .
        cd ..

        # Check pandas version
        python3 -c "import pandas; print(pandas.__version__, flush=True)"

        PRED_SAMP_FILES=(~{sep=' ' pred_samp_files}) # Load array into bash variable

        # Run DeeperNull model fitting
        python3 DeeperNull/deeper_null/fit_model.py \
            --covar_file ~{covar_file} \
            --pheno_file ~{pheno_file} \
            --model_config ~{model_config} \
            --train_samples ~{train_samp_file} \
            --pred_samples ${PRED_SAMP_FILES}
    >>>

    runtime {
        container: "gcr.io/ucsd-medicine-cast/fit_dn_model_gpu:latest"
    }

    output {
        File ens_preds = "ens_preds.csv"
        File ho_jointplot = "ho_jointplot.png"
        File ho_jointplot_male = "ho_male_jointplot.png"
        File ho_jointplot_female = "ho_female_jointplot.png"
        File ho_preds = "ho_preds.csv"
        File ho_scatter = "ho_scatter.png"
        File ho_scores = "ho_scores.json"
        File fit_model_config = "model_config.json"
    }
}