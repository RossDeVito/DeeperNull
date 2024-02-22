version 1.0

workflow fit_dn_model {
    input {
        File covar_file
        File pheno_file
        File model_config
        File train_samp_file
        Array[File] pred_samp_files
        String pat
    }

    call fit_dn_model_task {
        input:
            covar_file=covar_file,
            pheno_file=pheno_file,
            model_config=model_config,
            train_samp_file=train_samp_file,
            pred_samp_files=pred_samp_files,
            pat=pat
    }

    output {
        File ens_preds = fit_dn_model_task.ens_preds
        File ho_jointplot = fit_dn_model_task.ho_jointplot
        File ho_jointplot_male = fit_dn_model_task.ho_jointplot_male
        File ho_jointplot_female = fit_dn_model_task.ho_jointplot_female
        File ho_preds = fit_dn_model_task.ho_preds
        File ho_scatter = fit_dn_model_task.ho_scatter
        File ho_scores = fit_dn_model_task.ho_scores
        File fit_model_config = fit_dn_model_task.fit_model_config
        File? training_curve_0 = fit_dn_model_task.training_curve_0
        File? training_curve_1 = fit_dn_model_task.training_curve_1
        File? training_curve_2 = fit_dn_model_task.training_curve_2
        File? training_curve_3 = fit_dn_model_task.training_curve_3
        File? training_curve_4 = fit_dn_model_task.training_curve_4
    }

    meta {
        description: "Fit deep null style model on CPU"
    }
}

task fit_dn_model_task {
    input {
        File covar_file
        File pheno_file
        File model_config
        File train_samp_file
        Array[File] pred_samp_files
        String pat
    }

    command <<<
        pip3 list

        # Clone DeeperNull repo
        git clone https://~{pat}@github.com/RossDeVito/DeeperNull.git

        # Install DeeperNull
        cd DeeperNull
        pip3 install .
        cd ..

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
        docker: "gcr.io/ucsd-medicine-cast/fit_dn_model:latest"
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
        File? training_curve_0 = "training_curve_0.png" 
        File? training_curve_1 = "training_curve_1.png" 
        File? training_curve_2 = "training_curve_2.png" 
        File? training_curve_3 = "training_curve_3.png"
        File? training_curve_4 = "training_curve_4.png"
    }
}