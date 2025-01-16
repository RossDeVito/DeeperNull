# Define lists of model configs and covar sets
# MODEL_CONFIGS=("lin_reg_1" "lasso_1" "ridge_2")
# MODEL_CONFIGS=("deepnull_orig_1" "deepnull_es_1" "deepnull_eswp_1" "deepnull_eswp_sm_1")
# MODEL_CONFIGS=("xgb_4" "xgb_5")
MODEL_CONFIGS=(
	"lin_reg_1" "lasso_1" "ridge_2"
	"deepnull_orig_1" "deepnull_es_1" "deepnull_eswp_1" "deepnull_eswp_sm_1"
	"xgb_1" "xgb_2" "xgb_3"
)

COVAR_SETS=(
	"age_sex" "age_sex_pc"
	"age_sex_birth_coords" "age_sex_birth_coords_pc"
	"age_sex_home_coords" "age_sex_home_coords_pc"
	"age_sex_all_coords" "age_sex_all_coords_pc"
	"age_sex_tod" "age_sex_tod_pc"
	"age_sex_toy" "age_sex_toy_pc"
	"age_sex_time" "age_sex_time_pc"
	"age_sex_all_coords_time" "age_sex_all_coords_time_pc"
)

CONFIG_DIR=model_configs

PHENOTYPES=(
	# "standing_height_50"
	# "body_fat_percentage_23099"
	# "platelet_count_30080"
	# "glycated_haemoglobin_30750"
	# "vitamin_d_30890"
	# "diastolic_blood_pressure_4079"
	# "systolic_blood_pressure_4080"
	# "FEV1_3063"
	# "FVC_3062"
	# "HDL_cholesterol_30760"
	# "LDL_direct_30780"
	# "triglycerides_30870"
	# "c-reactive_protein_30710"
	# "creatinine_30700"
	# "alanine_aminotransferase_30620"
	# "aspartate_aminotransferase_30650"
	"grip_strength"
	# "heel_bone_mineral_density_3148"
	# "mean_time_to_identify_matches_20023"
	# "number_of_incorrect_matches_399"
	# "sleep_duration_1160"
	# "adjusted_telomere_ratio_22191"
	# "white_blood_cell_count_30000"
	# "red_blood_cell_count_30010"
	# "haemoglobin_concentration_30020"
	# "mean_corpuscular_volume_30040"
	# "glucose_30740"
	# "urate_30880"
	# "testosterone_30850"
	# "IGF1_30770"
	# "SHBG_30830" 
)

VERSION=V4

GPU=false
CPU_INSTANCE_TYPE=mem1_ssd1_v2_x16
# CPU_INSTANCE_TYPE=mem1_ssd1_v2_x36

PAT_FNAME=pat_dn_read.txt

# Loop through all combinations of model configs, covar sets, and phenotypes
for PHENO in "${PHENOTYPES[@]}"; do
	for MODEL_CONFIG in "${MODEL_CONFIGS[@]}"; do
		for COVAR_SET in "${COVAR_SETS[@]}"; do
			echo "Running with model config: $MODEL_CONFIG, covar set: $COVAR_SET, and phenotype: $PHENO"
			if [ "$GPU" = true ]; then
				echo "Launching Deep Null training on GPU"
				python fit_dn_model_launcher_ukbb.py \
					--model-desc $MODEL_CONFIG \
					--model-config ${CONFIG_DIR}/${MODEL_CONFIG}.json \
					--covar-set $COVAR_SET \
					--pheno $PHENO \
					--pat $PAT_FNAME \
					-v $VERSION \
					-g
			else
				echo "Launching Deep Null training on CPU"
				python fit_dn_model_launcher_ukbb.py \
					--model-desc $MODEL_CONFIG \
					--model-config ${CONFIG_DIR}/${MODEL_CONFIG}.json \
					--covar-set $COVAR_SET \
					--pheno $PHENO \
					--pat $PAT_FNAME \
					-v $VERSION \
					-i "$CPU_INSTANCE_TYPE"
			fi
		done
	done
done
