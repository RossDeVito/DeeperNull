# Define lists of model configs and covar sets
MODEL_CONFIGS=("xgb_3_bin_cls")

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
	"asthma_42015"
	"depression_20438"
	"diabetes_2443"
)

VERSION=V4_w_save_bin_cls

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
				python fit_dn_model_launcher_bin_cls.py \
					--model-desc $MODEL_CONFIG \
					--model-config ${CONFIG_DIR}/${MODEL_CONFIG}.json \
					--covar-set $COVAR_SET \
					--pheno $PHENO \
					--pat $PAT_FNAME \
					-v $VERSION \
					-g
			else
				echo "Launching Deep Null training on CPU"
				python fit_dn_model_launcher_bin_cls.py \
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
