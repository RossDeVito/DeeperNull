# Define lists of model configs and covar sets

MODEL_CONFIG=dev_5c1

COVAR_SET="age_sex_all_coords_time"

CONFIG_DIR=model_configs/mt

PHENOTYPES=(
	"standing_height_50"
	"body_fat_percentage_23099"
	"platelet_count_30080"
	"glycated_haemoglobin_30750"
	"vitamin_d_30890"
	"diastolic_blood_pressure_4079"
	"systolic_blood_pressure_4080"
	"FEV1_3063"
	"FVC_3062"
	"HDL_cholesterol_30760"
	"LDL_direct_30780"
	"triglycerides_30870"
	"c-reactive_protein_30710"
	"creatinine_30700"
	"alanine_aminotransferase_30620"
	"aspartate_aminotransferase_30650"
	# "grip_strength"
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

VERSION=V_dev_mt_1

GPU=false
# CPU_INSTANCE_TYPE=mem1_ssd1_v2_x16
CPU_INSTANCE_TYPE=mem1_ssd1_v2_x36

# Create PHENOS_ARG_STR, a space-separated string of phenotypes
PHENOS_ARG_STR=""
for PHENO in "${PHENOTYPES[@]}"; do
	PHENOS_ARG_STR="$PHENOS_ARG_STR $PHENO"
done

# Launch training job
if [ "$GPU" = true ]; then
	echo "Launching Deep Null training on GPU"
	python fit_dn_mt_model_launcher.py \
		--model-desc $MODEL_CONFIG \
		--model-config ${CONFIG_DIR}/${MODEL_CONFIG}.json \
		--covar-set $COVAR_SET \
		--pheno $PHENOS_ARG_STR \
		-v $VERSION \
		-g
else
	echo "Launching Deep Null training on CPU"
	python fit_dn_mt_model_launcher.py \
		--model-desc $MODEL_CONFIG \
		--model-config ${CONFIG_DIR}/${MODEL_CONFIG}.json \
		--covar-set $COVAR_SET \
		--pheno $PHENOS_ARG_STR \
		-v $VERSION \
		-i "$CPU_INSTANCE_TYPE"
fi
