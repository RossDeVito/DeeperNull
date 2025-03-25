PHENOTYPES=(
	# "standing_height_50"
	# "body_fat_percentage_23099"
	# "platelet_count_30080"
	# "glycated_haemoglobin_30750"
	"vitamin_d_30890"
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
)

SAVED_MODEL_FNAMES=(
	"model_0.json"
	"model_1.json"
	"model_2.json"
	"model_3.json"
	"model_4.json"
)

# Set the version of the model to use
VERSION=V4_w_save
COVAR_SET=age_sex_all_coords_time
NULL_MODEL=xgb_3
NULL_MODEL_TYPE=xgb
CLASSIFICATION=false

# Default locations
NULL_SAVE_DIR=/rdevito/deep_null/dn_output/${VERSION}

# Run SHAP job for each phenotype
for PHENO in "${PHENOTYPES[@]}"; do

	MODEL_SAVE_DIR=${NULL_SAVE_DIR}/${PHENO}/${COVAR_SET}/${NULL_MODEL}
	echo "Running SHAP for: $MODEL_SAVE_DIR"

	SAVED_MODEL_FILES=()
	for SAVED_MODEL_FNAME in "${SAVED_MODEL_FNAMES[@]}"; do
		SAVED_MODEL_FILES+=("${MODEL_SAVE_DIR}/${SAVED_MODEL_FNAME}")
	done
	# Join saved model files with a space
	SAVED_MODEL_FILES=$(IFS=" "; echo "${SAVED_MODEL_FILES[*]}")

	CLASS_ARG_STRING=""
	if [ "$CLASSIFICATION" = true ]; then
		CLASS_ARG_STRING="--classification"
	fi

	python launcher.py \
		--desc ${VERSION}/${PHENO}/${COVAR_SET}/${NULL_MODEL} \
		--covar_set ${COVAR_SET} \
		--model_files ${SAVED_MODEL_FILES} \
		--save_dir ${MODEL_SAVE_DIR} \
		--model_type ${NULL_MODEL_TYPE} \
		${CLASS_ARG_STRING}

done