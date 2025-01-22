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

# BASE_COVAR_SET=age_sex
# BASE_COVAR_SET=age_sex_all_coords
# BASE_COVAR_SET=age_sex_time
BASE_COVAR_SET=age_sex_all_coords_time

# Option to include PCs in null covariate set
PC_IN_NULL=true

# Set the covariate sets
COVAR_SET=${BASE_COVAR_SET}_pc

if [ "$PC_IN_NULL" = true ]; then
	NULL_COVAR_SET=${BASE_COVAR_SET}_pc
else
	NULL_COVAR_SET=${BASE_COVAR_SET}
fi

NULL_MODEL_TYPE=xgb_3

# Set the model type
MODEL_TYPE=lasso

# Set number of iterations for BASIL
NUM_ITER=25

# Iterate over each phenotype and launch the GWAS workflow
for PHENO in "${PHENOTYPES[@]}"; do
	echo "Running BASIL PRS for phenotype: $PHENO with covariate set: $COVAR_SET for $NUM_ITER iterations"
	python launcher_null.py \
		-p "${PHENO}" \
		--covar-set "${COVAR_SET}" \
		--null-covar-set "${NULL_COVAR_SET}" \
		--null-model "${NULL_MODEL_TYPE}" \
		-m "${MODEL_TYPE}" \
		-n "${NUM_ITER}"
done
