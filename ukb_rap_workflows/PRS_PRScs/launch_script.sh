PHENOTYPES=(
	# "standing_height_50"
	# "body_fat_percentage_23099"
	"platelet_count_30080"
	"glycated_haemoglobin_30750"
	# "vitamin_d_30890"
	# "diastolic_blood_pressure_4079"
	"systolic_blood_pressure_4080"
	"FEV1_3063"
	"FVC_3062"
	"HDL_cholesterol_30760"
	# "LDL_direct_30780"
	# "triglycerides_30870"
	"c-reactive_protein_30710"
	"creatinine_30700"
	"alanine_aminotransferase_30620"
	"aspartate_aminotransferase_30650"
)

# Set the covariate set

USE_NULL_MODEL=true

# COVAR_SET=age_sex_pc
# COVAR_SET=age_sex_all_coords_pc
# COVAR_SET=age_sex_time_pc
COVAR_SET=age_sex_all_coords_time_pc

# NULL_COVAR_SET=age_sex
# NULL_COVAR_SET=age_sex_all_coords
# NULL_COVAR_SET=age_sex_time
# NULL_COVAR_SET=age_sex_all_coords_time
NULL_COVAR_SET=age_sex_all_coords_time_pc

NULL_MODEL_TYPE=xgb_3

# Run with or without null model
if [ "$USE_NULL_MODEL" = true ]; then
	for PHENO in "${PHENOTYPES[@]}"; do
		echo "Running PRScs PRS for phenotype: $PHENO with $NULL_MODEL_TYPE $COVAR_SET null model."
		python launcher.py \
			-p "${PHENO}" --covar-set "${COVAR_SET}" \
			--null-covar-set "${NULL_COVAR_SET}" \
			--null-model "${NULL_MODEL_TYPE}"
	done
else
	for PHENO in "${PHENOTYPES[@]}"; do
		echo "Running PRScs PRS for phenotype: $PHENO with covariate set: $COVAR_SET"
		python launcher.py -p "${PHENO}" --covar-set "${COVAR_SET}" 
	done
fi



# --geno-fname allchr_wbqc_dev_xsmall