# For when covariates correlated with null predictions are removed

PHENOTYPES=(
	# "standing_height_50"
	# "body_fat_percentage_23099"
	# "platelet_count_30080"
	"glycated_haemoglobin_30750"
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
	# "asthma_42015"
	# "depression_20438"
	# "diabetes_2443"
)

# MOD_COVAR_SET=age_all_coords_time
MOD_COVAR_SET=sex_all_coords_time
NULL_BASE_COVAR_SET=age_sex_all_coords_time

NULL_DIR=/rdevito/deep_null/dn_output/V4_w_save
# NULL_DIR=/rdevito/deep_null/dn_output/V4_w_save_bin_cls

# Option to include PCs in null covariate set
PC_IN_NULL=false

# Set the covariate sets
COVAR_SET=${MOD_COVAR_SET}_pc

if [ "$PC_IN_NULL" = true ]; then
	NULL_COVAR_SET=${NULL_BASE_COVAR_SET}_pc
else
	NULL_COVAR_SET=${NULL_BASE_COVAR_SET}
fi


NULL_MODEL_TYPE=xgb_3
# NULL_MODEL_TYPE=xgb_3_bin_cls

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
		-n "${NUM_ITER}" \
		--null-dir "${NULL_DIR}"
done
