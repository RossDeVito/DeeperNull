PHENOTYPES=(
	"asthma_42015"
	"depression_20438"
	"diabetes_2443"
)

# Set the covariate set

USE_NULL_MODEL=true

COVAR_SET=age_sex_pc
# COVAR_SET=age_sex_all_coords_pc
# COVAR_SET=age_sex_time_pc
# COVAR_SET=age_sex_all_coords_time_pc

NULL_COVAR_SET=age_sex
# NULL_COVAR_SET=age_sex_all_coords
# NULL_COVAR_SET=age_sex_time
# NULL_COVAR_SET=age_sex_all_coords_time
# NULL_COVAR_SET=age_sex_all_coords_time_pc

NULL_MODEL_TYPE=xgb_3

# Run with or without null model
if [ "$USE_NULL_MODEL" = true ]; then
	for PHENO in "${PHENOTYPES[@]}"; do
		echo "Running PRScs PRS for phenotype: $PHENO with $NULL_MODEL_TYPE $COVAR_SET null model."
		python launcher_bin_cls.py \
			-p "${PHENO}" --covar-set "${COVAR_SET}" \
			--null-covar-set "${NULL_COVAR_SET}" \
			--null-model "${NULL_MODEL_TYPE}"
	done
else
	for PHENO in "${PHENOTYPES[@]}"; do
		echo "Running PRScs PRS for phenotype: $PHENO with covariate set: $COVAR_SET"
		python launcher_bin_cls.py -p "${PHENO}" --covar-set "${COVAR_SET}" 
	done
fi



# --geno-fname allchr_wbqc_dev_xsmall