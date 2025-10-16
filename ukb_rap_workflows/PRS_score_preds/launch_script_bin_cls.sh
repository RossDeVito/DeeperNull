# MODEL_TYPE=basil_age_sex_pc_lasso_25
# MODEL_TYPE=basil_age_sex_all_coords_pc_lasso_25
# MODEL_TYPE=basil_age_sex_time_pc_lasso_25
# MODEL_TYPE=basil_age_sex_all_coords_time_pc_lasso_25

# MODEL_TYPE=basil_age_sex_pc_null_xgb_3_age_sex_lasso_25
# MODEL_TYPE=basil_age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords_lasso_25
MODEL_TYPE=basil_age_sex_time_pc_null_xgb_3_age_sex_time_lasso_25
# MODEL_TYPE=basil_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_lasso_25
# MODEL_TYPE=basil_age_sex_all_coords_time_pc_null_xgb_3_age_sex_all_coords_time_pc_lasso_25

# MODEL_TYPE=prscs_age_sex_pc
# MODEL_TYPE=prscs_age_sex_all_coords_pc

PHENOTYPES=(
	# "asthma_42015"
	"depression_20438"
	# "diabetes_2443"
)

for PHENO in "${PHENOTYPES[@]}"; do
    echo "Scoring phenotype $PHENO for model $MODEL_TYPE"
	python launcher_bin_cls.py -m ${MODEL_TYPE} -p ${PHENO}
done
