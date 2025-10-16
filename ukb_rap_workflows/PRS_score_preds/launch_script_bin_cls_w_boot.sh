#!/bin/bash

MODEL_TYPES=(
	"basil_age_sex_pc_lasso_25"
	"basil_age_sex_all_coords_pc_lasso_25"
	"basil_age_sex_time_pc_lasso_25"
	"basil_age_sex_all_coords_time_pc_lasso_25"
	"basil_age_sex_pc_null_xgb_3_bin_cls_age_sex_lasso_25"
	"basil_age_sex_all_coords_pc_null_xgb_3_bin_cls_age_sex_all_coords_lasso_25"
	"basil_age_sex_time_pc_null_xgb_3_bin_cls_age_sex_time_lasso_25"
	"basil_age_sex_all_coords_time_pc_null_xgb_3_bin_cls_age_sex_all_coords_time_lasso_25"
	"basil_age_sex_all_coords_time_pc_null_xgb_3_bin_cls_age_sex_all_coords_time_pc_lasso_25"
	# "prscs_age_sex_pc"
	# "prscs_age_sex_all_coords_pc"
	# "prscs_age_sex_time_pc"
	# "prscs_age_sex_all_coords_time_pc"
	# "prscs_age_sex_pc_null_xgb_3_bin_cls_age_sex"
	# "prscs_age_sex_all_coords_pc_null_xgb_3_bin_cls_age_sex_all_coords"
	# "prscs_age_sex_time_pc_null_xgb_3_bin_cls_age_sex_time"
	# "prscs_age_sex_all_coords_time_pc_null_xgb_3_bin_cls_age_sex_all_coords_time"
	# "prscs_age_sex_all_coords_time_pc_null_xgb_3_bin_cls_age_sex_all_coords_time_pc"
)

PHENOTYPES=(
	"asthma_42015"
	"depression_20438"
	"diabetes_2443"
)

for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    for PHENO in "${PHENOTYPES[@]}"; do
        echo "Scoring phenotype $PHENO for model $MODEL_TYPE"
        python launcher_bin_cls_w_boot.py -m "${MODEL_TYPE}" -p "${PHENO}"
    done
done
