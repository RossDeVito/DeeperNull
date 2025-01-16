# MODEL_TYPE=prsice
# MODEL_TYPE=basil_age_sex_pc_lasso_25
# MODEL_TYPE=basil_age_sex_all_coords_pc_lasso_25
# MODEL_TYPE=basil_age_sex_pc_null_xgb_3_age_sex_lasso_25
MODEL_TYPE=basil_age_sex_all_coords_pc_null_xgb_3_age_sex_all_coords_lasso_25

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
	"c-reactive_protein_30710"
	# "creatinine_30700"
	# "alanine_aminotransferase_30620"
	# "aspartate_aminotransferase_30650"
)



for PHENO in "${PHENOTYPES[@]}"; do
    echo "Scoring phenotype $PHENO for model $MODEL_TYPE"
	python launcher.py -m ${MODEL_TYPE} -p ${PHENO}
done
