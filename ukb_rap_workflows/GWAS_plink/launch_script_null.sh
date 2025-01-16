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
	"num_cigarettes_day"
)

# BASE_COVAR_SET=age_sex
# BASE_COVAR_SET=age_sex_all_coords
# BASE_COVAR_SET=age_sex_time
BASE_COVAR_SET=age_sex_all_coords_time

COVAR_SET=${BASE_COVAR_SET}_pc
NULL_COVAR_SET=${BASE_COVAR_SET}

NULL_MODEL_TYPE=xgb_3

# Launch GWAS workflow for each phenotype
for PHENO in "${PHENOTYPES[@]}"; do
    echo "Running GWAS for phenotype: $PHENO with covariate set: $COVAR_SET and null covariate set: $NULL_COVAR_SET"
    python launcher_null.py \
        -p "${PHENO}" \
        --covar-set "${COVAR_SET}" \
        --null-covar-set "${NULL_COVAR_SET}" \
        --null-model "${NULL_MODEL_TYPE}"
done