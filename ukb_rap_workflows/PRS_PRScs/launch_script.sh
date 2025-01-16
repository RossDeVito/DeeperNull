PHENOTYPES=(
	"standing_height_50"
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
	# "c-reactive_protein_30710"
	# "creatinine_30700"
	# "alanine_aminotransferase_30620"
	# "aspartate_aminotransferase_30650"
)

# Set the covariate set
COVAR_SET=age_sex_pc
# COVAR_SET=age_sex_all_coords_pc
# COVAR_SET=age_sex_time_pc
# COVAR_SET=age_sex_all_coords_time_pc

# Iterate over each phenotype and launch the GWAS workflow
for PHENO in "${PHENOTYPES[@]}"; do
    echo "Running PRScs PRS for phenotype: $PHENO with covariate set: $COVAR_SET"
    python launcher.py -p "${PHENO}" --covar-set "${COVAR_SET}" \
		--geno-fname allchr_wbqc_dev_xsmall
done