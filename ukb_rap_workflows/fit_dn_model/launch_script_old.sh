# Launch Deep Null training

# MODEL_CONFIG=lin_reg_1
# MODEL_CONFIG=lasso_1
# MODEL_CONFIG=ridge_2
MODEL_CONFIG=deepnull_orig_1
# MODEL_CONFIG=deepnull_es_1
# MODEL_CONFIG=deepnull_eswp_1
# MODEL_CONFIG=deepnull_eswp_sm_1
CONFIG_DIR=model_configs

# COVAR_SET=age_sex
# COVAR_SET=age_sex_pc
# COVAR_SET=age_sex_birth_coords
# COVAR_SET=age_sex_birth_coords_pc
COVAR_SET=age_sex_all_coords_pc

PHENO=standing_height_50
# PHENO=body_fat_percentage_23099
# PHENO=platelet_count_30080
# PHENO=glycated_haemoglobin_30750

VERSION=V3

GPU=true
CPU_INSTANCE_TYPE=mem1_ssd1_v2_x16
# CPU_INSTANCE_TYPE=mem1_ssd1_v2_x36


PAT_FNAME=pat_dn_read.txt

# Launch Deep Null training
if [ "$GPU" = true ]; then
	echo "Launching Deep Null training on GPU"
	python fit_dn_model_launcher_ukbb.py \
		--model-desc $MODEL_CONFIG \
		--model-config ${CONFIG_DIR}/${MODEL_CONFIG}.json \
		--covar-set $COVAR_SET \
		--pheno $PHENO \
		--pat $PAT_FNAME \
		-v $VERSION \
		-g
else
	echo "Launching Deep Null training on CPU"
	python fit_dn_model_launcher_ukbb.py \
		--model-desc $MODEL_CONFIG \
		--model-config ${CONFIG_DIR}/${MODEL_CONFIG}.json \
		--covar-set $COVAR_SET \
		--pheno $PHENO \
		--pat $PAT_FNAME \
		-v $VERSION \
		-i $CPU_INSTANCE_TYPE
fi
