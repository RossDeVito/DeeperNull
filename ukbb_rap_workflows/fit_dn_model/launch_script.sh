# Launch Deep Null training

MODEL_CONFIG=xgb_2
CONFIG_DIR=model_configs

# COVAR_SET=age_sex
# COVAR_SET=age_sex_pc
# COVAR_SET=age_sex_birth_coords
COVAR_SET=age_sex_birth_coords_pc

# PHENO=standing_height_50
# PHENO=body_fat_percentage_23099
# PHENO=platelet_count_30080
PHENO=glycated_haemoglobin_30750

VERSION=V2
INSTANCE_TYPE=mem1_ssd1_v2_x16
GPU=false

PAT_FNAME=pat_dn_read.txt

# Launch Deep Null training
if [ "$GPU" = true ]; then
	echo "GPU workflow not implemented"
else
	echo "Launching Deep Null training on CPU"
	python fit_dn_model_launcher_ukbb.py \
		--model-desc $MODEL_CONFIG \
		--model-config ${CONFIG_DIR}/${MODEL_CONFIG}.json \
		--covar-set $COVAR_SET \
		--pheno $PHENO \
		--pat $PAT_FNAME \
		-v $VERSION \
		-i $INSTANCE_TYPE
fi
