# Launch Deep Null training

MODEL_CONFIG=lin_reg_1
CONFIG_DIR=model_configs

COVAR_SET=age_sex

PHENO=standing_height_50

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
