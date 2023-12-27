OUTPUT_PATH="./test"

HYDRA_FULL_ERROR=1 python ./src/core/eval.py \
# save_path="$SAVE_PATH" \
# Trainer.devices="1" \
# +module.save_path="$OUTPUT_PATH" \
# +module.model.device="cuda:1" \