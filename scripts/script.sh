OUTPUT_PATH="./test"

python ./src/core/train.py \
+save_path="$SAVE_PATH" \
Trainer.devices="1" \
+module.save_path="$OUTPUT_PATH" \
+module.model.device="cuda:1" \