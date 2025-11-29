#!/usr/bin/env bash

CONFIG_DIR="configs"
DATA_ROOT="MM5_SEG"
RGB_LIST=(RGB1 RGB2 RGB3 RGB4 RGB5 RGB6 RGB7 RGB8)

MODELS=(
    "rgb_baseline"
    "early_fusion"
    "mid_fusion"
    "late_fusion"
    "se_fusion"
    "attn_fusion"
)

BASE_EPOCHS=300
BASE_BATCH=32
BASE_LR=0.001
BASE_WD=0.0001

for MODEL in "${MODELS[@]}"; do
    for RGB in "${RGB_LIST[@]}"; do
        
        EXP_NAME="${MODEL}_${RGB}"
        SAVE_DIR="experiments/${EXP_NAME}"
        mkdir configs/${MODEL}
        CFG_FILE="configs/${MODEL}/auto_${EXP_NAME}.yaml"

        echo "--------------------------------------------"
        echo "Launching training: $EXP_NAME"
        echo "Config: $CFG_FILE"
        echo "--------------------------------------------"

        cat > "$CFG_FILE" <<EOF
exp_name: "${EXP_NAME}"

task: "segmentation"
data_root: "${DATA_ROOT}"
modalities: ["rgb", "depth"]
rgb_folder: "${RGB}"
img_size: [224, 224]

model_type: "${MODEL}"
pretrained: true

batch_size: ${BASE_BATCH}
epochs: ${BASE_EPOCHS}
learning_rate: ${BASE_LR}
weight_decay: ${BASE_WD}
num_workers: 4

save_dir: "${SAVE_DIR}"
save_freq: 1000
EOF

        uv run train.py --config "$CFG_FILE"

    done
done
