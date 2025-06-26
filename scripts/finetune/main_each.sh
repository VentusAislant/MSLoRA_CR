#!/bin/bash
export WANDB_MODE=disabled
source ./scripts/port_generator.sh
PORT=$(generate_random_port)
echo "Generated master port: $PORT"

DATASET_SPILT=$1
DEVICES=$2

EPOCH=3
# "slake" "vqarad" "pathvqa" "derm" "crc-val-he" "cbis-ddsm" "iu-x-ray" "nmi"
if [[ "$DATASET_SPILT" == "slake" || "$DATASET_SPILT" == "vqarad" || \
      "$DATASET_SPILT" == "pathvqa" || "$DATASET_SPILT" == "derm" || "$DATASET_SPILT" == "slake-vqarad" ]]; then
    TRAIN_DATA_PATH="./data/annotations/train/${DATASET_SPILT}_train.json"
    IMAGE_FOLDER="./data/all_med/images"
elif [[ "$DATASET_SPILT" == "CXP" ]]; then
    TRAIN_DATA_PATH="./data/classification/chest_xray_pneumonia/chest_xray_pneumonia_train.json"
    IMAGE_FOLDER="./data/classification/chest_xray_pneumonia"
    EPOCH=1
elif [[ "$DATASET_SPILT" == "HAM" ]]; then
    TRAIN_DATA_PATH="./data/classification/HAM10000/ham10000_train.json"
    IMAGE_FOLDER="./data/classification/HAM10000"
    EPOCH=1
elif [[ "$DATASET_SPILT" == "PCAM" ]]; then
    TRAIN_DATA_PATH="./data/classification/PCam/pcam_train.json"
    IMAGE_FOLDER="./data/classification/PCam"
    EPOCH=1
elif [[ "$DATASET_SPILT" == "iu-x-ray" ]]; then
    TRAIN_DATA_PATH="./data/report_gen/IU-X-RAY/iu-x-ray_report_gen_train.json"
    IMAGE_FOLDER="./data/report_gen/IU-X-RAY/"
    EPOCH=6
elif [[ "$DATASET_SPILT" == "nmi" ]]; then
    TRAIN_DATA_PATH="./data/report_gen/NMI/nmi_report_gen_train.json"
    IMAGE_FOLDER="./data/report_gen/NMI/"
    EPOCH=6
fi


#######################################################
# Important parameter about the version of training
#######################################################
TRAIN_VERSION="finetune_lora_each"
PRETRAINED_MODEL_PATH="./pretrained_models/llava_med_1_5"
MODEL_NAME=$(basename "$PRETRAINED_MODEL_PATH")
#######################################################
# parameter about the hyper-parameters of training
#######################################################
LR=2e-4
BATCH_SIZE=16
GRADIENT_ACC_STEPS=1
LORA_RANK=64
LORA_ALPHA=64

MAX_TASK=1
#######################################################
# The following content does not need modification.
#######################################################
OUTPUT_MODEL_NAME="${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/${DATASET_SPILT}"
deepspeed --include "localhost:${DEVICES}" --master_port "${PORT}" llava/train/train.py \
    --deepspeed ./scripts/base/zero3.json \
    --model_path $PRETRAINED_MODEL_PATH \
    --lora_enable True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.0 \
    --max_task $MAX_TASK \
    \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --bf16 True \
    --output_dir ./checkpoints/$OUTPUT_MODEL_NAME \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb
