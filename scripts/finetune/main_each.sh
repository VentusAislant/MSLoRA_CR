#!/bin/bash
export WANDB_MODE=disabled
source ./scripts/base/port_generator.sh
PORT=$(generate_random_port)
echo "Generated master port: $PORT"

DATASET_SPILT=$1
DEVICES=$2

EPOCH=0.000003
if [[ "$DATASET_SPLIT" == "pathvqa" ]]; then
    TRAIN_DATA_PATH="./data/MSLoRA_CR/PathVQA/train.json"
    IMAGE_FOLDER="./data/MSLoRA_CR/PathVQA"
elif [[ "$DATASET_SPLIT" == "slake-vqarad" ]]; then
    TRAIN_DATA_PATH="./data/MSLoRA_CR/Slake-VQARad/train.json"
    IMAGE_FOLDER="./data/MSLoRA_CR/Slake-VQARad"
elif [[ "$DATASET_SPLIT" == "derm" ]]; then
    TRAIN_DATA_PATH="./data/MSLoRA_CR/Fitzpatrick/train.json"
    IMAGE_FOLDER="./data/MSLoRA_CR/Fitzpatrick"
elif [[ "$DATASET_SPLIT" == "CXP" ]]; then
    TRAIN_DATA_PATH="./data/MSLoRA_CR/CXP/train.json"
    IMAGE_FOLDER="./data/MSLoRA_CR/CXP"
    EPOCH=0.000001
elif [[ "$DATASET_SPLIT" == "HAM" ]]; then
    TRAIN_DATA_PATH="./data/MSLoRA_CR/HAM/train.json"
    IMAGE_FOLDER="./data/MSLoRA_CR/HAM"
    EPOCH=0.000001
elif [[ "$DATASET_SPLIT" == "PCAM" ]]; then
    TRAIN_DATA_PATH="./data/MSLoRA_CR/PCam/train.json"
    IMAGE_FOLDER="./data/MSLoRA_CR/PCam"
    EPOCH=0.000001
elif [[ "$DATASET_SPLIT" == "iu-x-ray" ]]; then
    TRAIN_DATA_PATH="./data/MSLoRA_CR/IU-X-Ray/train.json"
    IMAGE_FOLDER="./data/MSLoRA_CR/IU-X-Ray"
    EPOCH=0.000006
elif [[ "$DATASET_SPLIT" == "nmi" ]]; then
    TRAIN_DATA_PATH="./data/MSLoRA_CR/WSI-DX/train.json"
    IMAGE_FOLDER="./data/MSLoRA_CR/WSI-DX"
    EPOCH=0.000006
fi

#######################################################
# Important parameter about the version of training
#######################################################
TRAIN_VERSION="finetune_lora_each"
PRETRAINED_MODEL_PATH="./pretrained_models/llava_med_v1.5"
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
