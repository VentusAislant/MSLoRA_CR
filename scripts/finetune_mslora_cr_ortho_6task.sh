export WANDB_MODE=disabled
source ./scripts/base/port_generator.sh
PORT=$(generate_random_port)
echo "Generated master port: $PORT"

# "pathvqa" "slake-vqarad" "PCAM" "CXP" "nmi" "iu-x-ray"
# 1 0 1 0 1 0
#######################################################
# Important parameter about the version of training
#######################################################
TRAIN_VERSION="finetune_MSLORA-CR-ORTHO_6TASK"
PRETRAINED_MODEL_PATH="./pretrained_models/llava_med_v1.5"
MODEL_NAME=$(basename "$PRETRAINED_MODEL_PATH")
#######################################################
# parameter about the hyper-parameters of training
#######################################################
DEVICES=1
LR=2e-4
EPOCH=0.000003
BATCH_SIZE=32
GRADIENT_ACC_STEPS=1
LORA_RANK=64
LORA_ALPHA=64

#######################################################
# The following content does not need modification.
#######################################################

MAX_TASK=1
DATASET_SPILT="pathvqa"
IMAGE_FOLDER="./data/MSLoRA_CR/PathVQA"
TRAIN_DATA_PATH="${IMAGE_FOLDER}/train.json"
OUTPUT_MODEL_NAME="${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/pathvqa/"

EPOCH=0.000003
deepspeed --include "localhost:${DEVICES}" --master_port "${PORT}" llava/train/train.py \
    --deepspeed ./scripts/base/zero1.json \
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
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 1 \
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


MAX_TASK=2
DATASET_SPILT="slake-vqarad"
IMAGE_FOLDER="./data/MSLoRA_CR/Slake-VQARad"
TRAIN_DATA_PATH="${IMAGE_FOLDER}/train.json"
pretrain_lora_path1="checkpoints/${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/pathvqa/cl_lora.bin"
OUTPUT_MODEL_NAME="${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/pathvqa_slake-vqarad/"

EPOCH=0.000004
deepspeed --include "localhost:${DEVICES}" --master_port "${PORT}" llava/train/train.py \
    --deepspeed ./scripts/base/zero1.json \
    --model_path $PRETRAINED_MODEL_PATH \
    --lora_enable True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.0 \
    --max_task $MAX_TASK \
    --previous_lora_path $pretrain_lora_path1 \
    --is_same_modality 0 \
    --seed 42 --alpha 0.1 --beta 0.01 \
    \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --bf16 True \
    --output_dir ./checkpoints/$OUTPUT_MODEL_NAME \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 1 \
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


MAX_TASK=3
DATASET_SPILT="PCAM"
IMAGE_FOLDER="./data/MSLoRA_CR/PCam"
TRAIN_DATA_PATH="${IMAGE_FOLDER}/train.json"
pretrain_lora_path2="checkpoints/${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/pathvqa_slake-vqarad/cl_lora.bin"
OUTPUT_MODEL_NAME="${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/pathvqa_slake-vqarad_PCAM/"

EPOCH=0.000001
deepspeed --include "localhost:${DEVICES}" --master_port "${PORT}" llava/train/train.py \
    --deepspeed ./scripts/base/zero1.json \
    --model_path $PRETRAINED_MODEL_PATH \
    --lora_enable True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.0 \
    --max_task $MAX_TASK \
    --previous_lora_path $pretrain_lora_path1 $pretrain_lora_path2 \
    --is_same_modality 1 0 \
    --seed 42 --alpha 0.1 --beta 0.01 \
    \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --bf16 True \
    --output_dir ./checkpoints/$OUTPUT_MODEL_NAME \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 1 \
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


# "PCAM" "CXP" "HAM" "nmi" "iu-x-ray"
MAX_TASK=4
DATASET_SPILT="CXP"
IMAGE_FOLDER="./data/MSLoRA_CR/CXP"
TRAIN_DATA_PATH="${IMAGE_FOLDER}/train.json"
pretrain_lora_path3="checkpoints/${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/pathvqa_slake-vqarad_PCAM/cl_lora.bin"
OUTPUT_MODEL_NAME="${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/pathvqa_slake-vqarad_PCAM_CXP/"

EPOCH=0.000001
deepspeed --include "localhost:${DEVICES}" --master_port "${PORT}" llava/train/train.py \
    --deepspeed ./scripts/base/zero1.json \
    --model_path $PRETRAINED_MODEL_PATH \
    --lora_enable True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.0 \
    --max_task $MAX_TASK \
    --previous_lora_path $pretrain_lora_path1 $pretrain_lora_path2 $pretrain_lora_path3 \
    --is_same_modality 0 1 0 \
    --seed 42 --alpha 0.1 --beta 0.01 \
    \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --bf16 True \
    --output_dir ./checkpoints/$OUTPUT_MODEL_NAME \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 1 \
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


# "PCAM" "CXP" "HAM" "nmi" "iu-x-ray"
MAX_TASK=5
EPOCH=0.000006
DATASET_SPILT="nmi"
IMAGE_FOLDER="./data/MSLoRA_CR/WSI-DX/"
TRAIN_DATA_PATH="${IMAGE_FOLDER}/train.json"
pretrain_lora_path4="checkpoints/${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/pathvqa_slake-vqarad_PCAM_CXP/cl_lora.bin"
OUTPUT_MODEL_NAME="${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/pathvqa_slake-vqarad_PCAM_CXP_nmi/"
deepspeed --include "localhost:${DEVICES}" --master_port "${PORT}" llava/train/train.py \
    --deepspeed ./scripts/base/zero1.json \
    --model_path $PRETRAINED_MODEL_PATH \
    --lora_enable True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.0 \
    --max_task $MAX_TASK \
    --previous_lora_path $pretrain_lora_path1 $pretrain_lora_path2 $pretrain_lora_path3 $pretrain_lora_path4 \
    --is_same_modality 1 0 1 0 \
    --seed 42 --alpha 0.1 --beta 0.01 \
    \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --bf16 True \
    --output_dir ./checkpoints/$OUTPUT_MODEL_NAME \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 1 \
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


# "PCAM" "CXP" "HAM" "nmi" "iu-x-ray"
MAX_TASK=6
BATCH_SIZE=16
EPOCH=0.000009
DATASET_SPILT="iu-x-ray"
IMAGE_FOLDER="./data/MSLoRA_CR/IU-X-Ray/"
TRAIN_DATA_PATH="${IMAGE_FOLDER}/train.json"
pretrain_lora_path5="checkpoints/${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/pathvqa_slake-vqarad_PCAM_CXP_nmi/cl_lora.bin"
OUTPUT_MODEL_NAME="${TRAIN_VERSION}-${LORA_RANK}-${LORA_ALPHA}_${MODEL_NAME}/pathvqa_slake-vqarad_PCAM_CXP_nmi_iu-x-ray/"
deepspeed --include "localhost:${DEVICES}" --master_port "${PORT}" llava/train/train.py \
    --deepspeed ./scripts/base/zero1.json \
    --model_path $PRETRAINED_MODEL_PATH \
    --lora_enable True \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.0 \
    --max_task $MAX_TASK \
    --previous_lora_path $pretrain_lora_path1 $pretrain_lora_path2 $pretrain_lora_path3 $pretrain_lora_path4 $pretrain_lora_path5\
    --is_same_modality 0 1 0 1 0 \
    --seed 42 --alpha 0.1 --beta 0.01 \
    \
    --data_path $TRAIN_DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --bf16 True \
    --output_dir ./checkpoints/$OUTPUT_MODEL_NAME \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 100 \
    --save_total_limit 1 \
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