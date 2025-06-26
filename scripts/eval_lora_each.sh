export CUDA_VISIBLE_DEVICES=0

###############################################
## Need To Set
###############################################
MODEL_PATH="./pretrained_models/llava_med_v1.5"
MODEL_NAME="finetune_lora_each-64-64_llava_med_v1.5"

# for 8 task
DATASET_SPLITS=("pathvqa" "slake-vqarad"  "derm" "PCAM" "CXP" "HAM" "nmi" "iu-x-ray")

# for 6 task
#DATASET_SPLITS=("pathvqa" "slake-vqarad" "PCAM" "CXP" "nmi" "iu-x-ray")

TASK_MASK=1
for i in "${!DATASET_SPLITS[@]}"; do
    dataset=${DATASET_SPLITS[$i]}
    lora_path="checkpoints/${MODEL_NAME}/${dataset}/cl_lora.bin"
    bash ./scripts/eval/main.sh "$dataset" "$MODEL_PATH" "$lora_path" "$TASK_MASK" "$MODEL_NAME"
done