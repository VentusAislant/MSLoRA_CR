export CUDA_VISIBLE_DEVICES=1

###############################################
## Need To Set
###############################################
MODEL_PATH="pretrained_models/llava_med_v1.5"
LORA_CKP_NAME="finetune_lora_MSLoRA-CR-ORTHO_6TASK-64-64_llava_med_v1.5"
DATASET_SPLITS=("pathvqa" "slake-vqarad" "PCAM" "CXP" "nmi" "iu-x-ray")

###############################################
## DO not care
###############################################
path_prefix="checkpoints/${LORA_CKP_NAME}"
previous_loras=()
for i in "${!DATASET_SPLITS[@]}"; do
    if [ "$i" -eq 0 ]; then
        path_suffix="${DATASET_SPLITS[$i]}"
    else
        path_suffix="${path_suffix}_${DATASET_SPLITS[$i]}"
    fi
    previous_loras+=("${path_prefix}/${path_suffix}/cl_lora.bin")
done

for i in "${!previous_loras[@]}"; do
    echo "Previous_Lora$((i+1)) = \"${previous_loras[$i]}\""
done

echo "========================================================"
TASK_MASK=1
previous_loras_string=$(printf "%s " "${previous_loras[@]:0:$TASK_MASK}")
echo "previous loras: $previous_loras_string"
dataset=${DATASET_SPLITS[$TASK_MASK-1]}
echo "========================================================"
echo "TASK MASK: $TASK_MASK"
bash ./scripts/eval/main.sh "$dataset" "$MODEL_PATH" "$previous_loras_string" "$TASK_MASK" "$LORA_CKP_NAME"

TASK_MASK=2
previous_loras_string=$(printf "%s " "${previous_loras[@]:0:$TASK_MASK}")
echo "previous loras: $previous_loras_string"
dataset=${DATASET_SPLITS[$TASK_MASK-1]}
echo "========================================================"
echo "TASK MASK: $TASK_MASK"
bash ./scripts/eval/main.sh "$dataset" "$MODEL_PATH" "$previous_loras_string" "$TASK_MASK" "$LORA_CKP_NAME"

TASK_MASK=3
previous_loras_string=$(printf "%s " "${previous_loras[@]:0:$TASK_MASK}")
echo "previous loras: $previous_loras_string"
dataset=${DATASET_SPLITS[$TASK_MASK-1]}
echo "========================================================"
echo "TASK MASK: $TASK_MASK"
bash ./scripts/eval/main.sh "$dataset" "$MODEL_PATH" "$previous_loras_string" "$TASK_MASK" "$LORA_CKP_NAME"

TASK_MASK=4
previous_loras_string=$(printf "%s " "${previous_loras[@]:0:$TASK_MASK}")
echo "previous loras: $previous_loras_string"
dataset=${DATASET_SPLITS[$TASK_MASK-1]}
echo "========================================================"
echo "TASK MASK: $TASK_MASK"
bash ./scripts/eval/main.sh "$dataset" "$MODEL_PATH" "$previous_loras_string" "$TASK_MASK" "$LORA_CKP_NAME"

TASK_MASK=5
previous_loras_string=$(printf "%s " "${previous_loras[@]:0:$TASK_MASK}")
echo "previous loras: $previous_loras_string"
dataset=${DATASET_SPLITS[$TASK_MASK-1]}
echo "========================================================"
echo "TASK MASK: $TASK_MASK"
bash ./scripts/eval/main.sh "$dataset" "$MODEL_PATH" "$previous_loras_string" "$TASK_MASK" "$LORA_CKP_NAME"

TASK_MASK=6
previous_loras_string=$(printf "%s " "${previous_loras[@]:0:$TASK_MASK}")
echo "previous loras: $previous_loras_string"
dataset=${DATASET_SPLITS[$TASK_MASK-1]}
echo "========================================================"
echo "TASK MASK: $TASK_MASK"
bash ./scripts/eval/main.sh "$dataset" "$MODEL_PATH" "$previous_loras_string" "$TASK_MASK" "$LORA_CKP_NAME"
