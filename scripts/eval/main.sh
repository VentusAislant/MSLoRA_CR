RESULT_DIR="results"

DATASET_SPLIT=$1
MODEL_PATH=$2
PREVIOUS_LORAS_STRING=$3
TASK_MASK=$4
MODEL_NAME=$5

echo "========================================================"
echo "DATASET_SPLIT: $DATASET_SPLIT"
echo "MODEL_PATH: $MODEL_PATH"
echo "PREVIOUS_LORAS_STRING: $PREVIOUS_LORAS_STRING"
echo "TASK_MASK: $TASK_MASK"
echo "MODEL_NAME: $MODEL_NAME"
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

if [[ "$DATASET_SPLIT" == "pathvqa" ]]; then
    QUESTION_FILE="./data/MSLoRA_CR/PathVQA/test.jsonl"
    IMAGE_FOLDER="./data/MSLoRA_CR/PathVQA"
elif [[ "$DATASET_SPLIT" == "slake-vqarad" ]]; then
    QUESTION_FILE="./data/MSLoRA_CR/Slake-VQARad/test.jsonl"
    IMAGE_FOLDER="./data/MSLoRA_CR/Slake-VQARad"
elif [[ "$DATASET_SPLIT" == "derm" ]]; then
    QUESTION_FILE="./data/MSLoRA_CR/Fitzpatrick/test.jsonl"
    IMAGE_FOLDER="./data/MSLoRA_CR/Fitzpatrick"
elif [[ "$DATASET_SPLIT" == "CXP" ]]; then
    QUESTION_FILE="./data/MSLoRA_CR/CXP/test.jsonl"
    IMAGE_FOLDER="./data/MSLoRA_CR/CXP"
elif [[ "$DATASET_SPLIT" == "HAM" ]]; then
    QUESTION_FILE="./data/MSLoRA_CR/HAM/test.jsonl"
    IMAGE_FOLDER="./data/MSLoRA_CR/HAM"
elif [[ "$DATASET_SPLIT" == "PCAM" ]]; then
    QUESTION_FILE="./data/MSLoRA_CR/PCam/test.jsonl"
    IMAGE_FOLDER="./data/MSLoRA_CR/PCam"
elif [[ "$DATASET_SPLIT" == "iu-x-ray" ]]; then
    QUESTION_FILE="./data/MSLoRA_CR/IU-X-Ray/test.jsonl"
    IMAGE_FOLDER="./data/MSLoRA_CR/IU-X-Ray"
elif [[ "$DATASET_SPLIT" == "nmi" ]]; then
    QUESTION_FILE="./data/MSLoRA_CR/WSI-DX/test.jsonl"
    IMAGE_FOLDER="./data/MSLoRA_CR/WSI-DX"
fi

echo "Quesiton File: $QUESTION_FILE"
echo "IMAGE_FOLDER: $IMAGE_FOLDER"

IFS=' ' read -r -a PREVIOUS_LORAS <<< "$PREVIOUS_LORAS_STRING"
echo "PREVIOUS_LORAS elements:"
for lora in "${PREVIOUS_LORAS[@]}"; do
    echo "$lora"
done

FINAL_RESULT_DIR="${RESULT_DIR}/${MODEL_NAME}/${DATASET_SPLIT}"

python llava/eval/eval_mslora_cr.py \
    --conv-mode mistral_instruct \
    --model-path $MODEL_PATH \
    --previous_lora_path $PREVIOUS_LORAS_STRING \
    --task_mask $TASK_MASK \
    --question-file $QUESTION_FILE \
    --image-folder $IMAGE_FOLDER \
    --answers-file "${FINAL_RESULT_DIR}/answer-file.jsonl" \
    --temperature 0.0

wait

python llava/eval/report_results.py \
    --result_file "${FINAL_RESULT_DIR}/answer-file.jsonl" \
    --metric_output_file "${FINAL_RESULT_DIR}/metrics.txt" \
    --hit_sample_file "${FINAL_RESULT_DIR}/hit.csv" \
    --not_hit_sample_file "${FINAL_RESULT_DIR}/not_hit.csv"