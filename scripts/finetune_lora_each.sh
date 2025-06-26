DATASET_SPLITS=("pathvqa" "slake-vqarad"  "derm" "nmi" "iu-x-ray" "PCAM" "CXP" "HAM")
DEVICES=0
for dataset in "${DATASET_SPLITS[@]}"; do
    bash ./scripts/finetune/main_each.sh "$dataset" "$DEVICES"
done