FINAL_RESULT_DIR=results/quilt-llava

for Folder in $(ls -d ${FINAL_RESULT_DIR}/*/ 2>/dev/null); do
    Folder=$(basename "$Folder")
    echo $FINAL_RESULT_DIR
    echo $Folder
    python llava/eval/report_results.py \
        --result_file "${FINAL_RESULT_DIR}/${Folder}/answer-file.jsonl" \
        --metric_output_file "${FINAL_RESULT_DIR}/${Folder}/metrics.txt" \
        --hit_sample_file "${FINAL_RESULT_DIR}/${Folder}/hit.csv" \
        --not_hit_sample_file "${FINAL_RESULT_DIR}/${Folder}/not_hit.csv"
done