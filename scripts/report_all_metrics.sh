#!/bin/bash
FINAL_RESULT_DIR="results/"

for work_dir in ${FINAL_RESULT_DIR}*/; do
    for Folder in ${work_dir}*/; do
        echo "Processing: ${Folder}"

        python llava/eval/report_results.py \
            --result_file "${Folder}/answer-file.jsonl" \
            --metric_output_file "${Folder}/metrics.txt" \
            --hit_sample_file "${Folder}/hit.csv" \
            --not_hit_sample_file "${Folder}/not_hit.csv"
    done
done
