#!/bin/bash

algos=('tango')
retain_ratios=(0.1 0.15 0.2)

for algo in "${algos[@]}"; do
    for retain_ratio in "${retain_ratios[@]}"; do

        echo "=================================================================="
        echo "Config: Algo=[$algo] Retain ratio=[$retain_ratio]"
        echo "=================================================================="

        dir_name="ratio_${retain_ratio}"
        work_dir="./outputs/Qwen2.5-VL-7B-Instruct/${algo}/${dir_name}"

        segment_type="holitom"  # can also be 'fastvid'
        num_neighbors=7
        context_ratio=0.4
        if [[ "$retain_ratio" == "0.1" ]]; then
            similarity_threshold=0.65
        else
            similarity_threshold=0.8
        fi
        cmd_args+=(
            "--segment-type" "$segment_type"
            "--similarity-threshold" "$similarity_threshold"
            "--context-ratio" "$context_ratio"
            "--num-neighbors" "$num_neighbors"
            "--retain-ratio" "$retain_ratio"
        )
        
        torchrun --nproc-per-node=8 run.py \
            --config "qwen2p5_run_config/${algo}.json" \
            --verbose \
            --retry 10 \
            --judge-args '{"use_azure": true}' \
            --work-dir "${work_dir}" \
            --reuse \
            "${cmd_args[@]}"
    done
done