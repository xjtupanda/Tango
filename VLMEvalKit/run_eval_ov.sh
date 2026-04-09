#!/bin/bash

algos=('tango')
pre_retain_ratios=('0.1' '0.15' '0.2')

for algo in "${algos[@]}"; do 
    for pre_retain_ratio in "${pre_retain_ratios[@]}"; do 

        echo "=================================================================="
        echo "Config: Algo=[$algo] Retain ratio=[$pre_retain_ratio]"
        echo "=================================================================="

        dir_name="ratio_${pre_retain_ratio}"
        work_dir="./outputs/llava_ov_qwen2_7b/${algo}/${dir_name}"
        
        cmd_args=()
        
        if [[ "$algo" == "tango" ]]; then
            if [[ "$pre_retain_ratio" == "0.1" ]]; then
                similarity_threshold="0.65"
            else
                similarity_threshold="0.8"
            fi
            segment_type='holitom'  # can also be 'fastvid'
            context_ratio="0.4"
            num_neighbors="7"

            cmd_args+=(
                "--segment-type" "$segment_type"
                "--similarity-threshold" "$similarity_threshold"
                "--context-ratio" "$context_ratio"
                "--num-neighbors" "$num_neighbors"
            )
        fi

        torchrun --nproc-per-node=8 run.py \
            --config "llava_run_config/${algo}_ov.json" \
            --verbose \
            --retry 10 \
            --judge-args '{"use_azure": true}' \
            --work-dir "${work_dir}" \
            --pre-retain-ratio "${pre_retain_ratio}" \
            --reuse \
            "${cmd_args[@]}"
            
    done
done