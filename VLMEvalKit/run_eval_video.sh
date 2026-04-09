#!/bin/bash

algos=('tango')
retain_ratios=('0.1' '0.15' '0.2')

for algo in "${algos[@]}"; do 
    for retain_ratio in "${retain_ratios[@]}"; do 

        echo "=================================================================="
        echo "Config: Algo=[$algo] Retain ratio=[$retain_ratio]"
        echo "=================================================================="

        dir_name="ratio_${retain_ratio}"
        work_dir="./outputs/llava_video_qwen2_7b/${algo}/${dir_name}"
        
        cmd_args=()
        
        if [[ "$algo" == "tango" ]]; then
            if [[ "$retain_ratio" == "0.1" ]]; then
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
        
        pruned_layer=18
        llm_retain_ratio=0.5

        set_ratio=$(awk -v r="$retain_ratio" -v k="$pruned_layer" -v p="$llm_retain_ratio" \
        'BEGIN { 
            res = (28 * r) / (k + (28 - k) * p);
            if (res > 1) res = 1.0; 
            print res 
        }')

        echo "=================================================================="
        echo "Config: Algo=[$algo] Pre Retain ratio=[$set_ratio]"
        echo "=================================================================="

        cmd_args+=(
            "--pruned-layer" "$pruned_layer"
            "--pre-retain-ratio" "$set_ratio"
            "--retain-ratio" "$llm_retain_ratio"
        )

        torchrun --nproc-per-node=8 run.py \
            --config "tmp_run_config/${algo}_video.json" \
            --verbose \
            --retry 10 \
            --judge-args '{"use_azure": true}' \
            --work-dir "${work_dir}" \
            --reuse \
            "${cmd_args[@]}"
            
    done
done