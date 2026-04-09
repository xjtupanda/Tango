PRUNE_CONFIG = {
    # General configs.
    "pre_prune_r": 0.1,    # Pre-llm retention ratio.
    "prune_k": 18,  # Prune at which LLM layer, starts from 1.
    "prune_r": 0.5, # Intra-LLM retention ratio
    "image_token_start_index": 0, # Dynamically set in func "prepare_inputs_labels_for_multimodal", for different input.
    "image_token_length": 0,      # Dynamically set in func "prepare_inputs_labels_for_multimodal", for different input.

    # pandayin: Vars stored for ST-RoPE.
    "time_intervals": [],   # frame timestamp list
    
    # pandayin: for attn-weight filter (mask sink token)
    "special_attn_token_list": [],
    
    # pandayin: For fastvid and holitom
    "similarity_threshold": 0.8,
    "num_neighbors": 3,   # number of neighbors for dpc-knn clustering algorithm.
    "context_ratio": 0.0,
    "alpha": 1.5,   # expansion coefficient
    "base_time": 10000.0, # Theta base for temporal dim.
    "base_space": 1000.0, # Theta base for spatial dim.
    "segment_type": "", # we reuse previous segmentation methods: 'holitom' or 'fastvid'
}