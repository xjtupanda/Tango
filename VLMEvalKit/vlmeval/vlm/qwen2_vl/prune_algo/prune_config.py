PRUNE_CONFIG = {
    "prune_r": 0.1, # Retention ratio
    "image_token_start_index": 0, # Dynamically set in func "prepare_inputs_labels_for_multimodal", for different input.
    "image_token_length": 0,      # Dynamically set in func "prepare_inputs_labels_for_multimodal", for different input.
    "video_grid_thw": [],              # video shape, num_tokens at T, H, W dimension.
    
    "time_intervals": [],
    "similarity_threshold": 0.8,
    "beta": 0.6,        # weight for aggregation of anchor features and cluster-merged features.
    "num_neighbors": 7,   # number of neighbors for dpc-knn clustering algorithm.
    "context_ratio": 0.0,
    "alpha": 1.5,   # expansion coefficient
    "base_time": 10000.0, # Theta base for temporal dim.
    "base_space": 1000.0, # Theta base for spatial dim.
    "segment_type": "", # 'fastvid' or 'holitom'
}