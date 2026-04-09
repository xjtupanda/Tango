
import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .utils import (fastvid_segmentation,
                    holitom_segmentation,
                    tango_token_merger
                    )

def tango_pruner(
    image_features: torch.Tensor,
    roped_image_features: torch.Tensor,
    sim_metric: torch.Tensor,          
    salient_score: torch.Tensor,  
    segment_type: str,  
    retain_ratio: float = 0.1,       
    context_ratio: float = 0.0,
    tau: float = 0.8,                
    beta: float = 0.6,               
    k_neighbors: int = 7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Input:
            image_features : (N, L, D)
            sim_metric: (N, L, D) or (N, D) (FastVid: pooled along token dim)
            salient_score: (N, L)
            segment_type: 'fastvid' or 'holitom'
        Output:
            output_features: (N, L, D): Same shape as input image features, kept positions replaced with pruned features.
            retained_indices: (num_kept, D) : indices of image tokens to be kept
    '''
    N, L, D = image_features.shape
    device = image_features.device
    dtype = image_features.dtype

    # =============================================================================
    #         Phase 1: Temporal Video Segmentation: FastVID or HoliTom type
    #                 Get frame num of each video segment.
    # =============================================================================
    # pandayin: upscale to fp32 for similarity calculation
    metric_normed = F.normalize(sim_metric.float(), p=2, dim=-1)    # (N, D) / (N, L, D)
    frame_similarity = (metric_normed[:-1] * metric_normed[1:]).sum(dim=-1)  # (N-1,) / (N-1, L)

    if segment_type == 'fastvid':
        segment_sizes = fastvid_segmentation(
            frame_similarity=frame_similarity,
            dyseg_c=8,
            dyseg_tau=tau
        )
    elif segment_type == 'holitom':
        segment_sizes = holitom_segmentation(
            frame_similarity=frame_similarity,
            dyseg_tau=tau
        )
    else:
        raise NotImplementedError(f"Unsupported segmentation algorithm: {segment_type}")


    # =============================================================================
    #         Step 2: Pruning within each segment: Tango / FastVID / HoliTom
    # =============================================================================
    
    # A list of length=num_segments. Each item of shape (num_frames, num_tokens, dim)
    image_features_segments = torch.split(image_features, segment_sizes, dim=0)
    # A list of length=num_segments. Each item of shape (num_frames, num_tokens)
    salient_score_segments = torch.split(salient_score, segment_sizes, dim=0)
    # A list of length=num_segments. Each item of shape (num_frames, num_tokens, dim)
    roped_image_features_segments = torch.split(roped_image_features, segment_sizes, dim=0)

    all_image_features = []
    all_kept_indices = []
    token_offset = 0

    for segment_idx, (image_feature_segment, salient_score_segment, roped_image_feature_segment) in enumerate(zip(image_features_segments, salient_score_segments, roped_image_features_segments)):
        cur_segment_length = len(image_feature_segment)
        
        
        cur_image_feature, cur_kept_indices = tango_token_merger(
            image_features=image_feature_segment,
            roped_image_features=roped_image_feature_segment,
            salient_score=salient_score_segment,
            retention_ratio=retain_ratio,
            context_ratio=context_ratio,
            k_neighbors=k_neighbors,
            beta=beta,
        )

        all_image_features.append(cur_image_feature)
        all_kept_indices.append(cur_kept_indices)
        cur_kept_indices += token_offset
        token_offset += cur_segment_length * L

    return torch.cat(all_image_features), torch.cat(all_kept_indices).sort().values



