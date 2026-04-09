import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import math

@torch.inference_mode()
def dpc_knn(x: torch.Tensor, 
            cluster_num: int,
            k: int = 7,
) -> torch.Tensor:
    """
    DPC-KNN implementation based on Euclidean distance.
    
    Input:
        x (Tensor): shape of (B, L, D) (batch_size, #tokens, hidden dimension)
        cluster_num (int): Number of target clusters. 
        k (int): Number of neighbors used for calculating density.
            FastVID set to 4, HoliTom set to 7 by default.
        
    Returns:
        index_center: (B, cluster_num) Index of cluster center
    """
    batch_size, seq_len, embed_dim = x.shape
    
    # pandayin: common implementation, L2 distance
    dist_matrix = torch.cdist(x.float(), x.float())
    
    
    actual_k = min(k + 1, seq_len)
    # pandayin: Caluclate local density (rho)
    dist_nearest, index_nearest = torch.topk(dist_matrix, actual_k, dim=-1, largest=False)
    
    # pandayin: Exclude the token itself (dist always the smallest, = 0)
    knn_dists = dist_nearest[:, :, 1:] 
    
    # Calculate with Gaussian kernel
    rho = (-(knn_dists ** 2).mean(dim=-1)).exp()
    
    # add a little noise to ensure no tokens have the same density.
    rho = rho + torch.rand_like(rho) * 1e-6

    # pandayin: Caluclate relative distance (delta)
    # mask[b, i, j] = True -> in batch b, j's density > i's density
    mask = (rho.unsqueeze(1) > rho.unsqueeze(2))
    max_dists = dist_matrix.flatten(1).max(dim=1).values[:, None, None]
    masked_dist = torch.where(mask, dist_matrix, max_dists)

    delta, _ = masked_dist.min(dim=-1) # (B, L)
    
    density_score = rho * delta         # (B, L)
    _, index_center = density_score.topk(cluster_num, dim=-1)
    return index_center

def fastvid_segmentation(
    frame_similarity: torch.Tensor, # (N-1)
    dyseg_c: int = 8,       # Number of top-k smallest similarity boundaries
    dyseg_tau: float = 0.9, # Threshold for similarity-based segmentation
):
    cut_indices_topk = torch.topk(frame_similarity, min(dyseg_c - 1, frame_similarity.shape[0]), largest=False).indices
    cut_indices_threshold = torch.nonzero(frame_similarity < dyseg_tau, as_tuple=False).squeeze(1)
    cut_indices = torch.unique(torch.cat([cut_indices_topk, cut_indices_threshold])).sort().values

    # pandayin: (s, e]. Add sentinel.
    padded = F.pad(cut_indices, (1, 1), value=-1)
    padded[-1] = frame_similarity.shape[0]
    segment_sizes = padded.diff().tolist()

    return segment_sizes

def holitom_segmentation(
    frame_similarity: torch.Tensor, # (N-1, L)
    dyseg_tau: float = 0.8, # number of threshold-based similarity segmentation
    max_window_size: int = 1024,
):
    
    num_intervals, L = frame_similarity.shape
    N = num_intervals + 1
    device = frame_similarity.device
    # pandayin: Dynamic Programming to find segments
    # with total maximum static token count.
    def get_pruned_static_count_vectorized(feature_sim, n, l, tau_val):
        sim_matrix = torch.ones((n, n, l), device=device)
        for start in range(n - 1):
            is_sim = (feature_sim[start:] > tau_val).float()
            cum_similarity = torch.cumprod(is_sim, dim=0)
            sim_matrix[start, start+1 : start+1+len(cum_similarity)] = cum_similarity
        indices = torch.arange(n, device=device)
        window_lengths = indices.unsqueeze(0) - indices.unsqueeze(1)
        window_lengths = window_lengths.clamp(min=0).float()
        return sim_matrix.sum(dim=-1) * window_lengths

    pruned_static_count = get_pruned_static_count_vectorized(frame_similarity, N, L, dyseg_tau)
    dp = torch.zeros(N, device=device)
    prev = torch.zeros(N, dtype=torch.long, device=device)

    for i in range(N):
        max_val = dp[i-1] if i > 0 else 0
        best_j = i
        for window_size in range(2, min(i + 1, max_window_size) + 1):
            j = i - window_size
            current_val = (dp[j] if j >= 0 else 0) + pruned_static_count[j+1, i]
            if current_val > max_val: 
                max_val = current_val
                best_j = j + 1
        dp[i] = max_val
        prev[i] = best_j

    windows = []
    curr = N - 1
    while curr >= 0:
        # pandayin: [s, e]
        windows.append((prev[curr].item(), curr))
        curr = prev[curr].item() - 1
    windows = windows[::-1]

    segment_sizes = [w[1]-w[0] + 1 for w in windows]
    return segment_sizes


def perform_cluster_merging(
    src: torch.Tensor, 
    roped_src: torch.Tensor,
    target: torch.Tensor, 
    roped_target: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Merge tokens in a cluster.
    Input:
        src:    [N_s, D] - Source tokens to be merged.
        target: [N_t, D] - Anchor tokens (cluster centers).
        beta:   float    - Weight for the anchor token.
    Output:
        updated_target: [N_t, D]
    """
    N_s, D = src.shape
    N_t, _ = target.shape
    device = target.device
    dtype = target.dtype

    roped_src = roped_src.float()
    roped_target = roped_target.float()

    
    dist = torch.cdist(roped_src, roped_target)
    assign_idx = dist.argmin(dim=-1)

    agg_feats = torch.zeros_like(target)
    agg_counts = torch.zeros((N_t, 1), device=device, dtype=dtype)
    
    ones = torch.ones((N_s, 1), device=device, dtype=dtype)
    
    agg_feats.index_add_(0, assign_idx, src)
    agg_counts.index_add_(0, assign_idx, ones)
    
    cluster_means = agg_feats / agg_counts.clamp(min=1)
        
    w = (1.0 / (agg_counts.clamp(min=1))).clamp(min=beta)
    
    return w * target + (1 - w) * cluster_means

def tango_compressor(
    image_features: torch.Tensor,         # [L, D] Original
    roped_image_features: torch.Tensor,   # [L, D] Roped
    salient_score: torch.Tensor,          # [L]
    retention_ratio: float,
    context_ratio: float,
    k_neighbors: int,
    beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
        Modified Salient selection with Robustness Fixes + Roped Clustering
    '''
    L, D = image_features.shape
    device = image_features.device
    
    raw_retain_num = math.ceil(L * retention_ratio)
    frame_retain_num = max(1, raw_retain_num) if L > 0 else 0
    
    frame_salient_num = int(frame_retain_num * (1.0 - context_ratio))
    frame_context_num = frame_retain_num - frame_salient_num
    
    full_indices = torch.arange(L, device=device)

    all_indices = []
    context_mask = torch.ones(L, dtype=torch.bool, device=device)

    ############ Salient token selection ############
    if frame_salient_num > 0:
        from ..prune_config import PRUNE_CONFIG
        alpha = PRUNE_CONFIG['alpha']
        num_candidates = min(int(alpha * frame_salient_num), L)
        actual_k = min(frame_salient_num, num_candidates)
        
        candidate_topk = torch.topk(salient_score, k=num_candidates)
        candidate_indices = candidate_topk.indices
        
        candidate_features_rope = roped_image_features[candidate_indices] 
        candidate_scores = salient_score[candidate_indices]

        if num_candidates <= actual_k or num_candidates <= 1:
            salient_idx = candidate_indices
        else:
            local_anchor_idx = dpc_knn(
                candidate_features_rope.unsqueeze(0),
                actual_k,
                k=k_neighbors,
            ).squeeze(0) 

            anchor_features_rope = candidate_features_rope[local_anchor_idx] 
            
            cand_feat_float = candidate_features_rope.float()
            anchor_feat_float = anchor_features_rope.float()

            dist = torch.cdist(cand_feat_float, anchor_feat_float)
            assignments = dist.argmin(dim=-1)

            # Sample Max-Score within cluster
            sampled_local_indices = []
            for i in range(actual_k):
                in_cluster_mask = (assignments == i)
                if in_cluster_mask.any():
                    cluster_member_idx = torch.where(in_cluster_mask)[0]
                    member_scores = candidate_scores[cluster_member_idx]
                    best_member_rel_idx = cluster_member_idx[member_scores.argmax()]
                    sampled_local_indices.append(best_member_rel_idx)
                else:
                    sampled_local_indices.append(local_anchor_idx[i])
            
            sampled_local_indices = torch.stack(sampled_local_indices)
            salient_idx = candidate_indices[sampled_local_indices] 
        
        all_indices.append(salient_idx)
        context_mask[salient_idx] = False

    ############ Contextual token merging ############
    if frame_context_num > 0:
        context_tokens = image_features[context_mask]
        context_tokens_rope = roped_image_features[context_mask]
        context_indices = full_indices[context_mask]

        if len(context_indices) > 0:
            actual_context_num = min(frame_context_num, len(context_indices))
            
            if len(context_indices) <= actual_context_num:
                all_indices.append(context_indices)
            else:
                anchor_token_local_idx = dpc_knn(
                    context_tokens_rope.unsqueeze(0),
                    actual_context_num,
                    k=k_neighbors,
                ).squeeze(0)

                anchor_token_idx = context_indices[anchor_token_local_idx]
                
                is_anchor_local_mask = torch.zeros(len(context_indices), dtype=torch.bool, device=device)
                is_anchor_local_mask[anchor_token_local_idx] = True
                
                anchor_tokens = context_tokens[is_anchor_local_mask]
                src_tokens = context_tokens
                
                anchor_tokens_rope = context_tokens_rope[is_anchor_local_mask]
                src_tokens_rope = context_tokens_rope
                
                merged_context_tokens = perform_cluster_merging(
                    src=src_tokens, 
                    roped_src=src_tokens_rope,
                    target=anchor_tokens, 
                    roped_target=anchor_tokens_rope,
                    beta=beta,
                )

                image_features[anchor_token_idx] = merged_context_tokens
                all_indices.append(anchor_token_idx)
    
    if len(all_indices) == 0:
        final_indices = torch.tensor([], dtype=torch.long, device=device)
    else:
        final_indices = torch.cat(all_indices).sort().values
        
    return image_features, final_indices


def tango_token_merger(
    image_features: torch.Tensor,      # [N, L, D]
    roped_image_features: torch.Tensor,# [N, L, D]
    salient_score: torch.Tensor,        # [N, L]
    tau: float = 0.8,
    retention_ratio: float = 0.10,
    context_ratio: float = 0.0,
    beta: float = 0.6,
    k_neighbors: int = 6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N, L, D = image_features.shape
    device = image_features.device
    dtype = image_features.dtype

    if N == 1:
        feat, indices = tango_compressor(
            image_features.squeeze(0),
            roped_image_features.squeeze(0),
            salient_score.squeeze(0),
            retention_ratio,
            context_ratio,
            k_neighbors,
            beta
        ) 
        return feat.unsqueeze(0), indices
    
    final_output_feats = image_features.reshape(-1, D)
    all_kept_indices = []

    metric_normed = F.normalize(image_features.float(), p=2, dim=-1)
    frame_similarity = (metric_normed[:-1] * metric_normed[1:]).sum(dim=-1)  # (N-1, L)
    
    global_indices_map = torch.arange(N*L, device=device).view(N, L)
    static_mask = torch.all(frame_similarity > tau, dim=0) # (L,)
    dynamic_mask = ~static_mask 

    total_tokens = N * L
    static_gain = (N-1)*(static_mask.sum().item())
    adjusted_retain_ratio = min(retention_ratio / ((total_tokens - static_gain) / total_tokens), 1.0)

    # ================= Process Static Part =================
    
    static_feat = image_features[:, static_mask].mean(dim=0)
    static_roped = roped_image_features[0, static_mask]
    static_salient_score = salient_score[:, static_mask].mean(dim=0)
    static_global_indices = global_indices_map[0, static_mask]

    updated_static_feat, static_kept_local_idx = tango_compressor(
        static_feat,
        static_roped,
        static_salient_score,
        adjusted_retain_ratio,
        context_ratio=context_ratio,
        k_neighbors=k_neighbors,
        beta=beta
    ) 

    final_output_feats = final_output_feats.reshape(N, -1, D)
    final_output_feats[0, static_mask] = updated_static_feat
    final_output_feats = final_output_feats.reshape(-1, D)
    all_kept_indices.append(static_global_indices[static_kept_local_idx])

    # ================= Process Dynamic Part =================
    
    num_dynamic_tokens = dynamic_mask.sum().item()
    
    if num_dynamic_tokens > 0:
        dynamic_feats = image_features[:, dynamic_mask]       
        dynamic_roped = roped_image_features[:, dynamic_mask] 
        dynamic_salient_score = salient_score[:, dynamic_mask]        
        dynamic_global_indices = global_indices_map[:, dynamic_mask] 

        raw_retain_num = math.ceil(num_dynamic_tokens * adjusted_retain_ratio)
        frame_salient_num = int(raw_retain_num * (1.0 - context_ratio))
        frame_context_num = raw_retain_num - frame_salient_num
        
        actual_k = min(frame_salient_num, num_dynamic_tokens)
        
        # pandayin: used to seperate selected salient tokens and 
        # remaining context tokens for the subsequent global merging step.
        full_context_mask = torch.ones((N, num_dynamic_tokens), dtype=torch.bool, device=device)

        ############ Frame-wise Salient Token Selection ############
        
        if actual_k > 0:
            from ..prune_config import PRUNE_CONFIG
            alpha = PRUNE_CONFIG['alpha']
            num_candidates = min(int(alpha * actual_k), num_dynamic_tokens)
            
            candidate_local_indices = torch.topk(dynamic_salient_score, k=num_candidates, dim=1).indices
            
            candidate_features_rope = torch.gather(
                dynamic_roped, 1,
                candidate_local_indices.unsqueeze(-1).expand(-1, -1, D)
            )
            candidate_scores = torch.gather(dynamic_salient_score, 1, candidate_local_indices)

            target_k = min(actual_k, num_candidates)
            
            if num_candidates <= target_k or num_candidates <= 1:
                selected_local_indices = candidate_local_indices
            else:
                center_rel_idx = dpc_knn(
                    candidate_features_rope,
                    cluster_num=target_k,
                    k=k_neighbors,
                ) 

                anchor_features_rope = torch.gather(
                    candidate_features_rope, 1,
                    center_rel_idx.unsqueeze(-1).expand(-1, -1, D)
                )
         
                dist = torch.cdist(candidate_features_rope, anchor_features_rope)
                assignments = dist.argmin(dim=-1)
                
                # Expand assignments to one-hot mask: [N, num_cand, target_k]
                # values are 0..target_k-1
                cluster_ids = torch.arange(target_k, device=device).view(1, 1, target_k)
                mask = (assignments.unsqueeze(-1) == cluster_ids) 
                
                min_val = torch.finfo(dtype).min
                masked_scores = candidate_scores.unsqueeze(-1).masked_fill(~mask, min_val)

                best_rel_indices = masked_scores.argmax(dim=1)
                
                cluster_exists = mask.any(dim=1) # [N, target_k]
                final_rel_indices = torch.where(cluster_exists, best_rel_indices, center_rel_idx)
                
                # [N, target_k]
                selected_local_indices = torch.gather(candidate_local_indices, 1, final_rel_indices)

            salient_global_idx = torch.gather(dynamic_global_indices, 1, selected_local_indices)
            all_kept_indices.append(salient_global_idx.flatten())

            full_context_mask.scatter_(1, selected_local_indices, False)

        ############ Segment-wise Spatio-Temporal Merging ############
        if frame_context_num > 0:
            valid_mask_flat = full_context_mask.reshape(-1) # [N * L_dyn]
            
            total_context_budget = frame_context_num * N
            
            if valid_mask_flat.sum() <= total_context_budget:
                flat_global_indices = dynamic_global_indices.flatten()
                context_global_indices = flat_global_indices[valid_mask_flat]
                all_kept_indices.append(context_global_indices)
            else:
                flat_dynamic_feats = dynamic_feats.reshape(-1, D)
                flat_dynamic_roped = dynamic_roped.reshape(-1, D)
                
                anchor_candidates = flat_dynamic_feats[valid_mask_flat]
                anchor_candidates_rope = flat_dynamic_roped[valid_mask_flat]
                
                anchor_token_local_indices = dpc_knn(
                    anchor_candidates_rope.unsqueeze(0),
                    cluster_num=total_context_budget,
                    k=k_neighbors,
                ).squeeze(0)
                
                flat_global_indices = dynamic_global_indices.flatten()
                candidates_global_indices = flat_global_indices[valid_mask_flat]
                anchor_token_global_indices = candidates_global_indices[anchor_token_local_indices]
                
                is_anchor = torch.zeros(len(anchor_candidates), dtype=torch.bool, device=device)
                is_anchor[anchor_token_local_indices] = True
                
                anchor_tokens = anchor_candidates[is_anchor]
                src_tokens = anchor_candidates#[~is_anchor]
                
                anchor_tokens_rope = anchor_candidates_rope[is_anchor]
                src_tokens_rope = anchor_candidates_rope#[~is_anchor]
                
                # Perform Merging
                merged_context_tokens = perform_cluster_merging(
                    src=src_tokens,
                    roped_src=src_tokens_rope,
                    target=anchor_tokens,
                    roped_target=anchor_tokens_rope,
                    beta=beta,
                )
                
                final_output_feats[anchor_token_global_indices] = merged_context_tokens
                all_kept_indices.append(anchor_token_global_indices)

    if len(all_kept_indices) == 0:
        kept_indices = torch.tensor([], dtype=torch.long, device=device)
    else:
        kept_indices = torch.cat(all_kept_indices)
    
    return final_output_feats.view(N, L, D), torch.sort(kept_indices).values