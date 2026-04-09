#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# ------------------------------------------------------------------------
# Modified from LLaVA-NeXT (https://github.com/LLaVA-VL/LLaVA-NeXT)
# A modified version to incorporate the Tango pruning method, 
# which can be applied to both LLaVA-OV and LLaVA-Video series.
# Copyright 2026 Shukang Yin
# ------------------------------------------------------------------------


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

import random
from llava.mm_utils import get_anyres_image_grid_shape


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


def encode_images_tango(self, images):
    image_features, attn_weights, global_image_features = self.get_model().get_vision_tower()(images)
    image_features = self.get_model().mm_projector(image_features)
    return image_features, attn_weights, global_image_features

def get_2dPool_tango(self, image_feature, attn_weights, stride=2):
    # image_feature: [B, L, D=3584]
    # metric: [3, B, L, D'=1152]

    height = width = self.get_vision_tower().num_patches_per_side   # siglip:384 res -> 27x27 grid
    num_frames, num_tokens, num_dim = image_feature.shape   # ov: [32, 729, 3584]
    
    image_feature = image_feature.view(num_frames, height, width, -1)
    image_feature = image_feature.permute(0, 3, 1, 2).contiguous()  # [B, D, grid_h, grid_w]
    

    attn_weights = attn_weights.view(num_frames, 1, height, width)

    # pandayin: for attention weights, we use mean pooling. 
    # For metric, align with the pool_mode of image_feature.   
    if self.config.mm_spatial_pool_mode == "average":
        # pandayin: llava-video goes this branch, note that config.json is 'bilinear', it's overwritten
        image_feature = nn.functional.avg_pool2d(image_feature, stride) # 13 x 13 = 169 tokens
        attn_weights = nn.functional.avg_pool2d(attn_weights, stride)  # [B, h=13, w=13]
    # pandayin: untested branch.
    elif self.config.mm_spatial_pool_mode == "max":
        image_feature = nn.functional.max_pool2d(image_feature, stride)
    elif self.config.mm_spatial_pool_mode == "bilinear":
        # pandayin: llava-ov goes this branch.
        height, width = image_feature.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]  # 14 x 14 = 196 tokens
        image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        attn_weights = nn.functional.interpolate(attn_weights, size=scaled_shape, mode='bilinear')
    else:
        raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
    
    image_feature = image_feature.permute(0, 2, 3, 1)
    image_feature = image_feature.view(num_frames, -1, num_dim)
    attn_weights = attn_weights.view(num_frames, -1)

    return image_feature, attn_weights

def add_token_per_grid_tango(self, image_feature, kept_indices):
    N, L, D = image_feature.shape
    H = W = int(math.sqrt(L))
    device = image_feature.device
    
    mask = torch.zeros(N * L, dtype=torch.bool, device=device)
    mask[kept_indices] = True
    mask = mask.view(N, H, W)
    
    row_active = mask.any(dim=2, keepdim=True)  # (N, H, 1)
    
    full_mask = torch.cat([mask, row_active], dim=2)  # (N, H, W + 1)
    
    feat_4d = image_feature.view(N, H, W, D)
    newline_col = self.model.image_newline.view(1, 1, 1, D).expand(N, H, 1, D)
    feat_with_nl = torch.cat([feat_4d, newline_col], dim=2) # (N, H, W + 1, D)
    
    return feat_with_nl[full_mask]

def prepare_inputs_labels_for_multimodal_tango(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
    vision_tower = self.get_vision_tower()
    
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if isinstance(modalities, str):
        modalities = [modalities]

    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
        video_idx_in_batch = []
        for _ in range(len(modalities)):
            if modalities[_] == "video":
                video_idx_in_batch.append(_)

        images_list = []
        for image in images:
            if image.ndim == 4:
                images_list.append(image)
            else:
                images_list.append(image.unsqueeze(0))

        concat_images = torch.cat([image for image in images_list], dim=0)
        split_sizes = [image.shape[0] for image in images_list]
        encoded_image_features, attn_weights, global_image_features = self.encode_images_tango(concat_images)

        # pandayin: A list, each element is [num_images, num_patch_H * num_patch_W, dim]
        encoded_image_features = torch.split(encoded_image_features, split_sizes)
        # pandayin: This naming is bad. But for convenience.
        encoded_attn_weights = torch.split(attn_weights, split_sizes)
        encoded_global_image_features = torch.split(global_image_features, split_sizes)

        image_features = []
        attn_weights_list = []
        global_image_features_list = []
        
        for idx, (image_feat, _attn_weights, _global_image_features) in enumerate(zip(encoded_image_features, encoded_attn_weights, encoded_global_image_features)):
            # pandayin: Video frames go this branch.
            if idx in video_idx_in_batch:
                # image_features.append(self.get_2dPool(image_feat))
                pooled_image_feat, pooled_attn_weights = self.get_2dPool_tango(image_feat, _attn_weights)
                image_features.append(pooled_image_feat)
                attn_weights_list.append(pooled_attn_weights)
                global_image_features_list.append(_global_image_features)
            # pandayin: untested branch. We only consider video.
            else:
                image_features.append(image_feat)
                
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")   # ov&video: spatial_unpad
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")   # ov&video: anyres_max_9
        mm_newline_position = getattr(self.config, "mm_newline_position", "one_token") # ov: one_token, video: grid

        if mm_patch_merge_type == "flat":
            image_features = [x.flatten(0, 1) for x in image_features]

        elif mm_patch_merge_type.startswith("spatial"):
            new_image_features = []
            for image_idx, (image_feature, _pooled_attn_weight, _global_image_features) in enumerate(zip(image_features, attn_weights_list, global_image_features_list)):
                #               (N, L, D),    (N, L)
                if image_idx in video_idx_in_batch:  # video operations
                    from .prune_config import PRUNE_CONFIG
                    from .pruner import tango_pruner
                    retention_ratio = PRUNE_CONFIG["pre_prune_r"]
                    similarity_threshold = PRUNE_CONFIG["similarity_threshold"]
                    context_ratio = PRUNE_CONFIG["context_ratio"]
                    segment_type = PRUNE_CONFIG["segment_type"]
                    num_neighbors = PRUNE_CONFIG["num_neighbors"]
                    if segment_type == "holitom":
                        sim_metric = image_feature
                    elif segment_type == "fastvid":
                        sim_metric = _global_image_features
                    else:
                        raise NotImplementedError(f"Unsupported type: {segment_type}. Should be one in ['holitom', 'fastvid']")
                    
                    salient_score = _pooled_attn_weight
                    
                    roped_image_feature = image_feature.clone()
                    roped_image_feature = F.normalize(roped_image_feature.float(), dim=-1)
                    
                    time_interval_tensor = torch.tensor(PRUNE_CONFIG["time_intervals"], device='cpu')
                    roped_image_feature = self.get_model().vision_rope(roped_image_feature, t_ids=time_interval_tensor)

                    merged_features, kept_vision_indices = tango_pruner(
                                                        image_features=image_feature,
                                                        roped_image_features=roped_image_feature,
                                                        sim_metric=sim_metric,
                                                        salient_score=salient_score,
                                                        segment_type=segment_type,
                                                        retain_ratio=retention_ratio,
                                                        context_ratio=context_ratio,
                                                        tau=similarity_threshold,
                                                        beta=0.6,
                                                        k_neighbors=num_neighbors,
                                                        )
                    image_feature = merged_features
                    
                    
                    # pandayin: image_feature = video frames: (B=64, 169, D)
                    # pandayin: llava-video goes this branch. Add a special <newline> token each row.
                    if mm_newline_position == "grid":
                        
                        image_feature = self.add_token_per_grid_tango(image_feature, kept_vision_indices)  # (64*182, D)
                        
                        # pandayin: Untested branch. This fast video branch is by default turned off. So skip this.
                        if getattr(self.config, "add_faster_video", False):
                            faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                            # Add a token for each frame
                            concat_slow_fater_token = []
                            # import pdb; pdb.set_trace()
                            for _ in range(image_feature.shape[0]):
                                if _ % self.config.faster_token_stride == 0:
                                    concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                else:
                                    concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                            # import pdb; pdb.set_trace()
                            image_feature = torch.cat(concat_slow_fater_token)

                            # print("!!!!!!!!!!!!")
                    
                        new_image_features.append(image_feature)
                    elif mm_newline_position == "frame":
                        # Frame-wise
                        image_feature = self.add_token_per_frame(image_feature)

                        new_image_features.append(image_feature.flatten(0, 1))
                    # pandayin: llava-ov goes this branch
                    elif mm_newline_position == "one_token":
                        # one-token
                        
                        image_feature = image_feature.flatten(0, 1) # (num_frames*num_tokens, D)

                        # prune features
                        image_feature = image_feature[kept_vision_indices]
                        # pandayin: Add one token at the end of this token sequence.
                        # i.e., (num_frames*num_tokens + 1, D)
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                        new_image_features.append(image_feature)     
                    elif mm_newline_position == "no_token":
                        new_image_features.append(image_feature.flatten(0, 1))
                    else:
                        raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                    # rank0_print("Single-images")
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]

                    if "anyres_max" in image_aspect_ratio:
                        matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                        if matched_anyres_max_num_patches:
                            max_num_patches = int(matched_anyres_max_num_patches.group(1))

                    if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                        if hasattr(self.get_vision_tower(), "image_size"):
                            vision_tower_image_size = self.get_vision_tower().image_size
                        else:
                            raise ValueError("vision_tower_image_size is not found in the vision tower.")
                        try:
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                        except Exception as e:
                            rank0_print(f"Error: {e}")
                            num_patch_width, num_patch_height = 2, 2
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        image_feature = image_feature.view(2, 2, height, width, -1)

                    if "maxpool2x2" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = nn.functional.max_pool2d(image_feature, 2)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                        unit = image_feature.shape[2]
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        c, h, w = image_feature.shape
                        times = math.sqrt(h * w / (max_num_patches * unit**2))
                        if times > 1.1:
                            image_feature = image_feature[None]
                            image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    if "nobase" in mm_patch_merge_type:
                        pass
                    else:
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    new_image_features.append(image_feature)
                else:  # single image operations
                    image_feature = image_feature[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                    new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        image_features = self.encode_images(images)

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
        raise NotImplementedError
    # rank_print(f"Total images : {len(image_features)}")

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- FIXME
    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    # rank_print("Inserting Images embedding")
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        # rank0_print(num_images)
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue
        
        image_token_start_index = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0][0].item()
        image_feat_len = image_features[0].shape[0]

        from .prune_config import PRUNE_CONFIG
        PRUNE_CONFIG['image_token_start_index'] = image_token_start_index
        PRUNE_CONFIG['image_token_length'] = image_feat_len

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                try:
                    cur_image_features = image_features[cur_image_idx]
                except IndexError:
                    cur_image_features = image_features[cur_image_idx - 1]
                
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        # import pdb; pdb.set_trace()
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
    # rank_print("Finishing Inserting")

    new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
    new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
    # TODO: Hard code for control loss spike
    # if tokenizer_model_max_length is not None:
    #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
    #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
    # rank0_print("Prepare pos id")

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, "tokenizer_padding_side", "right") == "left":
            new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
    # rank0_print("tokenizer padding")

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None
    if getattr(self.config, "use_pos_skipping", False) and self.training:
        position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
        split_position = random.randint(0, new_input_embeds.size(1))
        left_add = random.randint(0, self.config.pos_skipping_range)
        right_add = random.randint(left_add, self.config.pos_skipping_range)
        position_ids[:, :split_position] += left_add
        position_ids[:, split_position:] += right_add
    # import pdb; pdb.set_trace()
    # rank0_print("Finish preparing")
    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
