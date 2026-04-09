# pandayin: A monkey patch. Replace the original vision tower with the edited one.
from .siglip_encoder import SigLipVisionTower_Tango
from .llava_arch import (prepare_inputs_labels_for_multimodal_tango, 
                         encode_images_tango, get_2dPool_tango,
                         add_token_per_grid_tango)
def tango():
        
    print("################################")
    print("############ Tango ###########")
    print("################################")

    from llava.model.llava_arch import LlavaMetaForCausalLM
    LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_tango
    LlavaMetaForCausalLM.encode_images_tango = encode_images_tango
    LlavaMetaForCausalLM.get_2dPool_tango = get_2dPool_tango
    LlavaMetaForCausalLM.add_token_per_grid_tango = add_token_per_grid_tango

    # pandayin: A monkey patch. Replace the original vision tower with the edited one.
    from llava.model.multimodal_encoder import siglip_encoder as siglip_module
    
    siglip_module.SigLipVisionTower = SigLipVisionTower_Tango

    from llava.model.multimodal_encoder import builder as builder_module
    
    builder_module.SigLipVisionTower = SigLipVisionTower_Tango
