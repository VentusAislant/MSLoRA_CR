import json
import os.path

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

from llava.mslora_utils.lora_utils import get_all_linear_names, add_lora_into_model_by_name
from llava.model import LlavaMistralForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(
        model_path,
        model_name,
        lora_paths=None,
        load_8bit=False,
        load_4bit=False,
        device="cuda"
):
    print(f'Model Name: {model_name}')
    kwargs = {}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlavaMistralForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=False,
        use_flash_attention_2=False,
        **kwargs
    )

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if vision_tower is None:
        model.config.vision_tower = model.config.mm_vision_tower
        model.get_model().initialize_vision_modules(
            model.config,
        )
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    cl_lora_weights = []
    if lora_paths is not None:

        lora_names = get_all_linear_names(
            model,
            exclude_keywords=('vision', 'mm_projector', 'lm_head')
        )

        cfg_path = os.path.join(os.path.dirname(lora_paths[-1]), 'config.json')
        with open(cfg_path, 'r') as f:
            data = json.load(f)
        mslora_cfg = data['mslora_cfg']
        mslora_cfg['max_task'] = len(lora_paths)

        print(mslora_cfg)
        if 'adding_layers' in mslora_cfg:
            adding_layers = mslora_cfg.get('adding_layers')
            lora_names = [n for n in lora_names if any(f".{adding_layer}." in n for adding_layer in adding_layers)]

        print(f'lora names: ', lora_names)
        add_lora_into_model_by_name(model, names=lora_names, mslora_cfg=mslora_cfg)


        cl_lora_weights = []
        for lora_path in lora_paths:
            print(f'loading previous lora weight from {lora_path}')
            weight_to_load = torch.load(lora_path, 'cpu')
            # print(list(weight_to_load.keys()))
            cur_cl_lora_weight = {}
            for k, v in weight_to_load.items():
                if 'cl_lora_weight' in k:
                    cur_cl_lora_weight[k] = v
            cl_lora_weights.append(cur_cl_lora_weight)

            print('=' * 90)
            for k, v in model.state_dict().items():
                if any(k in kk for kk in weight_to_load.keys()):
                    print(f'matched: ', k)

            model.load_state_dict(weight_to_load, strict=False)

    model.only_ortho = False
    vision_tower.to(device=device, dtype=kwargs['torch_dtype'])
    model.model.mm_projector.to(device=device, dtype=kwargs['torch_dtype'])
    model.to(device=device, dtype=kwargs['torch_dtype'])
    return tokenizer, model, image_processor, context_len, cl_lora_weights
