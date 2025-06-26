import logging
import pathlib

from arguments import ModelArguments, TrainingArguments

from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN
from llava.mslora_utils.lora_utils import get_all_linear_names, add_lora_into_model_by_name
from llava.train.trainer import LLaVATrainer
from llava.model import *
from llava.train.data import *

local_rank = None

from transformers import set_seed

def set_all_seeds(seed):
    print(f'##########  SET SEDD: {seed}')
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_trainable_state_maybe_zero_3(named_params):
    to_return = {k: t for k, t in named_params if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def train(attn_implementation=None):
    ########################################################
    ## 1. Loading arguments
    ########################################################
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    seed = training_args.seed
    if seed == 42:
        print('###### Using Default random seed: 42')
        set_seed(42)
    else:
        set_all_seeds(seed)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    ########################################################
    ## 2. Loading Model
    ########################################################
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    config = LlavaMistralConfig.from_pretrained(model_args.model_path)
    config.max_task = model_args.max_task

    config.alpha = model_args.alpha
    config.beta = model_args.beta

    model = LlavaMistralForCausalLM.from_pretrained(
        model_args.model_path,
        config=config,
        low_cpu_mem_usage=False,
        use_flash_attention_2=False,
        **bnb_model_from_pretrained_args
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
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    model.model.mm_projector.to(dtype=compute_dtype, device=training_args.device)
    model.to(dtype=compute_dtype, device=training_args.device)

    ########################################################
    ## 3. Adding Lora
    ########################################################

    if model_args.lora_enable:
        model.requires_grad_(False)
        mslora_cfg = {
            "max_task": model_args.max_task,
            "lora_rank": model_args.lora_rank,
            "lora_alpha": model_args.lora_alpha,
            "lora_dropout": model_args.lora_dropout,
        }

        model.config.mslora_cfg = mslora_cfg

        lora_names = get_all_linear_names(
            model,
            exclude_keywords=('vision', 'mm_projector', 'lm_head')
        )


        if model_args.adding_layers is not None:
            model.config.mslora_cfg['adding_layers'] = model_args.adding_layers
            print(model_args.adding_layers)
            lora_names = [n for n in lora_names if any(f".{adding_layer}." in n for adding_layer in model_args.adding_layers)]
        print(f'lora names: ', lora_names)
        add_lora_into_model_by_name(model, names=lora_names, mslora_cfg=mslora_cfg)

        if model_args.previous_lora_path is not None:
            print(f'loading previous lora weight from {model_args.previous_lora_path}')
            for path in model_args.previous_lora_path:
                weight_to_load = torch.load(path, 'cpu')

                # not load weight
                for k, v in model.state_dict().items():
                    if 'mslora_weight' in k:
                        del weight_to_load[k]
                    if any(k in kk for kk in weight_to_load.keys()):
                        print(f'matched: ', k)
                model.load_state_dict(weight_to_load, strict=False)

        model.task_mask = torch.zeros(1024).to(device=model.device, dtype=torch.bool)
        model.task_mask[:model_args.max_task] = True
        # if model_args.max_task == 1:
        #     model.get_gate_net().requires_grad_(True)

        if model_args.is_same_modality is not None:
            print(f'Is same_modality {model_args.is_same_modality}')
            model.is_same_modality = model_args.is_same_modality
            assert len(model.is_same_modality) == model_args.max_task -1

    model.config.use_cache = False

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    training_args.use_im_start_end = model.config.mm_use_im_start_end

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_args.mm_use_im_start_end = model.config.mm_use_im_start_end
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    model.config.alpha = model_args.alpha
    model.config.beta = model_args.beta
    model.only_ortho = model.config.only_ortho = model_args.only_ortho
    if getattr(model.config, "alpha", None) is not None:
        alpha = model.config.alpha
        if alpha == -1:
            # model.alpha = torch.nn.Parameter(torch.tensor(-2.3, requires_grad=True))
            model.alpha = torch.nn.Parameter(torch.tensor(-4.6, requires_grad=True))
        else:
            model.alpha = model.config.alpha
    if getattr(model.config, "beta", None) is not None:
        beta = model.config.beta
        if beta == -1:
            # model.beta = torch.nn.Parameter(torch.tensor(-2.3, requires_grad=True))

            model.beta = torch.nn.Parameter(torch.tensor(-4.6, requires_grad=True))
        else:
            model.beta = model.config.beta

    print('=' * 90)
    print(f'alpha: {model.alpha},\n beta: {model.beta}')

    print('=' * 90)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"trainable_param: {name}, shape: {param.shape}")
    print('=' * 90)

    # exit(0)
    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True
    if model_args.alpha is not None and model_args.alpha == -1:
        print(model.alpha.data)
        model.config.alpha = model.alpha.data.detach().tolist()
        model.config.beta = model.beta.data.detach().tolist()
    trainer._save(training_args.output_dir)

if __name__ == "__main__":
    # train()
    train(attn_implementation="flash_attention_2")
