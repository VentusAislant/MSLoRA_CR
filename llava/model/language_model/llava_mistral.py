import functools
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
    MistralConfig, MistralModel, MistralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ...mslora_utils import mslora


def update_forward(module, task_mask):
    module.forward = functools.partial(module.forward, task_mask=task_mask)


def calculate_euclidean_distance(a, b):
    euclidean_distance = torch.norm(torch.abs(a - b))
    return euclidean_distance


def get_sim_score(cur_lora, other_lora):
    total_sim = 0
    for w1, w2 in zip(cur_lora, other_lora):
        dis = calculate_euclidean_distance(w1, w2.detach().clone())
        sim = 1 / torch.exp(dis)
        total_sim += sim
    total_sim /= len(cur_lora)
    return total_sim


def get_param(param):
    from deepspeed import zero
    if hasattr(param, "ds_id"):
        with zero.GatheredParameters([param]):
            param = param.data
    else:
        param = param
    return param


def compute_R(P_list, Q_list):
    R = 0.0
    I = torch.eye(P_list[0].size(0), device=P_list[0].device, dtype=P_list[0].dtype)
    for P, Q in zip(P_list, Q_list):
        PP_T = torch.matmul(P, P.T)
        QQ_T = torch.matmul(Q.T, Q)
        R += torch.norm(PP_T - I, p='fro') ** 2 + torch.norm(QQ_T - I, p='fro') ** 2
    R /= len(Q_list)
    return R


class LlavaMistralConfig(MistralConfig):
    model_type = "llava_mistral"


class LlavaMistralModel(LlavaMetaModel, MistralModel):
    config_class = LlavaMistralConfig

    def __init__(self, config: MistralConfig):
        super(LlavaMistralModel, self).__init__(config)


class LlavaMistralForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMistralConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        self.model = LlavaMistralModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[List[List[int]]] = None,
            return_dict: Optional[bool] = None,
            task_mask=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if task_mask is None:
            # for training
            task_mask = getattr(self, 'task_mask', None)

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                cls_tokens
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        if task_mask is not None:
            for name, module in self.named_modules():
                if isinstance(module, mslora.Linear):
                    update_forward(module, task_mask=task_mask)

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        if self.training:
            reg_loss = None
            ortho_loss = None
            if task_mask is not None:
                is_same_modality = getattr(self, 'is_same_modality', None)
                if is_same_modality is not None:
                    all_lora_weights = {}
                    num_task = torch.sum(task_mask).item()
                    for i in range(num_task):
                        cur_lora_weight = []
                        for n, param in self.named_parameters():
                            if f'task_{i}_lora' in n:
                                cur_lora_weight.append(get_param(param))
                        all_lora_weights[f"task_{i}_lora"] = cur_lora_weight

                    cur_lora = all_lora_weights.pop(f"task_{num_task - 1}_lora")
                    for i, (k, v) in enumerate(all_lora_weights.items()):
                        sim_score = get_sim_score(cur_lora, v)  # for different modality
                        if is_same_modality[i] == 1:
                            sim_score = 1 - sim_score  # for same modality
                        if reg_loss is None:
                            reg_loss = sim_score
                        else:
                            reg_loss += sim_score

                lora_As = []
                lora_Bs = []
                for n, param in self.named_parameters():
                    if 'lora_A' in n:
                        lora_As.append(get_param(param))
                    if 'lora_B' in n:
                        lora_Bs.append(get_param(param))
                ortho_loss = compute_R(lora_As, lora_Bs)

                if reg_loss is not None:
                    if ortho_loss is None:
                        if isinstance(self.alpha, torch.nn.Parameter):
                            with torch.no_grad():
                                print(
                                    f'reg_loss: {reg_loss}, ce_loss: {output.loss}, {torch.exp(self.alpha)}, {torch.exp(self.beta)}')
                            output.loss += torch.exp(self.alpha) * reg_loss
                        else:
                            print(f'reg_loss: {reg_loss}, ce_loss: {output.loss}, {self.alpha}, {self.beta}')
                            output.loss += self.alpha * reg_loss
                    else:
                        if isinstance(self.alpha, torch.nn.Parameter):
                            with torch.no_grad():
                                print(
                                    f'reg_loss: {reg_loss}, ce_loss: {output.loss}, ortho_loss: {ortho_loss},  {torch.exp(self.alpha)}, {torch.exp(self.beta)}')
                            output.loss += torch.exp(self.alpha) * reg_loss + torch.exp(self.beta) * ortho_loss

                        else:
                            print(
                                f'reg_loss: {reg_loss}, ce_loss: {output.loss}, ortho_loss: {ortho_loss},  {self.alpha}, {self.beta}')
                            output.loss += self.alpha * reg_loss + self.beta * ortho_loss
        return output

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            image_sizes: Optional[torch.Tensor] = None,
            task_mask=None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                cls_tokens
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
            self.cls_tokens = cls_tokens
            # print(self.cls_tokens)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            task_mask=task_mask,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, task_mask=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes

        if task_mask is not None:
            inputs['task_mask'] = task_mask

        return inputs


AutoConfig.register("llava_mistral", LlavaMistralConfig)
AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)
