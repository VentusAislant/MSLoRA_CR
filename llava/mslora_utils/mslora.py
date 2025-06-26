# no weight, just mask and multi-lora

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MSLoRALayer:
    def __init__(
            self,
            lora_rank: int,
            lora_alpha: int,
            lora_dropout: float,
    ):
        self.r = lora_rank
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x


class LoraModule(nn.Module):
    def __init__(self, in_features, out_features, lora_rank, lora_alpha, lora_dropout):
        super(LoraModule, self).__init__()
        self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, lora_rank))
        self.lora_alpha = lora_alpha
        self.r = lora_rank
        self.scaling = self.lora_alpha / self.r
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

    def forward(self, x):
        # shape: [bsz, xx, dim]
        result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        # print(self.lora_dropout(x).shape)
        # print((self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)).shape)
        # print(result.shape)
        return result


class Linear(nn.Linear, MSLoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            max_task: int,
            in_features: int,
            out_features: int,
            lora_rank: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            **kwargs
    ):
        """
            max_task means the cur task id, for example 4 for the fourth task
        """
        if 'adding_layers' in kwargs:
            kwargs.pop('adding_layers')
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MSLoRALayer.__init__(self, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.r = lora_rank
        self.fan_in_fan_out = fan_in_fan_out
        self.max_task = max_task
        # Actual trainable parameters
        if lora_rank > 0:
            self.cl_lora_pool = nn.ModuleDict({
                f"task_{i}_lora": LoraModule(in_features, out_features, lora_rank, lora_alpha, lora_dropout)
                for i in range(max_task)
            })
            # Freezing the pre-trained weight matrix
            # ONLY TRAIN THE CURRENT TASK WEIGHT
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
            for id, m in enumerate(self.cl_lora_pool.values()):
                if id == max_task - 1:
                    m.requires_grad_(True)
                else:
                    m.requires_grad_(False)

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'cl_lora_pool'):
            for module in self.cl_lora_pool.values():
                # print('initialize ', module)
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                nn.init.zeros_(module.lora_B)

    def forward(
            self,
            x: torch.Tensor,
            task_mask: torch.BoolTensor = None
    ):
        """
            x: torch.Size[bsz, ..., dim]
            task_weight: torch.Size[task]
            task_mask: torch.Size[task]
                for example:  there are 5 lora module in the linear
                    case1:
                        task_mask = [True, True, False, False, False]
                        task_weight = [0.2, 0.8]
                        return x+=(lora1(x)*0.2 + lora2(x)*0.8)
                     case2:
                        task_mask = [True, True, True, True, True]
                        task_weight = [0.2, 0.1, 0.05, 0.05, 0.6]
                        return x+=(lora1(x)*0.2 + lora2(x)*0.1, ...)
        """
        bsz = x.shape[0]

        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0:
            result = F.linear(x, T(self.weight), bias=self.bias)

            deltas = []
            num_task = 0
            for m, mask in zip(self.cl_lora_pool.values(), task_mask[:self.max_task]):
                if mask:
                    deltas.append(m(x))
                    num_task += 1

            deltas = torch.stack(deltas, dim=0).to(device=x.device, dtype=x.dtype)

            # global isFirst
            # if isFirst:
            #     print('=' * 90)
            #     print(task_mask[:self.max_task])
            #     print(deltas.shape, task_mask[:num_task].shape)
            assert deltas.shape[0] == task_mask[:num_task].shape[0]
            # exit(0)
            final_delta = deltas.sum(dim=0)
            # if isFirst:
            #     print(x.shape, result.shape, final_delta.shape)
            #     isFirst = False
            result += final_delta
            # exit(0)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


if __name__ == '__main__':
    model = Linear(
        max_task=6,
        in_features=100,
        out_features=200,
        lora_rank=64,
        lora_alpha=128
    )
    print(model)
    print('=' * 90)
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n)
    print('=' * 90)
    bsz = 1
    dim = 100
    max_task = 1000
    task_mask = torch.zeros(max_task)
    task_mask[:5] = 1
    task_mask = task_mask.to(dtype=torch.bool)
    task_weight = torch.tensor([0.2, 0.1, 0.05, 0.05, 0.6])
    x = torch.zeros(bsz, 100)

    o = model(x, task_weight, task_mask)
    print(o)

    task_mask = torch.zeros(max_task)
    task_mask[:2] = 1
    task_mask = task_mask.to(dtype=torch.bool)
    task_weight = torch.tensor([0.2, 0.8])
    x = torch.zeros(bsz, 100)

    o = model(x, task_weight, task_mask)
    print(o)
