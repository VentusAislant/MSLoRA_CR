from dataclasses import dataclass, field
from typing import Optional, List
import transformers


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default=None)
    lora_enable: bool = False
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.
    max_task: int = 1
    previous_lora_path: Optional[List[str]] = field(default=None)

    is_same_modality: Optional[List[int]] = field(default=None)

    adding_layers: Optional[List[int]] = field(default=None)

    # alpha: float = field(default=None)
    # beta: float = field(default=None)

    alpha: float = field(default=-1)
    beta: float = field(default=-1)
    only_ortho: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'pad'
    is_multimodal: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )

    seed: int = field(default=42, metadata={"help": "Random seed."})