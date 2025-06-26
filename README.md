# Contrastive Regularization with LoRA for Multimodal Biomedical Image Incremental Learning

## Release
- [July 4th, 2025] 🔥 We released **Contrastive Regularization with LoRA for Multimodal Biomedical Image Incremental Learning**, an innovative framework leveraging modality-specific LoRA and contrastive regularization for efficient multimodal biomedical image incremental learning in large vision-language models.

## Install

1. Clone this repository and navigate to `MSLoRA_CR` folder
```bash
git clone https://github.com/VentusAislant/MSLoRA_CR.git
cd MSLoRA_CR
```
2. Install Package: Create `conda` environment
```Shell
conda create -n mslora_cr python=3.10 -y
conda activate mslora_cr
pip install -e .

# optional
pip install -e .[train]
pip install -e .[eval]
```

3. Create soft link for dataset and pre-trained models

```shell
export LLAVA_MED_V1_5="path/to/llava_med_v1.5"
export MSLORA_DATA="path/to/data/MSLoRA_CR"
export CLIP_PATH="path/to/models/openai"

mkdir data
mkdir pretrained_models
mkdir checkpoints
ln -s $MSLORA_DATA ./data/
ln -s $LLAVA_MED_V1_5 ./pretrained_models/
ln -s $CLIP_PATH ./pretrained_models/
```
