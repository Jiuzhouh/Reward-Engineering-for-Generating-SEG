# Reward-Engineering-for-Generating-SEG
This is the official code for the paper: [Reward Engineering for Generating Semi-structured Explanation](https://arxiv.org/pdf/2309.08347.pdf).

## Setup
```
pip install -r requirements.txt
```

## Stage 1: Supervised Fine-tuning of the Flan-T5 Model
```
python finetune_flant5.py
```
To merge the adaptors into the base model we can use the `merge_peft_adapter.py`.
```
python merge_peft_adapter.py --adapter_model_name=flan-t5-xxl-lora-expla-graph-predict-stance --output_name flan-t5-xxl-lora-expla-graph-predict-stance-merged
```
