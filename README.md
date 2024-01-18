# Reward-Engineering-for-Generating-SEG
This is the official code for the paper: [Reward Engineering for Generating Semi-structured Explanation](https://arxiv.org/pdf/2309.08347.pdf) (EACL2024 accepted).

## Setup
```
pip install -r requirements.txt
```

## Stage 1: Supervised Fine-tuning of the Flan-T5 Model
```
python finetune_flant5.py
```
To merge the adapter into the base model we can use the `merge_peft_adapter.py`.
```
python merge_peft_adapter.py --adapter_model_name=flan-t5-xxl-lora-expla-graph-predict-stance --output_name=flan-t5-xxl-lora-expla-graph-predict-stance-merged
```

## Stage 2: Reward Modeling
First we fine-tune llama model on the task data: 
```
./run_llama_sft.sh
```
```
python merge_peft_adapter.py --adapter_model_name=explagraph-llama-7b-sft --output_name=explagraph-llama-7b-sft-merged
```
Then we train a reward model based on that fine-tuned checkpoint:
```
./run_reward_modeling.sh
```
```
python merge_peft_adapter.py --adapter_model_name=explagraph-reward-model-llama-7b-pretrained --output_name=explagraph-reward-model-llama-7b-pretrained-merged
```

## Stage 3: Reinforcement Learning
```
./run_rlhf.sh
```

## Inference
```
python inference_flant5.py
```

## Evaluation
For the evaluation, we use the official ExplaGraph [evaluation sctipts.](https://github.com/swarnaHub/ExplaGraphs/tree/main/eval_scripts)
