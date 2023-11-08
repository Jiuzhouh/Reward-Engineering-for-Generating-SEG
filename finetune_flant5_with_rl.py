import ast
import torch
import evaluate
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from datasets import load_dataset, concatenate_datasets
from trl.core import LengthSampler
from trl import AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
from transformers import Adafactor, AutoTokenizer, AutoModelForSeq2SeqLM, HfArgumentParser, pipeline
from graph_matching import split_to_edges, split_to_edges_explagraph, get_tokens, get_bleu_rouge, get_bert_score, get_ged

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the dataset name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=256, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=2, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(default="runs/", metadata={"help": "n steps to save the model"})
    seed: Optional[int] = field(default=1, metadata={"help": "the seed"})

parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
print("script_args: ", script_args)


dataset_name = script_args.dataset_name
print("dataset_name: ", dataset_name)

ppo_config = PPOConfig(
	model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    #project_kwargs={"logging_dir": "./runs"},
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    remove_unused_columns=False,
    adap_kl_ctrl=True,
    init_kl_coef=0.3,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
)

model_name = script_args.model_name

# Define LoRA Config
lora_config = LoraConfig(
 r=4,
 lora_alpha=16,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)

# load_from_full_model_checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name, device_map="auto", load_in_8bit=True, torch_dtype=torch.float16, peft_config=lora_config)

print("finetune model: ", type(model))
print("finetune model is_loaded_in_8bit: ", model.is_loaded_in_8bit)

# set seed before initializing value head for deterministic eval
set_seed(ppo_config.seed)

data = load_dataset("json", data_files=dataset_name)

#test_size = 1000 
#data = data["train"].train_test_split(test_size=test_size, shuffle=False)

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = data["train"].map(lambda x: tokenizer(x["input"], truncation=True), batched=True, remove_columns=["input", "output"])
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# take 100 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 100))
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = data["train"].map(lambda x: tokenizer(x["output"], truncation=True), batched=True, remove_columns=["input", "output"])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# take 100 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 100))
print(f"Max target length: {max_target_length}")

def preprocess_function(sample, padding="max_length"):
    
    # add prefix to the input for t5
    inputs = []
    for i in range(len(sample["input"])):
        input = sample["instruction"][i] + ' ' + sample["input"][i]
        inputs.append(input)

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["output"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["query"] = inputs
    model_inputs["reference"] = sample["output"]
    return model_inputs

tokenized_dataset = data.map(preprocess_function, batched=True, remove_columns=["input", "output", "instruction"])
tokenized_dataset.set_format(type="torch")
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# load reward model
reward_model_name = script_args.reward_model_name
reward_model_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    model_kwargs={"load_in_8bit": True, "torch_dtype": "torch.float16"},
    device_map="auto",
    tokenizer=reward_model_tokenizer,
    return_token_type_ids=False,
)
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1, "truncation": True}

import re
def extract_valid_graph(input_string):
    graph = []
    pattern = r'\[(.*?)\]'
    pattern_s = r"'s\b"
    pattern_t = r"'t\b"
    replacement_s = "\\'s"
    replacement_t = "\\'t"
    matches = re.findall(pattern, input_string)
    for match in matches:
        raw_triple = '[' + match + ']'
        raw_triple = re.sub(pattern_s, replacement_s, raw_triple)
        raw_triple = re.sub(pattern_t, replacement_t, raw_triple)
        try:
            triple = ast.literal_eval(raw_triple)
        except:
            continue
        if triple not in graph:
            graph.append(triple)
    return str(graph)

def compute_reward_copasse(prediction, reference):
    predicted_graphs = []
    gold_graphs = []
    pred_graph = extract_valid_graph(prediction[3:])
    #print(pred_graph)
    gold_graph = reference[2:]
    #print(gold_graph)
    predicted_graphs.append(pred_graph.lower())
    gold_graphs.append(gold_graph.lower())

    gold_edges = split_to_edges(gold_graphs)
    pred_edges = split_to_edges(predicted_graphs)

    #gold_tokens, pred_tokens, _ = get_tokens(gold_edges, pred_edges, None)

    #precisions_rouge, recalls_rouge, f1s_rouge, precisions_bleu, recalls_bleu, f1s_bleu = get_bleu_rouge(gold_tokens, pred_tokens, gold_edges, pred_edges)
    precisions_BS, recalls_BS, f1s_BS = get_bert_score(gold_edges, pred_edges)
    return f1s_BS[0]

def compute_reward_explagraph(prediction, reference):
    predicted_graphs = []
    gold_graphs = []
    pred_graph = prediction[8:]
    gold_graph = reference[8:]
    predicted_graphs.append(pred_graph.lower())
    gold_graphs.append(gold_graph.lower())

    gold_edges = split_to_edges_explagraph(gold_graphs)
    pred_edges = split_to_edges_explagraph(predicted_graphs)

    #gold_tokens, pred_tokens, _ = get_tokens(gold_edges, pred_edges, None)

    #precisions_rouge, recalls_rouge, f1s_rouge, precisions_bleu, recalls_bleu, f1s_bleu = get_bleu_rouge(gold_tokens, pred_tokens, gold_edges, pred_edges)
    precisions_BS, recalls_BS, f1s_BS = get_bert_score(gold_edges, pred_edges)
    return f1s_BS[0]

def compute_reward_ged(prediction, reference):
    predicted_graphs = []
    gold_graphs = []
    pred_graph = prediction[8:]
    gold_graph = reference[8:]
    predicted_graphs.append(pred_graph.lower())
    gold_graphs.append(gold_graph.lower())

    ged = get_ged(gold[0], pred[0])
    return 1-ged

def compute_reward_bleu(prediction, reference):
	google_bleu = evaluate.load('google_bleu')
	results = google_bleu.compute(predictions=[prediction], references=[[reference]])
	return results["google_bleu"]

def compute_reward_rouge(prediction, reference):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=[prediction], references=[[reference]])
    return results["rougeL"]

# save datasets to disk for later easy loading
# tokenized_dataset["train"].save_to_disk("data/train")
# tokenized_dataset["test"].save_to_disk("data/eval")

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=ppo_config.learning_rate)

if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=ppo_config.learning_rate,
    )
# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    ppo_config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=tokenized_dataset["train"],
    data_collator=collator,
    optimizer=optimizer,
)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    # "temperature": 0.7,
    "pad_token_id": tokenizer.pad_token_id,
    # "eos_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 256,
}
#output_min_length = 32
#output_max_length = script_args.output_max_length
#output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    response_tensors = ppo_trainer.generate(
        query_tensors,
        # length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # Compute rewards
    if "Premise" in batch["query"]:
        rewards_1 = [torch.tensor(compute_reward_copasse(batch["response"][i], batch["reference"][i])) for i in range(len(batch["response"]))]
    elif "Belief" in batch["query"]:
        rewards_1 = [torch.tensor(compute_reward_explagraph(batch["response"][i], batch["reference"][i])) for i in range(len(batch["response"]))]
    else:
        rewards_1 = [torch.tensor(compute_reward_bleu(batch["response"][i], batch["reference"][i])) for i in range(len(batch["response"]))]

    texts = [f"""{q}\nOutput:\n{r}""" for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards_2 = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]
	
    rewards = [x+y for x, y in zip(rewards_1, rewards_2)]

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    if script_args.save_freq and epoch and epoch % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"_step_{epoch}")
