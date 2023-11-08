from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

@dataclass
class ScriptArguments:
    """
    The name of the LM model we wish to fine with PPO
    """

    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the model name"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
assert script_args.adapter_model_name is not None, "please provide the name of the Adapter you would like to merge"
# assert script_args.base_model_name is not None, "please provide the name of the Base model"
assert script_args.output_name is not None, "please provide the output name of the merged model"

peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)

if peft_config.task_type == "SEQ_CLS":
    # peft is for reward model so load sequence classification
    base_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path, num_labels=1, torch_dtype=torch.bfloat16)
else:
    base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path, return_dict=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(base_model, script_args.adapter_model_name)

print("Peft model loaded")

merged_model = model.merge_and_unload()

merged_model.save_pretrained(f"{script_args.output_name}")
tokenizer.save_pretrained(f"{script_args.output_name}")
# model.push_to_hub(f"{script_args.output_name}", use_temp_dir=False)
