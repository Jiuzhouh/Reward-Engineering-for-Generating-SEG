import torch
import json
import gradio as gr
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from trl import AutoModelForSeq2SeqLMWithValueHead

model_name = "flan-t5-xxl-lora-expla-graph-predict-stance-merged-rlhf-llama-7b-reward-pretrained-kl0.3_step_140"

# Load peft config for pre-trained checkpoint etc.
load_trl_model = True
load_lora_model = False

gradio = False

if load_lora_model:
    peft_model_id = "flan-t5-xxl-lora-expla-graph-predict-stance"
    config = PeftConfig.from_pretrained(peft_model_id)
    model_name = config.base_model_name_or_path
    #model_name = "philschmid/flan-t5-xxl-sharded-fp16"
    #model_name = "flan-t5-xxl-lora-expla-graph-predict-stance-merged"

# load base LLM model and tokenizer
if load_trl_model:
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto")
    load_lora_model = False
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add Special Token for Inference
# new_tokens = ["<S>"]
# tokenizer.add_tokens(new_tokens)

# Load the Lora model
if load_lora_model:
    model = PeftModel.from_pretrained(model, peft_model_id)
    print("Peft model loaded")

model.eval()

def evaluate(instruction, input, max_new_tokens=256, temperature=0.1, top_p=0.9, top_k=40, num_beams=4, num_return_sequences=1):
    model_input = instruction + ' ' + input
    input_ids = tokenizer(model_input, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams, num_return_sequences=num_return_sequences)
    output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

    return output

import re
import ast
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

if gradio:
    gr.Interface(
            fn=evaluate,
            inputs=[
                gr.components.Textbox(
                    lines=2,
                    label="Instruction",
                    placeholder="none",
                ),
                gr.components.Textbox(lines=2, label="Input", placeholder="none"),
                gr.components.Slider(
                    minimum=1, maximum=2000, step=1, value=256, label="Max tokens"
                ),
                gr.components.Slider(
                    minimum=0, maximum=1, value=0.1, label="Temperature"
                ),
                gr.components.Slider(
                    minimum=0, maximum=1, value=0.75, label="Top p"
                ),
                gr.components.Slider(
                    minimum=0, maximum=100, step=1, value=40, label="Top k"
                ),
                gr.components.Slider(
                    minimum=1, maximum=4, step=1, value=4, label="Beams"
                ),
            ],
            outputs=[
                gr.inputs.Textbox(
                    lines=5,
                    label="Output",
                )
            ],
            title="🦙🌲 FLAN-T5-XXL-based Unified Graph-Text Model",
            description="FLAN-T5-XXL is a 11B-parameter T5-XXL model finetuned to follow instructions. The unified graph-text model is instruction-tuned using LoRA on diverse graph-text related tasks, including graph-to-text generation, text-to-graph generation, knowledge-based question generation."
        ).queue().launch(server_name="0.0.0.0", share=True)

else:
    dataset_path = 'data/expla_graph_predict_stance_dev.json'
    data = load_dataset("json", data_files=dataset_path)
    #data = data['train'].train_test_split(test_size=80, shuffle=False)
    samples = data['train']
    #outputs = []
    with open('output/flan-t5-xxl-lora-expla-graph-predict-stance-merged-rlhf-llama-7b-reward-pretrained-kl0.3_step_140-dev-output.txt', 'a') as output_file:
        #i = 0
        for sample in tqdm(samples):
            output_dict = {}
            output = evaluate(sample['instruction'], sample['input'], max_new_tokens=512, temperature=1.0, top_p=1.0, top_k=0.0, num_beams=4)
            output_file.write(str(output).strip() + '\n')
            # for copa-sse
            #output_file.write(str(output).strip()[:2] + extract_valid_graph(str(output).strip()[3:]) + '\n')
            #output_dict['index'] = i
            #output_dict['instruction'] = sample['instruction']
            #output_dict['input'] = sample['input']
            #output_dict['predicted'] = str(output).strip()
            #i+=1
            #outputs.append(output_dict)
        #output_file.write(json.dumps(outputs, indent=2, ensure_ascii=False))
