import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from tqdm import tqdm
from datasets import load_dataset
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = True,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "text_graph",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = True,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    #quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            #quantization_config=quantization_config
        )
        #model = PeftModel.from_pretrained(
        #    model,
        #    lora_weights,
        #    torch_dtype=torch.float16,
        #    cache_dir="/home/jiuzhouh/wj84_scratch/jiuzhouh/.cache/"
        #)
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    #model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    #model.config.bos_token_id = 1
    #model.config.eos_token_id = 2
    # for llama2-7b
    #model.config.pad_token = tokenizer.eos_token

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

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

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        #print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        #return output
        #return output.split("Output:")[1].split("</s>")[0].strip()
        #return output.split("[/INST] ")[1].split("</s>")[0].strip()
        return prompter.get_response(output)

    gradio = False

    if gradio:
        gr.Interface(
            fn=evaluate,
            inputs=[
                gr.components.Textbox(
                    lines=2,
                    label="Instruction",
                    placeholder="Tell me about alpacas.",
                ),
                gr.components.Textbox(lines=2, label="Input", placeholder="none"),
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
                gr.components.Slider(
                    minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
                )
            ],
            outputs=[
                gr.inputs.Textbox(
                    lines=5,
                    label="Output",
                )
            ],
            title="LLAMA2-13B-based Model",
            description="This model is finetuned on LLAMA2-13B on Explanation Generation Task: given a belief and an argument, the task requires a model to predict the stance (support/counter), whether a certain argument supports or counters a belief. Accordingly, it should also generate a commonsense explanation graph which reveals the internal reasoning process involved in inferring the predicted stance.",  # noqa: E501
        ).queue().launch(server_name="0.0.0.0", share=share_gradio)
        
    else:
        dataset_path = 'data/expla_graph_predict_stance_dev.json'
        data = load_dataset("json", data_files=dataset_path)
        samples = data['train']
        with open('output/llama-7b-sft-expla_graph_predict_stance_dev_sst-output.txt', 'a') as output_file:
            for sample in tqdm(samples):
                output = evaluate(sample['instruction'], sample['input'], max_new_tokens=256, temperature=1.0, top_p=1.0, top_k=0.0, num_beams=4)
                output_file.write(str(output).strip() + '\n')
                # for copa-sse
                #output_file.write(str(output).strip()[:2] + extract_valid_graph(str(output).strip()[3:]) + '\n')


if __name__ == "__main__":
    fire.Fire(main)
