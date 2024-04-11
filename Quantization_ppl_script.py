from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig#, QuantoConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from llama_cpp import Llama, llama_get_logits
from torch.nn import CrossEntropyLoss
# from awq import AutoAWQForCausalLM
from scipy.special import softmax
from tqdm import tqdm
import numpy as np
#import quanto
import torch
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = 'cuda:0'
checkpoint = "mistralai/Mistral-7B-Instruct-v0.2"
# list_configs_HF_BASE_LLMINT8_QLORA = [
#     (torch.float32, None),
#     (torch.float16, None),
#     (torch.bfloat16, None),
#     (torch.float16, BitsAndBytesConfig(load_in_8bit=True)),
#     (torch.float32, BitsAndBytesConfig(load_in_4bit=True,
#                             bnb_4bit_use_double_quant=False,
#                             bnb_4bit_quant_type="fp4",
#                             bnb_4bit_compute_dtype=torch.float16,
#                             )),
#     (torch.float32, BitsAndBytesConfig(load_in_4bit=True,
#                             bnb_4bit_use_double_quant=False,
#                             bnb_4bit_quant_type="nf4",
#                             bnb_4bit_compute_dtype=torch.float16,
#                             )),
#     (torch.float16, BitsAndBytesConfig(load_in_4bit=True,
#                             bnb_4bit_use_double_quant=False,
#                             bnb_4bit_quant_type="nf4",
#                             bnb_4bit_compute_dtype=torch.float16,
#                             )),
#     (torch.float16, BitsAndBytesConfig(load_in_4bit=True,
#                             bnb_4bit_use_double_quant=True,
#                             bnb_4bit_quant_type="nf4",
#                             bnb_4bit_compute_dtype=torch.float16,
#                             )),
#     (torch.float32, QuantoConfig(weights='float8')),
#     (torch.float32, QuantoConfig(weights='int8')),
#     (torch.float32, QuantoConfig(weights='int4')),
# ]

# list_configs_llamacpp = [
#     'mistral-7b-instruct-v0.2.Q8_0.gguf',
#     'mistral-7b-instruct-v0.2.Q4_K_M.gguf',
#     'mistral-7b-instruct-v0.2.Q3_K_L.gguf',
# ]

# list_configs_HF_AutoGPTQ = [
#     'Mistral-7B-Instruct-v0.2-GPTQ-4bit',
#     'Mistral-7B-Instruct-v0.2-GPTQ-8bit',
#     'mistral-7b-instruct-v0.2-GPTQM-Q4-GS128-DAT-TST-C4',
#     'mistral-7b-instruct-v0.2-GPTQM-Q4-GS128-DAT-TSF-C4',
#     'mistral-7b-instruct-v0.2-GPTQM-Q4-GS128-DAF-TST-C4',
#     'mistral-7b-instruct-v0.2-GPTQM-Q4-GSNone-DAT-TST-C4',
#     'mistral-7b-instruct-v0.2-GPTQM-Q3-GS128-DAT-TST-C4',
#     'mistral-7b-instruct-v0.2-GPTQM-Q3-GS32-DAT-TST-C4',
#     'mistral-7b-instruct-v0.2-GPTQM-Q2-GS32-DAT-TST-C4',
#     'mistral-7b-instruct-v0.2-GPTQM-Q2-GS8-DAT-TST-C4',
# ]

list_configs_AutoGPTQ = [
    # 'Mistral-7B-Instruct-v0.2-GPTQ-4bit',
    # 'mistral-7b-instruct-v0.2-GPTQM2-Q4-GS128-DAF-TSF-RP2',
    # 'mistral-7b-instruct-v0.2-GPTQM2-Q3-GS32-DAF-TSF-RP2',
    # 'mistral-7b-instruct-v0.2-GPTQM2-Q2-GS16-DAF-TSF-RP2',
    'mistral-7b-instruct-v0.2-GPTQM2-Q4-GS512-DAF-TSF-RP2',
    'mistral-7b-instruct-v0.2-GPTQM2-Q4-GS32-DAF-TSF-RP2',
    'mistral-7b-instruct-v0.2-GPTQM2-Q4-GSNone-DAF-TSF-RP2',
]

# list_configs_HF_AutoAWQ = [
#     "Mistral-7B-Instruct-v0.2-AWQ-4bit",
# ]

# list_configs_AutoAWQ = [
#     #(model_path, use_exllama2, fuse_layers)
#     ("mistral-7b-instruct-v0.2-AWQM-Q4-GS128-GEMM", True, False),
#     ("mistral-7b-instruct-v0.2-AWQM-Q4-GS128-GEMM", True, True),
#     ("mistral-7b-instruct-v0.2-AWQM-Q4-GS128-GEMV", False, False),
#     ("mistral-7b-instruct-v0.2-AWQM-Q4-GS128-GEMV", False, True),
#     ("mistral-7b-instruct-v0.2-AWQM-Q4-GS128-GEMVF", False, False),
#     ("mistral-7b-instruct-v0.2-AWQM-Q4-GS128-GEMVF", False, True),
#     ("mistral-7b-instruct-v0.2-AWQM-Q4-GS128-Marlin", False, False),
# 
# ]

def main():
    #### base llmint8 and qlora config
    # for config in tqdm(list_configs):
    #     torch_dtype, quantization_config = config

    #     tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast = False)
    #     model = AutoModelForCausalLM.from_pretrained(checkpoint,
    #                                             torch_dtype = torch_dtype,       
    #                                             device_map = device,
    #                                             quantization_config = quantization_config,
    #                                             )

    #     ppl, _ = evaluate_ppl(model, tokenizer)
        
    #     with open('./results_ppl.txt', 'a') as f:
    #         if 'Quanto' in str(quantization_config):
    #             quantization_config = 'Quanto ' + str(quantization_config.weights)
    #         model_config = f"{torch_dtype=}, quantization_config={str(quantization_config).replace('\n', ',')} {ppl=}\n"
    #         f.write(model_config)
        
    #     del model
    #     torch.cuda.empty_cache()

    #### llamacpp config
    # for model_name in tqdm(list_configs):
    #     model = Llama(
    #         model_path=f"./{model_name}",  # Download the model file first
    #         n_ctx=1024,             # The max sequence length to use - note that longer sequence lengths require much more resources
    #         n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
    #         n_gpu_layers=-1,         # The number of layers to offload to GPU, if you have GPU acceleration available, if -1 all layers are offloaded
    #         verbose=False,
    #         )

    #     ppl, _ = evaluate_ppl_llama_cpp(model)

    #     with open('./results_ppl.txt', 'a') as f:
    #         model_config = f"{model_name=}, {ppl=}\n"
    #         f.write(model_config)
        
    #     del model
    #     torch.cuda.empty_cache()

    #### base hf_autogptq config
    # for model_name in tqdm(list_configs_HF_AutoGPTQ):
        # framework = 'HF_AutoGPTQ'
        # model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        # tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ppl, _ = evaluate_ppl(model, tokenizer)
        # 
        # with open('./results_ppl.txt', 'a') as f:
            # model_config = f"{framework=}, {model_name=}, {ppl=}\n"
            # f.write(model_config)
        # 
        # del model
        # torch.cuda.empty_cache()

    for model_name in tqdm(list_configs_AutoGPTQ):
        framework = 'AutoGPTQ'
        model = AutoGPTQForCausalLM.from_quantized(model_name,
                                            use_marlin = False,
                                            disable_exllama = True,
                                            disable_exllamav2 = False,
                                            device=device,
                                            )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        ppl, _ = evaluate_ppl(model, tokenizer)
        
        with open('./results_ppl.txt', 'a') as f:
            model_config = f"{framework=}, {model_name=}, {ppl=}\n"
            f.write(model_config)
        
        del model
        torch.cuda.empty_cache()

    # for model_name in list_configs_HF_AutoAWQ:
    #     framework = 'HF_AutoAWQ'
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

    #     ppl, _ = evaluate_ppl(model, tokenizer)
    
    #     with open('./results_ppl.txt', 'a') as f:
    #         model_config = f"{framework=}, {model_name=}, {ppl=}\n"
    #         f.write(model_config)
        
    #     del model
    #     torch.cuda.empty_cache()
    
    # for config in list_configs_AutoAWQ:
    #     framework = 'AutoAWQ'
    #     model_name, use_exllama2, fuse_layers = config
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     model = AutoAWQForCausalLM.from_quantized(model_name, 
    #                                         fuse_layers=fuse_layers,
    #                                         use_exllama=False,
    #                                         use_exllama_v2=use_exllama2,
    #                                         max_seq_len=None,
    #                                         device_map={'': device},
    #                                         )

    #     ppl, _ = evaluate_ppl(model, tokenizer)
    
    #     with open('./results_ppl.txt', 'a') as f:
    #         model_config = f"{framework=}, {model_name=}, {use_exllama2=}, {fuse_layers=}, {ppl=}\n"
    #         f.write(model_config)
        
    #     del model
    #     torch.cuda.empty_cache()



def evaluate_ppl(model, tokenizer):
    path = '/mnt/datascience1/Adrien/quantization/wikitext2.json'
    with open(path) as f:
        data = json.load(f)['text']

    data = data[:300]

    ppl_list = []
    for j, prompt in tqdm(enumerate(data)):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        labels = input_ids.clone()
        with torch.no_grad():
            logits = model(input_ids=input_ids, labels=labels).logits
        
        loss_fct = CrossEntropyLoss()

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, 32_000)  ##32000 is the vocab size of mistral - to be adapted!
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        ppl = torch.exp(loss)
        ppl_list.append(ppl)

        print(f'{j}: {ppl=}')

    return sum(ppl_list)/len(ppl_list), ppl_list


def evaluate_ppl_llama_cpp(model):
    path = '/mnt/datascience1/Adrien/quantization/wikitext2.json'
    with open(path) as f:
        data = json.load(f)['text']

    data = data[:300]

    ppl_list = []
    error_raised = False
    for j, prompt in tqdm(enumerate(data)):
        input_ids = model.tokenize(prompt.encode('utf-8'))
        all_logits = None
        for i in range(len(input_ids)):
            input_ids_cut = input_ids[:i+1]
            try:
                prompt = model.detokenize(input_ids_cut).decode('utf-8')
            except:
                print(f'{j}: error with encoding ', model.detokenize(input_ids_cut))
                error_raised = True
                break;
            
            output = model(prompt, # Prompt
                    max_tokens=1,  # Generate up to 512 tokens
                    stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
                    echo=True,       # Whether to echo the prompt
                    )


            logits = llama_get_logits(model.ctx)
            logits_np = np.ctypeslib.as_array(logits, shape=(1,32000))

            try:
                all_logits = torch.concat([all_logits, torch.tensor(logits_np)], dim=0)
            except:
                all_logits = torch.tensor(logits_np)
        
        if error_raised:
            error_raised = False
            continue;

        loss_fct = CrossEntropyLoss()
        logits = all_logits.unsqueeze(0)
        labels = torch.tensor(input_ids).unsqueeze(0)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, 32_000)  ##32000 is the vocab size of mistral - to be adapted!
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        ppl = torch.exp(loss)
        ppl_list.append(ppl)

    return sum(ppl_list)/len(ppl_list), ppl_list


if __name__ == "__main__":
    main()