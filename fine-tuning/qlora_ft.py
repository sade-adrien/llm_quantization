
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import re

device = 'cuda:0'

TASK = 'gsm8k' # ['boolq', 'hellaswag', 'gsm8k']



################################### pure QLoRA #############################################
checkpoint = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast = False)
tokenizer.pad_token = tokenizer.eos_token
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                        bnb_4bit_quant_type='nf4',
                                        bnb_4bit_use_double_quant=False,
                                        )
model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                             torch_dtype = torch.bfloat16,       
                                             device_map=device,
                                             cache_dir='/mnt/esperanto/et/huggingface/hub',
                                             quantization_config=quantization_config,
                                            )


################################### AWQ + LoRA #############################################
# QUANTIZED_MODEL_ID = f"./models/mistral-7b-v0.1-AWQ-Q4-GS128-GEMM"
# model = AutoModelForCausalLM.from_pretrained(QUANTIZED_MODEL_ID,
#                                             device_map={'': device},
#                                         )
# tokenizer = AutoTokenizer.from_pretrained(QUANTIZED_MODEL_ID)
# tokenizer.pad_token = tokenizer.eos_token



lora_config = LoraConfig(r=16,
                        lora_alpha=8,
                        lora_dropout=0.05,
                        bias="none",
                        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj'])

model = prepare_model_for_kbit_training(model)   #! has to be applied BEFORE get_peft_model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

if TASK == 'boolq':
    dataset = load_dataset('google/boolq', split='train')
elif TASK == 'hellaswag':
    dataset = load_dataset('Rowan/hellaswag', split='train')
elif TASK == 'gsm8k':
    dataset = load_dataset('gsm8k', 'main', split='train')


def _preprocess(text):
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": _preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [_preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

def prepare_data(sample):
    if TASK == 'boolq':
        prompt = f"{sample['passage']}\nQuestion: {sample['question']}?\nAnswer: {'yes' if sample['answer'] else 'no'}"
    elif TASK == 'hellaswag':
        out_doc = _process_doc(sample)
        prompt = out_doc['query'] + ' ' + out_doc['choices'][out_doc['gold']]
    elif TASK == 'gsm8k':
        prompt = f"Question: {sample["question"]}\nAnswer: {sample["answer"]}"
    inputs = tokenizer(prompt, padding='max_length', truncation=True, max_length=512)
    inputs.update({'labels': inputs['input_ids']})
    return inputs


if TASK == 'boolq':
    col_to_remove = ['answer', 'passage', 'question']
elif TASK == 'hellaswag':
    col_to_remove = ['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type', 'label']
elif TASK == 'gsm8k':
    col_to_remove = ['answer', 'question']

train_data = dataset.select(list(range(0, int(0.9*len(dataset))))).map(prepare_data).remove_columns(col_to_remove)
test_data = dataset.select(list(range(int(0.9*len(dataset)), len(dataset)))).map(prepare_data).remove_columns(col_to_remove)

training_args = TrainingArguments(
    output_dir=f'./models/mistral-7b-v0.1_lora_{TASK.lower()}',
    #auto_find_batch_size=True, 
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate= 5e-5,
    num_train_epochs=1,
    evaluation_strategy="no",
    save_strategy="no",
    gradient_accumulation_steps=4,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    logging_steps=1,
    bf16=True,
    report_to='wandb',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

trainer.train()

trainer.model.save_pretrained(f'./models/mistral-7b-v0.1-AWQ-Q4-GS128-GEMM_qlora_{TASK.lower()}')
