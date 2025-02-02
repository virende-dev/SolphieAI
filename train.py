from datasets import load_dataset,Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import pandas as pd

ds = load_dataset("Virende/Solphie-1S-Foundation-Model-DS")
model_name = "<Llama-3.1-8B-Instruct>"
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

data_train = []
for row in ds["train"]:
    data_train.append({
        "question": row["question"],
        "answer": row["answer"],
        "think": row["think"]
    })
train_prompt_style = """Below are instructions describing the task, with input to provide more context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.
### Instructions:
You are an assistant who knows a lot about Solana。
### Question:
{}
### Response:
<think>
{}
</think>
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    inputs = examples["question"]
    cots = examples["think"]
    outputs = examples["answer"]
    texts = []
    for input_text, cot, output_text in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input_text, cot, output_text) + EOS_TOKEN
        
        texts.append(text)
    return {"text": texts}

dataset = Dataset.from_list(data_train)
dataset = dataset.map(formatting_prompts_func, batched=True)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)
trainer_stats = trainer.train()