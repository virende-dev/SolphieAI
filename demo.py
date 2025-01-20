from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 
dtype = None
load_in_4bit = True 


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./Solphie-1S-Foundation-Model",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

prompt_style = """Below are instructions describing the task, with input to provide more context.
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
"""

question = "How do I install the Solana CLI on macOS?"


FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])