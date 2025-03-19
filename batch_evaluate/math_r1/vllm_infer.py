#!/usr/bin/env python3
import os
from datasets import load_dataset, Dataset
from tqdm import tqdm  # For progress bar
from multiprocessing import freeze_support  # Needed on Windows

# Import vLLM components for offline inference
from vllm import LLM, SamplingParams
# Import the AutoTokenizer from transformers to use the chat template
from transformers import AutoTokenizer

def main():
    # === CONFIGURATION ===
    MODEL_NAME = "google/gemma-2-2b-it"  # Inference model to use.
    NUM_SAMPLES = 5       # Number of completions per prompt
    NUM_PROMPTS = 118200  # Process only the first 100000 examples from the dataset
    TEMPERATURE = 1       # Sampling temperature
    MAX_TOKENS = 2048
    TOP_P = 0.9           # Nucleus sampling parameter
    HF_USERNAME = "jwang2373"  # Replace with your HF username
    NEW_DATASET_NAME = f"{HF_USERNAME}/updated-code-gemma2-edu"
    SYSTEM_PROMPT = "You are a helpful coding assistant."

    # Initialize the tokenizer that supports chat template functionality.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # === LOAD DATASET ===
    DATASET_NAME = "ZHLiu627/code-opc2-edu"
    ds = load_dataset(DATASET_NAME)
    train_data = ds["train"]

    # Limit the dataset to the first NUM_PROMPTS examples.
    train_data = train_data.select(range(NUM_PROMPTS))

    # === PREPARE PROMPTS FOR BATCH PROCESSING USING CHAT TEMPLATE ===
    # Extract the base prompts from the dataset.
    base_prompts = [example["prompt"] for example in train_data]

    # Function to apply a chat template to a given prompt.
    def apply_chat_template_to_prompt(prompt):
        # Create a list of messages.
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        # Apply the chat template.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text

    # Build a batched list by processing each prompt with the chat template,
    # then repeating it NUM_SAMPLES times.
    batched_prompts = []
    for prompt in base_prompts:
        full_prompt = apply_chat_template_to_prompt(prompt)
        for _ in range(NUM_SAMPLES):
            batched_prompts.append(full_prompt)

    print(f"Total prompts to query: {len(batched_prompts)} "
          f"(which is {NUM_SAMPLES}x the original {len(base_prompts)} prompts)")

    # === INITIALIZE vLLM FOR OFFLINE BATCHED INFERENCE ===
    try:
        llm = LLM(model=MODEL_NAME)
    except Exception as e:
        print(f"Error initializing vLLM LLM: {e}")
        return

    # === SET SAMPLING PARAMETERS ===
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P,max_tokens=MAX_TOKENS)

    # === GENERATE RESPONSES USING OFFLINE BATCHED INFERENCE ===
    try:
        outputs = llm.generate(batched_prompts, sampling_params)
    except Exception as e:
        print(f"Error during vLLM inference: {e}")
        outputs = [None] * len(batched_prompts)

    # Extract generated text from each output.
    batched_responses = []
    for output in outputs:
        if output is None or not hasattr(output, "outputs") or not output.outputs:
            batched_responses.append("ERROR")
        else:
            # Each output is a RequestOutput-like object.
            generated_text = output.outputs[0].text
            batched_responses.append(generated_text)

    # === REASSEMBLE RESPONSES INTO THE ORIGINAL DATASET STRUCTURE ===
    new_data = []
    print("Reassembling responses and processing dataset...")
    # For each original example, group the corresponding NUM_SAMPLES responses.
    for i, example in enumerate(tqdm(train_data, desc="Processing Examples", unit="example")):
        start_idx = i * NUM_SAMPLES
        end_idx = (i + 1) * NUM_SAMPLES
        responses_group = batched_responses[start_idx:end_idx]

        example["new_prompt"] = example["prompt"]
        example["responses"] = responses_group
        new_data.append(example)

    new_ds = Dataset.from_list(new_data)
    print(new_ds)

    # === SAVE & UPLOAD TO HUGGING FACE HUB ===
    print(f"Uploading dataset to Hugging Face: {NEW_DATASET_NAME}")
    new_ds.push_to_hub(NEW_DATASET_NAME)
    print(f"? Dataset uploaded successfully: {NEW_DATASET_NAME}")

if __name__ == '__main__':
    freeze_support()  # Ensures proper bootstrapping on Windows
    main()
