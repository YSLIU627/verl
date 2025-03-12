# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets

from transformers import AutoTokenizer
from utils import copy, makedirs
import argparse

from utils import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def filter_level(example, keep_levels = ["3", "4", "5"]):
    return any(level in example["level"] for level in keep_levels)

def make_prefix(question):
    prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question} Assistant: <think>"""
    return prefix

def add_token_length(example):
    # Tokenize prompts without adding special tokens
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    tokenized = tokenizer(example["prompt"][0]["content"], add_special_tokens=False)
    # Calculate token counts
    example["prompt_length"] = len(tokenized["input_ids"])
    return example

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/math_r1_dataset')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    from pathlib import Path

    file_path = Path(os.path.join(args.local_dir, 'train.parquet'))
    if file_path.exists() and file_path.suffix == ".parquet":
        print("file existed")
        return
    data_source = 'DigitalLearningGmbH/MATH-lighteval-r1'

    dataset = datasets.load_dataset('DigitalLearningGmbH/MATH-lighteval', trust_remote_code=True)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            question = question + ' ' + instruction_following
            question = make_prefix(question)

            answer = example.pop('solution')
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

   # merged_dataset = datasets.concatenate_datasets([train_dataset, test_dataset])
    #merged_dataset = merged_dataset.filter(filter_level, keep_in_memory=True)
    #merged_dataset = merged_dataset.map(add_token_length, num_proc=32, keep_in_memory=True)
    max_length = 256
    #merged_dataset = merged_dataset.filter(lambda example: example["prompt_length"] <= max_length)
    #merged_dataset = merged_dataset.remove_columns("prompt_length")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    # test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
main()