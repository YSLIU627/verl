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

import os
import datasets
import argparse
from utils import copy, makedirs
import json

def make_prefix(question):
    prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question} Assistant: <think>"""
    return prefix

def make_format(example, idx):
    messages = example['messages']
    human_message = messages[0]
    assistant_message = messages[1]
    
    question = human_message["value"]
    question = question + " " + "Let's think step by step and output the final answer within \\boxed{}."
    question = make_prefix(question)
    answer = assistant_message["ground_truth"]["value"]
        
    data = {
        "data_source": "orz_dataset",
        "prompt": [{
            "role": "user",
            "content": question
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer
        },
        "extra_info": {
            "index": idx,
            "split": "train"
        }
    }
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument("--sample_start_idx", default=0, type=int)
    parser.add_argument("--sample_end_idx", default=999999999, type=int)
    parser.add_argument("--data_remote_dir", default='Tonic/OpenReasonerZero', type=str)
    args = parser.parse_args()

    print("Loading the local dataset...", flush=True)
    from pathlib import Path

    with open('./data_preprocess/orz_math_57k_collected.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    formatted_data = [{"messages": item} for item in raw_data]
    dataset = datasets.Dataset.from_list(formatted_data)

    dataset = dataset.select(range(args.sample_start_idx, min(args.sample_end_idx, len(dataset))))
    
    mapped_dataset = dataset.map(function=make_format, with_indices=True, remove_columns=dataset.column_names)
    print(mapped_dataset[0])
    print(f"len of training dataset is {len(mapped_dataset)}")
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    mapped_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

if __name__ == '__main__':
    main()
