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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--local_dir', default='data/aime_2024')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument("--data_remote_dir",default='HuggingFaceH4/MATH-500', type=str)
    args = parser.parse_args()

    data_source = 'AIME24'
    dataset = datasets.load_dataset("AI-MO/aimo-validation-aime", trust_remote_code=True)
    dataset = dataset.filter(lambda ex: ex['id'] >=60)

    train_dataset = dataset['train']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            question = question + ' ' + instruction_following
            question = make_prefix(question)

            solution = example.pop('answer')
            # solution = extract_solution(answer)
            # issue: "073" (train[15])
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(int(solution))
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset = train_dataset
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
