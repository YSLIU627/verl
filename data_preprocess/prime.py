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
import argparse
from utils import copy, makedirs, remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def make_format(example):
    return {
        "data_source": example['data_source'],
        "prompt": [
            {"content": example['prompt'][1]['content'] + "Let's think step by step and output the final answer within \\boxed{}.", "role": "user" }
        ],
        "ability": example['ability'],
        "reward_model": example['reward_model'],
        'extra_info': example['extra_info']
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument("--sample_start_idx", default=0, type=int)
    parser.add_argument("--sample_end_idx", default=999999999, type=int)
    parser.add_argument("--data_remote_dir",default = 'OpenCoder-LLM/opc-sft-stage2',type = str)
    parser.add_argument("--math_only", default=True)
    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = 'PRIME-RL/Eurus-2-RL-Data'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    from pathlib import Path

    file_path = Path(os.path.join(args.local_dir, 'train.parquet'))

    if file_path.exists() and file_path.suffix == ".parquet":
        print("file existed")
        #return 
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset = dataset['train']
    train_dataset = train_dataset.select(range(args.sample_start_idx, min(args.sample_end_idx, len(train_dataset)) ))
    test_dataset = dataset['validation']
    test_dataset = test_dataset.select(range(0, min(args.sample_end_idx, len(test_dataset)) ))
    if args.math_only:
        train_dataset = train_dataset.filter(lambda x: x['ability'] == 'math')
        test_dataset = test_dataset.filter(lambda x: x['ability'] == 'math')
    train_dataset = train_dataset.map(
        lambda x: make_format(x),
        remove_columns=train_dataset.column_names
    )
    test_dataset = test_dataset.map(
        lambda x: make_format(x),
        remove_columns=test_dataset.column_names
    )
    #instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    
    print(f"len of training dataset is {len(train_dataset)}, {len(dataset)}, {args.sample_start_idx},{args.sample_end_idx}")
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
main()