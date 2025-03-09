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
Preprocess the MATH500 dataset to parquet format
"""

import os
import datasets
import argparse
from utils import copy, makedirs, remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def make_format(example, question_key, answer_key):
    return {
        "data_source": "math-500",
        "prompt": [
            { "content": example[question_key] + "Let's think step by step and output the final answer within \\boxed{}.", "role": "user" }
        ],
        "ability": "math",
        "reward_model": {
            "ground_truth": example[answer_key],
            "style": "rule",
        },
        "extra_info": {
            "index": 0,
            "split": "dummy"
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument("--sample_start_idx", default=0, type=int)
    parser.add_argument("--sample_end_idx", default=999999999, type=int)
    parser.add_argument("--data_remote_dir",default='HuggingFaceH4/MATH-500', type=str)
    args = parser.parse_args()

    data_source = 'HuggingFaceH4/MATH-500'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    from pathlib import Path

    file_path = Path(os.path.join(args.local_dir, 'train.parquet'))

    if file_path.exists() and file_path.suffix == ".parquet":
        print("file existed")

    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    test_dataset = dataset['test']
    test_dataset = test_dataset.select(range(0, min(args.sample_end_idx, len(test_dataset)) ))
    test_dataset = test_dataset.map(
        lambda x: make_format(x, 'problem', 'answer'),
        remove_columns=test_dataset.column_names
    )
    #instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

if __name__ == '__main__':
    main()