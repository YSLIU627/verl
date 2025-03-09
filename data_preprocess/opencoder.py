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

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/code')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument("--sample_start_idx", default=0, type=int)
    parser.add_argument("--sample_end_idx", default=99999999, type=int)
    parser.add_argument("--data_remote_dir",default = 'OpenCoder-LLM/opc-sft-stage2',type = str)
    args = parser.parse_args()

    from pathlib import Path

    file_path = Path(os.path.join(args.local_dir, 'train.parquet'))

    if file_path.exists() and file_path.suffix == ".parquet":
        print("file existed")
        #return 

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = 'OpenCoder-LLM/opc-sft-stage2[educational_instruct]'
    print(f"Loading the {args.data_remote_dir} dataset from huggingface...", flush=True)
    if args.data_remote_dir == 'OpenCoder-LLM/opc-sft-stage2':
        dataset = datasets.load_dataset('OpenCoder-LLM/opc-sft-stage2', "educational_instruct", trust_remote_code=True)['train']
    else:
        dataset = datasets.load_dataset(args.data_remote_dir, trust_remote_code=True)['train']

    dataset = dataset.select(range(args.sample_start_idx, min(args.sample_end_idx, len(dataset)) ))
    dataset =  dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    #instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example['instruction']
            test = example["testcase"][0]
            question = f"You are an expert Python programmer, and here is your task:\n{question}\n Your provided code should be wrapped in one single markdown block" +  "(e.g. ```python \n{your provided code}\n```) " + f"and pass the following test:\n{test}"

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "tests": '\n'.join(example["testcase"]),
                        "entry_point": example['entry_point']
                    }
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

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
if __name__ == '__main__':
    main()