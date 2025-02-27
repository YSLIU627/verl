import os
import datasets
from typing import List
from verl.utils.hdfs_io import copy, makedirs
import argparse
import numpy as np
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from verl.utils.reward_score.opencoder import compute_score

import re
def main(args):
    def make_map_fn():
        def process_fn(example, idx):
            example['original_testcase'] = example['testcase']
            modified_tests = []
            for i in range(len(example['testcase'])):
                test = example['testcase'][i]
                match = re.search(example['entry_point'] + r"\((.*?)\) == (.+)", test)
                if match:
                    params = match.group(1).strip()  # 提取括号内的部分
                    result = match.group(2).strip()  # 提取 `==` 右侧的部分
                    test = example['testcase'][i] + f""", \"\"\"Fail to pass the test: {test} \"\"\"
                    """
                    modified_tests.append(test)
            example['testcase'] = modified_tests
            return example

        return process_fn
    dataset = datasets.load_dataset(args.get_data_dir, trust_remote_code=True)['train']
    train_dataset = dataset.map(function=make_map_fn(), with_indices=True,num_proc = os.cpu_count())
    if args.push_to_hub_dir:
        train_dataset.push_to_hub(args.push_to_hub_dir)
    elif args.save_dir:
        train_dataset.to_parquet(args.save_dir)
    result = [1,2]
    x = f""", "Fail to pass the test: assert `{result}`."
                    """
    print("assert f(x) == y"+ x)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_data_dir', default='~/data/code')
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument("--push_to_hub_dir", default=False, type=str)
    parser.add_argument("--sample_end_idx", default=99999999, type=int)
    parser.add_argument("--save_filter", action='store_true')
    args = parser.parse_args()
        # add a row to each data item that represents a unique id
    main(args)