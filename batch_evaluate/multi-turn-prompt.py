import os
import datasets
from typing import List
from verl.utils.hdfs_io import copy, makedirs
import argparse
import numpy as np
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from verl.utils.reward_score.opencoder import compute_score

def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def main(args):
    FAIL_PROMPT_TEMPLATE = """
You made an mistake such that the provided code has a bug or does not pass all the tests for the instruction. Please first make a reflection and then refine your code.
"""
    def make_map_fn():
        def process_fn(example, idx):
            first_prompt = example["prompt"]
            example["original_prompt"] = first_prompt
            if isinstance(first_prompt, str):
                first_prompt = [{"role": "user", "content": first_prompt}]
                
            for response, reward in zip(example['responses'], example['rewards']):
                if reward < 0.5:
                    break
            example["prompt"] = first_prompt + [{"role": "assistant", "content": response}] + [{"role": "user", "content": FAIL_PROMPT_TEMPLATE}]
            return example

        return process_fn
    dataset = datasets.load_dataset(args.get_data_dir, trust_remote_code=True)['train']
    dataset = dataset.filter(lambda ex: ex["mean_reward"] <0.99)
    train_dataset = dataset.map(function=make_map_fn(), with_indices=True,num_proc = os.cpu_count())
    if args.push_to_hub_dir:
        train_dataset.push_to_hub(args.push_to_hub_dir + "_v1")
    elif args.save_dir:
        train_dataset.to_parquet(args.save_dir)
    

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
    main(args)