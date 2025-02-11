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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_data_dir', default='~/data/code')
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument("--push_to_hub_dir", default=False, type=str)
    parser.add_argument("--sample_end_idx", default=99999999, type=int)
    parser.add_argument("--save_filter", action='store_true')
    args = parser.parse_args()
        # add a row to each data item that represents a unique id
    def make_map_fn():

        def process_fn(example, idx):
            question = example['instruction']
            test = example["testcase"]
            if isinstance(test, List):
                test = '\n'.join(test)
            responses = example["responses"]
            if isinstance(responses, List):
                rewards = []
                for response in responses:
                    try:
                        rewards.append(compute_score(response, {'tests':test, 'entry_point': example['entry_point']}))
                    except:
                        rewards.append(compute_score(response, test))
                example["rewards"] = rewards
                example["mean_reward"] = np.mean(rewards)
            else:
                try:
                    example["rewards"] = compute_score(response, {'tests':test, 'entry_point': example['entry_point']})
                except:
                    example["rewards"] = compute_score(response, test)
                    example["mean_reward"] = example["rewards"]
            return example

        return process_fn
    dataset = datasets.load_dataset(args.get_data_dir, trust_remote_code=True)['train']
    train_dataset = dataset.map(function=make_map_fn(), with_indices=True,num_proc = os.cpu_count())
    if args.push_to_hub_dir:
        train_dataset.push_to_hub(args.push_to_hub_dir)
    elif args.save_dir:
        train_dataset.to_parquet(args.save_dir)
        
    data_filtered = train_dataset.filter(lambda ex: ex["mean_reward"] < 0.99)
    data_filtered_0 = data_filtered.filter(lambda ex: ex["mean_reward"] > 0.01)
    filtered_rewards = data_filtered["mean_reward"]
    print(f"length: {len(train_dataset)}, reward < 1: {len(data_filtered)}")
    print(f"1 > reward >0, len = {len(data_filtered_0)},reward_mean for data_filtered {np.mean(filtered_rewards)}")
    if args.save_filter:
        if args.push_to_hub_dir:
            data_filtered.push_to_hub(args.push_to_hub_dir + "filtered")
        elif args.save_dir:
            data_filtered.to_parquet(args.save_dir + "filtered")
    