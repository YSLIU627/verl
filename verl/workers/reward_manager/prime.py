# Copyright 2024 PRIME team and/or its affiliates
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

import torch
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from multiprocessing import cpu_count


def single_compute_score(evaluation_func, completion, reference, task, extra_info, executor):
    try:
        return evaluation_func(task, completion, reference, extra_info)
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        return 0.0

        
def _safe_evaluation(evaluation_func, task, completion, reference, result_list, idx, extra_info):
    try:
        res = evaluation_func(task, completion, reference, extra_info)
        if isinstance(res, (int, float, bool)):
            result_list[idx] = float(res)
        elif isinstance(res, (list, tuple)) and isinstance(res[0], (int, float, bool)):
            result_list[idx] = float(res[0])
        else:
            result_list[idx] = 0.0
            
    except Exception as e:
        print(f"Error processing completion at index {idx}: {e}")
        result_list[idx] = 0.0


def parallel_compute_score(evaluation_func, completions, references, tasks, num_processes=32, extra_info=None, timeout=10):
    manager = multiprocessing.Manager()
    result_list = manager.list([0.0] * len(completions))
    processes = []

    for idx, (completion, reference, task) in enumerate(zip(completions, references, tasks)):
        p = multiprocessing.Process(target=_safe_evaluation, args=(evaluation_func, task, completion, reference, result_list, idx, extra_info))
        p.start()
        processes.append((p, idx))

    for p, idx in processes:
        p.join(timeout=timeout)
        if p.is_alive():
            warnings.warn(f"Timeout when processing completion at index {idx}")
            p.kill()
            p.join()

    return list(result_list)


# def parallel_compute_score(evaluation_func, completions, references, tasks, num_processes=32):
#     scores = [0.0] * len(completions)
#     with ProcessPoolExecutor(max_workers=num_processes) as executor:
#         futures = {
#             executor.submit(evaluation_func, task, completion, reference): idx
#             for idx, (completion, reference, task) in enumerate(zip(completions, references, tasks))
#         }

#         for future in as_completed(futures):
#             idx = futures[future]
#             try:
#                 result = future.result()
#                 if isinstance(result, (int, float, bool)):
#                     scores[idx] = float(result)
#                 elif isinstance(result, (list, tuple)) and isinstance(result[0], (int, float, bool)):
#                     scores[idx] = float(result[0])
#                 else:
#                     scores[idx] = 0.0
#             except Exception as e:
#                 print(f"Error processing completion at index {idx}: {e}")
#                 scores[idx] = 0.0
#     return scores


class PrimeRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, offset = 0, scale = 1.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.scale = scale
        self.offset = offset

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        num_processes = min(cpu_count()// 2, 16)
        assert len(sequences_str) == len(ground_truth) == len(data_sources)

        scores = parallel_compute_score(self.compute_score, sequences_str, ground_truth, data_sources, num_processes=num_processes)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = self.scale * scores[i] + self.offset

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor
