import os
import datasets
from typing import List
from verl.utils.hdfs_io import copy, makedirs
import argparse

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
    parser.add_argument("--batch_size",defualt = 10000, type = int)
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
                example["filtering"] = rewards > 0.9
            else:
                try:
                    example["rewards"] = compute_score(response, {'tests':test, 'entry_point': example['entry_point']})
                except:
                    example["rewards"] = compute_score(response, test)
            return example

        return process_fn
    dataset = datasets.load_dataset(args.get_data_dir, trust_remote_code=True)['train']

    from vllm import LLM, SamplingParams

    # 初始化模型（关键：启用内存优化）
    llm = LLM(
        model="qwen/",
        tensor_parallel_size=1,          # 单GPU模式（多GPU会增加显存占用）
        gpu_memory_utilization=0.8,      # 显存利用率限制（防止OOM）
        enforce_eager=True,              # 关闭图优化（减少显存峰值）
        disable_log_stats=True           # 禁用日志统计（减少内存开销）
    )

    sampling_params = SamplingParams(max_tokens=1024,temperature=1.0,n=5,top_p=0.9)

    # 流式分块处理
    current_batch = []
    for example in dataset:
        current_batch.append(example["prompt"])
        
        # 动态调整批次大小（根据内存情况）
        if len(current_batch) >= batch_size:
            outputs = llm.generate(current_batch, sampling_params)
            
            # 立即保存结果并释放内存（关键！）
            with open("results.jsonl", "a") as f:
                for prompt, output in zip(current_batch, outputs):
                    f.write(
                        json.dumps({
                            "prompt": prompt,
                            "response": output.outputs[0].text
                        }) + "\n"
                    )
            
            # 强制释放内存（Python GC不总是及时）
            del outputs
            current_batch.clear()
            
            # 手动触发垃圾回收（激进但有效）
            import gc
            gc.collect() 

    # 处理剩余批次
    if current_batch:
        outputs = llm.generate(current_batch, sampling_params)
    if args.push_to_hub_dir:
        train_dataset.push_to_hub(args.push_to_hub_dir)
    else:
        train_dataset.to_parquet(args.save_dir)