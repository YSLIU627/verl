import os
import datasets
from typing import List, Tuple
from verl.utils.hdfs_io import copy, makedirs
import argparse
import numpy as np
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from verl.utils.reward_score.opencoder import compute_score
import datasets
NAME="ZHLiu627/updated_qwen2.5_code_1.5b_grpo_iter0_full_data_miao_0212__self_correction_iter1_v1"
dataset = datasets.load_dataset(NAME, trust_remote_code=True)['train']
import re
response="""

```python
def anagrams(strs):
    anag = {}
    result = []
    return
"""
test = """
assert abc(2) == 4, \"aaa\"
"""
entry_point = "abc"
data = dataset[5]
code=data['code']
feedback = compute_score(code, {'tests':data['testcase'], 'entry_point': data['entry_point']})
print(code)
print('\n'.join(data['testcase']))
print(feedback)
#dataset_f = dataset.filter(lambda data: compute_score(data['code'], {'tests':data['testcase'], 'entry_point': data['entry_point']})[0]>0.9,num_proc=os.cpu_count())
#print(f"original len{len(dataset)}, filtered len{len(dataset_f)}")

from verl.workers.reward_manager.prime import PrimeRewardManager
reward_manger = PrimeRewardManager()



#print("""\"\"\"
 #     \"\"\"""")
#assert anagrams(["a"]) == [], "Fail to pass the test: assert anagrams(["a"]) == [] "