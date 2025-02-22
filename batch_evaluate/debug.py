import os
import datasets
from typing import List
from verl.utils.hdfs_io import copy, makedirs
import argparse
import numpy as np
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from verl.utils.reward_score.opencoder import compute_score




import re

text = "assert is_powerful_number(12sdf, ss2,33) == [dsfdf,f2,1]"

# 正则表达式提取括号内的参数部分和方括号内的内容
match = re.search(r"is_powerful_number\((.*?)\) == (.+)", text)
if match:
    params = match.group(1).strip()  # 提取括号内的部分
    result = match.group(2).strip()  # 提取 `==` 右侧的部分
else:
    print("No match found.")

#compute_score(response, {'tests':test, 'entry_point': example['entry_point']}