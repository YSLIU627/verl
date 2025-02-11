from typing import List, Union
from verl.utils.reward_score.code_utils import refine_text
from verl.utils.reward_score.execute import check_correctness
from verl.utils.reward_score.sanitize import sanitize


PYTHON_STOP = [ "\nif __name__",
                "\ndef main(",
                "\nprint("
                ]
    
PYTHON_IMPORTS = [  "import math",
                    "import re",
                    "import sys",
                    "import copy",
                    "import datetime",
                    "import itertools",
                    "import collections",
                    "import heapq",
                    "import functools",
                    "import hashlib",
                    "import numpy",
                    "import numpy as np",
                    "import string",
                    "from typing import *",
                    "from collections import *"
                    ]

def format_prompt(problem: str,
                      tests: Union[List[str],str],
                      code: str = None
                    ) -> str:
        problem = f"You are an expert Python programmer, and here is your task:\n{problem}"
        if isinstance(tests, List):
            test = "\n".join(tests)
        else:
            test = tests
        test = f"Your code should pass these tests:\n{test}\n"
        prompt = problem + test
        if code:
            code = refine_text(code)
            code = f"\n```python\n{code}\n```\n"
            prompt = prompt + code
        else:
            prompt = prompt + "\n```python\n"
        return prompt


def compute_score(solution_str: str, ground_truth: Union[str,dict]) -> float:
    #retval = 0.
    try:
        
        if isinstance(ground_truth, dict):
            
            solution = sanitize(solution_str,ground_truth.get("entry_point") )
            #print(solution)
            code =  "\n".join(PYTHON_IMPORTS)  + "\n"+ solution + "\n" + ground_truth['tests']
                
        else:
            solution = sanitize(solution_str )
            code =  "\n".join(PYTHON_IMPORTS)  + "\n" + solution + "\n" + ground_truth 
        feedback = check_correctness(task_id = 0, completion_id=0,    solution=code,time_out=6)
        #print(code,(feedback['result']))
        return float(feedback["passed"])
    except Exception as e:
        print(e)

    #eturn retval, feedback