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
# from . import gsm8k, math, prime_math, prime_code


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval', 'math-500']:
        from . import math
        res = math.compute_score(solution_str, ground_truth)
    elif data_source in [
            'numina_aops_forum', 'numina_synthetic_math', 'numina_amc_aime', 'numina_synthetic_amc', 'numina_cn_k12',
            'numina_olympiads',
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["math_500_train", "AIME_train", "RUC-AIBOX/STILL-3-Preview-RL-Data"]:
        from . import math_r1
        return math_r1.compute_score(solution_str, ground_truth)
    elif data_source in ["AIME24", "math-500-r1", "DigitalLearningGmbH/MATH-lighteval-r1"]:
        from . import math_r1
        return math_r1.compute_score_val(solution_str, ground_truth)
    elif "multiply" in data_source or "arithmetic" in data_source:
        from . import multiply
        return multiply.compute_score(solution_str, ground_truth)
    elif "countdown" in data_source:
        from . import countdown
        return countdown.compute_score(solution_str, ground_truth)
    elif data_source in ['orz_aime2024', 'orz_gpqa_diamond', 'orz_math500', 'orz_dataset']:
        from . import orz_verifier
        res = orz_verifier.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError

    if isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
