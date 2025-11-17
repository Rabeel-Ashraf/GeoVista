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
import torch

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math_score

        res = math_score.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        from . import prime_code

        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)

    elif data_source in ['rag_v2-train']:
        from . import agent
        res = agent.compute_score(solution_str, ground_truth)
    elif data_source in ['rag_v2-test']:
        from . import agent
        res = agent.compute_score_eval(solution_str, ground_truth)

    elif data_source in ['vstar', 'vl_agent', 'chart']:
        from . import vl_agent
        res = vl_agent.compute_score(solution_str, ground_truth, extra_info)

    elif data_source in ['geoguessr']:
        from . import vl_agent
        res = vl_agent.compute_common_reasoning(solution_str, ground_truth, extra_info)

    elif data_source in ['thinklite_eureka', 'xince']:
        from . import vl_agent
        res = vl_agent.compute_score_math(solution_str, ground_truth, extra_info)

    elif data_source in ['pixmo-points-image']:
        from . import vl_agent
        res = vl_agent.compute_score_counting(solution_str, ground_truth, extra_info)

    elif data_source in ['pixmo-docs-chart', 'charxiv']:
        from . import vl_agent
        res = vl_agent.compute_score_chart(solution_str, ground_truth, extra_info)

    elif data_source in ['thyme-2rounds', 'vstar_bench', 'thyme-rl', 'hrbench_4k']:
        from . import vl_agent
        # res = vl_agent.compute_score_crop(solution_str, ground_truth, extra_info)
        res = vl_agent.compute_score(solution_str, ground_truth, extra_info)

    elif data_source in ['maze']:
        from . import vl_agent
        res = vl_agent.compure_score_maze(solution_str, ground_truth, extra_info)
    
    elif data_source in ['google_map']:
        from . import search_agent
        res = search_agent.compure_score_search(solution_str, ground_truth, extra_info)
    elif data_source in ['google_map_v1']:
        # alignment reward & query penalty
        from . import search_agent_v1
        res = search_agent_v1.compure_score_search(solution_str, ground_truth, extra_info)
    elif data_source in ['google_map_v2']:
        # alignment reward
        from . import search_agent_v2
        res = search_agent_v2.compure_score_search(solution_str, ground_truth, extra_info)
    elif data_source in ['google_map_v3']:
        # alignment reward
        from . import search_agent_v3
        res = search_agent_v3.compure_score_search(solution_str, ground_truth, extra_info)
    elif data_source in ['google_map_v4']:
        # alignment reward
        from . import search_agent_v4
        res = search_agent_v4.compure_score_search(solution_str, ground_truth, extra_info)

    elif data_source in ["frozenlake"]:
        res = 0.0

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
