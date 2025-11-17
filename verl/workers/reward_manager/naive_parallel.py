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

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

import json
import datetime

class NaiveParallelRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

        self.step_cnt = 0

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        action_or_attn_mask = data.batch['action_mask'] if 'action_mask' in data.batch.keys() else data.batch['attention_mask']
        if 'env_reward' in data.batch.keys():
            reward_tensor += data.batch['env_reward']
            print(f' [DEBUG reward] mean={reward_tensor.mean().item()}, min={reward_tensor.min().item()}, max={reward_tensor.max().item()}')

        already_print_data_sources = {}

        # Build payloads sequentially to preserve order and avoid race in decoding
        payloads = []  # list of tuples: (data_source, response_str, ground_truth, extra_info)
        records = []   # list of tuples for ordered aggregation: (i, valid_response_length, response_str, ground_truth, data_source)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            attn = data_item.batch["attention_mask"]
            valid_prompt_length = int(attn[:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(attn[prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            # decode (keep prompt_str for parity though not printed by default)
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            payloads.append((data_source, response_str, ground_truth, extra_info))
            records.append((i, valid_response_length, response_str, ground_truth, data_source))

        # Define scoring function for threads
        def _score_one(p):
            ds, resp, gt, extra = p
            return self.compute_score(
                data_source=ds,
                solution_str=resp,
                ground_truth=gt,
                extra_info=extra,
            )

        # Parallel compute with order-preserving map
        num_workers = int(os.getenv("REWARD_MANAGER_WORKERS", "8"))
        if num_workers > 1 and len(payloads) > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as ex:
                results = list(ex.map(_score_one, payloads))
        else:
            results = [_score_one(p) for p in payloads]

        # Aggregate sequentially to keep behavior identical and maintain order
        for (i, vrl, resp_str, gt, ds), score in zip(records, results):
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            if vrl > 0:
                reward_tensor[i, vrl - 1] += reward

            # eos_idx = torch.nonzero(action_or_attn_mask[i, prompt_length: prompt_length + valid_response_length])[-1]
            # reward_tensor[i, eos_idx] = score

            if ds not in already_print_data_sources:
                already_print_data_sources[ds] = 0

            if already_print_data_sources[ds] < self.num_examine:
                already_print_data_sources[ds] += 1
                print(f"{len(data)=}")
                # print("[prompt]", prompt_str)
                print("[response]", resp_str)
                print("[ground_truth]", gt)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

            self.step_cnt += 1

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
