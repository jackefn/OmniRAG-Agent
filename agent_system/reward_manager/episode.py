# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from verl import DataProto
import torch
import numpy as np
import re

class EpisodeRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, normalize_by_length=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length
        self._re_search_legacy = re.compile(r"<search>.*?</search>", re.IGNORECASE | re.DOTALL)
        self._re_search_image = re.compile(r"<search_image>.*?</search_image>", re.IGNORECASE | re.DOTALL)
        self._re_search_audio = re.compile(r"<search_audio>.*?</search_audio>", re.IGNORECASE | re.DOTALL)
        self._re_think = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
        self._re_answer = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
        self._re_option = re.compile(r"^([A-Z])(?:[\\.:\\)])?", re.IGNORECASE)

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        format_rewards: list[float] = []
        answer_rewards: list[float] = []
        total_rewards: list[float] = []

        traj_debug_records = {} if self.num_examine < 0 else None

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')

            trajectory_outputs = data_item.non_tensor_batch.get('trajectory_outputs')
            if trajectory_outputs is None:
                trajectory_outputs = []
            full_outputs: list[str] = []
            answer_text = ""
            for output in trajectory_outputs:
                if "<answer>" in output and "</answer>" in output:
                    end_idx = output.lower().find("</answer>") + len("</answer>")
                    full_outputs.append(output[:end_idx])
                    answer_match = self._re_answer.search(output)
                    if answer_match:
                        answer_text = answer_match.group(1).strip()
                    break
                full_outputs.append(output)

            if not full_outputs:
                full_outputs = [response_str]
                if not answer_text:
                    answer_match = self._re_answer.search(response_str)
                    if answer_match:
                        answer_text = answer_match.group(1).strip()

            search_step_count = 0
            image_step_count = 0
            audio_step_count = 0
            has_valid_answer_step = False
            for idx, step_text in enumerate(full_outputs):
                has_think = bool(self._re_think.search(step_text))
                has_search_image = bool(self._re_search_image.search(step_text))
                has_search_audio = bool(self._re_search_audio.search(step_text))
                has_search_legacy = bool(self._re_search_legacy.search(step_text))
                has_answer = bool(self._re_answer.search(step_text))
                if idx == len(full_outputs) - 1 and has_answer:
                    if has_think and has_answer:
                        has_valid_answer_step = True
                else:
                    if has_think and (has_search_image or has_search_audio or has_search_legacy):
                        search_step_count += 1
                        if has_search_image:
                            image_step_count += 1
                        if has_search_audio:
                            audio_step_count += 1

            format_reward_pairs = search_step_count + (1 if has_valid_answer_step else 0)
            format_reward = 0.5 * format_reward_pairs
            if format_reward > 1.0:
                format_reward = 1.0

            correct_option = data_item.non_tensor_batch.get('correct_option')
            if correct_option is None:
                reward_gt = data_item.non_tensor_batch.get('reward_model_ground_truth') or {}
                correct_option = reward_gt.get('correct_option')
            if correct_option is None:
                reward_gt = data_item.non_tensor_batch.get('reward_model') or {}
                correct_option = reward_gt.get('correct_option')

            accuracy_reward = 0.0
            if answer_text and correct_option is not None:
                answer_norm = self._normalize_option(answer_text)
                correct_norm = self._normalize_option(str(correct_option))
                if answer_norm and correct_norm and answer_norm == correct_norm:
                    accuracy_reward = 1.0
            if format_reward >= 0.5:
                total_reward = -1.0 + format_reward + accuracy_reward
            else:
                total_reward = -1.0 + format_reward

            format_rewards.append(format_reward)
            answer_rewards.append(accuracy_reward)
            total_rewards.append(total_reward)

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = torch.tensor(
                    total_reward, dtype=torch.float32, device=reward_tensor.device
                )

            if traj_debug_records is not None:
                traj_uid = data_item.non_tensor_batch.get("traj_uid", "unknown")
                traj_debug_records[traj_uid] = {
                    "data_source": data_source,
                    "prompt": prompt_str,
                    "steps": list(full_outputs),
                    "search_image_steps": image_step_count,
                    "search_audio_steps": audio_step_count,
                    "format_reward": format_reward,
                    "answer_reward": accuracy_reward,
                    "total_reward": total_reward,
                }

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][format_reward]", format_reward)
                print(f"[{data_source}][answer_reward]", accuracy_reward)
                print(f"[{data_source}][total_reward]", total_reward)

        if traj_debug_records is not None:
            for traj_uid, record in traj_debug_records.items():
                print("=" * 80)
                print(f"[traj_uid]={traj_uid} [data_source]={record['data_source']}")
                print("[prompt]\n" + record["prompt"])
                for step_idx, step_text in enumerate(record["steps"]):
                    print(f"[step {step_idx}]\n{step_text}")
                print(
                    f"[format_reward]={record['format_reward']} [answer_reward]={record['answer_reward']} [total_reward]={record['total_reward']}"
                )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "format_reward": np.array(format_rewards, dtype=np.float32),
                    "answer_reward": np.array(answer_rewards, dtype=np.float32),
                    "total_reward": np.array(total_rewards, dtype=np.float32),
                },
            }
        else:
            return reward_tensor

    def _normalize_option(self, text: str) -> str:
        if not text:
            return ""
        text = str(text).strip()
        match = self._re_option.match(text)
        if match:
            return match.group(1).upper()
        return text.lower()
