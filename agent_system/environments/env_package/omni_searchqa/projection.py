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

from typing import List, Tuple, Optional
import re


_INVALID_PREFIX = "__INVALID__:"


def _postprocess_action(action: str) -> str:
    if "</search>" in action:
        return action.split("</search>", 1)[0] + "</search>"
    if "</search_image>" in action:
        return action.split("</search_image>", 1)[0] + "</search_image>"
    if "</search_audio>" in action:
        return action.split("</search_audio>", 1)[0] + "</search_audio>"
    if "</answer>" in action:
        return action.split("</answer>", 1)[0] + "</answer>"
    return action


def omni_searchqa_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    re_search_image = re.compile(r"<search_image>(.*?)</search_image>", re.IGNORECASE | re.DOTALL)
    re_search_audio = re.compile(r"<search_audio>(.*?)</search_audio>", re.IGNORECASE | re.DOTALL)
    re_answer = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
    re_think = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)

    for i, action in enumerate(actions):
        original_action = action or ""
        trimmed_action = _postprocess_action(original_action)

        image_matches = re_search_image.findall(trimmed_action)
        audio_matches = re_search_audio.findall(trimmed_action)
        answer_matches = re_answer.findall(trimmed_action)
        think_matches = re_think.findall(trimmed_action)

        reason: Optional[str] = None

        if think_matches:
            if len(think_matches) != 1:
                reason = "multiple_think_tags"
            elif not think_matches[0].strip():
                reason = "empty_think_tag"
        else:
            if image_matches or audio_matches or answer_matches:
                reason = "missing_think_tag"

        if reason is None:
            action_count = sum(
                [
                    len(image_matches),
                    len(audio_matches),
                    len(answer_matches),
                ]
            )
            if action_count != 1:
                reason = "multiple_action_tags" if action_count > 1 else "missing_action_tag"

        if reason is None and not (image_matches or audio_matches or answer_matches):
            if think_matches:
                reason = "missing_action_tag"
            elif trimmed_action.strip():
                reason = "missing_required_tags"
            else:
                reason = "empty_action"

        if reason is not None:
            results.append(f"{_INVALID_PREFIX}{reason}")
            valids[i] = 0
            continue

        if image_matches:
            query = image_matches[0].strip()
            if not query:
                results.append(f"{_INVALID_PREFIX}empty_search_query")
                valids[i] = 0
            else:
                results.append(f"SEARCH_IMAGE: {query}")
            continue

        if audio_matches:
            query = audio_matches[0].strip()
            if not query:
                results.append(f"{_INVALID_PREFIX}empty_search_query")
                valids[i] = 0
            else:
                results.append(f"SEARCH_AUDIO: {query}")
            continue

        if answer_matches:
            answer = answer_matches[0].strip()
            if not answer:
                results.append(f"{_INVALID_PREFIX}empty_answer")
                valids[i] = 0
            else:
                results.append(f"ANSWER: {answer}")
            continue

        # Fallback – should not be reached because of earlier checks, but keep for safety.
        results.append(f"{_INVALID_PREFIX}unknown_format")
        valids[i] = 0

    return results, valids
