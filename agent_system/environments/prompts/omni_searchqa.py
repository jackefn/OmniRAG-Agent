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

OMNI_SEARCHQA_ACTIONS = (
    """- <think>...</think><search_image>your image query</search_image>
- <think>...</think><search_audio>your audio query</search_audio>
- <think>...</think><answer>ONE_OPTION</answer>"""
)

OMNI_SEARCHQA_TEMPLATE_NO_HIS = """You are a multimodal QA agent with access to image and audio retrieval.

Task:
{question}
{context_block}
{options_block}
Video: <video>
Audio: <audio>

Available actions:
{available_actions}

Return exactly one action line with one of the formats above.
"""

OMNI_SEARCHQA_TEMPLATE = """You are a multimodal QA agent with access to image and audio retrieval.

Task:
{question}
{context_block}
{options_block}
Video: <video>
Audio: <audio>

Available actions:
{available_actions}

Step {current_step} (history length: {history_length}):
{action_history}

Return exactly one action line with one of the formats above.
"""
