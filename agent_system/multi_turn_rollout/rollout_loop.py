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

import torch
import numpy as np
import re
import ast
from PIL import Image
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.dataset.audio_utils import process_audio

class TrajectoryCollector:
    DEFAULT_SYSTEM_PROMPT = (
        "You are an agent for Audio/Video QA with retrieval tools over video frames and audio clips.\n"
        "You must act in iterative steps and choose exactly ONE action per turn.\n\n"
        "Output format rules:\n"
        "- Every turn, you MUST output exactly TWO XML-like tags in this order:\n"
        "  1) <think>...</think>\n"
        "  2) Exactly one of <search_image>...</search_image>, <search_audio>...</search_audio>, or <answer>...</answer>\n"
        "- The <think> tag is for reasoning.\n"
        "- <search_image> must contain a natural language query for IMAGE retrieval ONLY.\n"
        "- <search_audio> must contain a natural language query for AUDIO retrieval ONLY.\n"
        "- The <answer> tag must contain EXACTLY ONE option letter/word from the provided options ONLY.\n"
        "- Do NOT output any JSON, markdown, code fences, or extra text outside the tags.\n\n"
        "Allowed actions:\n"
        "1) <search_image>your image query</search_image>\n"
        "2) <search_audio>your audio query</search_audio>\n"
        "3) <answer>ONE_OPTION</answer>\n\n"
        "Rules:\n"
        "- The FIRST turn must use <search_image> or <search_audio>; do NOT answer immediately.\n"
        "- Prefer gathering evidence via the search actions before answering.\n"
        "- The user will provide: question, options, and videos.\n"
        "- When you answer, you must choose exactly one of the provided options (one word/letter)."
    )
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_video_frames = None
        if config is not None and hasattr(config, "env"):
            omni_cfg = config.env.get("omni_searchqa", {})
            max_frames = omni_cfg.get("max_video_frames")
            if max_frames is not None:
                try:
                    self.max_video_frames = int(max_frames)
                except (TypeError, ValueError):
                    self.max_video_frames = None

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        extra_keys = [
            "id",
            "original_id",
            "task_id",
            "question",
            "answer",
            "answer_text",
            "target",
            "options",
            "correct_option",
            "reward_model_ground_truth",
            "reward_model",
        ]
        extra_non_tensor = {}
        for key in extra_keys:
            if key in gen_batch.non_tensor_batch:
                extra_non_tensor[key] = gen_batch.non_tensor_batch[key][item]
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_retrieved = obs.get('retrieved_media', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        retrieved_media = obs_retrieved[item] if obs_retrieved is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ""
        if obs_text is not None:
            obs_content = str(obs_text)
        else:
            print(f"Warning: No text observation found!")

        multi_modal_data = None
        if "multi_modal_data" in gen_batch.non_tensor_batch:
            multi_modal_data = gen_batch.non_tensor_batch["multi_modal_data"][item]
            if isinstance(multi_modal_data, np.ndarray):
                multi_modal_data = multi_modal_data.item()
            if isinstance(multi_modal_data, dict):
                copied = {}
                for k, v in multi_modal_data.items():
                    if isinstance(v, np.ndarray) and v.dtype == object:
                        copied[k] = list(v)
                    elif isinstance(v, (list, tuple)):
                        copied[k] = list(v)
                    else:
                        copied[k] = v
                multi_modal_data = copied

        if retrieved_media:
            retrieved_images = retrieved_media.get("image") or []
            retrieved_audios = retrieved_media.get("audio") or []
        else:
            retrieved_images = []
            retrieved_audios = []

        if retrieved_images or retrieved_audios:
            if multi_modal_data is None:
                multi_modal_data = {}
            if retrieved_images:
                loaded_images = []
                for path in retrieved_images:
                    try:
                        with Image.open(path) as img:
                            loaded_images.append(img.convert("RGB").resize((224, 224), Image.BICUBIC))
                    except Exception as exc:
                        print(f"[WARN] failed to load retrieved image {path}: {exc}")
                if loaded_images:
                    multi_modal_data["image"] = (multi_modal_data.get("image") or []) + loaded_images
            if retrieved_audios:
                loaded_audios = []
                for path in retrieved_audios:
                    try:
                        audio_arr = process_audio(path, is_video=False)
                        if isinstance(audio_arr, np.ndarray) and audio_arr.size == 0:
                            print(f"[WARN] skip retrieved audio {path}: decoded length is 0")
                            continue
                        loaded_audios.append(audio_arr)
                    except Exception as exc:
                        print(f"[WARN] failed to load retrieved audio {path}: {exc}")
                if loaded_audios:
                    multi_modal_data["audio"] = (multi_modal_data.get("audio") or []) + loaded_audios

        if multi_modal_data:
            def _normalize_mm_list(value):
                if value is None:
                    return []
                if isinstance(value, np.ndarray) and value.dtype == object:
                    return list(value)
                if isinstance(value, (list, tuple)):
                    return list(value)
                return [value]

            def _limit_placeholders(text, token, desired):
                count = text.count(token)
                if count > desired:
                    remove_num = count - desired
                    for _ in range(remove_num):
                        idx = text.rfind(token)
                        if idx < 0:
                            break
                        text = text[:idx] + text[idx + len(token):]
                    return text
                if count < desired:
                    extra = desired - count
                    suffix = "\n".join([token] * extra)
                    if not text:
                        return suffix
                    if text.endswith("\n"):
                        return text + suffix
                    return text + "\n" + suffix
                return text

            multi_modal_data["image"] = _normalize_mm_list(multi_modal_data.get("image"))
            multi_modal_data["video"] = _normalize_mm_list(multi_modal_data.get("video"))
            multi_modal_data["audio"] = _normalize_mm_list(multi_modal_data.get("audio"))

            image_count = len(multi_modal_data.get("image") or [])
            video_count = len(multi_modal_data.get("video") or [])
            audio_count = len(multi_modal_data.get("audio") or [])
            obs_content = _limit_placeholders(obs_content, "<image>", image_count)
            obs_content = _limit_placeholders(obs_content, "<video>", video_count)
            obs_content = _limit_placeholders(obs_content, "<audio>", audio_count)

        user_content = obs_content
        if "<image>" in obs_content or "<video>" in obs_content or "<audio>" in obs_content:
            parts = re.split(r"(<image>|<video>|<audio>)", obs_content)
            content_list = []
            for part in parts:
                if not part:
                    continue
                if part == "<image>":
                    content_list.append({"type": "image"})
                elif part == "<video>":
                    content_list.append({"type": "video"})
                elif part == "<audio>":
                    content_list.append({"type": "audio"})
                else:
                    content_list.append({"type": "text", "text": part})
            user_content = content_list

        chat = np.array(
            [
                {"content": self.DEFAULT_SYSTEM_PROMPT, "role": "system"},
                {"content": user_content, "role": "user"},
            ]
        )
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Initialize return dict
        row_dict = {}
        
        raw_prompt = prompt_with_chat_template
        # Process multimodal data
        if multi_modal_data and self.processor is not None:
            if self.max_video_frames:
                trimmed_videos = []
                for v in (multi_modal_data.get("video") or []):
                    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > self.max_video_frames:
                        trimmed_videos.append(v[: self.max_video_frames])
                    else:
                        trimmed_videos.append(v)
                if trimmed_videos:
                    multi_modal_data["video"] = trimmed_videos
            images = multi_modal_data.get("image") or []
            videos = multi_modal_data.get("video") or []
            audios = multi_modal_data.get("audio") or []
            images = None if len(images) == 0 else images
            videos = None if len(videos) == 0 else videos
            audios = None if len(audios) == 0 else audios
            model_inputs = self.processor(
                text=[prompt_with_chat_template], images=images, videos=videos, audio=audios, return_tensors="pt"
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)
        elif is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        if multi_modal_data and self.processor is not None:
            input_ids = input_ids[0]
            attention_mask = attention_mask[0]
        else:
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )
        
        

        if multi_modal_data and self.processor is not None:
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=row_dict["multi_modal_inputs"].get("image_grid_thw"),
                    video_grid_thw=row_dict["multi_modal_inputs"].get("video_grid_thw"),
                    second_per_grid_ts=row_dict["multi_modal_inputs"].get("second_per_grid_ts"),
                    attention_mask=attention_mask,
                )
            ]
        elif is_multi_modal:

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask[0],
                )
              ]  # (1, 3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        if isinstance(attention_mask, torch.Tensor):
            total_tokens = int(attention_mask.sum().item())
        else:
            total_tokens = sum(attention_mask)
        text_tokens = len(raw_prompt_ids)
        mm_tokens = max(total_tokens - text_tokens, 0)
        # print(f"[TOKEN_STATS][idx={item}] total={total_tokens} text={text_tokens} mm={mm_tokens}")
        row_dict.update({
            'input_ids': input_ids[0] if isinstance(input_ids, torch.Tensor) and input_ids.dim() > 1 else input_ids,
            'attention_mask': attention_mask[0] if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() > 1 else attention_mask,
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        row_dict.update(extra_non_tensor)
        
        return row_dict

    def _build_env_kwargs_from_batch(self, gen_batch: DataProto) -> list[dict]:
        batch_size = len(gen_batch.batch["input_ids"])
        non_tensor = gen_batch.non_tensor_batch
        env_kwargs = []

        def _unwrap(value):
            if isinstance(value, np.ndarray):
                try:
                    return value.item()
                except ValueError:
                    return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            return value

        def _get(key: str, idx: int):
            if key in non_tensor:
                return _unwrap(non_tensor[key][idx])
            return None
        
        def _coerce_prompt_obj(raw_prompt):
            if isinstance(raw_prompt, str):
                raw_strip = raw_prompt.strip()
                if raw_strip.startswith("{") or raw_strip.startswith("["):
                    try:
                        return ast.literal_eval(raw_prompt)
                    except (ValueError, SyntaxError):
                        return raw_prompt
            return raw_prompt

        def _serialize_raw_prompt(raw_prompt):
            if not raw_prompt:
                return ""
            raw_prompt = _coerce_prompt_obj(raw_prompt)
            if isinstance(raw_prompt, str):
                return raw_prompt
            if isinstance(raw_prompt, dict):
                raw_prompt = [raw_prompt]
            if isinstance(raw_prompt, list):
                lines = []
                for msg in raw_prompt:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        parts = []
                        for seg in content:
                            seg_type = seg.get("type")
                            if seg_type == "text":
                                parts.append(seg.get("text", ""))
                            elif seg_type in ("image", "video", "audio"):
                                parts.append(f"<{seg_type}>")
                        content = "".join(parts)
                    lines.append(f"{role}: {content}".strip())
                return "\n".join(lines)
            return str(raw_prompt)

        def _extract_text_from_raw_prompt(raw_prompt):
            if not raw_prompt:
                return ""
            raw_prompt = _coerce_prompt_obj(raw_prompt)
            if isinstance(raw_prompt, str):
                return raw_prompt
            if isinstance(raw_prompt, dict):
                raw_prompt = [raw_prompt]
            if isinstance(raw_prompt, list):
                parts = []
                for msg in raw_prompt:
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for seg in content:
                            if seg.get("type") == "text":
                                parts.append(seg.get("text", ""))
                    elif isinstance(content, str):
                        parts.append(content)
                return " ".join(p.strip() for p in parts if p.strip())
            return str(raw_prompt)

        for i in range(batch_size):
            extra_info = _get("extra_info", i) or {}
            reward_ground_truth = _get("reward_model_ground_truth", i) or _get("reward_model", i) or {}
            raw_prompt = _get("raw_prompt", i)
            question = _get("question", i) or _get("query", i) or extra_info.get("question") or ""
            if not question or isinstance(question, (dict, list)):
                question = _extract_text_from_raw_prompt(raw_prompt)
            target = _get("target", i)
            answer = _get("answer", i) or _get("answer_text", i) or reward_ground_truth.get("answer_text")
            options = reward_ground_truth.get("multi_choice") or _get("options", i) or []
            correct_option = reward_ground_truth.get("correct_option") or _get("correct_option", i)
            task_id = _get("task_id", i) or _get("id", i) or ""
            raw_id = _get("id", i)
            original_id = _get("original_id", i)
            data_source = _get("data_source", i) or "unknown"
            prompt_text = _extract_text_from_raw_prompt(raw_prompt)
            env_kwargs.append(
                {
                    "task_id": task_id,
                    "id": raw_id,
                    "original_id": original_id,
                    "question": question,
                    "target": target if target is not None else answer,
                    "answer": answer,
                    "data_source": data_source,
                    "prompt": prompt_text,
                    "options": options,
                    "correct_option": correct_option,
                }
            )
        return env_kwargs

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
        
        if len(effective_batch) == 0:
            print("[WARN] effective_batch is empty in gather_rollout_data")
        else:
            has_tensor = any(isinstance(v, torch.Tensor) for v in effective_batch[0].values())
            if not has_tensor:
                key_types = {k: type(v).__name__ for k, v in effective_batch[0].items()}
                print(f"[WARN] effective_batch[0] has no tensor values, key_types={key_types}")

        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)
        # Initial observations from the environment
        env_kwargs = gen_batch.non_tensor_batch.pop('env_kwargs', None)
        if env_kwargs is None and "omni_searchqa" in self.config.env.env_name.lower():
            env_kwargs = self._build_env_kwargs_from_batch(gen_batch)
        obs, infos = envs.reset(kwargs=env_kwargs)
        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        has_answered = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        trajectory_outputs = [[] for _ in range(batch_size)]
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info
            # print("DEBUG batch_input:",batch_input.batch.keys())
            # print("DEBUG batch_input non tensor batch keys:",batch_input.non_tensor_batch.keys())
            # try:
            #     prompts = self.tokenizer.batch_decode(
            #         batch_input.batch["input_ids"], skip_special_tokens=False
            #     )
            #     for i, prompt in enumerate(prompts):
            #        print(f"[LLM][step={_step}][idx={i}] prompt:\n{prompt}")
            # except Exception:
            #     pass
            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            # try:
            #     for i, action in enumerate(text_actions):
            #         print(f"[LLM][step={_step}][idx={i}] response:\n{action}")
            # except Exception:
            #     pass
            has_answer = np.array(
                [("<answer>" in action and "</answer>" in action) for action in text_actions],
                dtype=bool,
            )
            has_answered = np.logical_or(has_answered, has_answer)

            for i, action in enumerate(text_actions):
                trajectory_outputs[i].append(action)

            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            batch.non_tensor_batch['trajectory_outputs'] = np.array(trajectory_outputs, dtype=object)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
            # Update observations for next step
            obs = next_obs
            # Break if all environments are done
            if is_done.all():
                break
            if np.all(has_answered):
                break
        for i, outputs in enumerate(trajectory_outputs):
            joined = "\n".join(outputs)
            # print(f"[TRAJECTORY][idx={i}] outputs:\n{joined}")
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        return gen_batch_output
