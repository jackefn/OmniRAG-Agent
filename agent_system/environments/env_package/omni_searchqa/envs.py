import asyncio
import concurrent.futures
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, ListConfig

from .retriever_client import OmniRetrieverClient


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


class OmniSearchQAEnv:
    def __init__(
        self,
        retriever: OmniRetrieverClient,
        max_turns: int,
        retriever_root: Optional[str] = None,
    ) -> None:
        self.retriever = retriever
        self.max_turns = max_turns
        self.retriever_root = retriever_root
        self.reset({})

    @staticmethod
    def _derive_video_id(extras: Dict[str, Any]) -> Optional[str]:
        video_id = extras.get("video_id") or extras.get("original_id") or extras.get("id")
        if isinstance(video_id, str) and "__" in video_id:
            return video_id.split("__", 1)[0]
        videos = extras.get("videos") or []
        if videos:
            video_path = videos[0].get("video")
            if isinstance(video_path, str) and video_path:
                base = os.path.splitext(os.path.basename(video_path))[0]
                if "_t" in base:
                    return base.split("_t", 1)[0]
                return base
        return video_id if isinstance(video_id, str) else None

    def reset(self, extras: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        self.question = str(extras.get("question", ""))
        self.target = extras.get("target")
        if self.target is None:
            self.target = extras.get("answer", "")
        self.task_id = extras.get("task_id") or extras.get("id") or ""
        self.data_source = extras.get("data_source", "unknown")
        self.prompt = extras.get("prompt", "")
        self.options = extras.get("options") or []
        self.correct_option = extras.get("correct_option")
        self.video_id = self._derive_video_id(extras)
        self.step_count = 0
        self.done = False
        return self.question, {
            "task_id": self.task_id,
            "data_source": self.data_source,
            "prompt": self.prompt,
            "options": self.options,
            "correct_option": self.correct_option,
        }

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self.done:
            return "", 0.0, True, {"won": False, "task_id": self.task_id}

        self.step_count += 1
        reward = 0.0
        done = False

        retrieved_media = {"image": [], "audio": []}
        if action.startswith("SEARCH_IMAGE:"):
            query = action[len("SEARCH_IMAGE:") :].strip()
            obs, retrieved_media = self._search_and_format(query, modality="image")
        elif action.startswith("SEARCH_AUDIO:"):
            query = action[len("SEARCH_AUDIO:") :].strip()
            obs, retrieved_media = self._search_and_format(query, modality="audio")
        elif action.startswith("SEARCH:"):
            # Backward compatibility for older checkpoints.
            query = action[len("SEARCH:") :].strip()
            obs, retrieved_media = self._search_and_format(query, modality="both")
        elif action.startswith("ANSWER:"):
            answer = action[len("ANSWER:") :].strip()
            reward = 1.0 if self._is_correct(answer) else 0.0
            done = True
            obs = "Answer submitted."
        else:
            obs = "Invalid action format. Use SEARCH_IMAGE(...), SEARCH_AUDIO(...), or ANSWER(...)."

        if self.step_count >= self.max_turns and not done:
            done = True

        self.done = done
        invalid_reason = None
        if action.startswith("__INVALID__:"):
            invalid_reason = action[len("__INVALID__:") :].strip()
            obs = self._format_invalid_message(invalid_reason)

        info = {
            "won": bool(done and reward >= 1.0),
            "task_id": self.task_id,
            "data_source": self.data_source,
            "retrieved_media": retrieved_media,
            "invalid_reason": invalid_reason,
        }
        return obs, reward, done, info

    @staticmethod
    def _format_invalid_message(reason: str) -> str:
        mapping = {
            "missing_think_tag": "Each action must include a <think>...</think> block before <search_image>/<search_audio>/<answer>.",
            "multiple_think_tags": "Provide exactly one <think>...</think> block per turn.",
            "empty_think_tag": "The <think>...</think> block cannot be empty.",
            "multiple_action_tags": "Output must contain exactly one of <search_image>...</search_image>, <search_audio>...</search_audio>, or <answer>...</answer> per turn.",
            "missing_action_tag": "After <think>, include one of <search_image>...</search_image>, <search_audio>...</search_audio>, or <answer>...</answer>.",
            "missing_required_tags": "Use the required XML tags: first <think>, then <search_image>/<search_audio>/<answer>.",
            "empty_action": "Action is empty. Follow the required tag format with content inside.",
            "empty_search_query": "The search tag must contain a non-empty natural language query.",
            "empty_answer": "The <answer> tag must specify one option letter/word.",
            "unknown_format": "Invalid action format. Use <think> then <search_image>/<search_audio>/<answer> with the required content.",
        }
        return mapping.get(reason, "Invalid action format. Use <think>...</think> followed by <search_image>/<search_audio>/<answer>.")

    def _is_correct(self, answer: str) -> bool:
        if self.correct_option:
            return _normalize_text(answer) == _normalize_text(str(self.correct_option))
        return _normalize_text(answer) == _normalize_text(str(self.target))

    def _search_and_format(self, query: str, modality: str) -> Tuple[str, Dict[str, List[str]]]:
        retrieved_media = {"image": [], "audio": []}
        retriever_url = (self.retriever.retriever_url or "").rstrip("/")
        if retriever_url.endswith("/query") or modality == "both":
            results, err = self.retriever.search(query, video_id=self.video_id)
            lines = [f"Search results for query: {query}"]
            if err:
                lines.append(f"retrieval error: {err}")
            if results:
                self._format_results(results, retrieved_media, target="both")
            return "\n".join(lines), retrieved_media

        lines = [f"Search results for query: {query}"]
        if modality == "image":
            image_results, image_err = self.retriever.search_image(query, video_id=self.video_id)
            if image_err:
                lines.append(f"image retrieval error: {image_err}")
            if image_results:
                self._format_results(image_results, retrieved_media, target="image")
        elif modality == "audio":
            audio_results, audio_err = self.retriever.search_audio(query, video_id=self.video_id)
            if audio_err:
                lines.append(f"audio retrieval error: {audio_err}")
            if audio_results:
                self._format_results(audio_results, retrieved_media, target="audio")

        return "\n".join(lines), retrieved_media

    def _format_results(
        self,
        results: List[Dict[str, Any]],
        retrieved_media: Dict[str, List[str]],
        target: str,
    ) -> List[str]:
        lines = []
        for idx, item in enumerate(results, start=1):
            segment = item.get("segment") or {}
            item_id = (
                segment.get("segment_id")
                or item.get("id")
                or item.get("_id")
                or ""
            )
            frame = segment.get("frame") or item.get("path") or item.get("file_path") or ""
            audio = segment.get("audio") or ""
            score = item.get("score")
            caption = item.get("caption") or item.get("text") or ""
            transcript = item.get("transcript") or segment.get("transcript") or ""
            t0 = segment.get("t0")
            t1 = segment.get("t1")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score or "")
            frame = self._resolve_media_path(frame)
            audio = self._resolve_media_path(audio)
            if frame and target in {"image", "both"}:
                retrieved_media["image"].append(frame)
            if audio and target in {"audio", "both"}:
                retrieved_media["audio"].append(audio)
            time_str = ""
            if t0 is not None and t1 is not None:
                time_str = f" t={t0}-{t1}"
            if target == "image":
                line = f"{idx}. id={item_id}{time_str} frame={frame} score={score_str} caption={caption}".strip()
            elif target == "audio":
                line = f"{idx}. id={item_id}{time_str} audio={audio} score={score_str}".strip()
                if transcript:
                    line += f" transcript={transcript}"
            else:
                line = (
                    f"{idx}. id={item_id}{time_str} frame={frame} audio={audio} score={score_str} caption={caption}".strip()
                )
                if transcript:
                    line += f" transcript={transcript}"
            lines.append(line)
        return lines

    def _resolve_media_path(self, path: str) -> str:
        if not path:
            return ""
        if os.path.isabs(path):
            return path
        if self.retriever_root and self.video_id:
            base = os.path.join(self.retriever_root, self.video_id)
            candidate = os.path.abspath(os.path.join(base, "segments", path))
            if os.path.exists(candidate):
                return candidate
            candidate = os.path.abspath(os.path.join(base, path))
            if os.path.exists(candidate):
                return candidate
        if self.retriever_root:
            candidate = os.path.abspath(os.path.join(self.retriever_root, "segments", path))
            if os.path.exists(candidate):
                return candidate
            candidate = os.path.abspath(os.path.join(self.retriever_root, path))
            if os.path.exists(candidate):
                return candidate
        return os.path.abspath(path)


class OmniSearchQAMultiProcessEnv:
    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config: Optional[DictConfig] = None,
    ) -> None:
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.max_steps = env_config.max_steps

        omni_cfg = env_config.get("omni_searchqa", {})
        retriever_url = omni_cfg.get("retriever_url")
        image_index = omni_cfg.get("image_index")
        audio_index = omni_cfg.get("audio_index")
        topk = omni_cfg.get("topk", 3)
        timeout = omni_cfg.get("timeout", 30)
        log_requests = omni_cfg.get("log_requests", False)
        retriever_root = omni_cfg.get("retriever_root")
        if not retriever_root:
            default_root = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "retrival_api",
                "rag_db",
                "videos",
            )
            if os.path.exists(default_root):
                retriever_root = default_root

        self.envs: List[OmniSearchQAEnv] = []
        for _ in range(self.batch_size):
            client = OmniRetrieverClient(
                retriever_url=retriever_url,
                image_index=image_index,
                audio_index=audio_index,
                topk=topk,
                timeout=timeout,
                log_requests=log_requests,
            )
            self.envs.append(OmniSearchQAEnv(client, max_turns=self.max_steps, retriever_root=retriever_root))

        max_workers = min(self.batch_size, 256)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def _sync_reset(self, env: OmniSearchQAEnv, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return env.reset(kwargs)

    def _sync_step(self, env: OmniSearchQAEnv, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        return env.step(action)

    def reset(self, kwargs: List[Dict[str, Any]]):
        if len(kwargs) > self.batch_size:
            raise ValueError(f"Got {len(kwargs)} kwarg dicts, but total_envs={self.batch_size}")

        pad_n = self.batch_size - len(kwargs)
        padded_kwargs = list(kwargs) + [{}] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_reset, env, kw)
            for env, kw in zip(self.envs, padded_kwargs)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        obs_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]
        return obs_list, info_list

    def step(self, actions: List[str]):
        if len(actions) > self.batch_size:
            raise ValueError(f"Got {len(actions)} actions, but total_envs={self.batch_size}")

        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_step, env, act)
            for env, act in zip(self.envs, padded_actions)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, reward_list, done_list, info_list = map(list, zip(*results))
        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list = [d for d, keep in zip(done_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]

        return obs_list, reward_list, done_list, info_list

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True

    def __del__(self):
        self.close()


def build_omni_searchqa_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config: Optional[DictConfig] = None,
):
    return OmniSearchQAMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
    )
