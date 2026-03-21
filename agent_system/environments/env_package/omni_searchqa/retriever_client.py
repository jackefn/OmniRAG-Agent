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

from typing import Any, Dict, List, Optional, Tuple
import logging
import requests

logger = logging.getLogger(__name__)


class OmniRetrieverClient:
    def __init__(
        self,
        retriever_url: Optional[str],
        image_index: Optional[str],
        audio_index: Optional[str],
        topk: int = 3,
        timeout: int = 30,
        log_requests: bool = False,
    ) -> None:
        self.retriever_url = retriever_url
        self.image_index = image_index
        self.audio_index = audio_index
        self.topk = topk
        self.timeout = timeout
        self.log_requests = log_requests
        self.session = requests.Session()

    def search_image(self, query: str, video_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        return self._search(query=query, index=self.image_index, modality="image", video_id=video_id)

    def search_audio(self, query: str, video_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        return self._search(query=query, index=self.audio_index, modality="audio", video_id=video_id)

    def search(self, query: str, video_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        return self._search(query=query, index=None, modality="unified", video_id=video_id)

    def _search(
        self,
        query: str,
        index: Optional[str],
        modality: str,
        video_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        if not self.retriever_url:
            return [], "retriever_url is not set"
        if not query:
            return [], "empty query"

        base = self.retriever_url.rstrip("/")
        url: str

        if base.endswith("/query") or base.endswith("/query_audio"):
            # Legacy single-endpoint mode where the full path is provided.
            url = base
        else:
            if modality == "audio":
                url = f"{base}/query_audio"
            else:
                url = f"{base}/query"

        payload: Dict[str, Any] = {
            "query": query,
            "top_k": self.topk,
        }
        if video_id is not None:
            payload["video_id"] = video_id

        # Optional backward-compatibility fields for services that still expect index-based routing.
        if index and (base.endswith("/query") or base.endswith("/query_audio")):
            payload.setdefault("index", index)
            payload.setdefault("modality", modality)

        try:
            if self.log_requests:
                logger.info("OmniRetrieverClient request: %s url=%s", payload, url)
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            results = self._extract_results(response.json())
            return results, None
        except Exception as exc:
            return [], f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _extract_results(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("results", "data", "hits"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
        return []
