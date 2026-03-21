import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from embeddings import embed_text

try:
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    WhisperModel = None  # type: ignore

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class ASRResult:
    transcript: str
    language: Optional[str]


class ASRTranscriber:
    """Wrapper that prefers faster-whisper, with a lightweight fallback."""

    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        self._backend = "fallback"
        self._model: Optional[WhisperModel] = None
        self._fallback = None

        if WhisperModel is not None:
            if device is None:
                if torch is not None and torch.cuda.is_available():  # pragma: no branch - best effort
                    device = "cuda"
                else:
                    device = "cpu"
            if compute_type is None:
                compute_type = "float16" if device == "cuda" else "int8"
            self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self._backend = "faster-whisper"
        else:
            from video_ingest import audio_to_text  # Local fallback based on SpeechRecognition

            self._fallback = audio_to_text
            self._backend = "speech_recognition"

    def transcribe(self, audio_path: str) -> ASRResult:
        if self._backend == "faster-whisper" and self._model is not None:
            segments, info = self._model.transcribe(audio_path, beam_size=5)
            pieces: List[str] = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    pieces.append(text)
            transcript = " ".join(pieces).strip()
            lang = getattr(info, "language", None)
            return ASRResult(transcript=transcript, language=lang)

        if self._fallback is None:
            raise RuntimeError(
                "No ASR backend available. Please install 'faster-whisper' or ensure SpeechRecognition fallback works."
            )

        transcript = self._fallback(audio_path)
        transcript = (transcript or "").strip()
        return ASRResult(transcript=transcript, language=None)


def _load_segment_entries(meta_path: str) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "audio" not in entry:
                continue
            entries.append(entry)
    return entries


def _should_skip(audio_index_path: str, audio_meta_path: str, force: bool) -> bool:
    if force:
        return False
    if not os.path.exists(audio_index_path) or not os.path.exists(audio_meta_path):
        return False

    try:
        index = faiss.read_index(audio_index_path)
    except Exception:
        return False

    meta_lines = 0
    with open(audio_meta_path, "r", encoding="utf-8") as f:
        for _ in f:
            meta_lines += 1

    if meta_lines == 0:
        return False
    return index.ntotal == meta_lines


@dataclass
class BuildStats:
    video_id: str
    processed: int
    indexed: int
    skipped_audio: int
    missing_audio: int
    empty_transcript: int


def build_audio_index_for_video(
    videos_root: str,
    video_id: str,
    transcriber: ASRTranscriber,
    *,
    force: bool = False,
) -> Optional[BuildStats]:
    video_dir = os.path.join(videos_root, video_id)
    meta_path = os.path.join(video_dir, "meta.jsonl")
    segments_dir = os.path.join(video_dir, "segments")
    audio_index_path = os.path.join(video_dir, "audio_index.faiss")
    audio_meta_path = os.path.join(video_dir, "audio_meta.jsonl")

    if not os.path.exists(meta_path):
        print(f"[WARN] meta.jsonl missing for video_id={video_id}, skipping")
        return None
    if not os.path.isdir(segments_dir):
        print(f"[WARN] segments directory missing for video_id={video_id}, skipping")
        return None

    if _should_skip(audio_index_path, audio_meta_path, force=force):
        print(f"[SKIP] audio index already built: {video_id}")
        return BuildStats(video_id, processed=0, indexed=0, skipped_audio=0, missing_audio=0, empty_transcript=0)

    entries = _load_segment_entries(meta_path)
    if not entries:
        print(f"[WARN] No segment entries with audio for {video_id}")
        return BuildStats(video_id, processed=0, indexed=0, skipped_audio=0, missing_audio=0, empty_transcript=0)

    vectors: List[np.ndarray] = []
    audio_meta: List[Dict[str, object]] = []
    missing_audio = 0
    empty_transcript = 0

    for entry in entries:
        audio_rel = entry.get("audio")
        if not isinstance(audio_rel, str) or not audio_rel:
            continue
        audio_abs = os.path.join(segments_dir, audio_rel)
        if not os.path.exists(audio_abs):
            print(f"[WARN] audio file missing: {audio_abs}")
            missing_audio += 1
            continue

        asr = transcriber.transcribe(audio_abs)
        if not asr.transcript:
            empty_transcript += 1
            continue

        vector = embed_text(asr.transcript)
        vector = np.asarray(vector, dtype="float32")
        if vector.shape[0] != 512:
            raise ValueError(f"Unexpected embedding dimension {vector.shape[0]}, expected 512")
        vectors.append(vector)

        audio_meta.append(
            {
                "type": "asr",
                "segment_id": entry.get("segment_id"),
                "t0": entry.get("t0"),
                "t1": entry.get("t1"),
                "audio": audio_rel,
                "frame": entry.get("frame"),
                "transcript": asr.transcript,
            }
        )

    index = faiss.IndexFlatL2(512)
    if vectors:
        stacked = np.stack(vectors, axis=0)
        index.add(stacked)

    os.makedirs(video_dir, exist_ok=True)
    faiss.write_index(index, audio_index_path)

    with open(audio_meta_path, "w", encoding="utf-8") as f:
        for item in audio_meta:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"[OK] audio index built for {video_id}: {len(audio_meta)} items, "
        f"missing_audio={missing_audio}, empty_transcript={empty_transcript}"
    )

    return BuildStats(
        video_id=video_id,
        processed=len(entries),
        indexed=len(audio_meta),
        skipped_audio=0,
        missing_audio=missing_audio,
        empty_transcript=empty_transcript,
    )


def iter_video_ids(videos_root: str, specific_ids: Optional[List[str]]) -> List[str]:
    if specific_ids:
        return specific_ids
    if not os.path.isdir(videos_root):
        return []
    ids = []
    for name in os.listdir(videos_root):
        meta_path = os.path.join(videos_root, name, "meta.jsonl")
        if os.path.exists(meta_path):
            ids.append(name)
    ids.sort()
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audio ASR FAISS indexes for video segments")
    parser.add_argument(
        "--videos-root",
        default="rag_db/videos",
        help="Root directory containing per-video subdirectories",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        help="Process a specific video id (can be supplied multiple times)"
    )
    parser.add_argument(
        "--model-size",
        default="small",
        help="Whisper model size when using faster-whisper (default: small)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for faster-whisper (auto-detect if omitted)",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="faster-whisper compute type (auto if omitted)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild indexes even if they appear up-to-date",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos_root = args.videos_root
    video_ids = iter_video_ids(videos_root, args.video_id)
    if not video_ids:
        print(f"[INFO] No videos to process under {videos_root}")
        return

    transcriber = ASRTranscriber(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )

    total_indexed = 0
    for vid in video_ids:
        stats = build_audio_index_for_video(
            videos_root,
            vid,
            transcriber,
            force=args.force,
        )
        if stats is None:
            continue
        total_indexed += stats.indexed

    print(f"[DONE] audio ASR indexing complete. Total indexed segments={total_indexed}")


if __name__ == "__main__":
    main()
