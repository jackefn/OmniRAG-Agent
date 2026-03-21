import argparse
import json
import math
import os
import re
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np

from embeddings import embed_text

try:  # Prefer faster-whisper
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    WhisperModel = None  # type: ignore

try:
    import whisper as openai_whisper  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    openai_whisper = None  # type: ignore


LAUGH_PATTERNS = [
    re.compile(r"ha\s?ha", re.IGNORECASE),
    re.compile(r"ha{2,}", re.IGNORECASE),
    re.compile(r"哈哈+"),
    re.compile(r"呵呵+"),
]
CLOSING_PHRASES = [
    "thanks for watching",
    "thank you",
    "subscribe",
    "like and subscribe",
]
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$")
CHINESE_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
TRAILING_T_PATTERN = re.compile(r"_t\d+s$", re.IGNORECASE)

THREAD_LOCAL = threading.local()


@dataclass
class SpeechSegment:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class BuildStats:
    video_id: str
    raw_segments: int
    after_filter: int
    after_merge: int
    final_segments: int
    skipped_short: int
    skipped_noise: int
    skipped_duplicates: int
    errors: Optional[str] = None


class Transcriber:
    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:
        self.beam_size = beam_size
        self.vad_filter = vad_filter

        if WhisperModel is not None:
            model_device = device
            if model_device is None:
                if _torch_available() and _torch_cuda_available():  # pragma: no branch - best effort
                    model_device = "cuda"
                else:
                    model_device = "cpu"
            if compute_type is None:
                compute_type = "float16" if model_device == "cuda" else "int8"
            self.backend = "faster-whisper"
            self.model = WhisperModel(model_size, device=model_device, compute_type=compute_type)
        elif openai_whisper is not None:
            model_device = device or ("cuda" if _torch_cuda_available() else "cpu")
            self.backend = "openai-whisper"
            self.model = openai_whisper.load_model(model_size, device=model_device)
        else:  # pragma: no cover - requires environment
            raise RuntimeError("No ASR backend available. Install 'faster-whisper' or 'whisper'.")

    def transcribe(self, audio_path: str) -> List[SpeechSegment]:
        if self.backend == "faster-whisper":
            segments_iter, _ = self.model.transcribe(
                audio_path,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
            )
            segments: List[SpeechSegment] = []
            for seg in segments_iter:
                segments.append(SpeechSegment(start=float(seg.start), end=float(seg.end), text=seg.text or ""))
            return segments

        # openai whisper fallback
        result = self.model.transcribe(
            audio_path,
            beam_size=self.beam_size,
            verbose=False,
        )
        segments: List[SpeechSegment] = []
        for seg in result.get("segments", []):
            segments.append(
                SpeechSegment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=seg.get("text", ""),
                )
            )
        return segments


def _torch_available() -> bool:
    try:
        import torch  # type: ignore

        return True
    except Exception:  # pragma: no cover - torch missing
        return False


def _torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return torch.cuda.is_available()
    except Exception:  # pragma: no cover - torch missing
        return False


def get_transcriber(config: Tuple[str, Optional[str], Optional[str], int, bool]) -> Transcriber:
    model_size, device, compute_type, beam_size, vad_filter = config
    transcriber = getattr(THREAD_LOCAL, "transcriber", None)
    if transcriber is None:
        transcriber = Transcriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        THREAD_LOCAL.transcriber = transcriber
    return transcriber


def video_id_from_filename(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    name = TRAILING_T_PATTERN.sub("", name)
    return name


def audio_index_exists(out_dir: str, video_id: str) -> bool:
    video_dir = os.path.join(out_dir, video_id)
    audio_meta_path = os.path.join(video_dir, "audio_meta.jsonl")
    audio_index_path = os.path.join(video_dir, "audio_index.faiss")
    return os.path.exists(audio_meta_path) and os.path.exists(audio_index_path)


def run_ffmpeg(cmd: Sequence[str]) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\nSTDERR: {proc.stderr.strip()}")


def extract_full_audio(video_path: str, temp_dir: str) -> str:
    full_wav = os.path.join(temp_dir, "full.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        full_wav,
    ]
    run_ffmpeg(cmd)
    return full_wav


def sanitize_text(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_chinese(text: str) -> bool:
    return bool(CHINESE_CHAR_RE.search(text))


def is_punctuation_only(text: str) -> bool:
    return bool(PUNCT_ONLY_RE.match(text))


def is_noise_text(lower_text: str) -> bool:
    for pattern in LAUGH_PATTERNS:
        if pattern.search(lower_text):
            return True
    for phrase in CLOSING_PHRASES:
        if phrase in lower_text:
            return True
    return False


def passes_length_filter(text: str) -> Tuple[bool, bool]:
    """Return (passes_filter, is_short)."""
    compact = text.replace(" ", "")
    if contains_chinese(text):
        if len(compact) < 4:
            return False, True
        is_short = len(compact) < 6
    else:
        tokens = [tok for tok in text.split(" ") if tok]
        if len(tokens) < 2:
            return False, True
        is_short = len(tokens) < 3
    return True, is_short


def initial_filter(segments: Iterable[SpeechSegment]) -> Tuple[List[SpeechSegment], int, int]:
    kept: List[SpeechSegment] = []
    skipped_short = 0
    skipped_noise = 0
    for seg in segments:
        text = sanitize_text(seg.text)
        if not text or is_punctuation_only(text):
            skipped_noise += 1
            continue
        lower = text.lower()
        if is_noise_text(lower):
            skipped_noise += 1
            continue
        passes, _ = passes_length_filter(text)
        if not passes:
            skipped_short += 1
            continue
        kept.append(SpeechSegment(start=seg.start, end=seg.end, text=text))
    return kept, skipped_short, skipped_noise


def merge_segments(segments: Sequence[SpeechSegment]) -> List[SpeechSegment]:
    if not segments:
        return []
    merged: List[SpeechSegment] = []
    current = segments[0]
    for nxt in segments[1:]:
        gap = nxt.start - current.end
        combined_duration = nxt.end - current.start
        _, current_short = passes_length_filter(current.text)
        _, next_short = passes_length_filter(nxt.text)
        should_merge = False
        if gap <= 0.3 and combined_duration <= 4.0:
            should_merge = True
        elif current_short or next_short:
            if gap <= 0.6 and combined_duration <= 5.0:
                should_merge = True
        if should_merge:
            separator = "、" if (contains_chinese(current.text) or contains_chinese(nxt.text)) else " "
            if current.text.endswith(tuple(["，", "。", "、", ",", ".", "?", "!", "?", "!"])):
                separator = " "
            merged_text = sanitize_text(f"{current.text}{separator}{nxt.text}")
            current = SpeechSegment(start=current.start, end=nxt.end, text=merged_text)
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    return merged


def deduplicate_segments(segments: Sequence[SpeechSegment]) -> Tuple[List[SpeechSegment], int]:
    seen = set()
    unique: List[SpeechSegment] = []
    skipped = 0
    for seg in segments:
        text = sanitize_text(seg.text)
        if not text:
            continue
        norm_key = re.sub(r"\s+", " ", text).lower()
        if norm_key in seen:
            skipped += 1
            continue
        seen.add(norm_key)
        unique.append(SpeechSegment(start=seg.start, end=seg.end, text=text))
    return unique, skipped


def finalize_segments(segments: Sequence[SpeechSegment]) -> List[SpeechSegment]:
    final: List[SpeechSegment] = []
    for seg in segments:
        text = sanitize_text(seg.text)
        if not text:
            continue
        passes, _ = passes_length_filter(text)
        if not passes:
            continue
        final.append(SpeechSegment(start=seg.start, end=seg.end, text=text))
    return final


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(vectors)
    return vectors


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cut_segment_audio(full_audio: str, out_path: str, start: float, end: float) -> None:
    duration = max(0.0, end - start)
    if duration <= 0.01:
        raise ValueError("Segment duration too small to extract audio")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, start):.2f}",
        "-t",
        f"{duration:.2f}",
        "-i",
        full_audio,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        out_path,
    ]
    run_ffmpeg(cmd)


def embed_transcripts(transcripts: Sequence[str]) -> np.ndarray:
    vectors: List[np.ndarray] = []
    for text in transcripts:
        vec = embed_text(text)
        arr = np.asarray(vec, dtype="float32")
        if arr.ndim != 1:
            raise ValueError("embed_text must return a 1D vector")
        norm = np.linalg.norm(arr)
        if norm <= 1e-8:
            continue
        vectors.append(arr)
    if not vectors:
        return np.zeros((0, 0), dtype="float32")
    stacked = np.stack(vectors, axis=0)
    normalize_vectors(stacked)
    return stacked


def build_faiss_index(vectors: np.ndarray, dim: int, out_path: str) -> None:
    index = faiss.IndexFlatIP(dim)
    if vectors.size:
        if vectors.shape[1] != dim:
            raise ValueError("Vector dimension mismatch when building FAISS index")
        index.add(vectors)
    faiss.write_index(index, out_path)


def process_video(
    video_path: str,
    out_dir: str,
    transcriber_config: Tuple[str, Optional[str], Optional[str], int, bool],
    overwrite: bool,
    cut_wav: bool,
) -> BuildStats:
    video_id = video_id_from_filename(video_path)
    video_dir = os.path.join(out_dir, video_id)
    audio_meta_path = os.path.join(video_dir, "audio_meta.jsonl")
    audio_index_path = os.path.join(video_dir, "audio_index.faiss")
    audio_segments_dir = os.path.join(video_dir, "audio")

    if not overwrite and os.path.exists(audio_meta_path) and os.path.exists(audio_index_path):
        print(f"[SKIP] {video_id}: audio index already exists")
        return BuildStats(
            video_id=video_id,
            raw_segments=0,
            after_filter=0,
            after_merge=0,
            final_segments=0,
            skipped_short=0,
            skipped_noise=0,
            skipped_duplicates=0,
        )

    ensure_dir(video_dir)
    if cut_wav:
        ensure_dir(audio_segments_dir)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            full_audio = extract_full_audio(video_path, tmp_dir)
            transcriber = get_transcriber(transcriber_config)
            raw_segments = transcriber.transcribe(full_audio)
            raw_count = len(raw_segments)

            filtered_segments, skipped_short, skipped_noise = initial_filter(raw_segments)
            merged_segments = merge_segments(filtered_segments)
            merged_filtered = finalize_segments(merged_segments)
            final_segments, skipped_duplicates = deduplicate_segments(merged_filtered)

            transcripts = [seg.text for seg in final_segments]
            vectors = embed_transcripts(transcripts)
            if vectors.size == 0 and final_segments:
                raise RuntimeError("Failed to embed transcripts; received zero vectors")

            if vectors.size == 0:
                dim = len(embed_text("test"))
                vectors = np.zeros((0, dim), dtype="float32")
            dim = vectors.shape[1] if vectors.size else len(embed_text("dummy"))

            meta_lines: List[str] = []
            for idx, seg in enumerate(final_segments):
                segment_id = f"seg_{idx:06d}"
                rel_audio = f"audio/{segment_id}.wav"
                if cut_wav:
                    out_wav = os.path.join(audio_segments_dir, f"{segment_id}.wav")
                    cut_segment_audio(full_audio, out_wav, seg.start, seg.end)

                meta = {
                    "type": "asr",
                    "segment_id": segment_id,
                    "t0": round(seg.start, 2),
                    "t1": round(seg.end, 2),
                    "audio": rel_audio if cut_wav else None,
                    "frame": None,
                    "transcript": seg.text,
                }
                meta_lines.append(json.dumps(meta, ensure_ascii=False))

            with open(audio_meta_path, "w", encoding="utf-8") as f_meta:
                for line in meta_lines:
                    f_meta.write(line + "\n")

            if vectors.size:
                build_faiss_index(vectors, vectors.shape[1], audio_index_path)
            else:
                dim = len(embed_text("dummy"))
                build_faiss_index(np.zeros((0, dim), dtype="float32"), dim, audio_index_path)

        return BuildStats(
            video_id=video_id,
            raw_segments=raw_count,
            after_filter=len(filtered_segments),
            after_merge=len(merged_segments),
            final_segments=len(final_segments),
            skipped_short=skipped_short,
            skipped_noise=skipped_noise,
            skipped_duplicates=skipped_duplicates,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return BuildStats(
            video_id=video_id,
            raw_segments=0,
            after_filter=0,
            after_merge=0,
            final_segments=0,
            skipped_short=0,
            skipped_noise=0,
            skipped_duplicates=0,
            errors=str(exc),
        )


def iter_videos(input_dir: str) -> List[str]:
    videos: List[str] = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            if not name.lower().endswith(".mp4"):
                continue
            videos.append(os.path.join(root, name))
    videos.sort()
    return videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audio ASR indexes for FastAPI audio retrieval")
    parser.add_argument("--input_dir", default="data/WorldSense")
    parser.add_argument("--out_dir", default="rag_db/videos")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker threads")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even if outputs exist")
    parser.add_argument("--no-cut-wav", action="store_true", help="Skip exporting per-segment wav files")
    parser.add_argument("--model-size", default="small", help="Whisper model size")
    parser.add_argument("--device", default=None, help="Device for ASR model (e.g. cuda, cpu)")
    parser.add_argument("--compute-type", default=None, help="faster-whisper compute type")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD filter in faster-whisper")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only process videos whose audio index/meta are missing under out_dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)
    videos = iter_videos(args.input_dir)
    if not videos:
        print(f"[INFO] No mp4 files found under {args.input_dir}")
        return

    if args.only_missing:
        original_count = len(videos)
        videos = [
            path
            for path in videos
            if not audio_index_exists(args.out_dir, video_id_from_filename(path))
        ]
        skipped = original_count - len(videos)
        print(
            f"[INFO] Filtering already indexed videos: skipped={skipped} remaining={len(videos)}"
        )
        if not videos:
            print("[INFO] Nothing to build; all videos already have audio indexes.")
            return

    transcriber_config = (
        args.model_size,
        args.device,
        args.compute_type,
        args.beam_size,
        not args.no_vad,
    )

    stats: List[BuildStats] = []

    def _task(path: str) -> BuildStats:
        return process_video(
            video_path=path,
            out_dir=args.out_dir,
            transcriber_config=transcriber_config,
            overwrite=args.overwrite,
            cut_wav=not args.no_cut_wav,
        )

    workers = max(1, args.workers)
    if workers == 1:
        for path in videos:
            stat = _task(path)
            stats.append(stat)
            _print_stats(stat)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_task, path): path for path in videos}
            for future in as_completed(future_map):
                stat = future.result()
                stats.append(stat)
                _print_stats(stat)

    summarize(stats)


def _print_stats(stat: BuildStats) -> None:
    if stat.errors:
        print(f"[ERR] {stat.video_id}: {stat.errors}")
        return
    print(
        f"[OK] {stat.video_id}: raw={stat.raw_segments} filter={stat.after_filter} merge={stat.after_merge} "
        f"final={stat.final_segments} skipped_short={stat.skipped_short} "
        f"skipped_noise={stat.skipped_noise} skipped_dup={stat.skipped_duplicates}"
    )


def summarize(stats: Sequence[BuildStats]) -> None:
    total = len(stats)
    success = sum(1 for s in stats if not s.errors)
    failures = [s for s in stats if s.errors]
    print("\n=== Summary ===")
    print(f"Processed videos: {total}, success: {success}, failed: {len(failures)}")
    if failures:
        for stat in failures:
            print(f" - {stat.video_id}: {stat.errors}")


if __name__ == "__main__":
    main()
