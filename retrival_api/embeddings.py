import clip
import torch
from PIL import Image
import faiss
import numpy as np
import json
from video_ingest import (
    audio_to_text,
    extract_frames,
    video_to_audio,
    extract_segments,
    _is_video_already_indexed,
)
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root="")
_embed_lock = threading.Lock()


def _shrink_text_for_clip(text: str) -> str:
    """Reduce text length while trying to keep semantic boundaries."""
    stripped = text.strip()
    if not stripped:
        return ""
    # Prefer trimming by words when possible
    parts = stripped.split()
    if len(parts) > 1:
        keep = max(1, int(len(parts) * 0.8))
        return " ".join(parts[:keep])
    # Fall back to character-level truncation for scripts without spaces
    keep_chars = max(1, int(len(stripped) * 0.8))
    return stripped[:keep_chars]


def embed_text(text: str):
    attempt = (text or "").strip()
    if attempt == "":
        attempt = ""

    last_error: RuntimeError | None = None
    for _ in range(6):
        try:
            with _embed_lock:
                tokens = clip.tokenize([attempt], truncate=True).to(device)
                with torch.no_grad():
                    vec = model.encode_text(tokens)
            return vec[0].cpu().numpy()
        except RuntimeError as exc:  # pragma: no cover - defensive path
            msg = str(exc)
            if "context length" not in msg.lower() and "too long" not in msg.lower():
                raise
            last_error = exc
            shortened = _shrink_text_for_clip(attempt)
            if shortened == attempt:
                break
            attempt = shortened

    if last_error is not None:
        print(
            f"[WARN] clip.tokenize kept failing for text len={len(text or '')}; "
            "using truncated content."
        )
    with _embed_lock:
        tokens = clip.tokenize([attempt[:77]], truncate=True).to(device)
        with torch.no_grad():
            vec = model.encode_text(tokens)
    return vec[0].cpu().numpy()


def embed_image(image_path: str):
    with _embed_lock:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model.encode_image(image)
    return vec[0].cpu().numpy()


def create_index(dim, index_path):
    index = faiss.IndexFlatL2(dim)
    faiss.write_index(index, index_path)


def load_index(index_path):
    return faiss.read_index(index_path)


def add_to_index(index_path, vector):
    index = load_index(index_path)
    index.add(np.array([vector]).astype("float32"))
    faiss.write_index(index, index_path)


def add_metadata(meta_path, item):
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_metadata(meta_path):
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            metas.append(json.loads(line))
    return metas


def split_asr_text(text, max_tokens=70):
    sentences = re.split(r"[.!?。！？]", text)

    clean_sentences = []
    for s in sentences:
        s = s.strip()
        if len(s) == 0:
            continue
        clean_sentences.append(s)

    return clean_sentences


def ingest_video(video_path, frame_dir, audio_path, index_path, meta_path):
    extract_frames(video_path, frame_dir, fps=1)
    video_to_audio(video_path, audio_path)
    asr_text = audio_to_text(audio_path)
    sentences = split_asr_text(asr_text)
    for idx, sent in enumerate(sentences):
        text_vec = embed_text(sent)
        add_to_index(index_path, text_vec)

        add_metadata(
            meta_path,
            {
                "type": "asr_sentence",
                "content": sent,
                "source": audio_path,
                "sentence_id": idx,
            },
        )

    for f in sorted(os.listdir(frame_dir)):
        img_path = os.path.join(frame_dir, f)
        img_vec = embed_image(img_path)
        add_to_index(index_path, img_vec)

        add_metadata(
            meta_path,
            {
                "type": "frame",
                "content": "",
                "source": img_path,
            },
        )

    print(
        f"Ingest finished: {len(sentences)} ASR sentences + {len(os.listdir(frame_dir))} frames processed."
    )


def ingest_video_segments(
    video_path,
    out_dir,
    index_path,
    meta_path,
    interval_sec=3.0,
    audio_win_sec=2.0,
):
    seg_meta_path = extract_segments(
        video_path=video_path,
        out_dir=out_dir,
        interval_sec=interval_sec,
        audio_win_sec=audio_win_sec,
    )
    with open(seg_meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seg = json.loads(line)
            frame_rel = seg["frame_relpath"]
            audio_rel = seg["audio_relpath"]
            frame_abs = os.path.join(out_dir, frame_rel)
            img_vec = embed_image(frame_abs)
            add_to_index(index_path, img_vec)

            add_metadata(
                meta_path,
                {
                    "type": "frame",
                    "modality": "image",
                    "segment_id": seg["segment_id"],
                    "t": seg["t"],
                    "t0": seg["t0"],
                    "t1": seg["t1"],
                    "frame": frame_rel,
                    "audio": audio_rel,
                },
            )


def ingest_video_segments_in_memory(
    video_path,
    out_dir,
    index,
    meta_items,
    interval_sec=3.0,
    audio_win_sec=2.0,
):
    seg_meta_path = extract_segments(
        video_path=video_path,
        out_dir=out_dir,
        interval_sec=interval_sec,
        audio_win_sec=audio_win_sec,
    )
    vectors = []
    skipped = 0
    with open(seg_meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seg = json.loads(line)
            frame_rel = seg["frame_relpath"]
            audio_rel = seg["audio_relpath"]
            frame_abs, frame_rel_fixed = resolve_frame_path(out_dir, frame_rel)
            if frame_abs is None:
                print(f"[WARN] Missing frame for segment {seg['segment_id']}, skipping.")
                skipped += 1
                continue
            try:
                img_vec = embed_image(frame_abs)
            except Exception as exc:
                print(f"[WARN] Failed to embed frame {frame_abs}: {exc}")
                skipped += 1
                continue
            vectors.append(img_vec.astype("float32"))
            meta_items.append(
                {
                    "type": "frame",
                    "modality": "image",
                    "segment_id": seg["segment_id"],
                    "t": seg["t"],
                    "t0": seg["t0"],
                    "t1": seg["t1"],
                    "frame": frame_rel_fixed,
                    "audio": audio_rel,
                }
            )
    if vectors:
        index.add(np.stack(vectors, axis=0))
    return len(vectors), skipped


def resolve_frame_path(out_dir, frame_rel):
    frame_abs = os.path.join(out_dir, frame_rel)
    if os.path.exists(frame_abs):
        return frame_abs, frame_rel

    stem = os.path.splitext(os.path.basename(frame_rel))[0]
    for base in [os.path.join(out_dir, "frames"), out_dir]:
        if not os.path.isdir(base):
            continue
        matches = []
        for name in os.listdir(base):
            if os.path.splitext(name)[0] == stem:
                matches.append(name)
        if matches:
            matches.sort()
            chosen = os.path.join(base, matches[0])
            return chosen, os.path.relpath(chosen, out_dir)
    return None, None


def resolve_video_path(videos_dir, video_id):
    exts = [".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"]
    exts_lower = {ext.lower() for ext in exts}

    vid = (video_id or "").strip()
    if not vid:
        return None

    for ext in exts:
        candidate = os.path.join(videos_dir, f"{vid}{ext}")
        if os.path.exists(candidate):
            return candidate

    direct_candidate = os.path.join(videos_dir, vid)
    if os.path.isfile(direct_candidate):
        return direct_candidate

    vid_lower = vid.lower()
    for root, _, files in os.walk(videos_dir):
        for name in files:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in exts_lower:
                continue
            stem_lower = stem.lower()
            if stem_lower == vid_lower or stem_lower.startswith(f"{vid_lower}_"):
                return os.path.join(root, name)
    return None


def extract_video_ids_from_json(data):
    """
    Supports:
      1) Original format: list of dicts, each dict contains "video" or "video_id".
      2) New format: dict mapping from video_id -> metadata dict
         e.g. { "hHwdqJxc": { "video_id": "hHwdqJxc", ... }, ... }
    """
    video_ids = []

    # Case A: list format
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            vid = item.get("video") or item.get("video_id")
            if isinstance(vid, str) and vid.strip():
                video_ids.append(vid.strip())

    # Case B: dict format (your example)
    elif isinstance(data, dict):
        for k, v in data.items():
            # key itself is the video_id in your format
            vid = None
            if isinstance(v, dict):
                vid = v.get("video") or v.get("video_id") or k
            else:
                vid = k
            if isinstance(vid, str) and vid.strip():
                video_ids.append(vid.strip())

    else:
        raise TypeError(
            f"Unsupported JSON root type: {type(data)}. Expected list or dict."
        )

    return sorted(set(video_ids))


def batch_ingest_from_test_json(
    test_json_path,
    videos_dir,
    rag_db_root,
    interval_sec=3.0,
    audio_win_sec=2.0,
    num_workers=4,
):
    os.makedirs(rag_db_root, exist_ok=True)
    videos_root = os.path.join(rag_db_root, "videos")
    os.makedirs(videos_root, exist_ok=True)

    with open(test_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # UPDATED: support both list and dict json formats
    video_ids = extract_video_ids_from_json(data)

    print(f"[INFO] Found {len(video_ids)} unique video ids in {test_json_path}")

    def _process_video(video_id):
        video_path = resolve_video_path(videos_dir, video_id)
        if video_path is None:
            print(f"[WARN] Missing video file for {video_id}, skipping.")
            return {
                "video_id": video_id,
                "status": "missing_video",
                "indexed": 0,
                "skipped_segments": 0,
            }

        video_dir = os.path.join(videos_root, video_id)
        segments_dir = os.path.join(video_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)
        index_path = os.path.join(video_dir, "index.faiss")
        meta_path = os.path.join(video_dir, "meta.jsonl")

        if _is_video_already_indexed(index_path, meta_path):
            index = faiss.read_index(index_path)
            print(f"[INFO] Skip built video {video_id}")
            return {
                "video_id": video_id,
                "status": "skipped_done",
                "indexed": int(index.ntotal),
                "skipped_segments": 0,
            }

        index = faiss.IndexFlatL2(512)
        meta_items = []

        print(f"[INFO] Ingesting {video_id} from {video_path}")
        indexed, skipped_segments = ingest_video_segments_in_memory(
            video_path=video_path,
            out_dir=segments_dir,
            index=index,
            meta_items=meta_items,
            interval_sec=interval_sec,
            audio_win_sec=audio_win_sec,
        )

        faiss.write_index(index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            for item in meta_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"[OK] {video_id}: {indexed} segments indexed, skipped {skipped_segments}.")
        return {
            "video_id": video_id,
            "status": "success",
            "indexed": indexed,
            "skipped_segments": skipped_segments,
        }

    success = 0
    skipped_missing = 0
    skipped_done = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_map = {executor.submit(_process_video, vid): vid for vid in video_ids}
        for future in as_completed(future_map):
            try:
                result = future.result()
            except Exception as exc:
                vid = future_map[future]
                print(f"[WARN] Failed to process {vid}: {exc}")
                continue
            if result["status"] == "success":
                success += 1
            elif result["status"] == "missing_video":
                skipped_missing += 1
            elif result["status"] == "skipped_done":
                skipped_done += 1

    print(
        "[OK] Batch ingest done. success="
        f"{success} skipped_missing_video={skipped_missing} skipped_done={skipped_done}"
    )


def search_in_video(video_id, query_text, topk=5, rag_db_root="rag_db"):
    video_dir = os.path.join(rag_db_root, "videos", video_id)
    index_path = os.path.join(video_dir, "index.faiss")
    meta_path = os.path.join(video_dir, "meta.jsonl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print(f"[WARN] Index or metadata missing for {video_id}.")
        return []

    index = faiss.read_index(index_path)
    metas = load_metadata(meta_path)
    if len(metas) == 0:
        return []

    vec = embed_text(query_text).astype("float32")
    k = min(topk, len(metas))
    distances, indices = index.search(np.array([vec]), k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metas):
            continue
        item = dict(metas[idx])
        item["score_l2"] = float(score)
        results.append(item)
    return results


if not os.path.exists("rag_db"):
    os.makedirs("rag_db")

if __name__ == "__main__":
    os.makedirs("rag_db", exist_ok=True)

    TEST_JSON = "/mnt/hpfs/xiangc/mxy/rl-omni/data/VideoOmniBench/video_ids.json"
    VIDEOS_DIR = "/mnt/hpfs/xiangc/mxy/rl-omni/retrival_api/VideoOmniBench/videos"
    RAG_DB_ROOT = "rag_db_videoomnibench"
    batch_ingest_from_test_json(
        test_json_path=TEST_JSON,
        videos_dir=VIDEOS_DIR,
        rag_db_root=RAG_DB_ROOT,
        interval_sec=1.0,
        audio_win_sec=1.0,
        num_workers=4,
    )
