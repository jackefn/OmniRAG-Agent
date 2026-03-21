from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Any, Optional, Tuple
import numpy as np
import os

from embeddings import embed_text, load_index, load_metadata

app = FastAPI(title="Video RAG API", description="RAG over video frames + ASR", version="1.0")

INDEX_PATH = "rag_db/video_index.faiss"
META_PATH = "rag_db/video_metadata.jsonl"

AUDIO_INDEX_TEMPLATE = "rag_db/videos/{video_id}/audio_index.faiss"
AUDIO_META_TEMPLATE = "rag_db/videos/{video_id}/audio_meta.jsonl"

def search(index_path, query_vector, top_k=5):
    index = load_index(index_path)
    query = np.array([query_vector]).astype("float32")
    D, I = index.search(query, top_k)
    return I[0], D[0]

def retrieve_results(meta_path):
    return load_metadata(meta_path)

def aggregate_by_segment(metas, ids, dists, want_k):
    best = {}
    for i, meta_id in enumerate(ids):
        if meta_id < 0 or meta_id >= len(metas):
            continue
        meta = metas[meta_id]
        seg_id = meta.get("segment_id")
        if seg_id is None:
            continue
        dist = float(dists[i])
        prev = best.get(seg_id)
        if prev is None or dist < prev["score"]:
            best[seg_id] = {
                "score": dist,
                "segment": {
                    "segment_id": seg_id,
                    "t0": meta.get("t0"),
                    "t1": meta.get("t1"),
                    "frame": meta.get("frame"),
                    "audio": meta.get("audio"),
                },
            }

    ranked = sorted(best.values(), key=lambda x: x["score"])
    return ranked[:want_k]

def resolve_index_meta_paths(video_id: Optional[str]) -> Tuple[str, str]:
    if video_id:
        index_path = f"rag_db/videos/{video_id}/index.faiss"
        meta_path = f"rag_db/videos/{video_id}/meta.jsonl"
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise HTTPException(
                status_code=404,
                detail=f"Index/meta not found for video_id={video_id}",
            )
        return index_path, meta_path

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        return INDEX_PATH, META_PATH

    raise HTTPException(
        status_code=404,
        detail="video_id is required or legacy index/meta not found",
    )


def resolve_audio_paths(video_id: Optional[str]) -> Tuple[str, str]:
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required for audio queries")

    index_path = AUDIO_INDEX_TEMPLATE.format(video_id=video_id)
    meta_path = AUDIO_META_TEMPLATE.format(video_id=video_id)

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise HTTPException(
            status_code=404,
            detail=f"Audio index/meta not found for video_id={video_id}",
        )
    return index_path, meta_path


def format_audio_results(metas, ids, dists, top_k):
    results = []
    seen = 0
    for idx, (meta_id, dist) in enumerate(zip(ids, dists)):
        if meta_id < 0 or meta_id >= len(metas):
            continue
        meta = metas[meta_id]
        segment = {
            "segment_id": meta.get("segment_id"),
            "t0": meta.get("t0"),
            "t1": meta.get("t1"),
            "frame": meta.get("frame"),
            "audio": meta.get("audio"),
        }
        results.append(
            {
                "rank": len(results),
                "score": float(dist),
                "segment": segment,
                "transcript": meta.get("transcript", ""),
            }
        )
        seen += 1
        if len(results) >= top_k:
            break
    return results


def query_rag(query: str, top_k: int = 5, video_id: Optional[str] = None):
    index_path, meta_path = resolve_index_meta_paths(video_id)
    v = embed_text(query)  # CPU embedding
    search_k = max(50, top_k * 10)
    ids, dist = search(index_path, v, top_k=search_k)
    metas = retrieve_results(meta_path)
    results = aggregate_by_segment(metas, ids, dist, want_k=top_k)
    print(f"Query: {query}")
    print(f"Results: {results}")
    return [
        {
            "rank": idx,
            "score": results[idx]["score"],
            "segment": results[idx]["segment"],
        }
        for idx in range(len(results))
    ]


def query_audio_rag(query: str, top_k: int, video_id: Optional[str]):
    index_path, meta_path = resolve_audio_paths(video_id)
    vec = embed_text(query).astype("float32")
    search_k = max(50, top_k * 10)
    ids, dists = search(index_path, vec, top_k=search_k)
    metas = retrieve_results(meta_path)
    return format_audio_results(metas, ids, dists, top_k)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    video_id: Optional[str] = None


class QueryResponse(BaseModel):
    rank: int
    score: float
    segment: Any


class QueryAudioResponse(BaseModel):
    rank: int
    score: float
    segment: Any
    transcript: str


@app.get("/")
def root():
    return {"message": "Video RAG API is running!"}


@app.post("/query", response_model=List[QueryResponse])
def rag_query(req: QueryRequest):
    return query_rag(req.query, top_k=req.top_k, video_id=req.video_id)


@app.get("/query", response_model=List[QueryResponse])
def rag_query_get(
    query: str = Query(...),
    top_k: int = 5,
    video_id: Optional[str] = None,
):
    return query_rag(query, top_k=top_k, video_id=video_id)


@app.post("/query_audio", response_model=List[QueryAudioResponse])
def rag_query_audio(req: QueryRequest):
    return query_audio_rag(req.query, top_k=req.top_k, video_id=req.video_id)


@app.get("/query_audio", response_model=List[QueryAudioResponse])
def rag_query_audio_get(
    query: str = Query(...),
    top_k: int = 5,
    video_id: Optional[str] = Query(...),
):
    return query_audio_rag(query, top_k=top_k, video_id=video_id)
