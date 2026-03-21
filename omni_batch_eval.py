import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor



DEFAULT_SYSTEM_PROMPT = """You are an agent for Audio/Video QA with retrieval tools over video frames and audio clips.
You must act in iterative steps and choose exactly ONE action per turn.

Output format rules:
- Every turn, you MUST output exactly TWO XML-like tags in this order:
  1) <think>...</think>
  2) Exactly one of <search_image>...</search_image>, <search_audio>...</search_audio>, or <answer>...</answer>
- The <think> tag is for reasoning.
- The <search_image> tag must contain a natural-language query for IMAGE retrieval ONLY.
- The <search_audio> tag must contain a natural-language query for AUDIO retrieval ONLY.
- The <answer> tag must contain EXACTLY ONE option letter/word from the provided options ONLY.
- Do NOT output any JSON, markdown, code fences, or extra text outside the tags.

Allowed actions:
1) <search_image>your query</search_image>
2) <search_audio>your query</search_audio>
3) <answer>ONE_OPTION</answer>

Rules:
- The FIRST turn must use <search_image> or <search_audio>; do NOT answer immediately.
- Prefer gathering evidence via the search actions before answering.
- The user will provide: question, options, and videos.
- When you answer, you must choose exactly one of the provided options (one word/letter).
"""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json_rows(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
        raise ValueError(f"Unsupported json root type: {type(obj)}")
    return load_jsonl(path)


def parse_action(text: str) -> Dict[str, Any]:
    t = text.strip()

    think_tags = re.findall(r"<think>(.*?)</think>", t, flags=re.DOTALL | re.IGNORECASE)
    image_tags = re.findall(r"<search_image>(.*?)</search_image>", t, flags=re.DOTALL | re.IGNORECASE)
    audio_tags = re.findall(r"<search_audio>(.*?)</search_audio>", t, flags=re.DOTALL | re.IGNORECASE)
    answer_tags = re.findall(r"<answer>(.*?)</answer>", t, flags=re.DOTALL | re.IGNORECASE)

    if not (think_tags or image_tags or audio_tags or answer_tags):
        return {"action": "invalid", "format_ok": False, "error": "unparseable_output"}

    if len(think_tags) != 1:
        return {"action": "invalid", "format_ok": False, "error": "missing_or_multiple_think"}

    action_counts = sum(
        [
            len(image_tags),
            len(audio_tags),
            len(answer_tags),
        ]
    )
    if action_counts != 1:
        return {"action": "invalid", "format_ok": False, "error": "missing_or_multiple_action_tags"}

    think_text = think_tags[0].strip()
    if answer_tags:
        return {
            "action": "answer",
            "answer": answer_tags[0].strip(),
            "think": think_text,
            "format_ok": True,
        }
    if image_tags:
        return {
            "action": "search_image",
            "query": image_tags[0].strip(),
            "think": think_text,
            "format_ok": True,
        }
    if audio_tags:
        return {
            "action": "search_audio",
            "query": audio_tags[0].strip(),
            "think": think_text,
            "format_ok": True,
        }

    return {"action": "invalid", "format_ok": False, "error": "missing_action"}


def _extract_answer_tag(text: str) -> Optional[str]:
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"<answer>(.*?)</anawer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def normalize_choice(ans: str) -> Optional[str]:
   
    if not ans:
        return None
    s = ans.strip()
    m = re.match(r"^\s*([A-Za-z])\b", s)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-Za-z])\b", s)
    if m:
        return m.group(1).upper()
    return None


def build_user_text(sample: Dict[str, Any]) -> str:
    q = sample["extra_info"]["question"]
    choices = sample["reward_model_ground_truth"]["multi_choice"]
    opt_text = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
    return (
        "You are doing Audio/Video QA.\n"
        f"Question: {q}\n"
        f"Options:\n{opt_text}\n\n"
        "Output format:\n"
        "- First turn MUST use <search_image> or <search_audio> (do NOT answer).\n"
        "- Every turn must output exactly:\n"
        "  <think>...</think>\n"
        "  and either <search_image>...</search_image>, <search_audio>...</search_audio>, or <answer>...</answer>\n"
        "- Do not output JSON or any text outside the tags.\n"
    )


def make_qwen_mm_user_message(
    text: str,
    image_paths: Optional[List[str]] = None,
    audio_paths: Optional[List[str]] = None,
    video_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    for p in (image_paths or []):
        content.append({"type": "image", "path": p})
    for p in (video_paths or []):
        content.append({"type": "video", "path": p})
    for p in (audio_paths or []):
        content.append({"type": "audio", "path": p})
    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def _tags_only_reminder(reason: str) -> str:
    return (
        "Format error: reply with EXACTLY two tags only:\n"
        "<think>...</think>\n"
        "and one of <search_image>...</search_image>, <search_audio>...</search_audio>, or <answer>...</answer>.\n"
        "No JSON, markdown, or extra text.\n"
        f"Reason: {reason}"
    )


def messages_have_video(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages:
        for block in msg.get("content", []) or []:
            if block.get("type") == "video":
                return True
    return False



def rag_search_image(rag_url: str, query: str, top_k: int, *, video_id: str) -> List[Dict[str, Any]]:
    payload = {"query": query, "top_k": top_k, "video_id": video_id}
    r = requests.post(
        rag_url.rstrip("/") + "/query",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"RAG error: {r.status_code} - {r.text}")
    return r.json()


def rag_search_audio(rag_url: str, query: str, top_k: int, *, video_id: str) -> List[Dict[str, Any]]:
    payload = {"query": query, "top_k": top_k, "video_id": video_id}
    r = requests.post(
        rag_url.rstrip("/") + "/query_audio",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"RAG audio error: {r.status_code} - {r.text}")
    return r.json()


def format_image_results(
    results: List[Dict[str, Any]],
    image_prefix: str,
    attach_k: int,
) -> Tuple[str, List[str]]:
    lines: List[str] = []
    imgs: List[str] = []

    for item in results[:attach_k]:
        seg = item.get("segment") or {}
        frame_rel = seg.get("frame", "")
        t0, t1 = seg.get("t0"), seg.get("t1")
        frame_abs = os.path.join(image_prefix, frame_rel) if frame_rel else ""

        lines.append(
            f"- rank={item.get('rank')} score={float(item.get('score', 0.0)):.6f} "
            f"t0={t0} t1={t1} frame={frame_rel}"
        )

        if frame_abs and os.path.exists(frame_abs):
            imgs.append(frame_abs)

    text = "RAG image results:\n" + "\n".join(lines) if lines else "RAG image results: <empty>"
    return text, imgs


def format_audio_results(
    results: List[Dict[str, Any]],
    audio_prefix: str,
    attach_k: int,
) -> Tuple[str, List[str]]:
    lines: List[str] = []
    audios: List[str] = []

    for item in results[:attach_k]:
        seg = item.get("segment") or {}
        audio_rel = seg.get("audio", "")
        t0, t1 = seg.get("t0"), seg.get("t1")
        transcript = item.get("transcript") or seg.get("transcript")
        audio_abs = os.path.join(audio_prefix, audio_rel) if audio_rel else ""

        line = (
            f"- rank={item.get('rank')} score={float(item.get('score', 0.0)):.6f} "
            f"t0={t0} t1={t1} audio={audio_rel}"
        )
        if transcript:
            line += f" transcript={transcript}"
        lines.append(line)

        if audio_abs and os.path.exists(audio_abs):
            audios.append(audio_abs)

    text = "RAG audio results:\n" + "\n".join(lines) if lines else "RAG audio results: <empty>"
    return text, audios



@torch.inference_mode()
def qwen_local_chat(
    model: Qwen2_5OmniForConditionalGeneration,
    processor: Qwen2_5OmniProcessor,
    messages: List[Dict[str, Any]],
    *,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    fps: int = 1,
    use_audio_in_video: bool = True,
    load_audio_from_video: bool = True,
) -> str:
    def _build_inputs(use_audio: bool, load_audio: bool) -> Dict[str, Any]:
        inputs_local = processor.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            fps=fps,
            use_audio_in_video=use_audio,
            load_audio_from_video=load_audio,
        )
        return inputs_local

    has_video = messages_have_video(messages)
    fallback = False
    try:
        inputs = _build_inputs(use_audio=use_audio_in_video, load_audio=load_audio_from_video)
    except StopIteration:
        if has_video and (use_audio_in_video or load_audio_from_video):
            fallback = True
            inputs = _build_inputs(use_audio=False, load_audio=False)
            use_audio_in_video = False
        else:
            raise

    if has_video and (use_audio_in_video or load_audio_from_video):
        audio_lengths = inputs.get("audio_lengths")
        if torch.is_tensor(audio_lengths) and audio_lengths.numel() == 0:
            fallback = True
            inputs = _build_inputs(use_audio=False, load_audio=False)
            use_audio_in_video = False

    device = getattr(getattr(model, "thinker", None), "device", None) or next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        use_audio_in_video=use_audio_in_video,
    )
    out = model.generate(**inputs, **gen_kwargs)
    in_len = inputs["input_ids"].shape[1]
    new_tokens = out[:, in_len:] if out.shape[1] >= in_len else out
    text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()



def ensure_video_path(sample: Dict[str, Any], videos_dir: str, video_suffix: str) -> None:
    
    vid = sample.get("video_id") or ""
    if not vid:
        return
    cand = os.path.join(videos_dir, f"{vid}{video_suffix}")
    if os.path.exists(cand):
        sample["videos"] = [cand]
        return
    
    old = (sample.get("videos") or [""])[0]
    if old and os.path.exists(old):
        return
    print(f"[WARN] video file missing: cand={cand} old={old}")



def run_agent_for_sample(
    sample: Dict[str, Any],
    *,
    model,
    processor,
    rag_url: str,
    rag_db_root: str,
    top_k: int,
    max_turns: int,
    attach_segments: int,
    include_video_in_first_turn: bool,
) -> Dict[str, Any]:
    video_id = sample.get("video_id", "")
    correct = (sample.get("reward_model_ground_truth", {}) or {}).get("correct_option", "")
    correct = (correct or "").strip().upper()

    image_prefix = os.path.join(rag_db_root, "videos", video_id, "segments")
    audio_prefix = os.path.join(rag_db_root, "videos", video_id)

    options = sample["reward_model_ground_truth"]["multi_choice"]

    video_paths: List[str] = []
    if include_video_in_first_turn:
        vlist = sample.get("videos") or []
        if vlist and os.path.exists(vlist[0]):
            video_paths = [vlist[0]]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]},
        make_qwen_mm_user_message(text=build_user_text(sample), video_paths=video_paths),
    ]

    traj_turns: List[Dict[str, Any]] = []
    all_searches: List[str] = []
    last_search_query = ""

    for turn in range(1, max_turns + 1):
        raw = qwen_local_chat(
            model=model,
            processor=processor,
            messages=messages,
            temperature=0.2,
            max_new_tokens=512,
            fps=1,
            use_audio_in_video=True,
            load_audio_from_video=True,
        )

        action = parse_action(raw)
        answer_from_tag = _extract_answer_tag(raw)

        turn_record: Dict[str, Any] = {
            "turn": turn,
            "assistant_raw": raw,
            "parsed": action,
        }
        traj_turns.append(turn_record)

        messages.append({"role": "assistant", "content": [{"type": "text", "text": raw}]})

        
        if answer_from_tag is not None:
            pred_norm = normalize_choice(answer_from_tag)
            is_ok = (pred_norm == correct) if (pred_norm and correct) else False
            return {
                "id": sample.get("id"),
                "video_id": video_id,
                "qid": sample.get("qid"),
                "question_type": sample.get("question_type"),
                "question": sample["extra_info"]["question"],
                "options": options,
                "correct_option": correct,
                "prediction_raw": answer_from_tag,
                "prediction": pred_norm,
                "is_correct": is_ok,
                "agent_turns": turn,
                "searches": all_searches,
                "last_search": last_search_query,
                "trajectory": traj_turns,
            }

        
        if action.get("action") == "answer":
            pred_raw = (action.get("answer") or "").strip()
            pred_norm = normalize_choice(pred_raw)
            is_ok = (pred_norm == correct) if (pred_norm and correct) else False
            return {
                "id": sample.get("id"),
                "video_id": video_id,
                "qid": sample.get("qid"),
                "question_type": sample.get("question_type"),
                "question": sample["extra_info"]["question"],
                "options": options,
                "correct_option": correct,
                "prediction_raw": pred_raw,
                "prediction": pred_norm,
                "is_correct": is_ok,
                "agent_turns": turn,
                "searches": all_searches,
                "last_search": last_search_query,
                "trajectory": traj_turns,
            }

        
        if not action.get("format_ok", False):
            reason = action.get("error", "invalid_output")
            messages.append(make_qwen_mm_user_message(text=_tags_only_reminder(reason)))
            traj_turns.append({"turn": turn, "tool": "format_reminder", "reason": reason})
            continue

        
        act = (action.get("action") or "").strip().lower()
        if act in {"search_image", "search_audio"}:
            query = (action.get("query") or "").strip()
            if not query:
                messages.append(make_qwen_mm_user_message(text=_tags_only_reminder("empty_search_query")))
                traj_turns.append({"turn": turn, "tool": "format_reminder", "reason": "empty_search_query"})
                continue

            last_search_query = query
            all_searches.append(query)
            if act == "search_image":
                results = rag_search_image(rag_url, query=query, top_k=top_k, video_id=video_id)
                tool_text, img_paths = format_image_results(
                    results=results,
                    image_prefix=image_prefix,
                    attach_k=attach_segments,
                )
                traj_turns.append(
                    {
                        "turn": turn,
                        "tool": "rag_search_image",
                        "video_id": video_id,
                        "query": query,
                        "top_k": top_k,
                        "image_prefix": image_prefix,
                        "results": results,
                        "attached": {"images": img_paths},
                    }
                )
                msg_text = (
                    "TOOL(search_image) executed.\n"
                    f"Query: {query}\n\n"
                    f"{tool_text}\n\n"
                    "Retrieved image segments are attached.\n"
                    "Now decide next action using the required tags only."
                )
                messages.append(
                    make_qwen_mm_user_message(
                        text=msg_text,
                        image_paths=img_paths,
                    )
                )
                continue

            if act == "search_audio":
                results = rag_search_audio(rag_url, query=query, top_k=top_k, video_id=video_id)
                tool_text, audio_paths = format_audio_results(
                    results=results,
                    audio_prefix=audio_prefix,
                    attach_k=attach_segments,
                )
                traj_turns.append(
                    {
                        "turn": turn,
                        "tool": "rag_search_audio",
                        "video_id": video_id,
                        "query": query,
                        "top_k": top_k,
                        "audio_prefix": audio_prefix,
                        "results": results,
                        "attached": {"audios": audio_paths},
                    }
                )
                msg_text = (
                    "TOOL(search_audio) executed.\n"
                    f"Query: {query}\n\n"
                    f"{tool_text}\n\n"
                    "Retrieved audio segments are attached.\n"
                    "Now decide next action using the required tags only."
                )
                messages.append(
                    make_qwen_mm_user_message(
                        text=msg_text,
                        audio_paths=audio_paths,
                    )
                )
                continue

        
        messages.append(make_qwen_mm_user_message(text=_tags_only_reminder(f"unknown_action '{act}'")))
        traj_turns.append({"turn": turn, "tool": "format_reminder", "reason": f"unknown_action '{act}'"})

    
    return {
        "id": sample.get("id"),
        "video_id": video_id,
        "qid": sample.get("qid"),
        "question_type": sample.get("question_type"),
        "question": sample["extra_info"]["question"],
        "options": options,
        "correct_option": correct,
        "error": "max_turns_reached",
        "prediction_raw": None,
        "prediction": None,
        "is_correct": False,
        "agent_turns": max_turns,
        "searches": all_searches,
        "last_search": last_search_query,
        "trajectory": traj_turns,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="input jsonl test set")
    ap.add_argument("--out-traj", required=True, help="output trajectory jsonl")
    ap.add_argument("--out-summary", required=True, help="output summary json")
    ap.add_argument("--model-path", required=True, help="local path to Qwen2.5-Omni model")

    ap.add_argument("--rag-url", default="http://127.0.0.1:8001", help="RAG service base url")
    ap.add_argument(
        "--rag-db-root",
        default="retrieval/rag_db",
        help="rag_db root; will use rag_db/videos/<video_id>/segments as media_prefix",
    )

    ap.add_argument(
        "--videos-dir",
        default="datasets/OmniVideoBench/videos_downsampled",
        help="downsampled videos dir",
    )
    ap.add_argument(
        "--video-suffix",
        default="_k1_g5.mp4",
        help="video file suffix, e.g. _k1_g5.mp4; final path: <videos-dir>/<video_id><suffix>",
    )

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1)

    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--attach-segments", type=int, default=3)
    ap.add_argument("--include-video", action="store_true", help="include original video in first turn")

    ap.add_argument("--flash-attn2", action="store_true")
    ap.add_argument("--enable-audio-output", action="store_true")
    ap.add_argument("--resume", action="store_true", help="resume from existing output trajectory jsonl")

    args = ap.parse_args()

    attn_impl = "flash_attention_2" if args.flash_attn2 else None
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=attn_impl,
        enable_audio_output=args.enable_audio_output,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    rows = load_json_rows(args.jsonl)
    sub = rows[args.start:] if args.limit < 0 else rows[args.start: args.start + args.limit]

    total = 0
    correct = 0
    missing_gt = 0

    os.makedirs(os.path.dirname(args.out_traj) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)

    done_ids = set()
    if args.resume and os.path.exists(args.out_traj):
        done_total = 0
        done_correct = 0
        done_missing_gt = 0
        with open(args.out_traj, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done_total += 1
                if obj.get("is_correct"):
                    done_correct += 1
                correct_option = obj.get("correct_option")
                if not correct_option:
                    done_missing_gt += 1
                sid = obj.get("id")
                if sid is not None:
                    done_ids.add(sid)
        total = done_total
        correct = done_correct
        missing_gt = done_missing_gt
        print(
            f"Found {done_total} done samples in {args.out_traj}, "
            f"correct={done_correct}, missing_gt={done_missing_gt}; will skip done ids"
        )

    write_mode = "a" if args.resume else "w"
    with open(args.out_traj, write_mode, encoding="utf-8") as f_out:
        for i, sample in enumerate(sub):
            
            ensure_video_path(sample, videos_dir=args.videos_dir, video_suffix=args.video_suffix)

            sid = sample.get("id")
            vid = sample.get("video_id")
            gt = (sample.get("reward_model_ground_truth", {}) or {}).get("correct_option", None)

            if args.resume and sid in done_ids:
                continue

            print(f"\n========== SAMPLE {args.start + i} | id={sid} video_id={vid} ==========")

            result = run_agent_for_sample(
                sample=sample,
                model=model,
                processor=processor,
                rag_url=args.rag_url,
                rag_db_root=args.rag_db_root,
                top_k=args.top_k,
                max_turns=args.max_turns,
                attach_segments=args.attach_segments,
                include_video_in_first_turn=args.include_video,
            )

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
            f_out.flush()

            if sid is not None:
                done_ids.add(sid)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total += 1
            if not gt:
                missing_gt += 1
            if result.get("is_correct"):
                correct += 1

            pred = result.get("prediction")
            print(f"[RESULT] pred={pred} gt={(gt or '').strip()} correct={result.get('is_correct')}")

    acc = (correct / total) if total > 0 else 0.0
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "missing_gt": missing_gt,
        "args": vars(args),
    }

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n====================")
    print(f"Done. total={total} correct={correct} acc={acc:.4f}")
    print(f"Wrote trajectories: {args.out_traj}")
    print(f"Wrote summary:      {args.out_summary}")


if __name__ == "__main__":
    main()
