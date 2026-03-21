from moviepy import VideoFileClip
import os
import speech_recognition as sr
import json
import subprocess
import faiss

def extract_frames(video_path, frame_dir, fps=1):
    os.makedirs(frame_dir, exist_ok=True)
    clip = VideoFileClip(video_path)
    clip.write_images_sequence(f"{frame_dir}/frame_%06d.jpg", fps=fps)

def video_to_audio(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    audio = clip.audio
    audio.write_audiofile(output_audio_path)

def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)
    with audio as source:
        # Record the audio data
        audio_data = recognizer.record(source)
        try:
            # Recognize the speech
            text = recognizer.recognize_whisper(audio_data)
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from service; {e}")
    return text

def get_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())

def _run_ffmpeg(cmd: list) -> None:
    subprocess.run(cmd, check=True, capture_output=True)

def extract_segments(
    video_path: str,
    out_dir: str,
    interval_sec: float = 3.0,
    audio_win_sec: float = 2.0,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    duration = get_duration(video_path)
    frame_margin = max(0.5, interval_sec * 0.1)
    last_valid_time = duration if duration <= frame_margin else max(0.0, duration - frame_margin)
    meta_path = os.path.join(out_dir, "meta.jsonl")

    seg_idx = 0
    t = 0.0
    with open(meta_path, "w", encoding="utf-8") as f:
        while t <= last_valid_time + 1e-6:
            t0 = max(0.0, t - audio_win_sec / 2.0)
            t1 = min(duration, t0 + audio_win_sec)
            seg_id = f"seg_{seg_idx:06d}"

            frame_rel = os.path.join("frames", f"{seg_id}.jpg")
            audio_rel = os.path.join("audio", f"{seg_id}.wav")
            frame_path = os.path.join(out_dir, frame_rel)
            audio_path = os.path.join(out_dir, audio_rel)

            _run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{t:.3f}",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    frame_path,
                ]
            )

            _run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{t0:.3f}",
                    "-t",
                    f"{(t1 - t0):.3f}",
                    "-i",
                    video_path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-vn",
                    "-c:a",
                    "pcm_s16le",
                    audio_path,
                ]
            )

            f.write(
                json.dumps(
                    {
                        "segment_id": seg_id,
                        "t": t,
                        "t0": t0,
                        "t1": t1,
                        "frame_relpath": frame_rel,
                        "audio_relpath": audio_rel,
                    }
                )
                + "\n"
            )

            seg_idx += 1
            t = seg_idx * interval_sec

    return meta_path


def _is_video_already_indexed(index_path: str, meta_path: str) -> bool:
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return False

    meta_lines = 0
    with open(meta_path, "r", encoding="utf-8") as f:
        for _ in f:
            meta_lines += 1
    if meta_lines == 0:
        return False

    try:
        index = faiss.read_index(index_path)
    except Exception:
        return False

    return int(index.ntotal) == meta_lines

