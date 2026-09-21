#!/usr/bin/env python3
"""Build a multi-speaker Fish Audio soundtrack and align the final mix.

Example:
    python3 scripts/build_audio.py \
        --script script.json --roles script_roles.json --out audio

The Fish Audio timestamps returned for individual lines are intentionally not
used as the final alignment.  The mixed ``full.wav`` is aligned again with
timestamps_cpu.py, then converted with make_timing.py.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tts_fishaudio as tts  # noqa: E402  - supports direct script execution


EPSILON = 1e-6


@dataclass(frozen=True)
class TimelineLine:
    sentence_index: int
    speaker: str
    text: str
    start: float
    end: float
    reference_id: str
    reference_id_env: str | None


@dataclass
class RenderedLine:
    plan: TimelineLine
    pcm: bytes
    path: Path
    sample_rate: int

    @property
    def actual_duration(self) -> float:
        return len(self.pcm) / 2 / self.sample_rate

    @property
    def actual_end(self) -> float:
        return self.plan.start + self.actual_duration


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"JSON file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _role_duration(roles_path: Path) -> float | None:
    """Read an optional generated duration, preserving trailing planned gaps."""
    data = read_json(roles_path)
    if not isinstance(data, dict) or "duration" not in data:
        return None
    duration = _finite_number(data["duration"], "script_roles.duration")
    if duration <= 0:
        raise ValueError("script_roles.duration must be greater than zero")
    return duration


def _script_sentences(path: Path) -> list[str]:
    """Use the existing script contract, while rejecting index-shifting blanks."""
    raw = read_json(path) if path.suffix.lower() == ".json" else None
    if raw is None:
        sentences = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    else:
        data = raw.get("sentences") if isinstance(raw, dict) else raw
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError("Expected {'sentences': [...]} or a list of strings")
        sentences = [item.strip() for item in data]
    if not sentences or any(not sentence or not tts.normalized_text(sentence) for sentence in sentences):
        raise ValueError("Script must contain nonempty sentences with spoken characters")
    return sentences


def load_plan(script_path: Path, roles_path: Path) -> tuple[list[str], list[TimelineLine]]:
    sentences = _script_sentences(script_path)
    data = read_json(roles_path)
    if not isinstance(data, dict):
        raise ValueError(f"{roles_path} must contain a JSON object")

    voice_cast = data.get("voice_cast")
    timeline = data.get("timeline")
    if not isinstance(voice_cast, dict) or not voice_cast:
        raise ValueError("script_roles.json must contain a nonempty voice_cast object")
    if not isinstance(timeline, list) or not timeline:
        raise ValueError("script_roles.json must contain a nonempty timeline array")

    used_speakers = {
        item.get("speaker")
        for item in timeline
        if isinstance(item, dict) and isinstance(item.get("speaker"), str)
    }
    resolved_cast: dict[str, tuple[str, str | None]] = {}
    for speaker, role in voice_cast.items():
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("voice_cast keys must be nonempty speaker names")
        if speaker not in used_speakers:
            continue
        if not isinstance(role, dict):
            raise ValueError(f"voice_cast.{speaker} must be an object")
        env_name = role.get("reference_id_env")
        if env_name is not None and (not isinstance(env_name, str) or not env_name.strip()):
            raise ValueError(f"voice_cast.{speaker}.reference_id_env must be a nonempty string")
        direct_reference = role.get("reference_id")
        if direct_reference is not None and (not isinstance(direct_reference, str) or not direct_reference.strip()):
            raise ValueError(f"voice_cast.{speaker}.reference_id must be a nonempty string")
        if direct_reference:
            reference_id = direct_reference.strip()
        elif env_name:
            reference_id = os.getenv(env_name)
            if not reference_id:
                raise ValueError(
                    f"Missing reference voice for speaker '{speaker}': "
                    f"environment variable {env_name} is not set"
                )
        else:
            raise ValueError(
                f"voice_cast.{speaker} needs reference_id or reference_id_env"
            )
        resolved_cast[speaker] = (reference_id, env_name)

    seen: set[int] = set()
    entries: list[TimelineLine] = []
    for position, item in enumerate(timeline):
        if not isinstance(item, dict):
            raise ValueError(f"timeline[{position}] must be an object")
        index = item.get("sentence_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"timeline[{position}].sentence_index must be an integer")
        if index < 0 or index >= len(sentences):
            raise ValueError(
                f"timeline[{position}].sentence_index {index} is outside script.sentences"
            )
        if index in seen:
            raise ValueError(f"sentence_index {index} appears more than once in timeline")
        seen.add(index)

        speaker = item.get("speaker")
        if not isinstance(speaker, str) or speaker not in resolved_cast:
            known = ", ".join(sorted(resolved_cast))
            raise ValueError(
                f"timeline[{position}].speaker '{speaker}' is not in voice_cast "
                f"(known: {known})"
            )
        start = _finite_number(item.get("start"), f"timeline[{position}].start")
        end = _finite_number(item.get("end"), f"timeline[{position}].end")
        if start < 0 or end < start:
            raise ValueError(
                f"timeline[{position}] has invalid range: start={start}, end={end}"
            )
        reference_id, env_name = resolved_cast[speaker]
        entries.append(TimelineLine(index, speaker, sentences[index], start, end, reference_id, env_name))

    missing = sorted(set(range(len(sentences))) - seen)
    if missing:
        raise ValueError(
            "timeline is missing sentence_index values: " + ", ".join(map(str, missing))
        )
    by_sentence = sorted(entries, key=lambda line: line.sentence_index)
    if any(next_line.start + EPSILON < current.start for current, next_line in zip(by_sentence, by_sentence[1:])):
        raise ValueError(
            "timeline.start must be nondecreasing in script sentence order; "
            "timestamps_cpu.py aligns sentences in that order"
        )
    return sentences, sorted(entries, key=lambda line: (line.start, line.sentence_index))


def synthesize_line(
    line: TimelineLine,
    api_key: str,
    model: str,
    format_type: str,
    latency: str,
    sample_rate: int,
) -> bytes:
    """Call the existing Fish Audio implementation and return mono s16le PCM."""
    raw_audio, _ = tts.call_fish_audio_stream(
        text=line.text,
        api_key=api_key,
        model=model,
        reference_id=line.reference_id,
        format_type=format_type,
        latency=latency,
    )
    return tts.decode_audio(raw_audio, format_type, sample_rate)


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    if not pcm or len(pcm) % 2:
        raise ValueError("PCM audio must contain a nonempty number of complete int16 samples")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


def read_wav(path: Path, sample_rate: int) -> bytes:
    try:
        with wave.open(str(path), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise ValueError(f"Cached line must be mono 16-bit PCM WAV: {path}")
            if source.getframerate() != sample_rate:
                raise ValueError(
                    f"Cached line sample rate {source.getframerate()} != requested {sample_rate}: {path}"
                )
            pcm = source.readframes(source.getnframes())
    except (wave.Error, EOFError) as error:
        raise ValueError(f"Invalid cached line WAV: {path}") from error
    if not pcm:
        raise ValueError(f"Cached line is empty: {path}")
    return pcm


def speaker_filename(speaker: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", speaker).strip("_") or "speaker"
    return safe


def render_lines(
    timeline: Iterable[TimelineLine],
    lines_dir: Path,
    sample_rate: int,
    api_key: str,
    model: str,
    format_type: str,
    latency: str,
    reuse_lines: bool = False,
    synthesize: Callable[[TimelineLine, str, str, str, str, int], bytes] | None = None,
) -> list[RenderedLine]:
    lines_dir.mkdir(parents=True, exist_ok=True)
    synthesize = synthesize or synthesize_line
    rendered: list[RenderedLine] = []
    for line in timeline:
        output = lines_dir / f"{line.sentence_index:03d}_{speaker_filename(line.speaker)}.wav"
        if reuse_lines and output.exists():
            pcm = read_wav(output, sample_rate)
            print(f"[MultiVoice] Reusing {output}")
        else:
            print(f"[MultiVoice] Synthesizing sentence {line.sentence_index} ({line.speaker})")
            pcm = synthesize(line, api_key, model, format_type, latency, sample_rate)
            write_wav(output, pcm, sample_rate)
        if len(pcm) == 0 or len(pcm) % 2:
            raise ValueError(f"Sentence {line.sentence_index} returned invalid PCM")
        rendered.append(RenderedLine(line, pcm, output, sample_rate))
    return rendered


def find_overlaps(rendered: list[RenderedLine]) -> list[tuple[RenderedLine, RenderedLine]]:
    ordered = sorted(rendered, key=lambda item: (item.plan.start, item.plan.sentence_index))
    return [
        (current, following)
        for current, following in zip(ordered, ordered[1:])
        if current.actual_end > following.plan.start + EPSILON
    ]


def validate_timeline(rendered: list[RenderedLine], allow_overlap: bool) -> list[tuple[RenderedLine, RenderedLine]]:
    overlaps = find_overlaps(rendered)
    if overlaps and not allow_overlap:
        current, following = overlaps[0]
        raise ValueError(
            f"Sentence {current.plan.sentence_index} overlaps next sentence.\n"
            f"planned:\n"
            f"start={current.plan.start:g}\n"
            f"next_start={following.plan.start:g}\n\n"
            f"actual:\n"
            f"duration={current.actual_duration:.3f}\n"
            f"end={current.actual_end:.3f}\n\n"
            "Please:\n"
            "1. shorten the text\n"
            "2. change the timeline\n"
            "3. adjust TTS speed\n"
            "or pass --allow-overlap for intentional PCM mixing"
        )
    # planned_end is retained for reporting; only crossing the next line's
    # planned start is a timing conflict.
    return overlaps


def pcm_samples(pcm: bytes) -> list[int]:
    if len(pcm) % 2:
        raise ValueError("PCM contains an incomplete int16 sample")
    samples = array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    return list(samples)


def samples_pcm(samples: list[int]) -> bytes:
    clipped = array("h", (max(-32768, min(32767, value)) for value in samples))
    if sys.byteorder != "little":
        clipped.byteswap()
    return clipped.tobytes()


def mix_pcm(rendered: list[RenderedLine], total_frames: int, sample_rate: int) -> bytes:
    mixed = [0] * total_frames
    for item in rendered:
        start_frame = round(item.plan.start * sample_rate)
        samples = pcm_samples(item.pcm)
        end_frame = start_frame + len(samples)
        if start_frame < 0 or end_frame > total_frames:
            raise ValueError(
                f"Sentence {item.plan.sentence_index} exceeds fixed output duration "
                f"at {item.actual_end:.3f}s"
            )
        for offset, sample in enumerate(samples):
            mixed[start_frame + offset] += sample
    return samples_pcm(mixed)


def _summary(text: str, limit: int = 24) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def write_report(path: Path, rendered: list[RenderedLine], total_frames: int, sample_rate: int,
                 overlaps: list[tuple[RenderedLine, RenderedLine]]) -> None:
    payload = {
        "duration": round(total_frames / sample_rate, 3),
        "sample_rate": sample_rate,
        "channels": 1,
        "lines": [
            {
                "sentence_index": item.plan.sentence_index,
                "speaker": item.plan.speaker,
                "text": item.plan.text,
                "planned_start": item.plan.start,
                "planned_end": item.plan.end,
                "actual_duration": round(item.actual_duration, 3),
                "actual_end": round(item.actual_end, 3),
                "line_file": str(item.path),
            }
            for item in rendered
        ],
        "overlaps": [
            {
                "sentence_index": current.plan.sentence_index,
                "next_sentence_index": following.plan.sentence_index,
                "actual_end": round(current.actual_end, 3),
                "next_start": following.plan.start,
            }
            for current, following in overlaps
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tick_step(duration: float) -> float:
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300):
        if duration / step <= 12:
            return float(step)
    return 600.0


def write_timeline_svg(path: Path, rendered: list[RenderedLine], total_duration: float) -> None:
    roles: list[str] = []
    for item in rendered:
        if item.plan.speaker not in roles:
            roles.append(item.plan.speaker)
    colors = ["#2f6fed", "#d44a3a", "#1a9c70", "#8b5cf6", "#d28a18", "#0f766e"]
    left, plot_width, right = 190, 1120, 40
    header, row_height, detail_gap, detail_row = 64, 48, 32, 26
    detail_top = header + len(roles) * row_height + detail_gap
    height = detail_top + max(1, len(rendered)) * detail_row + 34
    scale = plot_width / max(total_duration, 1.0)
    step = _tick_step(total_duration)
    role_index = {role: i for i, role in enumerate(roles)}

    out: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + plot_width + right}" '
        f'height="{height}" viewBox="0 0 {left + plot_width + right} {height}">',
        '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#182230}'
        '.muted{fill:#607080;font-size:12px}.label{font-size:14px;font-weight:600}'
        '.small{font-size:11px}.bar{stroke:#fff;stroke-width:1}</style>',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        f'<text x="{left}" y="24" class="label">Audio timeline · {total_duration:.3f}s</text>',
        f'<text x="{left}" y="44" class="muted">ticks every {step:g}s · fixed mono {rendered[0].sample_rate if rendered else 24000}Hz</text>',
    ]
    for tick in range(0, int(math.floor(total_duration / step)) + 1):
        seconds = tick * step
        x = left + seconds * scale
        out.append(f'<line x1="{x:.2f}" y1="{header - 8}" x2="{x:.2f}" y2="{detail_top - 8}" stroke="#d8e0ea"/>')
        out.append(f'<text x="{x + 2:.2f}" y="{header - 16}" class="muted">{seconds:g}</text>')
    if total_duration > 0 and (int(total_duration / step) * step) < total_duration:
        x = left + total_duration * scale
        out.append(f'<line x1="{x:.2f}" y1="{header - 8}" x2="{x:.2f}" y2="{detail_top - 8}" stroke="#d8e0ea"/>')
        out.append(f'<text x="{x + 2:.2f}" y="{header - 16}" class="muted">{total_duration:g}</text>')

    for role, index in role_index.items():
        y = header + index * row_height
        out.append(f'<text x="18" y="{y + 29}" class="label">{html.escape(role)}</text>')
        out.append(f'<line x1="{left}" y1="{y + row_height}" x2="{left + plot_width}" y2="{y + row_height}" stroke="#e4e9f0"/>')

    for item in rendered:
        y = header + role_index[item.plan.speaker] * row_height + 12
        x = left + item.plan.start * scale
        width = max(2.0, item.actual_duration * scale)
        color = colors[role_index[item.plan.speaker] % len(colors)]
        label = f"#{item.plan.sentence_index} {_summary(item.plan.text, 18)}"
        title = (f"sentence {item.plan.sentence_index} · {item.plan.speaker} · "
                 f"start {item.plan.start:.3f}s · end {item.actual_end:.3f}s · {item.plan.text}")
        out.append(f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="24" rx="3" fill="{color}" class="bar"><title>{html.escape(title)}</title></rect>')
        if width >= 90:
            out.append(f'<text x="{x + 5:.2f}" y="{y + 16}" fill="#fff" style="fill:#fff" class="small">{html.escape(label)}</text>')

    out.append(f'<text x="18" y="{detail_top - 14}" class="label">Lines</text>')
    for index, item in enumerate(rendered):
        y = detail_top + index * detail_row
        detail = (f"#{item.plan.sentence_index}  {item.plan.speaker}  "
                  f"{item.plan.start:.3f}–{item.actual_end:.3f}s  {_summary(item.plan.text, 72)}")
        out.append(f'<text x="18" y="{y + 17}" class="small">{html.escape(detail)}</text>')
    out.append("</svg>\n")
    path.write_text("\n".join(out), encoding="utf-8")


def run_alignment(
    script_path: Path,
    full_wav: Path,
    out_dir: Path,
    backend: str,
    model: str,
    model_dir: str | None,
    chunk_sec: float,
) -> None:
    timestamps = out_dir / "timestamps.json"
    timing = out_dir / "timing.json"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "timestamps_cpu.py"),
        str(full_wav),
        str(script_path),
        str(timestamps),
        "--backend",
        backend,
        "--chunk-sec",
        str(chunk_sec),
    ]
    if backend == "whisper":
        command.extend(["--model", model])
    if model_dir:
        command.extend(["--model-dir", model_dir])
    subprocess.run(command, check=True)
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "make_timing.py"), str(timestamps), str(timing)],
        check=True,
    )


def build_project(
    script_path: Path,
    roles_path: Path,
    out_dir: Path,
    api_key: str,
    model: str = tts.DEFAULT_MODEL,
    format_type: str = tts.DEFAULT_FORMAT,
    latency: str = tts.DEFAULT_LATENCY,
    sample_rate: int = 24000,
    duration: float | None = None,
    allow_overlap: bool = False,
    reuse_lines: bool = False,
    run_alignment_step: bool = True,
    alignment_backend: str = "firered",
    alignment_model: str = "small",
    alignment_model_dir: str | None = None,
    alignment_chunk_sec: float = 75.0,
    synthesize: Callable[[TimelineLine, str, str, str, str, int], bytes] | None = None,
    alignment_runner: Callable[..., None] | None = None,
) -> dict[str, Path]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be a positive finite number")
    _, timeline = load_plan(script_path, roles_path)
    rendered = render_lines(
        timeline,
        out_dir / "lines",
        sample_rate,
        api_key,
        model,
        format_type,
        latency,
        reuse_lines,
        synthesize,
    )
    overlaps = validate_timeline(rendered, allow_overlap)
    planned_duration = max(item.plan.end for item in rendered)
    actual_duration = max(item.actual_end for item in rendered)
    role_duration = _role_duration(roles_path)
    total_duration = duration if duration is not None else (role_duration or planned_duration)
    if total_duration + EPSILON < planned_duration:
        raise ValueError(
            f"fixed duration {total_duration:.3f}s is shorter than timeline end {planned_duration:.3f}s"
        )
    if actual_duration > total_duration + EPSILON:
        raise ValueError(
            f"audio ends at {actual_duration:.3f}s beyond fixed output duration {total_duration:.3f}s"
        )
    total_frames = round(total_duration * sample_rate)
    full_pcm = mix_pcm(rendered, total_frames, sample_rate)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_wav = out_dir / "full.wav"
    write_wav(full_wav, full_pcm, sample_rate)
    report = out_dir / "audio_report.json"
    write_report(report, rendered, total_frames, sample_rate, overlaps)
    timeline_svg = out_dir / "audio_timeline.svg"
    write_timeline_svg(timeline_svg, rendered, total_frames / sample_rate)

    outputs = {"full_wav": full_wav, "report": report, "timeline_svg": timeline_svg}
    if run_alignment_step:
        runner = alignment_runner or run_alignment
        runner(
            script_path,
            full_wav,
            out_dir,
            alignment_backend,
            alignment_model,
            alignment_model_dir,
            alignment_chunk_sec,
        )
        outputs["timestamps"] = out_dir / "timestamps.json"
        outputs["timing"] = out_dir / "timing.json"
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True, help="script_roles.json")
    parser.add_argument("--out", type=Path, required=True, help="Output directory, normally audio/")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=os.getenv("FISH_AUDIO_MODEL", tts.DEFAULT_MODEL),
                        choices=["s1", "s2-pro", "s2.1-pro", "s2.1-pro-free", "drama-3-preview"])
    parser.add_argument("--format", dest="format_type", default=os.getenv("FISH_AUDIO_FORMAT", tts.DEFAULT_FORMAT),
                        choices=["mp3", "wav", "opus"], help="Fish Audio response format")
    parser.add_argument("--latency", default=os.getenv("FISH_AUDIO_LATENCY", tts.DEFAULT_LATENCY),
                        choices=["normal", "balanced", "low"])
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--duration", type=float, help="Fixed output duration in seconds; default=max timeline.end")
    parser.add_argument("--allow-overlap", action="store_true", help="Mix overlapping lines by PCM addition")
    parser.add_argument("--reuse-lines", action="store_true", help="Reuse existing line WAVs in --out/lines")
    parser.add_argument("--skip-alignment", action="store_true", help="Only build full.wav/report/SVG")
    parser.add_argument("--alignment-backend", choices=["firered", "whisper"], default="firered")
    parser.add_argument("--alignment-model", default="small", help="faster-whisper model when backend=whisper")
    parser.add_argument("--alignment-model-dir")
    parser.add_argument("--alignment-chunk-sec", type=float, default=75.0)
    return parser


def main() -> None:
    tts.load_env()
    args = build_parser().parse_args()
    api_key = args.api_key or os.getenv("FISH_AUDIO_API_KEY") or os.getenv("FISH_API_KEY")
    if not api_key:
        raise SystemExit("Set FISH_AUDIO_API_KEY in .env or pass --api-key")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required to decode Fish Audio responses")
    if args.alignment_chunk_sec <= 0 or not math.isfinite(args.alignment_chunk_sec):
        raise SystemExit("--alignment-chunk-sec must be positive and finite")
    try:
        outputs = build_project(
            args.script,
            args.roles,
            args.out,
            api_key,
            model=args.model,
            format_type=args.format_type,
            latency=args.latency,
            sample_rate=args.sample_rate,
            duration=args.duration,
            allow_overlap=args.allow_overlap,
            reuse_lines=args.reuse_lines,
            run_alignment_step=not args.skip_alignment,
            alignment_backend=args.alignment_backend,
            alignment_model=args.alignment_model,
            alignment_model_dir=args.alignment_model_dir,
            alignment_chunk_sec=args.alignment_chunk_sec,
        )
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"[MultiVoice] Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("✓ TTS generated")
    print("✓ timeline validated")
    print("✓ audio mixed")
    print(f"✓ full.wav generated: {outputs['full_wav']}")
    print(f"✓ audio report: {outputs['report']}")
    print(f"✓ timeline SVG: {outputs['timeline_svg']}")
    if "timestamps" in outputs:
        print(f"✓ timestamps generated: {outputs['timestamps']}")
        print(f"✓ timing.json generated: {outputs['timing']}")


if __name__ == "__main__":
    main()
