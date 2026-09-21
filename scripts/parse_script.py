#!/usr/bin/env python3
"""Parse a Markdown screenplay into video-talkcraft audio project files.

This layer only extracts source dialogue. It never invents narration or
rewrites dialogue. Timing is an explicit estimate used to create an editable
audio plan; the final duration and word timing come from build_audio.py and
the existing CPU alignment pipeline.

Parse:
    python3 scripts/parse_script.py \
        --script episodes/ep01/script.md \
        --voices voices.json \
        --out episodes/ep01/generated \
        --review

Validate generated files:
    python3 scripts/parse_script.py --validate episodes/ep01/generated
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EPSILON = 1e-6
DEFAULT_SPEECH_RATE = 4.0
DEFAULT_LINE_GAP = 0.35
DEFAULT_SCENE_GAP = 1.5
DEFAULT_ACTION_DURATION = 1.0

SCENE_TIME_WORDS = {
    "日", "夜", "晨", "早", "午", "午后", "黄昏", "昏", "傍晚", "凌晨", "清晨", "深夜",
}
SETTING_WORDS = {
    "内": "内", "外": "外", "内外": "内外", "INT": "内", "INT.": "内",
    "EXT": "外", "EXT.": "外", "INT/EXT": "内外", "INT./EXT.": "内外",
}
SCENE_DURATION_RE = re.compile(r"\s*\{duration\s*=\s*([0-9]+(?:\.[0-9]+)?)\}\s*$", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
SPEAKER_ONLY_RE = re.compile(r"^\s*([^:#\n][^:\n]{0,60}?)\s*[：:]\s*$")
SPEAKER_INLINE_RE = re.compile(r"^\s*([^:#\n][^:\n]{0,60}?)\s*[：:]\s*(.+?)\s*$")
PAREN_RE = re.compile(r"[（(\[【][^）)\]】]*[）)\]】]")
SPEECH_CHAR_RE = re.compile(r"[一-鿿㐀-䶿A-Za-z0-9]")


class ParseError(ValueError):
    """A user-fixable screenplay or voice configuration error."""


@dataclass(frozen=True)
class Voice:
    key: str
    description: str
    reference_id_env: str | None
    reference_id: str | None
    speech_rate: float
    aliases: tuple[str, ...]


@dataclass
class VoiceCatalog:
    voices: dict[str, Voice]
    aliases: dict[str, str]
    alias_display: dict[str, str]
    default_speech_rate: float
    line_gap: float
    scene_gap: float
    action_duration: float

    def resolve(self, label: str) -> Voice:
        key = normalize_speaker_label(label)
        canonical = self.aliases.get(key)
        if canonical is None:
            known = ", ".join(sorted(self.voices))
            raise ParseError(
                f"Unknown speaker: {label.strip()}\n"
                f"Please map:\n"
                f"{label.strip()} -> existing voice\n"
                f"Known voices: {known}"
            )
        return self.voices[canonical]


@dataclass
class Scene:
    scene_id: str
    heading: str
    location: str
    time: str
    setting: str
    source_line: int
    events: list[dict[str, Any]] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0
    duration: float | None = None


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ParseError(f"JSON file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ParseError(f"Invalid JSON in {path}: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: Any, field_name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ParseError(f"{field_name} must be finite and >= {minimum:g}")
    return result


def normalize_speaker_label(label: str) -> str:
    """Normalize presentation variants without guessing a different character."""
    value = label.strip().strip("*`_")
    value = PAREN_RE.sub("", value)
    value = re.sub(r"\s+", "", value)
    return value.casefold()


def _voice_entries(data: dict[str, Any]) -> dict[str, Any]:
    entries = data.get("voices", data.get("voice_cast"))
    if not isinstance(entries, dict) or not entries:
        raise ParseError("voices.json must contain a nonempty 'voices' object")
    return entries


def load_voices(path: Path) -> VoiceCatalog:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ParseError("voices.json must contain a JSON object")
    entries = _voice_entries(data)
    timing = data.get("timing") if isinstance(data.get("timing"), dict) else {}
    rate_config = data.get("speech_rate", timing.get("speech_rate", {}))
    if not isinstance(rate_config, dict):
        raise ParseError("speech_rate must be an object mapping speaker to characters/second")
    default_rate = data.get("default_speech_rate", timing.get("default_speech_rate", DEFAULT_SPEECH_RATE))
    default_rate = number(default_rate, "default_speech_rate", minimum=0.1)

    voices: dict[str, Voice] = {}
    alias_map: dict[str, str] = {}
    alias_display: dict[str, str] = {}

    def add_alias(alias: str, canonical: str) -> None:
        if not isinstance(alias, str) or not alias.strip():
            raise ParseError(f"Alias for {canonical} must be a nonempty string")
        normalized = normalize_speaker_label(alias)
        previous = alias_map.get(normalized)
        if previous is not None and previous != canonical:
            raise ParseError(f"Speaker alias '{alias}' maps to both {previous} and {canonical}")
        alias_map[normalized] = canonical
        alias_display[normalized] = alias

    for canonical, raw_role in entries.items():
        if not isinstance(canonical, str) or not canonical.strip():
            raise ParseError("voices keys must be nonempty speaker ids")
        if not isinstance(raw_role, dict):
            raise ParseError(f"voices.{canonical} must be an object")
        reference_id_env = raw_role.get("reference_id_env")
        if reference_id_env is not None and (not isinstance(reference_id_env, str) or not reference_id_env.strip()):
            raise ParseError(f"voices.{canonical}.reference_id_env must be a nonempty string")
        reference_id = raw_role.get("reference_id")
        if reference_id is not None and (not isinstance(reference_id, str) or not reference_id.strip()):
            raise ParseError(f"voices.{canonical}.reference_id must be a nonempty string")
        if reference_id_env is None and reference_id is None:
            raise ParseError(
                f"voices.{canonical} needs reference_id_env or reference_id for Fish Audio"
            )
        role_rate = raw_role.get("speech_rate", rate_config.get(canonical, default_rate))
        role_rate = number(role_rate, f"speech_rate.{canonical}", minimum=0.1)
        description = raw_role.get("description", raw_role.get("display_name", canonical))
        if not isinstance(description, str) or not description.strip():
            description = canonical
        raw_aliases = raw_role.get("aliases", [])
        if isinstance(raw_aliases, str):
            raw_aliases = [raw_aliases]
        if not isinstance(raw_aliases, list) or not all(isinstance(alias, str) for alias in raw_aliases):
            raise ParseError(f"voices.{canonical}.aliases must be a string array")
        aliases = tuple([description, canonical, *raw_aliases])
        voices[canonical] = Voice(
            key=canonical,
            description=description,
            reference_id_env=reference_id_env,
            reference_id=reference_id,
            speech_rate=role_rate,
            aliases=aliases,
        )
        for alias in aliases:
            add_alias(alias, canonical)

    top_aliases = data.get("aliases", {})
    if not isinstance(top_aliases, dict):
        raise ParseError("aliases must be an object mapping screenplay names to voice ids")
    for alias, canonical in top_aliases.items():
        if canonical not in voices:
            raise ParseError(f"Alias '{alias}' points to unknown voice '{canonical}'")
        add_alias(alias, canonical)

    def pause(name: str, default: float) -> float:
        return number(timing.get(name, default), f"timing.{name}")

    return VoiceCatalog(
        voices=voices,
        aliases=alias_map,
        alias_display=alias_display,
        default_speech_rate=default_rate,
        line_gap=pause("line_gap", DEFAULT_LINE_GAP),
        scene_gap=pause("scene_gap", DEFAULT_SCENE_GAP),
        action_duration=pause("action_duration", DEFAULT_ACTION_DURATION),
    )


def parse_scene_heading(text: str, source_line: int, scene_number: int) -> Scene | None:
    original = text.strip()
    duration: float | None = None
    duration_match = SCENE_DURATION_RE.search(original)
    if duration_match:
        duration = number(float(duration_match.group(1)), "scene duration", minimum=0.001)
        heading_text = original[:duration_match.start()].strip()
    else:
        heading_text = original
        if "{duration" in original.lower():
            raise ParseError(
                f"Invalid scene duration in heading at line {source_line}: {original}"
            )

    tokens = heading_text.split()
    if not tokens:
        return None
    first_is_id = bool(re.match(r"^(?:\d+(?:[-.]\d+)+|S\d+[A-Za-z0-9_-]*)$", tokens[0], re.IGNORECASE))
    slash_time_setting = re.fullmatch(r"([^/\s]+)/([^/\s]+)", tokens[-1]) if tokens else None
    if slash_time_setting:
        time_token = slash_time_setting.group(1)
        setting_token = slash_time_setting.group(2).upper()
    else:
        setting_token = tokens[-1].upper() if tokens else ""
        time_token = tokens[-2] if len(tokens) >= 2 else ""
    is_scene = first_is_id or (time_token in SCENE_TIME_WORDS and setting_token in SETTING_WORDS)
    if not is_scene:
        return None
    scene_id = tokens[0] if first_is_id else f"scene-{scene_number:02d}"
    body = tokens[1:] if first_is_id else tokens
    setting = SETTING_WORDS.get(setting_token, "")
    time = time_token if setting else ""
    if slash_time_setting:
        body = body[:-1]
    elif setting:
        body = body[:-2]
    location = " ".join(body).strip() or heading_text
    return Scene(scene_id, original, location, time, setting, source_line, duration=duration)


def is_stage_direction(text: str) -> bool:
    return bool(re.match(r"^(?:（.*）|\(.*\)|\[.*\]|【.*】|\*.*\*)$", text.strip()))


def _new_scene(scene_number: int) -> Scene:
    return Scene(f"scene-{scene_number:02d}", "未命名场景", "", "", "", 1)


def parse_markdown(path: Path, catalog: VoiceCatalog) -> tuple[str, list[Scene]]:
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ParseError(f"Script not found: {path}") from error
    lines = source.splitlines()
    title = ""
    scenes: list[Scene] = []
    current_scene: Scene | None = None
    current_voice: Voice | None = None
    current_voice_label = ""
    dialogue_buffer: list[str] = []
    dialogue_start_line = 0

    def ensure_scene() -> Scene:
        nonlocal current_scene
        if current_scene is None:
            current_scene = _new_scene(1)
            scenes.append(current_scene)
        return current_scene

    def flush_dialogue_with_line() -> None:
        nonlocal dialogue_buffer, current_voice, current_voice_label, dialogue_start_line
        if current_voice is not None and dialogue_buffer:
            text = "\n".join(part for part in dialogue_buffer).strip()
            if text:
                ensure_scene().events.append({
                    "kind": "dialogue",
                    "speaker": current_voice.key,
                    "speaker_label": current_voice_label,
                    "text": text,
                    "source_line": dialogue_start_line,
                })
        dialogue_buffer = []
        current_voice = None
        current_voice_label = ""
        dialogue_start_line = 0

    for line_number, raw_line in enumerate(lines, 1):
        stripped = raw_line.strip()
        heading_match = HEADING_RE.match(raw_line)
        if heading_match:
            flush_dialogue_with_line()
            heading = heading_match.group(1).strip()
            scene = parse_scene_heading(heading, line_number, len(scenes) + 1)
            if scene is not None:
                current_scene = scene
                scenes.append(scene)
            elif not title:
                title = heading
            continue
        if not stripped:
            flush_dialogue_with_line()
            continue
        if is_stage_direction(stripped):
            flush_dialogue_with_line()
            ensure_scene().events.append({
                "kind": "action",
                "text": stripped,
                "source_line": line_number,
            })
            continue

        inline = SPEAKER_INLINE_RE.match(stripped)
        if inline:
            flush_dialogue_with_line()
            current_voice = catalog.resolve(inline.group(1))
            current_voice_label = inline.group(1).strip()
            dialogue_start_line = line_number
            ensure_scene().events.append({
                "kind": "dialogue",
                "speaker": current_voice.key,
                "speaker_label": current_voice_label,
                "text": inline.group(2).strip(),
                "source_line": line_number,
            })
            current_voice = None
            current_voice_label = ""
            dialogue_start_line = 0
            continue

        speaker_only = SPEAKER_ONLY_RE.match(stripped)
        if speaker_only:
            flush_dialogue_with_line()
            current_voice = catalog.resolve(speaker_only.group(1))
            current_voice_label = speaker_only.group(1).strip()
            dialogue_start_line = line_number
            continue

        if current_voice is not None:
            if dialogue_start_line == 0:
                dialogue_start_line = line_number
            dialogue_buffer.append(stripped)
        else:
            ensure_scene().events.append({
                "kind": "action",
                "text": stripped,
                "source_line": line_number,
            })
    flush_dialogue_with_line()
    if not any(event["kind"] == "dialogue" for scene in scenes for event in scene.events):
        raise ParseError("No dialogue lines found in the Markdown screenplay")
    return title, scenes


def estimate_duration(text: str, speech_rate: float) -> float:
    units = len(SPEECH_CHAR_RE.findall(text))
    punctuation_pause = (
        text.count("，") * 0.10
        + text.count(",") * 0.10
        + text.count("、") * 0.10
        + text.count("；") * 0.18
        + text.count(";") * 0.18
        + text.count("：") * 0.12
        + text.count(":") * 0.12
        + text.count("。") * 0.28
        + text.count("！") * 0.30
        + text.count("!") * 0.30
        + text.count("？") * 0.30
        + text.count("?") * 0.30
        + text.count("……") * 0.42
        + text.count("...") * 0.42
    )
    return round(max(0.25, units / speech_rate + punctuation_pause), 3)


def build_timeline(title: str, scenes: list[Scene], catalog: VoiceCatalog) -> dict[str, Any]:
    dialogue: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    cursor = 0.0
    sentence_index = 0
    has_scene_durations = any(scene.duration is not None for scene in scenes)

    def add_gap(start: float, end: float, reason: str, scene_id: str) -> None:
        if end - start > EPSILON:
            gaps.append({"start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3),
                         "reason": reason, "scene_id": scene_id})

    for scene_index, scene in enumerate(scenes):
        if scene_index > 0:
            previous = cursor
            previous_scene = scenes[scene_index - 1]
            if not (
                has_scene_durations
                and previous_scene.duration is not None
                and scene.duration is not None
            ):
                cursor += catalog.scene_gap
                add_gap(previous, cursor, "scene_gap", scene.scene_id)
        scene_start = cursor
        previous_kind = "scene"
        scene_actions: list[dict[str, Any]] = []
        for event in scene.events:
            if event["kind"] == "action":
                action_start = cursor
                cursor += catalog.action_duration
                action = {
                    "kind": "action",
                    "scene_id": scene.scene_id,
                    "text": event["text"],
                    "source_line": event["source_line"],
                    "start": round(action_start, 3),
                    "end": round(cursor, 3),
                    "estimated_duration": round(catalog.action_duration, 3),
                }
                actions.append(action)
                scene_actions.append(action)
                add_gap(action_start, cursor, "action", scene.scene_id)
                previous_kind = "action"
                continue
            if previous_kind == "dialogue":
                previous = cursor
                cursor += catalog.line_gap
                add_gap(previous, cursor, "line_gap", scene.scene_id)
            voice = catalog.voices[event["speaker"]]
            estimated = estimate_duration(event["text"], voice.speech_rate)
            start = cursor
            end = start + estimated
            row = {
                "sentence_index": sentence_index,
                "speaker": voice.key,
                "speaker_label": event["speaker_label"],
                "text": event["text"],
                "scene_id": scene.scene_id,
                "source_line": event["source_line"],
                "estimated_duration": estimated,
                "planned_start": round(start, 3),
                "planned_end": round(end, 3),
                "start": round(start, 3),
                "end": round(end, 3),
                "timing_confidence": "estimated",
            }
            dialogue.append(row)
            sentence_index += 1
            cursor = end
            previous_kind = "dialogue"
        content_end = cursor
        if scene.duration is not None:
            required = content_end - scene_start
            if scene.duration + EPSILON < required:
                raise ParseError(
                    f"Scene {scene.scene_id} duration is too short. "
                    f"planned={scene.duration:.3f}s, required={required:.3f}s"
                )
            scene.start = round(scene_start, 3)
            scene.end = round(scene_start + scene.duration, 3)
            cursor = scene.end
            gap_end = content_end
            if scene_index < len(scenes) - 1 and scenes[scene_index + 1].duration is not None:
                gap_end = min(content_end + catalog.scene_gap, cursor)
                add_gap(content_end, gap_end, "scene_gap", scene.scene_id)
            add_gap(gap_end, cursor, "scene_duration", scene.scene_id)
        else:
            scene.start = round(scene_start, 3)
            scene.end = round(content_end, 3)

    scene_rows = []
    dialogue_by_scene = {scene.scene_id: [] for scene in scenes}
    for row in dialogue:
        dialogue_by_scene[row["scene_id"]].append(row["sentence_index"])
    actions_by_scene = {scene.scene_id: [] for scene in scenes}
    for action in actions:
        actions_by_scene[action["scene_id"]].append(action)
    for scene in scenes:
        scene_dialogue = [row for row in dialogue if row["scene_id"] == scene.scene_id]
        scene_actions = actions_by_scene[scene.scene_id]
        scene_points = [point for row in scene_dialogue for point in (row["start"], row["end"])]
        scene_points.extend(point for action in scene_actions for point in (action["start"], action["end"]))
        if scene.duration is not None:
            start, end = scene.start, scene.end
        else:
            start = min(scene_points) if scene_points else scene.start
            end = max(scene_points) if scene_points else scene.end
        scene_rows.append({
            "scene_id": scene.scene_id,
            "heading": scene.heading,
            "location": scene.location,
            "time": scene.time,
            "setting": scene.setting,
            "source_line": scene.source_line,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "duration_planned": scene.duration,
            "duration_source": "specified" if scene.duration is not None else "estimated",
            "dialogue_indices": dialogue_by_scene[scene.scene_id],
            "actions": scene_actions,
        })
    return {
        "title": title,
        "scene_duration_schema": "v1",
        "timing_confidence": "estimated",
        "duration_source": (
            "specified"
            if all(scene.duration is not None for scene in scenes)
            else "mixed" if any(scene.duration is not None for scene in scenes) else "estimated"
        ),
        "duration": round(cursor, 3),
        "estimated_duration": round(cursor, 3),
        "scenes": scene_rows,
        "dialogue": dialogue,
        "actions": actions,
        "gaps": gaps,
    }


def generated_voice_cast(catalog: VoiceCatalog) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for voice in catalog.voices.values():
        role: dict[str, Any] = {
            "description": voice.description,
            "speech_rate": voice.speech_rate,
        }
        if voice.reference_id_env:
            role["reference_id_env"] = voice.reference_id_env
        if voice.reference_id:
            role["reference_id"] = voice.reference_id
        result[voice.key] = role
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _tick_step(duration: float) -> float:
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300):
        if duration / step <= 12:
            return float(step)
    return 600.0


def write_audio_timeline_svg(path: Path, timeline: dict[str, Any]) -> None:
    rows = timeline["dialogue"]
    scenes = timeline.get("scenes", [])
    roles: list[str] = []
    for row in rows:
        if row["speaker"] not in roles:
            roles.append(row["speaker"])
    duration = max(float(timeline["estimated_duration"]), 1.0)
    left, plot_width, right = 190, 1120, 40
    header, row_height, detail_top = 64, 48, 64 + len(roles) * 48 + 32
    scene_detail_row = 25
    lines_top = detail_top + max(1, len(scenes)) * scene_detail_row + 28
    detail_row = 26
    height = lines_top + max(1, len(rows)) * detail_row + 34
    scale = plot_width / duration
    role_index = {role: index for index, role in enumerate(roles)}
    colors = ["#2f6fed", "#d44a3a", "#1a9c70", "#8b5cf6", "#d28a18", "#0f766e"]
    step = _tick_step(duration)
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + plot_width + right}" height="{height}" viewBox="0 0 {left + plot_width + right} {height}">',
        '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#182230}.muted{fill:#607080;font-size:12px}.label{font-size:14px;font-weight:600}.small{font-size:11px}</style>',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        f'<text x="{left}" y="24" class="label">Audio timeline · {duration:.3f}s</text>',
        f'<text x="{left}" y="44" class="muted">duration source: {html.escape(timeline.get("duration_source", "estimated"))}</text>',
    ]
    for tick in range(int(math.floor(duration / step)) + 1):
        seconds = tick * step
        x = left + seconds * scale
        out.append(f'<line x1="{x:.2f}" y1="56" x2="{x:.2f}" y2="{lines_top - 8}" stroke="#d8e0ea"/>')
        out.append(f'<text x="{x + 2:.2f}" y="48" class="muted">{seconds:g}</text>')
    for scene in scenes:
        start_x = left + scene["start"] * scale
        end_x = left + scene["end"] * scale
        out.append(f'<line x1="{start_x:.2f}" y1="56" x2="{start_x:.2f}" y2="{lines_top - 8}" stroke="#7a8798" stroke-width="1.5"/>')
        out.append(f'<line x1="{end_x:.2f}" y1="56" x2="{end_x:.2f}" y2="{lines_top - 8}" stroke="#7a8798" stroke-width="1.5"/>')
        out.append(f'<title>Scene {html.escape(scene["scene_id"])}: {scene["start"]:.3f}-{scene["end"]:.3f}s</title>')
    for role, index in role_index.items():
        y = header + index * row_height
        out.append(f'<text x="18" y="{y + 29}" class="label">{html.escape(role)}</text>')
        out.append(f'<line x1="{left}" y1="{y + row_height}" x2="{left + plot_width}" y2="{y + row_height}" stroke="#e4e9f0"/>')
    for row in rows:
        x = left + row["start"] * scale
        width = max(2.0, row["estimated_duration"] * scale)
        y = header + role_index[row["speaker"]] * row_height + 12
        color = colors[role_index[row["speaker"]] % len(colors)]
        title = f"#{row['sentence_index']} {row['speaker']} {row['start']:.3f}-{row['end']:.3f}s {row['text']}"
        out.append(f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="24" rx="3" fill="{color}"><title>{html.escape(title)}</title></rect>')
        if width >= 90:
            label = f"#{row['sentence_index']} {html.escape(row['text'][:18])}"
            out.append(f'<text x="{x + 5:.2f}" y="{y + 16}" fill="#fff" style="fill:#fff" class="small">{label}</text>')
    out.append(f'<text x="18" y="{detail_top - 14}" class="label">Scenes</text>')
    for index, scene in enumerate(scenes):
        y = detail_top + index * scene_detail_row
        detail = (f"{scene['scene_id']}  {scene['start']:.3f}-{scene['end']:.3f}s  "
                  f"duration={scene['duration']:.3f}s  {scene['duration_source']}  {scene['heading']}")
        out.append(f'<text x="18" y="{y + 17}" class="small">{html.escape(detail)}</text>')
    out.append(f'<text x="18" y="{lines_top - 14}" class="label">Lines</text>')
    for index, row in enumerate(rows):
        y = lines_top + index * detail_row
        detail = f"#{row['sentence_index']}  {row['speaker']}  {row['start']:.3f}-{row['end']:.3f}s  {row['text'][:72]}"
        out.append(f'<text x="18" y="{y + 17}" class="small">{html.escape(detail)}</text>')
    out.append("</svg>\n")
    path.write_text("\n".join(out), encoding="utf-8")


def parse_project(script_path: Path, voices_path: Path, out_dir: Path) -> dict[str, Any]:
    catalog = load_voices(voices_path)
    title, scenes = parse_markdown(script_path, catalog)
    timeline = build_timeline(title, scenes, catalog)
    source_hash = sha256_file(script_path)
    voices_hash = sha256_file(voices_path)
    sentences = [row["text"] for row in timeline["dialogue"]]
    roles_timeline = [
        {
            "sentence_index": row["sentence_index"],
            "speaker": row["speaker"],
            "speaker_label": row["speaker_label"],
            "text": row["text"],
            "scene_id": row["scene_id"],
            "source_line": row["source_line"],
            "estimated_duration": row["estimated_duration"],
            "planned_start": row["planned_start"],
            "planned_end": row["planned_end"],
            "start": row["start"],
            "end": row["end"],
            "timing_confidence": "estimated",
        }
        for row in timeline["dialogue"]
    ]
    roles = {
        "schema": "video-talkcraft.script-roles.v1",
        "source_sha256": source_hash,
        "voices_sha256": voices_hash,
        "timing_confidence": "estimated",
        "duration_source": timeline["duration_source"],
        "duration": timeline["estimated_duration"],
        "voice_cast": generated_voice_cast(catalog),
        "timeline": roles_timeline,
    }
    timeline["source_sha256"] = source_hash
    timeline["voices_sha256"] = voices_hash
    timeline["voice_cast"] = generated_voice_cast(catalog)
    _write_json(out_dir / "script.json", {"sentences": sentences})
    _write_json(out_dir / "script_roles.json", roles)
    _write_json(out_dir / "timeline.json", timeline)
    write_audio_timeline_svg(out_dir / "audio_timeline.svg", timeline)
    result = review_generated(out_dir)
    result["script_path"] = script_path
    result["voices_path"] = voices_path
    result["out_dir"] = out_dir
    return result


def generated_is_current(out_dir: Path, script_path: Path, voices_path: Path) -> bool:
    timeline_path = out_dir / "timeline.json"
    if not timeline_path.is_file():
        return False
    try:
        data = read_json(timeline_path)
        return (
            data.get("scene_duration_schema") == "v1"
            and data.get("source_sha256") == sha256_file(script_path)
            and data.get("voices_sha256") == sha256_file(voices_path)
            and all(
                isinstance(scene, dict)
                and "duration" in scene
                and "duration_source" in scene
                for scene in data.get("scenes", [])
            )
        )
    except (ParseError, OSError, KeyError, TypeError):
        return False


def validate_generated(out_dir: Path) -> dict[str, Any]:
    script_path = out_dir / "script.json"
    roles_path = out_dir / "script_roles.json"
    timeline_path = out_dir / "timeline.json"
    for path in (script_path, roles_path, timeline_path, out_dir / "audio_timeline.svg"):
        if not path.is_file():
            raise ParseError(f"Missing generated file: {path}")
    script = read_json(script_path)
    roles = read_json(roles_path)
    timeline = read_json(timeline_path)
    if not isinstance(script, dict) or not isinstance(script.get("sentences"), list):
        raise ParseError("generated/script.json must contain sentences[]")
    if not isinstance(timeline, dict):
        raise ParseError("generated/timeline.json must contain an object")
    sentences = script["sentences"]
    if not sentences or any(not isinstance(text, str) or not text.strip() for text in sentences):
        raise ParseError("generated/script.json contains empty or invalid text")
    if not isinstance(roles, dict) or not isinstance(roles.get("voice_cast"), dict) or not roles["voice_cast"]:
        raise ParseError("generated/script_roles.json must contain a nonempty voice_cast")
    for speaker, role in roles["voice_cast"].items():
        if not isinstance(role, dict):
            raise ParseError(f"voice_cast.{speaker} must be an object")
        if not role.get("reference_id_env") and not role.get("reference_id"):
            raise ParseError(f"voice_cast.{speaker} needs reference_id_env or reference_id")
    rows = roles.get("timeline")
    if not isinstance(rows, list):
        raise ParseError("generated/script_roles.json must contain timeline[]")
    if len(rows) != len(sentences):
        raise ParseError(f"script.json has {len(sentences)} sentences but timeline has {len(rows)} lines")
    seen: set[int] = set()
    ordered: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ParseError(f"timeline[{position}] must be an object")
        index = row.get("sentence_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in seen or index < 0 or index >= len(sentences):
            raise ParseError(f"timeline[{position}] has invalid or duplicate sentence_index")
        seen.add(index)
        speaker = row.get("speaker")
        if not isinstance(speaker, str) or speaker not in roles["voice_cast"]:
            raise ParseError(f"timeline[{position}] uses unknown speaker: {speaker}")
        text = row.get("text", sentences[index])
        if text != sentences[index] or not isinstance(text, str) or not text.strip():
            raise ParseError(f"timeline[{position}] text differs from script.json sentence {index}")
        start = number(row.get("start"), f"timeline[{position}].start")
        end = number(row.get("end"), f"timeline[{position}].end")
        if end <= start:
            raise ParseError(f"timeline[{position}] must have start < end")
        ordered.append(row)
    if seen != set(range(len(sentences))):
        raise ParseError("timeline sentence_index values must be continuous")
    timeline_rows = timeline.get("dialogue") if isinstance(timeline, dict) else None
    if not isinstance(timeline_rows, list) or len(timeline_rows) != len(rows):
        raise ParseError("timeline.json dialogue must match script_roles.json timeline")
    for position, row in enumerate(rows):
        mirror = timeline_rows[position]
        if not isinstance(mirror, dict):
            raise ParseError(f"timeline.json dialogue[{position}] must be an object")
        for field in ("sentence_index", "speaker", "text", "start", "end"):
            if mirror.get(field) != row.get(field):
                raise ParseError(
                    f"timeline.json dialogue[{position}].{field} differs from script_roles.json"
                )
    ordered.sort(key=lambda row: (float(row["start"]), row["sentence_index"]))
    overlaps = []
    for current, following in zip(ordered, ordered[1:]):
        if float(current["end"]) > float(following["start"]) + EPSILON:
            overlaps.append((current["sentence_index"], following["sentence_index"]))
    if overlaps:
        raise ParseError(f"planned timeline overlaps: {overlaps}")

    scene_rows = timeline.get("scenes")
    if not isinstance(scene_rows, list) or not scene_rows:
        raise ParseError("generated/timeline.json must contain scenes[]")
    scene_map: dict[str, dict[str, Any]] = {}
    previous_end = None
    for position, scene in enumerate(scene_rows):
        if not isinstance(scene, dict):
            raise ParseError(f"timeline.scenes[{position}] must be an object")
        scene_id = scene.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id or scene_id in scene_map:
            raise ParseError(f"timeline.scenes[{position}] has invalid or duplicate scene_id")
        start = number(scene.get("start"), f"timeline.scenes[{position}].start")
        end = number(scene.get("end"), f"timeline.scenes[{position}].end")
        duration = number(scene.get("duration", end - start), f"timeline.scenes[{position}].duration")
        if end <= start or abs(duration - (end - start)) > EPSILON:
            raise ParseError(f"timeline.scenes[{position}] duration does not match start/end")
        if scene.get("duration_source") not in (None, "estimated", "specified"):
            raise ParseError(f"timeline.scenes[{position}] has invalid duration_source")
        if scene.get("duration_source") == "specified":
            planned = number(scene.get("duration_planned"), f"timeline.scenes[{position}].duration_planned")
            if abs(planned - duration) > EPSILON:
                raise ParseError(f"timeline.scenes[{position}] duration differs from duration_planned")
        scene_map[scene_id] = scene
        if timeline.get("duration_source") == "specified" and previous_end is not None:
            if abs(start - previous_end) > EPSILON:
                raise ParseError("specified scene durations must form a continuous timeline")
        previous_end = end
    if timeline.get("duration_source") == "specified":
        total = number(timeline.get("duration", timeline.get("estimated_duration")), "timeline.duration")
        if abs(total - previous_end) > EPSILON:
            raise ParseError("timeline.duration must equal the end of the final scene")
    for row in timeline_rows:
        scene = scene_map.get(row.get("scene_id"))
        if scene is None:
            raise ParseError(f"dialogue uses unknown scene: {row.get('scene_id')}")
        if float(row["start"]) < float(scene["start"]) - EPSILON or float(row["end"]) > float(scene["end"]) + EPSILON:
            raise ParseError(f"dialogue {row['sentence_index']} falls outside scene {row['scene_id']}")
    used = {row["speaker"] for row in rows}
    warnings = [f"unused voice: {key}" for key in sorted(set(roles["voice_cast"]) - used)]
    if not isinstance(timeline, dict) or timeline.get("timing_confidence") != "estimated":
        warnings.append("timeline.json has no estimated timing_confidence marker")
    return {
        "scenes": len(scene_rows),
        "scene_rows": scene_rows,
        "duration_source": timeline.get("duration_source", "estimated"),
        "speakers": sorted(used),
        "speaker_labels": {
            speaker: sorted({row.get("speaker_label", speaker) for row in rows if row.get("speaker") == speaker})
            for speaker in sorted(used)
        },
        "dialogue_lines": len(rows),
        "duration": float(roles.get("duration", 0)),
        "warnings": warnings,
        "out_dir": out_dir,
    }


def review_generated(out_dir: Path) -> dict[str, Any]:
    data = read_json(out_dir / "timeline.json")
    roles = read_json(out_dir / "script_roles.json")
    rows = roles.get("timeline", [])
    used = {row["speaker"] for row in rows}
    return {
        "scenes": len(data.get("scenes", [])),
        "scene_rows": data.get("scenes", []),
        "duration_source": data.get("duration_source", "estimated"),
        "speakers": sorted(used),
        "dialogue_lines": len(rows),
        "duration": float(data.get("estimated_duration", roles.get("duration", 0))),
        "speaker_labels": {
            speaker: sorted({row.get("speaker_label", speaker) for row in rows if row.get("speaker") == speaker})
            for speaker in sorted({row["speaker"] for row in rows})
        },
        "warnings": [f"unused voice: {key}" for key in sorted(set(roles.get("voice_cast", {})) - used)],
        "out_dir": out_dir,
    }


def print_review(summary: dict[str, Any]) -> None:
    print(f"Detected scenes: {summary['scenes']}")
    print(f"Detected speakers: {len(summary['speakers'])}")
    print(f"Detected dialogue lines: {summary['dialogue_lines']}")
    print("\nSpeakers:")
    for speaker in summary["speakers"]:
        labels = ", ".join(summary.get("speaker_labels", {}).get(speaker, []))
        print(f"  {speaker} -> {labels or speaker}")
    print("\nScenes:")
    for scene in summary.get("scene_rows", []):
        print(
            f"  {scene['scene_id']:<8} {scene['start']:.1f} -> {scene['end']:.1f} "
            f"({scene.get('duration', scene['end'] - scene['start']):.1f}s)"
        )
    print(f"\nTotal duration: {summary['duration']:.1f}s ({summary.get('duration_source', 'estimated')})")
    for warning in summary.get("warnings", []):
        print(f"WARNING: {warning}")
    print(f"\nPlease review:\n{summary['out_dir'] / 'script_roles.json'}\n{summary['out_dir'] / 'audio_timeline.svg'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate", type=Path, help="Validate an existing generated directory")
    modes.add_argument("--script", type=Path, help="Markdown screenplay to parse")
    parser.add_argument("--voices", type=Path, help="voices.json; required when parsing")
    parser.add_argument("--out", type=Path, help="Generated output directory; required when parsing")
    parser.add_argument("--review", action="store_true", help="Print the review summary after parsing")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.validate:
            summary = validate_generated(args.validate)
            print("✓ generated files valid")
            print_review(summary)
            return
        if not args.voices or not args.out:
            raise ParseError("--voices and --out are required when parsing --script")
        result = parse_project(args.script, args.voices, args.out)
        print("✓ generated script.json")
        print("✓ generated script_roles.json")
        print("✓ generated timeline.json")
        print("✓ generated audio_timeline.svg")
        if args.review:
            print_review(result)
    except (ParseError, OSError) as error:
        print(f"[ParseScript] Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
