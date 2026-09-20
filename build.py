#!/usr/bin/env python3
"""Build an episode's screenplay data, or its complete audio pipeline.

Review first:
    python3 build.py episodes/ep01

After reviewing generated/script_roles.json and generated/audio_timeline.svg:
    python3 build.py episodes/ep01 --all --confirm
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_audio, parse_script  # noqa: E402


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def find_voices(episode_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        path = resolve_path(requested)
        if not path.is_file():
            raise parse_script.ParseError(f"voices.json not found: {path}")
        return path
    candidates = [episode_dir / "voices.json", ROOT / "voices.json"]
    for path in candidates:
        if path.is_file():
            return path
    raise parse_script.ParseError(
        "No voices.json found. Put one in the episode directory or repository root, "
        "or pass --voices PATH."
    )


def ensure_generated(
    script_path: Path,
    voices_path: Path,
    generated_dir: Path,
    *,
    force_parse: bool = False,
) -> tuple[dict, bool]:
    """Return a validated summary and whether parsing wrote new files."""
    current = parse_script.generated_is_current(generated_dir, script_path, voices_path)
    if force_parse or not current:
        summary = parse_script.parse_project(script_path, voices_path, generated_dir)
        return summary, True
    try:
        summary = parse_script.validate_generated(generated_dir)
    except parse_script.ParseError as error:
        raise parse_script.ParseError(
            f"Generated files are current by hash but invalid: {error}\n"
            "Fix or remove the generated files, then rerun the review."
        ) from error
    return summary, False


def print_generation_status(summary: dict, generated_dir: Path, *, wrote: bool) -> None:
    print("✓ generated audio project" if wrote else "✓ reused current generated audio project")
    parse_script.print_review(summary)
    print(f"\nGenerated directory: {generated_dir}")


def require_confirmation(confirm: bool) -> None:
    if not confirm:
        raise RuntimeError(
            "Audio build stopped before Fish Audio. Review generated/script_roles.json "
            "and generated/audio_timeline.svg, then rerun with --confirm."
        )


def build_audio_for_episode(
    episode_dir: Path,
    generated_dir: Path,
    audio_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Path]:
    build_audio.tts.load_env()
    api_key = args.api_key or os.getenv("FISH_AUDIO_API_KEY") or os.getenv("FISH_API_KEY")
    if not api_key:
        raise RuntimeError("Set FISH_AUDIO_API_KEY in .env or pass --api-key")
    if not args.reuse_lines and not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for Fish Audio response decoding")
    if args.alignment_chunk_sec <= 0 or not math.isfinite(args.alignment_chunk_sec):
        raise RuntimeError("--alignment-chunk-sec must be positive and finite")

    return build_audio.build_project(
        generated_dir / "script.json",
        generated_dir / "script_roles.json",
        audio_dir,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", type=Path, help="Episode directory containing script.md")
    parser.add_argument("--voices", type=Path, help="Shared or episode-local voices.json")
    parser.add_argument("--audio", action="store_true", help="Build audio after parsing and confirmation")
    parser.add_argument("--all", action="store_true", help="Parse, validate, and build audio")
    parser.add_argument("--confirm", action="store_true", help="Confirm the generated project for audio synthesis")
    parser.add_argument("--force-parse", action="store_true", help="Regenerate generated/* before review")
    parser.add_argument("--api-key")
    parser.add_argument("--model", default=os.getenv("FISH_AUDIO_MODEL", build_audio.tts.DEFAULT_MODEL))
    parser.add_argument("--format", dest="format_type", default=os.getenv("FISH_AUDIO_FORMAT", build_audio.tts.DEFAULT_FORMAT),
                        choices=["mp3", "wav", "opus"])
    parser.add_argument("--latency", default=os.getenv("FISH_AUDIO_LATENCY", build_audio.tts.DEFAULT_LATENCY),
                        choices=["normal", "balanced", "low"])
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--duration", type=float, help="Override generated output duration in seconds")
    parser.add_argument("--allow-overlap", action="store_true", help="Allow intentional PCM mixing of overlapping lines")
    parser.add_argument("--reuse-lines", action="store_true", help="Reuse existing audio/lines/*.wav")
    parser.add_argument("--skip-alignment", action="store_true", help="Debug only: stop after full.wav")
    parser.add_argument("--alignment-backend", choices=["firered", "whisper"], default="firered")
    parser.add_argument("--alignment-model", default="small")
    parser.add_argument("--alignment-model-dir")
    parser.add_argument("--alignment-chunk-sec", type=float, default=75.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    episode_dir = resolve_path(args.episode)
    script_path = episode_dir / "script.md"
    generated_dir = episode_dir / "generated"
    audio_dir = episode_dir / "audio"
    if not script_path.is_file():
        print(f"[Build] script.md not found: {script_path}", file=sys.stderr)
        return 1
    try:
        voices_path = find_voices(episode_dir, args.voices)
        summary, wrote = ensure_generated(
            script_path,
            voices_path,
            generated_dir,
            force_parse=args.force_parse,
        )
        print_generation_status(summary, generated_dir, wrote=wrote)
        wants_audio = args.audio or args.all
        if not wants_audio:
            return 0
        require_confirmation(args.confirm)
        outputs = build_audio_for_episode(episode_dir, generated_dir, audio_dir, args)
        print("\n✓ TTS generated")
        print("✓ timeline validated")
        print("✓ audio mixed")
        print(f"✓ full.wav generated: {outputs['full_wav']}")
        print(f"✓ audio report: {outputs['report']}")
        print(f"✓ timeline SVG: {outputs['timeline_svg']}")
        if "timestamps" in outputs:
            print(f"✓ timestamps generated: {outputs['timestamps']}")
            print(f"✓ timing.json generated: {outputs['timing']}")
        return 0
    except (parse_script.ParseError, RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"[Build] Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
