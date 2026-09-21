import json
import os
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from scripts import build_audio


def tone(value: int, samples: int) -> bytes:
    return struct.pack(f"<{samples}h", *([value] * samples))


class BuildAudioTests(unittest.TestCase):
    def plan_files(self, root: Path, sentences=None, timeline=None):
        sentences = sentences or ["甲说。", "乙说。"]
        timeline = timeline or [
            {"sentence_index": 0, "speaker": "aheng", "start": 0, "end": 1},
            {"sentence_index": 1, "speaker": "skeptic", "start": 1.5, "end": 2.5},
        ]
        script = root / "script.json"
        roles = root / "script_roles.json"
        script.write_text(json.dumps({"sentences": sentences}, ensure_ascii=False), encoding="utf-8")
        roles.write_text(json.dumps({
            "voice_cast": {
                "aheng": {"description": "阿衡", "reference_id_env": "TEST_REF_AHENG"},
                "skeptic": {"description": "质疑者", "reference_id_env": "TEST_REF_SKEPTIC"},
            },
            "timeline": timeline,
        }, ensure_ascii=False), encoding="utf-8")
        return script, roles

    def fake_synth(self, durations):
        def synth(line, api_key, model, format_type, latency, sample_rate):
            self.assertEqual(api_key, "test-key")
            self.assertEqual(model, build_audio.tts.DEFAULT_MODEL)
            self.assertEqual(format_type, "mp3")
            self.assertEqual(latency, "balanced")
            return tone(1000 + line.sentence_index * 1000, durations[line.sentence_index])

        return synth

    def test_roles_map_to_env_reference_ids_without_sending_speaker_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script, roles = self.plan_files(root)
            with patch.dict(os.environ, {"TEST_REF_AHENG": "ref-a", "TEST_REF_SKEPTIC": "ref-s"}, clear=False):
                _, timeline = build_audio.load_plan(script, roles)
            self.assertEqual(timeline[0].reference_id, "ref-a")
            self.assertEqual(timeline[0].text, "甲说。")
            self.assertNotIn("aheng", timeline[0].text)

    def test_roles_can_use_direct_reference_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "script.json"
            roles = root / "script_roles.json"
            script.write_text(json.dumps({"sentences": ["甲说。"]}, ensure_ascii=False), encoding="utf-8")
            roles.write_text(json.dumps({
                "voice_cast": {
                    "aheng": {"description": "阿衡", "reference_id": "direct-ref"},
                    "unused": {"description": "未使用", "reference_id_env": "MISSING_UNUSED_REF"},
                },
                "timeline": [{"sentence_index": 0, "speaker": "aheng", "start": 0, "end": 1}],
            }, ensure_ascii=False), encoding="utf-8")
            _, timeline = build_audio.load_plan(script, roles)
            self.assertEqual(timeline[0].reference_id, "direct-ref")
            self.assertIsNone(timeline[0].reference_id_env)

    def test_generated_duration_preserves_trailing_silence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "script.json"
            roles = root / "script_roles.json"
            script.write_text(json.dumps({"sentences": ["甲说。"]}, ensure_ascii=False), encoding="utf-8")
            roles.write_text(json.dumps({
                "duration": 3,
                "voice_cast": {"aheng": {"reference_id": "direct-ref"}},
                "timeline": [{"sentence_index": 0, "speaker": "aheng", "start": 0, "end": 1}],
            }, ensure_ascii=False), encoding="utf-8")
            outputs = build_audio.build_project(
                script,
                roles,
                root / "audio",
                "test-key",
                sample_rate=10,
                run_alignment_step=False,
                synthesize=lambda *args: tone(1000, 5),
            )
            with wave.open(str(outputs["full_wav"]), "rb") as audio:
                self.assertEqual(audio.getnframes(), 30)

    def test_timeline_is_sorted_by_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script, roles = self.plan_files(root, timeline=[
                {"sentence_index": 1, "speaker": "skeptic", "start": 5, "end": 6},
                {"sentence_index": 0, "speaker": "aheng", "start": 1, "end": 2},
            ])
            with patch.dict(os.environ, {"TEST_REF_AHENG": "ref-a", "TEST_REF_SKEPTIC": "ref-s"}, clear=False):
                _, timeline = build_audio.load_plan(script, roles)
            self.assertEqual([line.sentence_index for line in timeline], [0, 1])

    def test_missing_reference_identifies_speaker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script, roles = self.plan_files(root)
            with patch.dict(os.environ, {"TEST_REF_AHENG": "ref-a"}, clear=False):
                os.environ.pop("TEST_REF_SKEPTIC", None)
                with self.assertRaisesRegex(ValueError, "speaker 'skeptic'.*TEST_REF_SKEPTIC"):
                    build_audio.load_plan(script, roles)

    def test_pcm_mixing_adds_overlaps_and_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script, roles = self.plan_files(root, timeline=[
                {"sentence_index": 0, "speaker": "aheng", "start": 0, "end": 1},
                {"sentence_index": 1, "speaker": "skeptic", "start": 0.5, "end": 1.5},
            ])
            with patch.dict(os.environ, {"TEST_REF_AHENG": "ref-a", "TEST_REF_SKEPTIC": "ref-s"}, clear=False):
                outputs = build_audio.build_project(
                    script,
                    roles,
                    root / "audio",
                    "test-key",
                    sample_rate=10,
                    allow_overlap=True,
                    run_alignment_step=False,
                    synthesize=self.fake_synth({0: 10, 1: 10}),
                )
            with wave.open(str(outputs["full_wav"]), "rb") as audio:
                samples = struct.unpack("<15h", audio.readframes(15))
            self.assertEqual(samples[:5], (1000,) * 5)
            self.assertEqual(samples[5:10], (3000,) * 5)
            self.assertEqual(samples[10:], (2000,) * 5)

            clipped = build_audio.mix_pcm([
                build_audio.RenderedLine(
                    build_audio.TimelineLine(0, "aheng", "甲说。", 0, 1, "ref-a", "TEST_REF_AHENG"),
                    tone(30000, 1),
                    root / "a.wav",
                    10,
                ),
                build_audio.RenderedLine(
                    build_audio.TimelineLine(1, "skeptic", "乙说。", 0, 1, "ref-s", "TEST_REF_SKEPTIC"),
                    tone(30000, 1),
                    root / "b.wav",
                    10,
                ),
            ], 1, 10)
            self.assertEqual(struct.unpack("<h", clipped)[0], 32767)

    def test_pcm_mixing_preserves_silence_between_lines(self):
        first = build_audio.RenderedLine(
            build_audio.TimelineLine(0, "aheng", "甲说。", 0, 0.1, "ref-a", "TEST_REF_AHENG"),
            tone(1200, 1),
            Path("first.wav"),
            10,
        )
        second = build_audio.RenderedLine(
            build_audio.TimelineLine(1, "skeptic", "乙说。", 0.3, 0.4, "ref-s", "TEST_REF_SKEPTIC"),
            tone(2400, 1),
            Path("second.wav"),
            10,
        )
        mixed = build_audio.mix_pcm([first, second], 5, 10)
        self.assertEqual(struct.unpack("<5h", mixed), (1200, 0, 0, 2400, 0))

    def test_overlap_fails_before_full_mix_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script, roles = self.plan_files(root, timeline=[
                {"sentence_index": 0, "speaker": "aheng", "start": 0, "end": 1},
                {"sentence_index": 1, "speaker": "skeptic", "start": 0.5, "end": 1.5},
            ])
            with patch.dict(os.environ, {"TEST_REF_AHENG": "ref-a", "TEST_REF_SKEPTIC": "ref-s"}, clear=False):
                with self.assertRaisesRegex(ValueError, "Sentence 0 overlaps next sentence"):
                    build_audio.build_project(
                        script,
                        roles,
                        root / "audio",
                        "test-key",
                        sample_rate=10,
                        run_alignment_step=False,
                        synthesize=self.fake_synth({0: 10, 1: 10}),
                    )
            self.assertFalse((root / "audio" / "full.wav").exists())

    def test_actual_end_may_exceed_planned_end_before_next_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script, roles = self.plan_files(root, timeline=[
                {"sentence_index": 0, "speaker": "aheng", "start": 0, "end": 0.8},
                {"sentence_index": 1, "speaker": "skeptic", "start": 1.5, "end": 2.5},
            ])
            with patch.dict(os.environ, {"TEST_REF_AHENG": "ref-a", "TEST_REF_SKEPTIC": "ref-s"}, clear=False):
                outputs = build_audio.build_project(
                    script,
                    roles,
                    root / "audio",
                    "test-key",
                    sample_rate=10,
                    run_alignment_step=False,
                    synthesize=self.fake_synth({0: 10, 1: 5}),
                )
            report = json.loads(outputs["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["lines"][0]["planned_end"], 0.8)
            self.assertEqual(report["lines"][0]["actual_end"], 1.0)
            self.assertTrue(outputs["full_wav"].exists())

    def test_alignment_runs_only_after_full_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script, roles = self.plan_files(root)
            calls = []

            def alignment(*args):
                calls.append(args)
                args[2].joinpath("timestamps.json").write_text("{}", encoding="utf-8")
                args[2].joinpath("timing.json").write_text("{}", encoding="utf-8")

            with patch.dict(os.environ, {"TEST_REF_AHENG": "ref-a", "TEST_REF_SKEPTIC": "ref-s"}, clear=False):
                outputs = build_audio.build_project(
                    script,
                    roles,
                    root / "audio",
                    "test-key",
                    sample_rate=10,
                    run_alignment_step=True,
                    alignment_runner=alignment,
                    synthesize=self.fake_synth({0: 5, 1: 5}),
                )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1], outputs["full_wav"])
            self.assertTrue(outputs["full_wav"].exists())
            self.assertTrue(outputs["timestamps"].exists())
            self.assertTrue(outputs["timing"].exists())
            report = json.loads(outputs["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["sample_rate"], 10)
            self.assertEqual(report["channels"], 1)
            self.assertEqual(report["lines"][0]["actual_duration"], 0.5)
            self.assertTrue(outputs["timeline_svg"].read_text(encoding="utf-8").startswith("<?xml"))

    def test_alignment_wrapper_calls_existing_scripts_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "script.json"
            full = root / "full.wav"
            out = root / "audio"
            out.mkdir()
            calls = []
            with patch.object(build_audio.subprocess, "run", side_effect=lambda command, check: calls.append(command)):
                build_audio.run_alignment(script, full, out, "whisper", "small", None, 75)
            self.assertEqual(len(calls), 2)
            self.assertIn("timestamps_cpu.py", calls[0][1])
            self.assertEqual(calls[0][-2:], ["--model", "small"])
            self.assertIn("make_timing.py", calls[1][1])
            self.assertEqual(calls[1][-2:], [str(out / "timestamps.json"), str(out / "timing.json")])

    def test_fish_call_is_reused_for_each_role_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script, roles = self.plan_files(root)
            calls = []

            def fake_call(**kwargs):
                calls.append(kwargs)
                return b"raw", []

            with patch.dict(os.environ, {"TEST_REF_AHENG": "ref-a", "TEST_REF_SKEPTIC": "ref-s"}, clear=False), \
                    patch.object(build_audio.tts, "call_fish_audio_stream", side_effect=fake_call), \
                    patch.object(build_audio.tts, "decode_audio", return_value=tone(100, 5)):
                build_audio.build_project(
                    script,
                    roles,
                    root / "audio",
                    "test-key",
                    sample_rate=10,
                    run_alignment_step=False,
                )
            self.assertEqual([call["reference_id"] for call in calls], ["ref-a", "ref-s"])
            self.assertEqual([call["text"] for call in calls], ["甲说。", "乙说。"])


if __name__ == "__main__":
    unittest.main()
