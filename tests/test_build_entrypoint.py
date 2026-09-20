import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import build


class BuildEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.episode = self.root / "ep01"
        self.episode.mkdir()
        (self.episode / "script.md").write_text(
            "# Test\n\n## 1-1 村口 日 外\n\n阿衡：\n未必。\n",
            encoding="utf-8",
        )
        (self.root / "voices.json").write_text(json.dumps({
            "aliases": {"阿衡": "aheng"},
            "voices": {"aheng": {"description": "阿衡", "reference_id": "direct-ref"}},
        }, ensure_ascii=False), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_review_command_only_generates_project_data(self):
        with patch.object(build, "ROOT", self.root):
            result = build.main([str(self.episode)])
        self.assertEqual(result, 0)
        self.assertTrue((self.episode / "generated" / "script_roles.json").is_file())
        self.assertFalse((self.episode / "audio" / "full.wav").exists())

    def test_audio_requires_explicit_confirmation(self):
        with patch.object(build, "ROOT", self.root):
            result = build.main([str(self.episode), "--all"])
        self.assertEqual(result, 1)
        self.assertFalse((self.episode / "audio" / "full.wav").exists())

    def test_confirmed_all_delegates_to_existing_audio_builder(self):
        fake_outputs = {
            "full_wav": self.episode / "audio" / "full.wav",
            "report": self.episode / "audio" / "audio_report.json",
            "timeline_svg": self.episode / "audio" / "audio_timeline.svg",
            "timestamps": self.episode / "audio" / "timestamps.json",
            "timing": self.episode / "audio" / "timing.json",
        }
        with patch.object(build, "ROOT", self.root), patch.object(
            build, "build_audio_for_episode", return_value=fake_outputs
        ) as audio_build:
            result = build.main([str(self.episode), "--all", "--confirm"])
        self.assertEqual(result, 0)
        audio_build.assert_called_once()
        self.assertEqual(audio_build.call_args.args[0], self.episode)
        self.assertEqual(audio_build.call_args.args[1], self.episode / "generated")


if __name__ == "__main__":
    unittest.main()
