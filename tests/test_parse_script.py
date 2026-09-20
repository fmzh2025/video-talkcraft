import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import build_audio, parse_script


SCRIPT = """# 白鹿 EP01：山南山北尽寻鹿

## 1-1 山村溪口 日 外

小孩们：
深山白鹿，见者得富贵——获者改命数！

村中质疑者：鹿若真在山里，早被旁人得去了。

财主：
鹿出没的道儿，你早该报上来。
财主：报出路，这些，都是你的。

阿衡（低声）：
未必。得鹿者，未必肯说。

## 1-2 深山兽径 昏 外

村民：追了半辈子……连个影儿都没。

阿衡OS：
可我追了半生，连一句“见过”都没挣上。

阿衡沿着兽迹前行。

## 1-3 溪边宿地 夜 外

旁白：得鹿的人，未必肯说……
"""


VOICES = {
    "aliases": {
        "阿衡": "aheng",
        "阿衡OS": "aheng_os",
        "财主": "tycoon",
        "村中质疑者": "skeptic",
        "小孩们": "children",
        "村民": "villagers",
        "旁白": "narrator",
    },
    "voices": {
        "aheng": {
            "description": "阿衡",
            "reference_id": "ref-aheng",
            "speech_rate": 3.0,
        },
        "aheng_os": {
            "description": "阿衡OS",
            "reference_id": "ref-aheng-os",
            "speech_rate": 3.0,
        },
        "tycoon": {"description": "财主", "reference_id": "ref-tycoon", "speech_rate": 3.5},
        "skeptic": {"description": "村中质疑者", "reference_id": "ref-skeptic", "speech_rate": 4.0},
        "children": {"description": "小孩们", "reference_id": "ref-children", "speech_rate": 4.5},
        "villagers": {"description": "村民", "reference_id": "ref-villagers", "speech_rate": 4.0},
        "narrator": {"description": "旁白", "reference_id": "ref-narrator", "speech_rate": 3.0},
    },
    "timing": {"line_gap": 0.35, "scene_gap": 1.5, "action_duration": 1.0},
}


class ParseScriptTests(unittest.TestCase):
    def files(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        script = root / "script.md"
        voices = root / "voices.json"
        script.write_text(SCRIPT, encoding="utf-8")
        voices.write_text(json.dumps(VOICES, ensure_ascii=False), encoding="utf-8")
        return temp, root, script, voices

    def test_markdown_scenes_and_dialogue_are_extracted_without_actions(self):
        temp, root, script, voices = self.files()
        self.addCleanup(temp.cleanup)
        catalog = parse_script.load_voices(voices)
        title, scenes = parse_script.parse_markdown(script, catalog)
        self.assertEqual(title, "白鹿 EP01：山南山北尽寻鹿")
        self.assertEqual([scene.scene_id for scene in scenes], ["1-1", "1-2", "1-3"])
        events = [event for scene in scenes for event in scene.events]
        dialogue = [event for event in events if event["kind"] == "dialogue"]
        actions = [event for event in events if event["kind"] == "action"]
        self.assertEqual(len(dialogue), 8)
        self.assertEqual([event["speaker"] for event in dialogue], [
            "children", "skeptic", "tycoon", "tycoon", "aheng", "villagers", "aheng_os", "narrator",
        ])
        self.assertEqual(len(actions), 1)
        self.assertIn("沿着兽迹", actions[0]["text"])
        self.assertNotIn("沿着兽迹", [event["text"] for event in dialogue])

    def test_alias_parenthetical_and_os_normalize_to_canonical_speakers(self):
        temp, root, script, voices = self.files()
        self.addCleanup(temp.cleanup)
        catalog = parse_script.load_voices(voices)
        self.assertEqual(catalog.resolve("阿衡（低声）").key, "aheng")
        self.assertEqual(catalog.resolve("阿衡OS").key, "aheng_os")
        self.assertEqual(catalog.resolve("**财主**").key, "tycoon")

    def test_multiline_dialogue_keeps_source_line_breaks(self):
        temp, root, script, voices = self.files()
        self.addCleanup(temp.cleanup)
        script.write_text("阿衡：\n第一句。\n第二句。\n", encoding="utf-8")
        catalog = parse_script.load_voices(voices)
        _, scenes = parse_script.parse_markdown(script, catalog)
        dialogue = [event for event in scenes[0].events if event["kind"] == "dialogue"]
        self.assertEqual(dialogue[0]["text"], "第一句。\n第二句。")

    def test_unknown_speaker_is_blocking_and_names_the_label(self):
        temp, root, script, voices = self.files()
        self.addCleanup(temp.cleanup)
        script.write_text("## 1-1 村口 日 外\n\n老人：\n别走。\n", encoding="utf-8")
        with self.assertRaisesRegex(parse_script.ParseError, r"Unknown speaker: 老人"):
            parse_script.parse_project(script, voices, root / "generated")

    def test_estimation_keeps_line_and_scene_gaps(self):
        temp, root, script, voices = self.files()
        self.addCleanup(temp.cleanup)
        catalog = parse_script.load_voices(voices)
        _, scenes = parse_script.parse_markdown(script, catalog)
        timeline = parse_script.build_timeline("title", scenes, catalog)
        self.assertEqual(timeline["timing_confidence"], "estimated")
        self.assertGreater(timeline["estimated_duration"], sum(row["estimated_duration"] for row in timeline["dialogue"]))
        reasons = {gap["reason"] for gap in timeline["gaps"]}
        self.assertIn("line_gap", reasons)
        self.assertIn("scene_gap", reasons)
        self.assertIn("action", reasons)
        self.assertTrue(all(row["end"] > row["start"] for row in timeline["dialogue"]))

    def test_generated_files_are_consistent_and_source_is_unchanged(self):
        temp, root, script, voices = self.files()
        self.addCleanup(temp.cleanup)
        original = script.read_bytes()
        summary = parse_script.parse_project(script, voices, root / "generated")
        self.assertEqual(summary["dialogue_lines"], 8)
        self.assertEqual(script.read_bytes(), original)
        valid = parse_script.validate_generated(root / "generated")
        self.assertEqual(valid["dialogue_lines"], 8)
        self.assertEqual(valid["scenes"], 3)
        generated_script = json.loads((root / "generated" / "script.json").read_text(encoding="utf-8"))
        generated_roles = json.loads((root / "generated" / "script_roles.json").read_text(encoding="utf-8"))
        self.assertEqual(generated_script["sentences"][0], "深山白鹿，见者得富贵——获者改命数！")
        self.assertEqual(len(generated_roles["timeline"]), len(generated_script["sentences"]))
        self.assertTrue((root / "generated" / "audio_timeline.svg").is_file())

    def test_generated_project_feeds_audio_builder_with_mock_tts(self):
        temp, root, script, voices = self.files()
        self.addCleanup(temp.cleanup)
        out = root / "generated"
        parse_script.parse_project(script, voices, out)
        outputs = build_audio.build_project(
            out / "script.json",
            out / "script_roles.json",
            root / "audio",
            "mock-key",
            sample_rate=100,
            run_alignment_step=False,
            synthesize=lambda *args: b"\x00\x00" * 10,
        )
        self.assertTrue(outputs["full_wav"].is_file())
        report = json.loads(outputs["report"].read_text(encoding="utf-8"))
        self.assertEqual(report["channels"], 1)
        self.assertEqual(len(report["lines"]), 8)

    def test_cli_validate(self):
        temp, root, script, voices = self.files()
        self.addCleanup(temp.cleanup)
        out = root / "generated"
        parse_script.parse_project(script, voices, out)
        result = subprocess.run(
            [sys.executable, "scripts/parse_script.py", "--validate", str(out)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("generated files valid", result.stdout)


if __name__ == "__main__":
    unittest.main()
