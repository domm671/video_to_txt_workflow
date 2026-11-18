import io
import json
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from workflow import ANALYSIS_MARKER, CORRECTED_MARKER, api_settings, complete, run


class WorkflowTests(unittest.TestCase):
    def test_api_settings_reads_toml(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "api_config.toml"
            path.write_text(
                'base_url = "https://example.com/v1/"\nmodel = "test-model"\napi_key = "test-key"\n',
                encoding="utf-8-sig",
            )
            self.assertEqual(
                api_settings(path), ("https://example.com/v1/chat/completions", "test-key", "test-model", {})
            )

    def test_api_settings_rejects_incomplete_config(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "api_config.toml"
            path.write_text('api_key = ""\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "请在 api_config.toml 中填写"):
                api_settings(path)

    @patch("workflow.urlopen")
    def test_ark_stream_uses_pi_request_parameters_and_collects_only_answer(self, urlopen):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "api_config.toml"
            path.write_text(
                'base_url = "https://ark.cn-beijing.volces.com/api/plan/v3"\n'
                'model = "deepseek-v4.1-flash"\napi_key = "test-key"\n'
                'stream = true\nmax_completion_tokens = 393216\nreasoning_effort = "low"\n',
                encoding="utf-8",
            )
            settings = api_settings(path)

        corrected = "这是校订正文。"
        analysis = "# 思维结构与思维模型分析\n\n分析正文。"
        events = [
            {"choices": [{"delta": {"reasoning_content": "中间推理"}}]},
            {"choices": [{"delta": {"content": f"{CORRECTED_MARKER}\n{corrected}\n<<<思"}}]},
            {"choices": [{"delta": {"content": f"维分析>>>\n{analysis}"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 120}},
        ]
        data = "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events)
        urlopen.return_value = io.BytesIO((data + "data: [DONE]\n\n").encode("utf-8"))
        self.assertEqual(complete(settings, "原文"), (corrected, analysis))
        urlopen.assert_called_once()
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://ark.cn-beijing.volces.com/api/plan/v3/chat/completions")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "deepseek-v4.1-flash")
        self.assertEqual(payload["max_completion_tokens"], 393216)
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertNotIn("max_tokens", payload)

    def test_api_settings_rejects_invalid_reasoning_effort(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "api_config.toml"
            for value in ('""', '"   "', "1", "true"):
                with self.subTest(value=value):
                    path.write_text(
                        'base_url = "https://example.com/v1"\nmodel = "test-model"\napi_key = "test-key"\n'
                        f'reasoning_effort = {value}\n', encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, "reasoning_effort"):
                        api_settings(path)

    @patch("workflow.urlopen")
    def test_interrupted_stream_is_rejected(self, urlopen):
        urlopen.return_value = io.BytesIO(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
        settings = ("https://example.com/chat/completions", "test-key", "test-model", {"stream": True})
        with self.assertRaisesRegex(RuntimeError, "未返回完整"):
            complete(settings, "原文")

    @patch("workflow.urlopen")
    def test_full_transcript_and_both_outputs_use_one_request(self, urlopen):
        transcript = "这是连续的原始段落。\n" * 600
        corrected = "这是一段校订后的正文。"
        analysis = "作者围绕实际问题展开思考。\n\n- 先比较，再验证。"
        content = f"{CORRECTED_MARKER}\r\n{corrected}\r\n{ANALYSIS_MARKER}\r\n{analysis}"
        response = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        urlopen.return_value = io.BytesIO(json.dumps(response).encode("utf-8"))
        settings = ("https://example.com/chat/completions", "test-key", "test-model", {"max_tokens": 8000})

        self.assertEqual(complete(settings, transcript), (corrected, analysis))
        urlopen.assert_called_once()
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["messages"][1]["content"], transcript)
        self.assertEqual(payload["max_tokens"], 8000)

    def test_output_files_include_original_video_name(self):
        def transcribe(command, **kwargs):
            if command[0] == "python":
                Path(command[3]).write_text("原始转录。", encoding="utf-8-sig")
            return CompletedProcess(command, 0)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "原视频 名称.第二期.mp4"
            with (
                patch("workflow.BASE_DIR", root),
                patch("workflow.python_with_dependencies", return_value="python"),
                patch("workflow.ffmpeg_executable", return_value="ffmpeg"),
                patch("workflow.subprocess.run", side_effect=transcribe),
                patch("workflow.complete", return_value=("校订稿。", "自由组织的分析。")),
            ):
                output = run(video, ())

            expected = {
                f"{video.stem}_原始转录.txt": "原始转录。",
                f"{video.stem}_勘误后.txt": "校订稿。",
                f"{video.stem}_思维分析.md": "自由组织的分析。",
            }
            self.assertEqual({file.name for file in output.iterdir()}, set(expected))
            for name, content in expected.items():
                self.assertEqual((output / name).read_text(encoding="utf-8-sig").strip(), content)

    @patch("workflow.urlopen")
    def test_response_missing_analysis_is_rejected(self, urlopen):
        content = f"{CORRECTED_MARKER}\n校订后的正文。"
        response = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        urlopen.return_value = io.BytesIO(json.dumps(response).encode("utf-8"))
        with self.assertRaisesRegex(RuntimeError, "返回格式有误"):
            complete(("https://example.com/chat/completions", "test-key", "test-model", {}), "原文")

    @patch("workflow.urlopen")
    def test_truncated_completion_is_rejected(self, urlopen):
        response = {"choices": [{"finish_reason": "length", "message": {"content": "不完整"}}]}
        urlopen.return_value = io.BytesIO(json.dumps(response).encode("utf-8"))
        with self.assertRaisesRegex(RuntimeError, "达到长度上限"):
            complete(("https://example.com/chat/completions", "test-key", "test-model", {}), "原文")
