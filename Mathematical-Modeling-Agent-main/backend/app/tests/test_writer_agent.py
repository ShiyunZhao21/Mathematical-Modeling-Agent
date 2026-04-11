import asyncio
import os
import tempfile
import unittest
import uuid

from app.core.agents.writer_agent import WriterAgent
from app.schemas.A2A import GeneratedFigure, RequiredFigure

TEST_TMP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".tmp", "tests")
)
os.makedirs(TEST_TMP_ROOT, exist_ok=True)


def _make_tmp_dir() -> str:
    path = os.path.join(TEST_TMP_ROOT, f"writer_{uuid.uuid4().hex}")
    os.makedirs(path, exist_ok=True)
    return path


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = []


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeModel:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    async def chat(self, **kwargs):
        content = self.responses.pop(0) if self.responses else ""
        return _FakeResponse(content)


class TestWriterAgentImageValidation(unittest.TestCase):
    def test_extract_inserted_images_uses_file_names(self):
        content = """
        分析段落。

        ![图一](nested/path/figure_a.png)
        ![图二](figure_b.jpg?raw=1)
        """

        inserted = WriterAgent._extract_inserted_images(content)

        self.assertEqual(inserted, {"figure_a.png", "figure_b.jpg"})

    def test_find_missing_images_detects_uninserted_files(self):
        content = """
        前文分析。

        ![图一](figure_a.png)
        """

        missing = WriterAgent._find_missing_images(
            content,
            ["figure_a.png", "subdir/figure_b.png", "figure_c.jpg"],
        )

        self.assertEqual(missing, ["figure_b.png", "figure_c.jpg"])

    def test_find_invalid_image_references_detects_nonexistent_or_nested_paths(self):
        content = """
        ![图一](figure_a.png)
        ![图二](nested/figure_b.png)
        ![图三](ghost.png)
        """

        invalid = WriterAgent._find_invalid_image_references(
            content,
            {"figure_a.png", "figure_b.png"},
        )

        self.assertEqual(invalid, ["nested/figure_b.png", "ghost.png"])

    def test_ensure_images_inserted_raises_when_hard_validation_still_fails(self):
        tmp_dir = _make_tmp_dir()
        open(os.path.join(tmp_dir, "real_chart.png"), "wb").close()

        writer = WriterAgent(
            task_id="test",
            model=_FakeModel(
                [
                    "正文分析\n\n![错误图片](ghost.png)",
                    "正文分析\n\n![还是错误](nested/real_chart.png)",
                ]
            ),
            work_dir=tmp_dir,
        )
        writer.available_images = ["real_chart.png"]

        with self.assertRaises(ValueError):
            asyncio.run(
                writer._ensure_images_inserted(
                    response_content="正文分析\n\n![错误图片](ghost.png)",
                    tools=None,
                    tool_choice=None,
                    sub_title="ques1",
                )
            )

    def test_ensure_images_inserted_rejects_disk_image_not_in_manifest(self):
        tmp_dir = _make_tmp_dir()
        open(os.path.join(tmp_dir, "real_chart.png"), "wb").close()
        open(os.path.join(tmp_dir, "rogue_chart.png"), "wb").close()

        writer = WriterAgent(
            task_id="test",
            model=_FakeModel(
                [
                    "正文分析\n\n![错误图片](rogue_chart.png)",
                    "正文分析\n\n![还是错误](rogue_chart.png)",
                ]
            ),
            work_dir=tmp_dir,
        )
        writer.required_figures = [
            RequiredFigure(
                figure_id="ques1_real_chart",
                filename="real_chart.png",
                purpose="核心结论图",
            )
        ]
        writer.generated_figures = [
            GeneratedFigure(
                figure_id="ques1_real_chart",
                filename="real_chart.png",
                purpose="核心结论图",
                required=True,
                generated=True,
            )
        ]
        writer.allowed_images, writer.required_images, writer.missing_required_generation = (
            writer._build_image_contract(
                available_images=None,
                required_figures=writer.required_figures,
                generated_figures=writer.generated_figures,
            )
        )

        with self.assertRaises(ValueError):
            asyncio.run(
                writer._ensure_images_inserted(
                    response_content="正文分析\n\n![错误图片](rogue_chart.png)",
                    tools=None,
                    tool_choice=None,
                    sub_title="ques1",
                )
            )

    def test_run_raises_when_required_figure_was_not_generated(self):
        tmp_dir = _make_tmp_dir()

        writer = WriterAgent(
            task_id="test",
            model=_FakeModel(["这条响应不应该被消费"]),
            work_dir=tmp_dir,
        )

        with self.assertRaises(ValueError):
            asyncio.run(
                writer.run(
                    prompt="请撰写问题一章节",
                    available_images=[],
                    required_figures=[
                        RequiredFigure(
                            figure_id="ques1_real_chart",
                            filename="real_chart.png",
                            purpose="核心结论图",
                        )
                    ],
                    generated_figures=[
                        GeneratedFigure(
                            figure_id="ques1_real_chart",
                            filename="real_chart.png",
                            purpose="核心结论图",
                            required=True,
                            generated=False,
                        )
                    ],
                    sub_title="ques1",
                )
            )


if __name__ == "__main__":
    unittest.main()
