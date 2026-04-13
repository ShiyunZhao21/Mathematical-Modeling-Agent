import asyncio
import os
import tempfile
import unittest
import uuid

from app.core.agents.coder_agent import CoderAgent
from app.schemas.A2A import RequiredFigure

TEST_TMP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".tmp", "tests")
)
os.makedirs(TEST_TMP_ROOT, exist_ok=True)


def _make_tmp_dir() -> str:
    path = os.path.join(TEST_TMP_ROOT, f"coder_{uuid.uuid4().hex}")
    os.makedirs(path, exist_ok=True)
    return path


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = []

    def model_dump(self):
        return {"role": "assistant", "content": self.content, "tool_calls": []}


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeModel:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    async def chat(self, **kwargs):
        content = self.responses[self.calls]
        self.calls += 1
        return _FakeResponse(content)


class _FakeInterpreter:
    def __init__(self, created_images_snapshots: list[list[str]]):
        self.created_images_snapshots = list(created_images_snapshots)
        self.sections: list[str] = []
        self.last_snapshot: list[str] = []

    def add_section(self, section_name: str):
        self.sections.append(section_name)

    async def execute_code(self, code: str):
        return "", False, ""

    async def get_created_images(self, section: str):
        if self.created_images_snapshots:
            self.last_snapshot = list(self.created_images_snapshots.pop(0))
        return list(self.last_snapshot)

    def get_code_output(self, section: str):
        return "图表输出摘要"


class TestCoderAgentImageValidation(unittest.TestCase):
    def test_run_retries_when_images_are_insufficient(self):
        tmp_dir = _make_tmp_dir()
        model = _FakeModel(["第一版总结", "补图完成"])
        interpreter = _FakeInterpreter(
            [
                ["ques1_model_fit.png"],
                [
                    "ques1_model_fit.png",
                    "ques1_residual_diagnostics.png",
                    "ques1_parameter_comparison.png",
                ],
            ]
        )
        agent = CoderAgent(
            task_id="test",
            model=model,
            work_dir=tmp_dir,
            max_chat_turns=6,
            max_retries=3,
            code_interpreter=interpreter,
        )

        result = asyncio.run(agent.run("请完成问题一建模", "ques1"))

        self.assertEqual(model.calls, 2)
        self.assertEqual(result.code_response, "补图完成")
        self.assertEqual(
            result.created_images,
            [
                "ques1_model_fit.png",
                "ques1_residual_diagnostics.png",
                "ques1_parameter_comparison.png",
            ],
        )
        self.assertIn("图表产出不足", agent.chat_history[-1]["content"])

    def test_run_retries_when_required_figure_filename_is_missing(self):
        tmp_dir = _make_tmp_dir()
        model = _FakeModel(["第一版总结", "补齐指定图片"])
        interpreter = _FakeInterpreter(
            [
                [
                    "step0_temp_plot.png",
                    "ques1_residual_diagnostics.png",
                    "ques1_parameter_comparison.png",
                ],
                [
                    "ques1_correlation_heatmap.png",
                    "ques1_residual_diagnostics.png",
                    "ques1_parameter_comparison.png",
                ],
            ]
        )
        agent = CoderAgent(
            task_id="test",
            model=model,
            work_dir=tmp_dir,
            max_chat_turns=6,
            max_retries=3,
            code_interpreter=interpreter,
        )

        result = asyncio.run(
            agent.run(
                "请完成问题一建模",
                "ques1",
                required_figures=[
                    RequiredFigure(
                        figure_id="ques1_correlation_heatmap",
                        filename="ques1_correlation_heatmap.png",
                        purpose="展示核心变量相关性",
                    ),
                    RequiredFigure(
                        figure_id="ques1_residual_diagnostics",
                        filename="ques1_residual_diagnostics.png",
                        purpose="展示残差诊断结果",
                    ),
                ],
            )
        )

        self.assertEqual(model.calls, 2)
        self.assertEqual(result.code_response, "补齐指定图片")
        self.assertIn("缺少的必需图片", agent.chat_history[-1]["content"])
        self.assertIn("ques1_correlation_heatmap.png", result.created_images)
        self.assertEqual(
            [figure.filename for figure in result.generated_figures if figure.generated],
            [
                "ques1_correlation_heatmap.png",
                "ques1_residual_diagnostics.png",
                "ques1_parameter_comparison.png",
            ],
        )

    def test_run_honors_required_figure_count_over_generic_question_minimum(self):
        tmp_dir = _make_tmp_dir()
        model = _FakeModel(["两张必需图已完成"])
        interpreter = _FakeInterpreter(
            [
                [
                    "ques1_correlation_heatmap.png",
                    "ques1_prediction_scatter.png",
                ],
                [
                    "ques1_correlation_heatmap.png",
                    "ques1_prediction_scatter.png",
                ],
            ]
        )
        agent = CoderAgent(
            task_id="test",
            model=model,
            work_dir=tmp_dir,
            max_chat_turns=6,
            max_retries=3,
            code_interpreter=interpreter,
        )

        result = asyncio.run(
            agent.run(
                "请完成问题一建模",
                "ques1",
                required_figures=[
                    RequiredFigure(
                        figure_id="ques1_correlation_heatmap",
                        filename="ques1_correlation_heatmap.png",
                        purpose="展示核心变量相关性",
                    ),
                    RequiredFigure(
                        figure_id="ques1_prediction_scatter",
                        filename="ques1_prediction_scatter.png",
                        purpose="展示预测值与真实值对比",
                    ),
                ],
            )
        )

        self.assertEqual(model.calls, 1)
        self.assertEqual(result.code_response, "两张必需图已完成")
        self.assertEqual(
            result.created_images,
            [
                "ques1_correlation_heatmap.png",
                "ques1_prediction_scatter.png",
            ],
        )

    def test_run_preserves_existing_images_when_max_retries_is_reached(self):
        tmp_dir = _make_tmp_dir()
        model = _FakeModel(["第一版总结"])
        interpreter = _FakeInterpreter(
            [
                ["ques1_model_fit.png"],
                ["ques1_model_fit.png"],
            ]
        )
        agent = CoderAgent(
            task_id="test",
            model=model,
            work_dir=tmp_dir,
            max_chat_turns=6,
            max_retries=1,
            code_interpreter=interpreter,
        )

        result = asyncio.run(agent.run("请完成问题一建模", "ques1"))

        self.assertEqual(model.calls, 1)
        self.assertEqual(result.code_response, "第一版总结")
        self.assertEqual(result.created_images, ["ques1_model_fit.png"])


if __name__ == "__main__":
    unittest.main()
