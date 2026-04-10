import asyncio
import os
import tempfile
import unittest
import uuid

from app.core.agents.coder_agent import CoderAgent

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

    def add_section(self, section_name: str):
        self.sections.append(section_name)

    async def execute_code(self, code: str):
        return "", False, ""

    async def get_created_images(self, section: str):
        return self.created_images_snapshots.pop(0)

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


if __name__ == "__main__":
    unittest.main()
