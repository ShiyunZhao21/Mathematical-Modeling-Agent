import os
import tempfile
import unittest
import uuid

from app.tools.base_interpreter import BaseCodeInterpreter

TEST_TMP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".tmp", "tests")
)
os.makedirs(TEST_TMP_ROOT, exist_ok=True)


def _make_tmp_dir() -> str:
    path = os.path.join(TEST_TMP_ROOT, f"base_{uuid.uuid4().hex}")
    os.makedirs(path, exist_ok=True)
    return path


class DummyInterpreter(BaseCodeInterpreter):
    async def initialize(self):
        return None

    async def _pre_execute_code(self):
        return None

    async def execute_code(self, code: str):
        return "", False, ""

    async def cleanup(self):
        return None

    async def get_created_images(self, section: str):
        return []


class TestBaseInterpreterFontSetup(unittest.TestCase):
    def setUp(self):
        self.interpreter = DummyInterpreter(
            task_id="test",
            work_dir=".",
            notebook_serializer=None,
        )

    def test_prepare_code_rewrites_arial_family(self):
        original = """
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial Unicode MS', 'SimHei', 'DejaVu Sans'],
})
"""

        prepared = self.interpreter.prepare_code_for_execution(original)

        self.assertIn("Auto-injected matplotlib Chinese font setup", prepared)
        self.assertIn("mpl.rcParams['font.family'] = ['sans-serif']", prepared)
        self.assertIn("'font.family': 'sans-serif'", prepared)
        self.assertIn("'Microsoft YaHei'", prepared)
        self.assertNotIn("'font.family': 'Arial'", prepared)

    def test_prepare_code_normalizes_direct_font_assignment(self):
        original = """
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'Arial'
"""

        prepared = self.interpreter.prepare_code_for_execution(original)

        self.assertIn("mpl.rcParams['font.family'] = ['sans-serif']", prepared)
        self.assertIn("plt.rcParams['font.family'] = ['sans-serif']", prepared)
        self.assertNotIn("= 'Arial'", prepared)

    def test_collect_created_images_renames_to_stable_semantic_names(self):
        tmp_dir = _make_tmp_dir()
        interpreter = DummyInterpreter(
            task_id="test",
            work_dir=tmp_dir,
            notebook_serializer=None,
        )

        open(os.path.join(tmp_dir, "fig1.png"), "wb").close()
        open(os.path.join(tmp_dir, "step0_residual_diagnostics.png"), "wb").close()

        created_images = interpreter.collect_created_images("ques1")

        self.assertIn("ques1_figure_01.png", created_images)
        self.assertIn("ques1_residual_diagnostics.png", created_images)
        self.assertTrue(
            os.path.exists(os.path.join(tmp_dir, "ques1_figure_01.png"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(tmp_dir, "ques1_residual_diagnostics.png"))
        )
        self.assertFalse(os.path.exists(os.path.join(tmp_dir, "fig1.png")))


if __name__ == "__main__":
    unittest.main()
