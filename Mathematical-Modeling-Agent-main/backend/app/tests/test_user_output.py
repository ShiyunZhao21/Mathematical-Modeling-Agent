import os
import tempfile
import unittest

from app.models.user_output import UserOutput
from app.schemas.A2A import WriterResponse

TEST_TMP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".tmp", "tests")
)
os.makedirs(TEST_TMP_ROOT, exist_ok=True)


class TestUserOutput(unittest.TestCase):
    def _make_tmp_dir(self) -> str:
        return tempfile.mkdtemp(dir=TEST_TMP_ROOT)

    def test_reference_formats_are_collected_into_footnotes(self):
        tmp_dir = self._make_tmp_dir()
        output = UserOutput(work_dir=tmp_dir, ques_count=1)
        output.set_res(
            "firstPage",
            WriterResponse(
                response_content="背景文献{[^1] Reference A.} 与再次引用{[^1]: Reference A.}",
                footnotes=None,
            ),
        )

        rendered = output.get_result_to_save()

        self.assertIn("[^1]", rendered)
        self.assertIn("[^1]: Reference A", rendered)
        self.assertEqual(rendered.count("[^1]: Reference A"), 1)

    def test_reference_section_is_omitted_when_no_footnotes_exist(self):
        tmp_dir = self._make_tmp_dir()
        output = UserOutput(work_dir=tmp_dir, ques_count=1)
        output.set_res(
            "firstPage",
            WriterResponse(response_content="纯正文内容", footnotes=None),
        )

        rendered = output.get_result_to_save()

        self.assertNotIn("## 参考文献", rendered)

    def test_build_paper_tex_returns_standalone_document(self):
        tmp_dir = self._make_tmp_dir()
        output = UserOutput(work_dir=tmp_dir, ques_count=1)
        output.footnotes = {
            "ref-1": {
                "content": "Reference A",
                "number": 1,
            }
        }
        output.set_latex_section(
            "firstPage",
            "\\title{示例标题}\n\n\\begin{abstract}摘要内容\\end{abstract}",
        )
        output.set_latex_section("ques1", "\\section{问题一}\n正文内容")

        rendered = output.build_paper_tex()

        self.assertIn("\\documentclass[12pt,a4paper]{ctexart}", rendered)
        self.assertIn("\\begin{document}", rendered)
        self.assertIn("\\maketitle", rendered)
        self.assertIn("\\begin{thebibliography}{99}", rendered)
        self.assertIn("\\bibitem{ref1} Reference A", rendered)
        self.assertIn("\\end{document}", rendered)


if __name__ == "__main__":
    unittest.main()
