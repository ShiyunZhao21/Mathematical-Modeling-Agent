import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class DocumentManager:
    def __init__(self, work_dir: str):
        self.work_dir = Path(work_dir)
        self.docs_dir = self.work_dir / "docs"
        self.questions_dir = self.docs_dir / "questions"
        self.latex_dir = self.work_dir / "latex"
        self.sections_dir = self.latex_dir / "sections"
        self.formulas_dir = self.latex_dir / "formulas"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.questions_dir.mkdir(parents=True, exist_ok=True)
        self.latex_dir.mkdir(parents=True, exist_ok=True)
        self.sections_dir.mkdir(parents=True, exist_ok=True)
        self.formulas_dir.mkdir(parents=True, exist_ok=True)

    def _question_dir(self, question_key: str) -> Path:
        question_dir = self.questions_dir / question_key
        question_dir.mkdir(parents=True, exist_ok=True)
        return question_dir

    def save_json(self, path: Path, payload: dict[str, Any] | list[Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def save_text(self, path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def save_model(self, path: Path, model: BaseModel) -> Path:
        return self.save_json(path, model.model_dump(mode="json"))

    def save_problem_digest(self, payload: BaseModel, markdown: str) -> None:
        self.save_model(self.docs_dir / "document_1_problem_digest.json", payload)
        self.save_text(self.docs_dir / "document_1_problem_digest.md", markdown)

    def save_problem_analysis(self, payload: BaseModel, markdown: str) -> None:
        self.save_model(self.docs_dir / "document_2_problem_analysis.json", payload)
        self.save_text(self.docs_dir / "document_2_problem_analysis.md", markdown)

    def save_question_model_plan(self, question_key: str, payload: BaseModel, markdown: str) -> None:
        question_dir = self._question_dir(question_key)
        self.save_model(question_dir / "document_3_model_plan.json", payload)
        self.save_text(question_dir / "document_3_model_plan.md", markdown)
        self.save_text(question_dir / "formulas.md", payload.formula_spec)
        self.save_text(self.formulas_dir / f"{question_key}.tex", payload.formula_spec)

    def save_question_section_markdown(self, question_key: str, content: str) -> None:
        question_dir = self._question_dir(question_key)
        self.save_text(question_dir / "section.md", content)

    def save_question_section_latex(self, question_key: str, content: str) -> None:
        self.save_text(self.sections_dir / f"{question_key}.tex", content)

    def save_conclusion_memory(self, payload: BaseModel, markdown: str) -> None:
        self.save_model(self.docs_dir / "conclusion_memory.json", payload)
        self.save_text(self.docs_dir / "conclusion_memory.md", markdown)

    def save_global_latex_assets(
        self,
        figures_tex: str,
        paper_tex: str,
        paper_body_tex: str | None = None,
    ) -> None:
        self.save_text(self.latex_dir / "figures.tex", figures_tex)
        if paper_body_tex is not None:
            self.save_text(self.latex_dir / "paper_body.tex", paper_body_tex)
        self.save_text(self.work_dir / "paper.tex", paper_tex)

    def load_text(self, relative_path: str) -> str:
        return (self.work_dir / relative_path).read_text(encoding="utf-8")
