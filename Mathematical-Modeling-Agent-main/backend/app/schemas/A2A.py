from pydantic import BaseModel, Field
from typing import Any


class CoordinatorToModeler(BaseModel):
    questions: dict
    ques_count: int


class ProblemDigest(BaseModel):
    title: str = ""
    background: str = ""
    conditions: list[str] = Field(default_factory=list)
    questions: dict[str, str] = Field(default_factory=dict)
    files_summary: list[str] = Field(default_factory=list)
    research_summary: str = ""


class ProblemAnalysis(BaseModel):
    overall_analysis: str = ""
    question_links: dict[str, str] = Field(default_factory=dict)
    per_question_guidance: dict[str, str] = Field(default_factory=dict)


class RequiredFigure(BaseModel):
    figure_id: str
    filename: str
    purpose: str = ""
    section_hint: str = ""
    caption_hint: str = ""
    required: bool = True


class GeneratedFigure(BaseModel):
    figure_id: str
    filename: str
    purpose: str = ""
    section_hint: str = ""
    caption_hint: str = ""
    required: bool = False
    generated: bool = True


class CompressedImageRecord(BaseModel):
    filename: str
    section: str = ""
    required: bool = False
    generated: bool = True
    purpose: str = ""


class CompressedSectionState(BaseModel):
    section_key: str
    allowed_images: list[str] = Field(default_factory=list)
    required_images: list[str] = Field(default_factory=list)
    generated_images: list[str] = Field(default_factory=list)
    locked_identifiers: list[str] = Field(default_factory=list)
    open_tasks: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)


class CompressedAgentMemory(BaseModel):
    current_section: str = ""
    completed_sections: list[str] = Field(default_factory=list)
    open_tasks: list[str] = Field(default_factory=list)
    allowed_images: list[str] = Field(default_factory=list)
    required_images: list[str] = Field(default_factory=list)
    generated_images: list[str] = Field(default_factory=list)
    locked_identifiers: list[str] = Field(default_factory=list)
    image_records: list[CompressedImageRecord] = Field(default_factory=list)
    section_states: dict[str, CompressedSectionState] = Field(default_factory=dict)
    section_facts: dict[str, list[str]] = Field(default_factory=dict)


class QuestionModelPlan(BaseModel):
    question_key: str
    goal: str = ""
    assumptions: list[str] = Field(default_factory=list)
    variables_and_parameters: dict[str, str] = Field(default_factory=dict)
    parameter_estimation: str = ""
    model_method: str = ""
    solution_steps: list[str] = Field(default_factory=list)
    validation_plan: str = ""
    formula_spec: str = ""
    coder_prompt: str = ""
    writer_context: str = ""
    required_figures: list[RequiredFigure] = Field(default_factory=list)
    plan_markdown: str = ""


class ConclusionMemory(BaseModel):
    global_findings: list[str] = Field(default_factory=list)
    per_question_findings: dict[str, list[str]] = Field(default_factory=dict)
    shared_assumptions: list[str] = Field(default_factory=list)
    shared_symbols: dict[str, str] = Field(default_factory=dict)


class ModelerToCoder(BaseModel):
    questions_solution: dict[str, str]
    question_model_docs: dict[str, QuestionModelPlan] = Field(default_factory=dict)


class CoderToWriter(BaseModel):
    code_response: str | None = None
    code_output: str | None = None
    created_images: list[str] | None = None
    generated_figures: list[GeneratedFigure] | None = None
    status: str = "ok"
    warnings: list[str] = Field(default_factory=list)


class WriterResponse(BaseModel):
    response_content: Any
    footnotes: list[tuple[str, str]] | None = None
    warnings: list[str] = Field(default_factory=list)
