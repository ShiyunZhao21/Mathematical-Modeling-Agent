from __future__ import annotations

from app.schemas.A2A import ConclusionMemory, QuestionModelPlan


class ConclusionMemoryManager:
    def __init__(self, memory: ConclusionMemory | None = None):
        self.memory = memory or ConclusionMemory()

    def add_global_finding(self, content: str) -> None:
        if content and content not in self.memory.global_findings:
            self.memory.global_findings.append(content)

    def add_question_finding(self, question_key: str, content: str) -> None:
        if not content:
            return
        bucket = self.memory.per_question_findings.setdefault(question_key, [])
        if content not in bucket:
            bucket.append(content)

    def merge_question_plan(self, plan: QuestionModelPlan) -> None:
        for item in plan.assumptions:
            if item not in self.memory.shared_assumptions:
                self.memory.shared_assumptions.append(item)

        for key, value in plan.variables_and_parameters.items():
            if key not in self.memory.shared_symbols:
                self.memory.shared_symbols[key] = value

        self.add_question_finding(
            plan.question_key,
            f"{plan.question_key}：{plan.goal}；核心方法：{plan.model_method}",
        )

    def merge_question_section(self, question_key: str, section_markdown: str) -> None:
        excerpt = section_markdown.strip()
        if not excerpt:
            return
        if len(excerpt) > 600:
            excerpt = excerpt[:600] + "..."
        self.add_question_finding(question_key, excerpt)

    def to_markdown(self) -> str:
        lines = ["# 结论库"]

        if self.memory.global_findings:
            lines.extend(["", "## 全局结论"])
            lines.extend([f"- {item}" for item in self.memory.global_findings])

        if self.memory.shared_assumptions:
            lines.extend(["", "## 共享假设"])
            lines.extend([f"- {item}" for item in self.memory.shared_assumptions])

        if self.memory.shared_symbols:
            lines.extend(["", "## 共享符号"])
            lines.extend(
                [f"- {key}: {value}" for key, value in self.memory.shared_symbols.items()]
            )

        if self.memory.per_question_findings:
            lines.extend(["", "## 分题结论"])
            for question_key, findings in self.memory.per_question_findings.items():
                lines.extend(["", f"### {question_key}"])
                lines.extend([f"- {item}" for item in findings])

        return "\n".join(lines).strip() + "\n"
