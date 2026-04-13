from app.core.agents import WriterAgent, CoderAgent, CoordinatorAgent, ModelerAgent
from app.schemas.request import Problem
from app.schemas.response import SystemMessage
from app.tools.openalex_scholar import OpenAlexScholar
from app.utils.log_util import logger
from app.utils.common_utils import (
    create_work_dir,
    get_config_template,
    md_2_docx,
    export_paper_pdf,
)
from app.models.user_output import UserOutput
from app.config.setting import settings
from app.tools.interpreter_factory import create_interpreter
from app.services.redis_manager import task_store
from app.tools.notebook_serializer import NotebookSerializer
from app.core.flows import Flows
from app.core.llm.llm_factory import LLMFactory
from app.core.document_manager import DocumentManager
from app.core.conclusion_memory import ConclusionMemoryManager
from app.core.prompts import LATEX_WRITER_PROMPT
from app.core.llm.llm import simple_chat
from app.schemas.A2A import ConclusionMemory, ModelerToCoder


class WorkFlow:
    def __init__(self):
        pass

    def execute(self) -> str:
        pass


class MathModelWorkFlow(WorkFlow):
    task_id: str
    work_dir: str
    ques_count: int = 0
    questions: dict[str, str | int] = {}

    async def _publish_stage(self, stage: str, message: str, content: str | None = None):
        await task_store.update_task(
            self.task_id,
            status="running",
            stage=stage,
            message=message,
        )
        if content:
            await task_store.publish_message(self.task_id, SystemMessage(content=content))

    async def _build_research_summary(self, scholar: OpenAlexScholar | None, coordinator_response) -> str:
        if scholar is None:
            return ""
        title = str(coordinator_response.questions.get("title", "")).strip()
        background = str(coordinator_response.questions.get("background", "")).strip()
        query = title or background[:100]
        if not query:
            return ""
        try:
            papers = await scholar.search_papers(query, limit=3)
            return scholar.papers_to_str(papers)
        except Exception as exc:
            logger.warning(f"背景研究失败: {exc}")
            return ""

    async def _build_latex_section(self, writer_llm, markdown_content: str) -> str:
        history = [
            {"role": "system", "content": LATEX_WRITER_PROMPT},
            {"role": "user", "content": markdown_content},
        ]
        return await simple_chat(writer_llm, history)

    async def execute(self, problem: Problem):
        self.task_id = problem.task_id
        self.work_dir = create_work_dir(self.task_id)
        code_interpreter = None
        notebook_serializer = None

        llm_factory = LLMFactory(self.task_id)
        coordinator_llm, modeler_llm, coder_llm, writer_llm = llm_factory.get_all_llms()

        coordinator_agent = CoordinatorAgent(self.task_id, coordinator_llm)
        modeler_agent = ModelerAgent(self.task_id, modeler_llm)

        scholar = (
            OpenAlexScholar(task_id=self.task_id, email=settings.OPENALEX_EMAIL)
            if settings.OPENALEX_EMAIL
            else None
        )

        document_manager = DocumentManager(self.work_dir)
        conclusion_memory_manager = ConclusionMemoryManager(ConclusionMemory())

        await self._publish_stage(
            "master:collect",
            "Coordinator is collecting problem inputs",
            "总领 agent 开始读取题目与附件",
        )

        try:
            coordinator_response = await coordinator_agent.run(problem.ques_all)
            self.questions = coordinator_response.questions
            self.ques_count = coordinator_response.ques_count
        except Exception as e:
            logger.error(f"CoordinatorAgent 执行失败: {e}")
            raise e

        files_summary: list[str] = []
        try:
            request_payload = await task_store.get_task_request(self.task_id)
            files_summary = request_payload.get("payload", {}).get("files", [])
        except Exception:
            files_summary = []

        await self._publish_stage(
            "master:research",
            "Coordinator is researching background materials",
            "总领 agent 开始背景分析与文献梳理",
        )
        research_summary = await self._build_research_summary(scholar, coordinator_response)

        problem_digest = coordinator_agent.build_problem_digest(
            coordinator_response,
            files_summary=files_summary,
            research_summary=research_summary,
        )
        digest_markdown = "\n\n".join(
            [
                f"# {problem_digest.title}",
                f"## 问题背景\n{problem_digest.background}",
                f"## 附件摘要\n" + ("\n".join(f"- {item}" for item in problem_digest.files_summary) or "无"),
                f"## 条件\n" + ("\n".join(f"- {item}" for item in problem_digest.conditions) or "无"),
                "## 问题列表\n" + "\n".join(
                    f"### {key}\n{value}" for key, value in problem_digest.questions.items()
                ),
                f"## 背景研究\n{problem_digest.research_summary or '无'}",
            ]
        )
        document_manager.save_problem_digest(problem_digest, digest_markdown)

        await self._publish_stage(
            "master:document_2",
            "Coordinator is building overall problem analysis",
            "总领 agent 正在生成文档2：整体问题分析",
        )
        problem_analysis = await coordinator_agent.analyze_problem_links(problem_digest)
        analysis_markdown = "\n\n".join(
            [
                "# 整体问题分析",
                problem_analysis.overall_analysis,
                "## 问题衔接",
                "\n".join(
                    f"### {key}\n{value}" for key, value in problem_analysis.question_links.items()
                ),
                "## 分题建模指导",
                "\n".join(
                    f"### {key}\n{value}"
                    for key, value in problem_analysis.per_question_guidance.items()
                ),
            ]
        )
        document_manager.save_problem_analysis(problem_analysis, analysis_markdown)
        conclusion_memory_manager.add_global_finding(problem_analysis.overall_analysis)
        document_manager.save_conclusion_memory(
            conclusion_memory_manager.memory,
            conclusion_memory_manager.to_markdown(),
        )

        user_output = UserOutput(work_dir=self.work_dir, ques_count=self.ques_count)

        await self._publish_stage(
            "interpreter:prepare",
            "Preparing local code interpreter",
            "正在创建代码沙盒环境",
        )

        notebook_serializer = NotebookSerializer(work_dir=self.work_dir)
        code_interpreter = await create_interpreter(
            kind="local",
            task_id=self.task_id,
            work_dir=self.work_dir,
            notebook_serializer=notebook_serializer,
            timeout=3000,
        )

        try:
            await task_store.publish_message(
                self.task_id,
                SystemMessage(content="代码沙盒环境创建完成", type="success"),
            )

            coder_agent = CoderAgent(
                task_id=problem.task_id,
                model=coder_llm,
                work_dir=self.work_dir,
                max_chat_turns=settings.MAX_CHAT_TURNS,
                max_retries=settings.MAX_RETRIES,
                code_interpreter=code_interpreter,
            )

            writer_agent = WriterAgent(
                task_id=problem.task_id,
                model=writer_llm,
                comp_template=problem.comp_template,
                format_output=problem.format_output,
                scholar=scholar,
                work_dir=self.work_dir,
            )

            flows = Flows(self.questions)
            config_template = get_config_template(problem.comp_template)

            legacy_modeler_response = await modeler_agent.run(coordinator_response)
            question_model_docs = {}
            for question_key, question_text in problem_digest.questions.items():
                await self._publish_stage(
                    f"modeler:{question_key}",
                    f"Modeler is building {question_key}",
                    f"建模手开始处理 {question_key}",
                )
                question_plan = await modeler_agent.run_for_question(
                    question_key=question_key,
                    question_text=question_text,
                    problem_digest=problem_digest,
                    problem_analysis=problem_analysis,
                    conclusion_memory=conclusion_memory_manager.memory,
                )
                question_model_docs[question_key] = question_plan
                document_manager.save_question_model_plan(
                    question_key, question_plan, question_plan.plan_markdown
                )
                conclusion_memory_manager.merge_question_plan(question_plan)
                document_manager.save_conclusion_memory(
                    conclusion_memory_manager.memory,
                    conclusion_memory_manager.to_markdown(),
                )

            modeler_response = ModelerToCoder(
                questions_solution=legacy_modeler_response.questions_solution,
                question_model_docs=question_model_docs,
            )

            solution_flows = flows.get_solution_flows(self.questions, modeler_response)

            for key, value in solution_flows.items():
                await self._publish_stage(
                    f"coder:{key}",
                    f"Coder is solving {key}",
                    f"代码手开始求解 {key}",
                )
                coder_response = await coder_agent.run(
                    prompt=value["coder_prompt"],
                    subtask_title=key,
                    required_figures=value.get("required_figures"),
                )
                await task_store.publish_message(
                    self.task_id,
                    SystemMessage(content=f"代码手求解成功 {key}", type="success"),
                )

                question_plan = question_model_docs.get(key)
                writer_prompt = flows.get_writer_prompt(
                    key,
                    coder_response.code_response,
                    code_interpreter,
                    config_template,
                    question_plan=question_plan,
                    problem_digest=problem_digest,
                    problem_analysis=problem_analysis,
                    conclusion_memory_markdown=conclusion_memory_manager.to_markdown(),
                    available_images=coder_response.created_images,
                    generated_figures=coder_response.generated_figures,
                )

                await self._publish_stage(
                    f"writer:{key}",
                    f"Writer is drafting {key}",
                    f"论文手开始写 {key} 部分",
                )
                writer_response = await writer_agent.run(
                    writer_prompt,
                    available_images=coder_response.created_images,
                    required_figures=question_plan.required_figures if question_plan else [],
                    generated_figures=coder_response.generated_figures,
                    sub_title=key,
                )
                user_output.set_res(key, writer_response)
                document_manager.save_question_section_markdown(
                    key, writer_response.response_content
                )

                latex_section = await self._build_latex_section(
                    writer_llm, writer_response.response_content
                )
                user_output.set_latex_section(key, latex_section)
                document_manager.save_question_section_latex(key, latex_section)

                if key.startswith("ques"):
                    conclusion_memory_manager.merge_question_section(
                        key, writer_response.response_content
                    )
                    document_manager.save_conclusion_memory(
                        conclusion_memory_manager.memory,
                        conclusion_memory_manager.to_markdown(),
                    )
                    await self._publish_stage(
                        f"memory:{key}",
                        f"Updating memory for {key}",
                        f"结论库已更新 {key}",
                    )

            write_flows = flows.get_write_flows(
                user_output,
                config_template,
                problem_digest=problem_digest,
                problem_analysis=problem_analysis,
                conclusion_memory_markdown=conclusion_memory_manager.to_markdown(),
                bg_ques_all=problem.ques_all,
            )
            for key, value in write_flows.items():
                await self._publish_stage(
                    f"writer:{key}",
                    f"Writer is drafting {key}",
                    f"论文手开始写 {key} 部分",
                )
                writer_response = await writer_agent.run(prompt=value, sub_title=key)
                user_output.set_res(key, writer_response)
                latex_section = await self._build_latex_section(
                    writer_llm, writer_response.response_content
                )
                user_output.set_latex_section(key, latex_section)

            user_output.save_result()
            paper_body_tex = user_output.build_paper_body_tex()
            document_manager.save_global_latex_assets(
                figures_tex="% figures placeholder\n",
                paper_tex=user_output.build_paper_tex(),
                paper_body_tex=paper_body_tex,
            )
            notebook_serializer.export_code_script()
            md_2_docx(self.task_id)
            export_paper_pdf(self.task_id)

            artifacts = await task_store.list_artifacts(self.task_id)
            await task_store.update_task(
                self.task_id,
                stage="outputs",
                message="Artifacts exported",
                artifacts=artifacts,
            )
        finally:
            if code_interpreter is not None:
                await code_interpreter.cleanup()
