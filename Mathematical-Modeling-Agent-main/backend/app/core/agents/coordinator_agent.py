from app.core.agents.agent import Agent
from app.core.llm.llm import LLM, simple_chat
from app.core.prompts import COORDINATOR_PROMPT, MASTER_ANALYSIS_PROMPT
from app.core.agents.modeler_agent import repair_json
import json
import re
from app.utils.log_util import logger
from app.schemas.A2A import CoordinatorToModeler, ProblemAnalysis, ProblemDigest


class CoordinatorAgent(Agent):
    def __init__(
        self,
        task_id: str,
        model: LLM,
        max_chat_turns: int = 30,
    ) -> None:
        super().__init__(task_id, model, max_chat_turns)
        self.system_prompt = COORDINATOR_PROMPT

    async def run(self, ques_all: str) -> CoordinatorToModeler:
        """用户输入问题 使用LLM 格式化 questions"""
        await self.append_chat_history(
            {"role": "system", "content": self.system_prompt}
        )
        await self.append_chat_history({"role": "user", "content": ques_all})
        max_retries = 3
        attempt = 0
        while attempt <= max_retries:
            try:
                response = await self.model.chat(
                    history=self.chat_history,
                    agent_name=self.__class__.__name__,
                )
                json_str = response.choices[0].message.content
                if not json_str:
                    raise ValueError("返回的 JSON 字符串为空")

                questions = repair_json(json_str)
                if not questions:
                    raise ValueError("返回内容无法解析为 JSON 对象")

                ques_count = questions["ques_count"]
                logger.info(f"questions:{questions}")
                return CoordinatorToModeler(questions=questions, ques_count=ques_count)

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                attempt += 1
                logger.warning(f"解析失败 (尝试 {attempt}/{max_retries}): {str(e)}")

                if attempt > max_retries:
                    logger.error("超过最大重试次数，放弃解析")
                    raise RuntimeError(f"无法解析模型响应: {str(e)}")

                error_prompt = f"⚠️ 上次响应格式错误: {str(e)}。请严格输出JSON格式"
                await self.append_chat_history(
                    {
                        "role": "system",
                        "content": self.system_prompt + "\n" + error_prompt,
                    }
                )

        raise RuntimeError("意外的流程终止")

    def build_problem_digest(
        self,
        coordinator_to_modeler: CoordinatorToModeler,
        files_summary: list[str] | None = None,
        research_summary: str = "",
    ) -> ProblemDigest:
        questions = coordinator_to_modeler.questions
        conditions = [
            value
            for key, value in questions.items()
            if key not in {"title", "background", "ques_count"}
            and not str(key).startswith("ques")
        ]
        normalized_questions = {
            key: value
            for key, value in questions.items()
            if str(key).startswith("ques") and key != "ques_count"
        }
        return ProblemDigest(
            title=questions.get("title", ""),
            background=questions.get("background", ""),
            conditions=[str(item) for item in conditions],
            questions={key: str(value) for key, value in normalized_questions.items()},
            files_summary=files_summary or [],
            research_summary=research_summary,
        )

    async def analyze_problem_links(self, problem_digest: ProblemDigest) -> ProblemAnalysis:
        history = [
            {"role": "system", "content": MASTER_ANALYSIS_PROMPT},
            {
                "role": "user",
                "content": json.dumps(problem_digest.model_dump(mode="json"), ensure_ascii=False),
            },
        ]
        raw_content = await simple_chat(self.model, history)
        payload = repair_json(raw_content)
        if not payload:
            raise ValueError("整体问题分析解析失败")
        return ProblemAnalysis(**payload)
