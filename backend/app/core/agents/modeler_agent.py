from app.core.agents.agent import Agent
from app.core.llm.llm import LLM, simple_chat
from app.core.prompts import MODELER_PROMPT, QUESTION_MODELER_PROMPT
from app.schemas.A2A import (
    ConclusionMemory,
    CoordinatorToModeler,
    ModelerToCoder,
    ProblemAnalysis,
    ProblemDigest,
    QuestionModelPlan,
)
from app.utils.log_util import logger
import json
import re
from icecream import ic


def repair_json(json_str: str) -> dict | None:
    """Try to repair malformed JSON from LLM output."""
    json_str = json_str.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", json_str)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            json_str = candidate

    try:
        fixed = re.sub(
            r'(?<=: ")(.*?)(?=",\s*\n\s*"|"\s*\n\s*})',
            lambda m: m.group(0).replace('"', '\\"'),
            json_str,
            flags=re.DOTALL,
        )
        parsed = json.loads(fixed)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, re.error):
        pass

    try:
        pattern = r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.|"(?!,\s*\n)|"(?!\s*\n\s*}))*)"'
        matches = re.findall(pattern, json_str, re.DOTALL)
        if matches:
            return {k: v.replace('\\"', '"') for k, v in matches}
    except re.error:
        pass

    return None


class ModelerAgent(Agent):
    def __init__(
        self,
        task_id: str,
        model: LLM,
        max_chat_turns: int = 30,
    ) -> None:
        super().__init__(task_id, model, max_chat_turns)
        self.system_prompt = MODELER_PROMPT

    async def run(self, coordinator_to_modeler: CoordinatorToModeler) -> ModelerToCoder:
        await self.append_chat_history(
            {"role": "system", "content": self.system_prompt}
        )
        await self.append_chat_history(
            {
                "role": "user",
                "content": json.dumps(coordinator_to_modeler.questions),
            }
        )

        max_parse_retries = 3
        for attempt in range(max_parse_retries):
            response = await self.model.chat(
                history=self.chat_history,
                agent_name=self.__class__.__name__,
            )

            json_str = response.choices[0].message.content
            if not json_str:
                raise ValueError("返回的 JSON 字符串为空，请检查输入内容。")

            questions_solution = repair_json(json_str)
            if questions_solution:
                ic(questions_solution)
                return ModelerToCoder(questions_solution=questions_solution)

            logger.warning(
                f"JSON 解析失败 (第{attempt + 1}次)，请求模型重新生成"
            )
            await self.append_chat_history(
                {"role": "assistant", "content": json_str}
            )
            await self.append_chat_history(
                {
                    "role": "user",
                    "content": '你返回的JSON格式有误，请严格按照JSON格式重新输出，注意字符串值内的双引号必须转义为\\"，不要包含未转义的特殊字符。',
                }
            )

        raise ValueError(
            f"经过{max_parse_retries}次尝试仍无法解析JSON，请检查模型输出"
        )

    async def run_for_question(
        self,
        question_key: str,
        question_text: str,
        problem_digest: ProblemDigest,
        problem_analysis: ProblemAnalysis,
        conclusion_memory: ConclusionMemory,
    ) -> QuestionModelPlan:
        history = [
            {"role": "system", "content": QUESTION_MODELER_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "problem_digest": problem_digest.model_dump(mode="json"),
                        "problem_analysis": problem_analysis.model_dump(mode="json"),
                        "conclusion_memory": conclusion_memory.model_dump(mode="json"),
                        "current_question": {
                            "question_key": question_key,
                            "question_text": question_text,
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        raw_content = await simple_chat(self.model, history)
        payload = repair_json(raw_content)
        if not payload:
            raise ValueError(f"{question_key} 建模方案解析失败")
        return QuestionModelPlan(**payload)
