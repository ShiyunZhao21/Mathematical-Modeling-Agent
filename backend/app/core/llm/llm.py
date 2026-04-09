from app.utils.common_utils import transform_link, split_footnotes
from app.utils.log_util import logger
import time
import json
import httpx
from app.schemas.response import (
    CoderMessage,
    WriterMessage,
    ModelerMessage,
    SystemMessage,
    CoordinatorMessage,
)
from app.services.redis_manager import redis_manager
from litellm import acompletion
import litellm
from app.schemas.enums import AgentType
from app.utils.track import agent_metrics
from icecream import ic

litellm.callbacks = [agent_metrics]

class LLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        task_id: str,
        max_tokens: int | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.chat_count = 0
        self.max_tokens = max_tokens
        self.task_id = task_id

    @staticmethod
    def _extract_text_content(message) -> str | None:
        if message is None:
            return None

        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if text:
                        text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)

        return None

    async def _fallback_anthropic_chat(self, history: list, top_p: float | None = None) -> dict:
        if not self.base_url:
            raise ValueError("缺少 base_url，无法执行回退请求")

        system_prompt = None
        messages = []
        for item in history or []:
            role = item.get("role")
            content = item.get("content")
            if role == "system":
                system_prompt = f"{system_prompt}\n\n{content}" if system_prompt else content
                continue
            if role not in {"user", "assistant"}:
                continue
            if content is None:
                continue
            messages.append({"role": role, "content": content})

        payload = {
            "model": (self.model or "").split("/", 1)[-1],
            "messages": messages,
            "max_tokens": self.max_tokens or 4096,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if top_p is not None:
            payload["top_p"] = top_p

        async with httpx.AsyncClient(timeout=300.0, trust_env=False) as client:
            response = await client.post(
                f"{self.base_url.rstrip('/')}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def chat(
        self,
        history: list = None,
        tools: list = None,
        tool_choice: str = None,
        max_retries: int = 8,  # 添加最大重试次数
        retry_delay: float = 1.0,  # 添加重试延迟
        top_p: float | None = None,  # 添加top_p参数,
        agent_name: AgentType = AgentType.SYSTEM,  # CoderAgent or WriterAgent
        sub_title: str | None = None,
    ) -> str:
        logger.info(f"subtitle是:{sub_title}")

        # 验证和修复工具调用完整性
        if history:
            history = self._validate_and_fix_tool_calls(history)

        kwargs = {
            "api_key": self.api_key,
            "model": self.model,
            "messages": history,
            "stream": False,
            "top_p": top_p,
            "metadata": {"agent_name": agent_name},
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        if self.base_url:
            kwargs["base_url"] = self.base_url
        litellm.enable_json_schema_validation = True #加入json格式验证

        # TODO: stream 输出
        for attempt in range(max_retries):
            try:
                # completion = self.client.chat.completions.create(**kwargs)
                response = await acompletion(**kwargs)
                logger.info(f"API返回: {response}")
                if not response or not hasattr(response, "choices"):
                    raise ValueError("无效的API响应")

                message = response.choices[0].message
                if not getattr(message, "tool_calls", None):
                    extracted_content = self._extract_text_content(message)
                    if extracted_content is None:
                        raw_response = await self._fallback_anthropic_chat(history, top_p=top_p)
                        extracted_content = "\n".join(
                            block.get("text", "")
                            for block in raw_response.get("content", [])
                            if isinstance(block, dict) and block.get("type") == "text"
                        ).strip()
                        if not extracted_content:
                            raise ValueError("回退请求未获取到文本内容")
                        message.content = extracted_content

                self.chat_count += 1
                await self.send_message(response, agent_name, sub_title)
                return response
            except Exception as e:
                logger.error(f"第{attempt + 1}次重试: {str(e)}")
                if attempt < max_retries - 1:  # 如果不是最后一次尝试
                    time.sleep(retry_delay * (attempt + 1))  # 指数退避
                    continue
                logger.debug(f"请求参数: {kwargs}")
                raise  # 如果所有重试都失败，则抛出异常

    def _validate_and_fix_tool_calls(self, history: list) -> list:
        """验证并修复工具调用完整性"""
        if not history:
            return history

        ic(f"🔍 开始验证工具调用，历史消息数量: {len(history)}")

        # 查找所有未匹配的tool_calls
        fixed_history = []
        i = 0

        while i < len(history):
            msg = history[i]

            # 如果是包含tool_calls的消息
            if isinstance(msg, dict) and "tool_calls" in msg and msg["tool_calls"]:
                ic(f"📞 发现tool_calls消息在位置 {i}")

                # 检查每个tool_call是否都有对应的response，分别处理
                valid_tool_calls = []
                invalid_tool_calls = []

                for tool_call in msg["tool_calls"]:
                    tool_call_id = tool_call.get("id")
                    ic(f"  检查tool_call_id: {tool_call_id}")

                    if tool_call_id:
                        # 查找对应的tool响应
                        found_response = False
                        for j in range(i + 1, len(history)):
                            if (
                                history[j].get("role") == "tool"
                                and history[j].get("tool_call_id") == tool_call_id
                            ):
                                ic(f"  ✅ 找到匹配响应在位置 {j}")
                                found_response = True
                                break

                        if found_response:
                            valid_tool_calls.append(tool_call)
                        else:
                            ic(f"  ❌ 未找到匹配响应: {tool_call_id}")
                            invalid_tool_calls.append(tool_call)

                # 根据检查结果处理消息
                if valid_tool_calls:
                    # 有有效的tool_calls，保留它们
                    fixed_msg = msg.copy()
                    fixed_msg["tool_calls"] = valid_tool_calls
                    fixed_history.append(fixed_msg)
                    ic(
                        f"  🔧 保留 {len(valid_tool_calls)} 个有效tool_calls，移除 {len(invalid_tool_calls)} 个无效的"
                    )
                else:
                    # 没有有效的tool_calls，移除tool_calls但可能保留其他内容
                    cleaned_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                    if cleaned_msg.get("content"):
                        fixed_history.append(cleaned_msg)
                        ic(f"  🔧 移除所有tool_calls，保留消息内容")
                    else:
                        ic(f"  🗑️ 完全移除空的tool_calls消息")

            # 如果是tool响应消息，检查是否是孤立的
            elif isinstance(msg, dict) and msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                ic(f"🔧 检查tool响应消息: {tool_call_id}")

                # 查找对应的tool_calls
                found_call = False
                for j in range(len(fixed_history)):
                    if fixed_history[j].get("tool_calls") and any(
                        tc.get("id") == tool_call_id
                        for tc in fixed_history[j]["tool_calls"]
                    ):
                        found_call = True
                        break

                if found_call:
                    fixed_history.append(msg)
                    ic(f"  ✅ 保留有效的tool响应")
                else:
                    ic(f"  🗑️ 移除孤立的tool响应: {tool_call_id}")

            else:
                # 普通消息，直接保留
                fixed_history.append(msg)

            i += 1

        if len(fixed_history) != len(history):
            ic(f"🔧 修复完成: {len(history)} -> {len(fixed_history)} 条消息")
        else:
            ic(f"✅ 验证通过，无需修复")

        return fixed_history

    async def send_message(self, response, agent_name, sub_title=None):
        logger.info(f"subtitle是:{sub_title}")
        content = response.choices[0].message.content

        match agent_name:
            case AgentType.CODER:
                agent_msg: CoderMessage = CoderMessage(content=content)
            case AgentType.WRITER:
                # 处理 Markdown 格式的图片语法
                content, _ = split_footnotes(content)
                content = transform_link(self.task_id, content)
                agent_msg: WriterMessage = WriterMessage(
                    content=content,
                    sub_title=sub_title,
                )
            case AgentType.MODELER:
                agent_msg: ModelerMessage = ModelerMessage(content=content)
            case AgentType.SYSTEM:
                agent_msg: SystemMessage = SystemMessage(content=content)
            case AgentType.COORDINATOR:
                agent_msg: CoordinatorMessage = CoordinatorMessage(content=content)
            case _:
                raise ValueError(f"不支持的agent类型: {agent_name}")

        await redis_manager.publish_message(
            self.task_id,
            agent_msg,
        )


# class DeepSeekModel(LLM):
#     def __init__(
#         self,
#         api_key: str,
#         model: str,
#         base_url: str,
#         task_id: str,
#     ):
#         super().__init__(api_key, model, base_url, task_id)
# self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)


async def simple_chat(model: LLM, history: list) -> str:
    """
    Description of the function.

    Args:
        model (LLM): 模型
        history (list): 构造好的历史记录（包含system_prompt,user_prompt）

    Returns:
        return_type: Description of the return value.
    """
    kwargs = {
        "api_key": model.api_key,
        "model": model.model,
        "messages": history,
        "stream": False,
    }

    if model.base_url:
        kwargs["base_url"] = model.base_url

    response = await acompletion(**kwargs)
    content = model._extract_text_content(response.choices[0].message)
    if content is not None:
        return content

    raw_response = await model._fallback_anthropic_chat(history)
    content = "\n".join(
        block.get("text", "")
        for block in raw_response.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
    if not content:
        raise ValueError("模型未返回可用文本内容")
    return content
