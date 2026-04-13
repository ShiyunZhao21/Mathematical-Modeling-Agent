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

    def _model_name(self) -> str:
        model_name = (self.model or "").strip()
        if "/" in model_name:
            return model_name.split("/", 1)[1]
        return model_name

    def _provider_name(self) -> str:
        model_name = (self.model or "").strip().lower()
        if "/" in model_name:
            return model_name.split("/", 1)[0]
        if model_name.startswith("claude"):
            return "anthropic"
        return model_name

    def _supports_anthropic_fallback(self) -> bool:
        provider = self._provider_name()
        return provider in {"anthropic", "claude"}

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

    @staticmethod
    def _normalize_tool_call(tool_call: dict | None) -> dict | None:
        if tool_call is None:
            return None
        if hasattr(tool_call, "model_dump"):
            tool_call = tool_call.model_dump()
        if not isinstance(tool_call, dict):
            return None

        function = tool_call.get("function")
        if hasattr(function, "model_dump"):
            function = function.model_dump()
        if not isinstance(function, dict):
            function = {}

        function_name = function.get("name")
        if not function_name:
            return None

        arguments = function.get("arguments", "")
        if arguments is None:
            arguments = ""
        elif not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)

        normalized = {
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": arguments,
            },
        }

        tool_call_id = tool_call.get("id")
        if tool_call_id:
            normalized["id"] = tool_call_id

        return normalized

    @classmethod
    def _normalize_history_message(cls, message: dict | None) -> dict | None:
        if message is None:
            return None
        if hasattr(message, "model_dump"):
            message = message.model_dump()
        if not isinstance(message, dict):
            return None

        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            return None

        content = message.get("content")
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if text:
                        text_parts.append(text)
            content = "\n".join(text_parts) if text_parts else None
        elif content is not None and not isinstance(content, str):
            content = str(content)

        normalized = {"role": role}

        if role in {"system", "user"}:
            if not content:
                return None
            normalized["content"] = content
            return normalized

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not tool_call_id:
                return None
            normalized["tool_call_id"] = tool_call_id
            normalized["content"] = content or ""
            return normalized

        normalized_tool_calls = []
        for tool_call in message.get("tool_calls") or []:
            normalized_tool_call = cls._normalize_tool_call(tool_call)
            if normalized_tool_call:
                normalized_tool_calls.append(normalized_tool_call)

        if normalized_tool_calls:
            normalized["tool_calls"] = normalized_tool_calls

        if content is not None:
            normalized["content"] = content
        elif not normalized_tool_calls:
            return None

        return normalized

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
            "model": self._model_name(),
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
        max_retries: int = 8,
        retry_delay: float = 1.0,
        top_p: float | None = None,
        agent_name: AgentType = AgentType.SYSTEM,
        sub_title: str | None = None,
    ) -> str:
        # FIX(subtitle=None): 对 None 做兜底，避免日志中持续出现 subtitle是:None
        effective_sub_title = sub_title or f"task_{self.task_id[:8]}"
        logger.info(f"subtitle是:{effective_sub_title}")

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
        litellm.enable_json_schema_validation = True

        for attempt in range(max_retries):
            try:
                response = await acompletion(**kwargs)
                logger.info(f"API返回: {response}")
                if not response or not hasattr(response, "choices"):
                    raise ValueError("无效的API响应")

                message = response.choices[0].message
                if not getattr(message, "tool_calls", None):
                    extracted_content = self._extract_text_content(message)
                    if extracted_content is None:
                        if not self._supports_anthropic_fallback():
                            raise ValueError(
                                "模型未返回可用文本内容，且当前 provider 未启用 Anthropic 回退协议"
                            )
                        raw_response = await self._fallback_anthropic_chat(
                            history, top_p=top_p
                        )
                        extracted_content = "\n".join(
                            block.get("text", "")
                            for block in raw_response.get("content", [])
                            if isinstance(block, dict) and block.get("type") == "text"
                        ).strip()
                        if not extracted_content:
                            raise ValueError("Anthropic 回退请求未获取到文本内容")
                        message.content = extracted_content

                self.chat_count += 1
                await self.send_message(response, agent_name, effective_sub_title)
                return response
            except Exception as e:
                logger.error(f"第{attempt + 1}次重试: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                logger.debug(f"请求参数: {kwargs}")
                raise

    def _validate_and_fix_tool_calls(self, history: list) -> list:
        """验证并修复工具调用完整性"""
        if not history:
            return history

        ic(f"🔍 开始验证工具调用，历史消息数量: {len(history)}")

        normalized_history = []
        for msg in history:
            normalized_msg = self._normalize_history_message(msg)
            if normalized_msg:
                normalized_history.append(normalized_msg)

        fixed_history = []
        i = 0

        while i < len(normalized_history):
            msg = normalized_history[i]

            if isinstance(msg, dict) and "tool_calls" in msg and msg["tool_calls"]:
                ic(f"📞 发现tool_calls消息在位置 {i}")

                valid_tool_calls = []
                invalid_tool_calls = []

                for tool_call in msg["tool_calls"]:
                    tool_call_id = tool_call.get("id")
                    ic(f"  检查tool_call_id: {tool_call_id}")

                    if tool_call_id:
                        found_response = False
                        for j in range(i + 1, len(normalized_history)):
                            if (
                                normalized_history[j].get("role") == "tool"
                                and normalized_history[j].get("tool_call_id") == tool_call_id
                            ):
                                ic(f"  ✅ 找到匹配响应在位置 {j}")
                                found_response = True
                                break

                        if found_response:
                            valid_tool_calls.append(tool_call)
                        else:
                            ic(f"  ❌ 未找到匹配响应: {tool_call_id}")
                            invalid_tool_calls.append(tool_call)

                if valid_tool_calls:
                    fixed_msg = msg.copy()
                    fixed_msg["tool_calls"] = valid_tool_calls
                    fixed_history.append(fixed_msg)
                    ic(
                        f"  🔧 保留 {len(valid_tool_calls)} 个有效tool_calls，移除 {len(invalid_tool_calls)} 个无效的"
                    )
                else:
                    cleaned_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                    if cleaned_msg.get("content"):
                        fixed_history.append(cleaned_msg)
                        ic(f"  🔧 移除所有tool_calls，保留消息内容")
                    else:
                        ic(f"  🗑️ 完全移除空的tool_calls消息")

            elif isinstance(msg, dict) and msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                ic(f"🔧 检查tool响应消息: {tool_call_id}")

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
                fixed_history.append(msg)

            i += 1

        if len(fixed_history) != len(history):
            ic(
                f"🔧 修复完成: 原始 {len(history)} -> 归一化 {len(normalized_history)} -> 最终 {len(fixed_history)} 条消息"
            )
        else:
            ic(f"✅ 验证通过，无需修复")

        return fixed_history

    async def send_message(self, response, agent_name, sub_title=None):
        # FIX(subtitle=None): 兜底，保持与 chat() 一致
        effective_sub_title = sub_title or f"task_{self.task_id[:8]}"
        logger.info(f"subtitle是:{effective_sub_title}")
        content = response.choices[0].message.content

        match agent_name:
            case AgentType.CODER:
                agent_msg: CoderMessage = CoderMessage(content=content)
            case AgentType.WRITER:
                content, _ = split_footnotes(content)
                content = transform_link(self.task_id, content)
                agent_msg: WriterMessage = WriterMessage(
                    content=content,
                    sub_title=effective_sub_title,
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


async def simple_chat(model: LLM, history: list) -> str:
    """
    Args:
        model (LLM): 模型
        history (list): 构造好的历史记录（包含system_prompt,user_prompt）
    Returns:
        str: 模型返回的文本内容
    """
    history = model._validate_and_fix_tool_calls(history)

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

    if not model._supports_anthropic_fallback():
        raise ValueError("模型未返回可用文本内容，且当前 provider 未启用 Anthropic 回退协议")

    raw_response = await model._fallback_anthropic_chat(history)
    content = "\n".join(
        block.get("text", "")
        for block in raw_response.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
    if not content:
        raise ValueError("模型未返回可用文本内容")
    return content