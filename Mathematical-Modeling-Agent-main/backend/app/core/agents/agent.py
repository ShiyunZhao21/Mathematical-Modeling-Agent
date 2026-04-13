from app.core.llm.llm import LLM
from app.schemas.A2A import (
    CompressedAgentMemory,
    CompressedImageRecord,
    CompressedSectionState,
)
from app.utils.log_util import logger
from icecream import ic
import os
import re

# TODO: Memory 的管理
# TODO: 评估任务完成情况，rethinking


class Agent:
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
    SECTION_PATTERN = re.compile(
        r"\b(?:eda|ques\d+|sensitivity_analysis|firstPage|RepeatQues|analysisQues|modelAssumption|symbol|judge)\b"
    )
    FILE_PATTERN = re.compile(
        r"([A-Za-z0-9_\-\u4e00-\u9fff./\\]+\.(?:png|jpg|jpeg|gif|bmp|webp))",
        re.IGNORECASE,
    )
    IMAGE_MARKDOWN_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    GENERIC_IMAGE_STEM_PATTERN = re.compile(
        r"^(?:fig(?:ure)?|image|img|plot|chart|graph|visual(?:ization)?|"
        r"output|result|photo|pic|picture|tmp|temp)(?:[_-]?\d+)?$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        task_id: str,
        model: LLM,
        max_chat_turns: int = 30,
        max_memory: int = 24,
    ) -> None:
        self.task_id = task_id
        self.model = model
        self.chat_history: list[dict] = []
        self.max_chat_turns = max_chat_turns
        self.current_chat_turns = 0
        self.max_memory = max_memory
        self.compressed_memory = CompressedAgentMemory()

    async def run(self, prompt: str, system_prompt: str, sub_title: str) -> str:
        try:
            logger.info(f"{self.__class__.__name__}:开始:执行对话")
            self.current_chat_turns = 0

            await self.append_chat_history({"role": "system", "content": system_prompt})
            await self.append_chat_history({"role": "user", "content": prompt})

            response = await self.model.chat(
                history=self.chat_history,
                agent_name=self.__class__.__name__,
                sub_title=sub_title,
            )
            response_content = response.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": response_content})
            logger.info(f"{self.__class__.__name__}:完成:执行对话")
            return response_content
        except Exception as e:
            error_msg = f"执行过程中遇到错误: {str(e)}"
            logger.error(f"Agent执行失败: {str(e)}")
            return error_msg

    async def append_chat_history(self, msg: dict) -> None:
        ic(f"添加消息: role={msg.get('role')}, 当前历史长度={len(self.chat_history)}")
        self.chat_history.append(msg)
        ic(f"添加后历史长度={len(self.chat_history)}")

        if msg.get("role") != "tool":
            ic("触发内存清理")
            await self.clear_memory()
        else:
            ic("跳过内存清理(tool消息)")

    async def clear_memory(self):
        """当聊天历史超过最大记忆轮次时，压缩为结构化记忆块。"""
        ic(f"检查内存清理: 当前={len(self.chat_history)}, 最大={self.max_memory}")

        if len(self.chat_history) <= self.max_memory:
            ic("无需清理内存")
            return

        ic("开始内存清理")
        logger.info(
            f"{self.__class__.__name__}:开始清除记忆，当前记录数：{len(self.chat_history)}"
        )

        try:
            system_msg = (
                self.chat_history[0]
                if self.chat_history and self.chat_history[0]["role"] == "system"
                else None
            )

            preserve_start_idx = self._find_safe_preserve_point()
            ic(f"保留起始索引: {preserve_start_idx}")

            start_idx = 1 if system_msg else 0
            end_idx = preserve_start_idx
            ic(f"压缩范围: {start_idx} -> {end_idx}")

            if end_idx > start_idx:
                evicted_history = self.chat_history[start_idx:end_idx]
                new_memory = self._build_memory_from_history(evicted_history)
                self.compressed_memory = self._merge_compressed_memory(
                    self.compressed_memory, new_memory
                )

                new_history = []
                if system_msg:
                    new_history.append(system_msg)

                memory_block = self._render_compressed_memory(self.compressed_memory)
                if memory_block:
                    new_history.append(
                        {
                            "role": "assistant",
                            "content": f"[结构化记忆压缩]\n{memory_block}",
                        }
                    )

                new_history.extend(self.chat_history[preserve_start_idx:])

                self.chat_history = new_history
                ic(f"内存清理完成，新历史长度: {len(self.chat_history)}")
                logger.info(
                    f"{self.__class__.__name__}:记忆清除完成，压缩至：{len(self.chat_history)}条记录"
                )
            else:
                logger.info(f"{self.__class__.__name__}:无需清除记忆，记录数量合理")

        except Exception as e:
            logger.error(f"记忆清除失败，使用简单切片策略: {str(e)}")
            self.chat_history = self._get_safe_fallback_history()

    def _merge_unique(self, existing: list[str], new_items: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for item in [*(existing or []), *(new_items or [])]:
            normalized = (item or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
        return merged

    def _normalize_file_identifier(self, value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return ""
        raw = raw.split("?", 1)[0].split("#", 1)[0].strip()
        raw = raw.replace("\\", "/")
        if raw.startswith("http://") or raw.startswith("https://"):
            raw = raw.rsplit("/", 1)[-1]
        if "/site-packages/" in raw.lower() or raw.lower().startswith("site-packages/"):
            return ""
        basename = os.path.basename(raw)
        if not basename or basename in {".", ".."}:
            return ""
        _, ext = os.path.splitext(basename)
        if ext.lower() not in self.IMAGE_EXTENSIONS:
            return ""
        stem = os.path.splitext(basename)[0]
        if self.GENERIC_IMAGE_STEM_PATTERN.fullmatch(stem or ""):
            return ""
        return basename

    def _coerce_message_content(self, msg: dict) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return str(content)

    def _extract_section_keys(self, text: str) -> list[str]:
        return list(dict.fromkeys(self.SECTION_PATTERN.findall(text or "")))

    def _extract_candidate_files(self, text: str) -> list[str]:
        candidates = []
        for match in self.FILE_PATTERN.findall(text or ""):
            normalized = self._normalize_file_identifier(match)
            if normalized:
                candidates.append(normalized)
        for match in self.IMAGE_MARKDOWN_PATTERN.findall(text or ""):
            normalized = self._normalize_file_identifier(match)
            if normalized:
                candidates.append(normalized)
        return self._merge_unique([], candidates)

    def _extract_contract_image_list(self, text: str, prefix: str) -> list[str]:
        for line in (text or "").splitlines():
            stripped = line.strip()
            if not stripped.startswith(prefix):
                continue
            payload = stripped.split("=", 1)[-1].strip()
            if payload in {"", "无"}:
                return []
            images = []
            for item in payload.split(","):
                normalized = self._normalize_file_identifier(item)
                if normalized:
                    images.append(normalized)
            return self._merge_unique([], images)
        return []

    def _extract_generated_image_list(self, text: str) -> list[str]:
        images: list[str] = []
        for line in (text or "").splitlines():
            stripped = line.strip()
            if stripped.startswith(("唯一允许引用的图片文件名=", "必须插入的图片文件名=")):
                continue
            if "文件名=" in stripped:
                filename_part = stripped.split("文件名=", 1)[1].split("；", 1)[0].strip()
                normalized = self._normalize_file_identifier(filename_part)
                if normalized:
                    images.append(normalized)
            if stripped.startswith("[IMAGE_SAVED]"):
                normalized = self._normalize_file_identifier(
                    stripped.split("]", 1)[-1].strip()
                )
                if normalized:
                    images.append(normalized)
            if stripped.startswith("[IMAGE_MANIFEST]"):
                for item in stripped.split("]", 1)[-1].split(","):
                    normalized = self._normalize_file_identifier(item)
                    if normalized:
                        images.append(normalized)
            if any(
                keyword in stripped
                for keyword in ["生成图片", "已生成", "图片文件名", "保存为", "保存成", "输出图片"]
            ):
                images = self._merge_unique(images, self._extract_candidate_files(stripped))
        return self._merge_unique([], images)

    def _extract_open_tasks(self, text: str) -> list[str]:
        tasks: list[str] = []
        for line in (text or "").splitlines():
            stripped = line.strip(" -：:；;")
            if not stripped:
                continue
            if any(
                keyword in stripped
                for keyword in ["缺少", "补齐", "待", "TODO", "必须", "禁止", "只允许"]
            ):
                tasks.append(stripped[:180])
        return self._merge_unique([], tasks)[:8]

    def _extract_facts(self, text: str) -> list[str]:
        facts: list[str] = []
        for line in (text or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) > 220:
                continue
            if stripped.startswith(("{", "}", "[", "]", '"', "'")):
                continue
            if "\\n" in stripped or "\": " in stripped:
                continue
            if any(
                keyword in stripped
                for keyword in ["结论", "数据特征", "模型类型", "核心指标", "核心结论"]
            ):
                facts.append(stripped[:220])
        return self._merge_unique([], facts)[:8]

    def _update_section_state(
        self,
        memory: CompressedAgentMemory,
        section_key: str,
        files: list[str],
        text: str,
    ) -> None:
        if not section_key:
            return

        state = memory.section_states.get(section_key)
        if state is None:
            state = CompressedSectionState(section_key=section_key)

        allowed_from_contract = self._extract_contract_image_list(
            text, "唯一允许引用的图片文件名="
        )
        required_from_contract = self._extract_contract_image_list(
            text, "必须插入的图片文件名="
        )
        generated_from_text = self._extract_generated_image_list(text)
        locked_files = self._merge_unique(
            files,
            self._merge_unique(
                allowed_from_contract,
                self._merge_unique(required_from_contract, generated_from_text),
            ),
        )

        state.locked_identifiers = self._merge_unique(
            state.locked_identifiers, locked_files
        )
        state.facts = self._merge_unique(state.facts, self._extract_facts(text))
        state.open_tasks = self._merge_unique(state.open_tasks, self._extract_open_tasks(text))

        if allowed_from_contract:
            state.allowed_images = self._merge_unique(
                state.allowed_images, allowed_from_contract
            )
        if required_from_contract:
            state.required_images = self._merge_unique(
                state.required_images, required_from_contract
            )
        if generated_from_text:
            state.generated_images = self._merge_unique(
                state.generated_images, generated_from_text
            )

        memory.section_states[section_key] = state

    def _build_memory_from_history(self, history: list[dict]) -> CompressedAgentMemory:
        memory = CompressedAgentMemory()
        current_section = ""

        for msg in history:
            text = self._coerce_message_content(msg)
            if not text:
                continue

            sections = self._extract_section_keys(text)
            if sections:
                current_section = sections[-1]
                memory.current_section = current_section
                if any(keyword in text for keyword in ["完成", "成功", "求解成功", "写作完成"]):
                    memory.completed_sections = self._merge_unique(
                        memory.completed_sections, sections
                    )

            files = self._extract_candidate_files(text)
            allowed_from_contract = self._extract_contract_image_list(
                text, "唯一允许引用的图片文件名="
            )
            required_from_contract = self._extract_contract_image_list(
                text, "必须插入的图片文件名="
            )
            generated_from_text = self._extract_generated_image_list(text)
            image_files = self._merge_unique(
                allowed_from_contract,
                self._merge_unique(required_from_contract, generated_from_text),
            )

            if image_files:
                memory.locked_identifiers = self._merge_unique(
                    memory.locked_identifiers, image_files
                )
                memory.allowed_images = self._merge_unique(
                    memory.allowed_images, allowed_from_contract
                )
                memory.required_images = self._merge_unique(
                    memory.required_images, required_from_contract
                )
                memory.generated_images = self._merge_unique(
                    memory.generated_images, generated_from_text
                )
                for image_name in image_files:
                    memory.image_records.append(
                        CompressedImageRecord(
                            filename=image_name,
                            section=current_section,
                            required=image_name in memory.required_images,
                            generated=(
                                image_name in memory.generated_images
                                or image_name in allowed_from_contract
                            ),
                        )
                    )

            if current_section:
                self._update_section_state(memory, current_section, files, text)

            memory.open_tasks = self._merge_unique(
                memory.open_tasks, self._extract_open_tasks(text)
            )
            if current_section:
                existing_facts = memory.section_facts.get(current_section, [])
                memory.section_facts[current_section] = self._merge_unique(
                    existing_facts, self._extract_facts(text)
                )

        deduped_records: list[CompressedImageRecord] = []
        seen_records: set[tuple[str, str]] = set()
        for record in memory.image_records:
            key = (record.filename, record.section)
            if key in seen_records:
                continue
            seen_records.add(key)
            deduped_records.append(record)
        memory.image_records = deduped_records
        return memory

    def _merge_compressed_memory(
        self,
        base: CompressedAgentMemory,
        incoming: CompressedAgentMemory,
    ) -> CompressedAgentMemory:
        merged = CompressedAgentMemory(
            current_section=incoming.current_section or base.current_section,
            completed_sections=self._merge_unique(
                base.completed_sections, incoming.completed_sections
            ),
            open_tasks=self._merge_unique(base.open_tasks, incoming.open_tasks),
            allowed_images=self._merge_unique(base.allowed_images, incoming.allowed_images),
            required_images=self._merge_unique(base.required_images, incoming.required_images),
            generated_images=self._merge_unique(base.generated_images, incoming.generated_images),
            locked_identifiers=self._merge_unique(
                base.locked_identifiers, incoming.locked_identifiers
            ),
            image_records=list(base.image_records),
            section_states=dict(base.section_states),
            section_facts=dict(base.section_facts),
        )

        existing_record_keys = {(record.filename, record.section) for record in merged.image_records}
        for record in incoming.image_records:
            key = (record.filename, record.section)
            if key not in existing_record_keys:
                merged.image_records.append(record)
                existing_record_keys.add(key)

        for section_key, state in incoming.section_states.items():
            existing = merged.section_states.get(section_key)
            if existing is None:
                merged.section_states[section_key] = state
                continue
            merged.section_states[section_key] = CompressedSectionState(
                section_key=section_key,
                allowed_images=self._merge_unique(existing.allowed_images, state.allowed_images),
                required_images=self._merge_unique(existing.required_images, state.required_images),
                generated_images=self._merge_unique(existing.generated_images, state.generated_images),
                locked_identifiers=self._merge_unique(
                    existing.locked_identifiers, state.locked_identifiers
                ),
                open_tasks=self._merge_unique(existing.open_tasks, state.open_tasks),
                facts=self._merge_unique(existing.facts, state.facts),
            )

        for section_key, facts in incoming.section_facts.items():
            merged.section_facts[section_key] = self._merge_unique(
                merged.section_facts.get(section_key, []), facts
            )

        return merged

    def _render_compressed_memory(self, memory: CompressedAgentMemory) -> str:
        if not any(
            [
                memory.current_section,
                memory.completed_sections,
                memory.open_tasks,
                memory.allowed_images,
                memory.required_images,
                memory.generated_images,
                memory.locked_identifiers,
            ]
        ):
            return ""

        lines = [
            "以下为系统保留的结构化记忆；它比旧 prose 更可信，尤其是 section key、图片文件名、引用标识。",
            "如果旧上下文中的自然语言描述与这里冲突，永远以这里的精确标识为准。",
        ]
        if memory.current_section:
            lines.append(f"当前章节: {memory.current_section}")
        if memory.completed_sections:
            lines.append(f"已完成章节: {', '.join(memory.completed_sections)}")
        if memory.allowed_images:
            lines.append(f"允许图片: {', '.join(memory.allowed_images)}")
        if memory.required_images:
            lines.append(f"必须图片: {', '.join(memory.required_images)}")
        if memory.generated_images:
            lines.append(f"已确认生成图片: {', '.join(memory.generated_images)}")
        if memory.locked_identifiers:
            lines.append(f"锁定标识: {', '.join(memory.locked_identifiers[:20])}")
        if memory.open_tasks:
            lines.append("待办/约束:")
            lines.extend(f"- {item}" for item in memory.open_tasks[:8])

        for section_key, state in list(memory.section_states.items())[:6]:
            lines.append(f"章节状态[{section_key}]:")
            if state.allowed_images:
                lines.append(f"- 允许图片: {', '.join(state.allowed_images)}")
            if state.required_images:
                lines.append(f"- 必须图片: {', '.join(state.required_images)}")
            if state.generated_images:
                lines.append(f"- 已生成图片: {', '.join(state.generated_images)}")
            if state.open_tasks:
                lines.extend(f"- 待办: {item}" for item in state.open_tasks[:4])
            if state.facts:
                lines.extend(f"- 事实: {item}" for item in state.facts[:4])

        return "\n".join(lines)

    def _find_safe_preserve_point(self) -> int:
        min_preserve = min(3, len(self.chat_history))
        preserve_start = len(self.chat_history) - min_preserve
        ic(
            f"寻找安全保留点: 历史长度={len(self.chat_history)}, 最少保留={min_preserve}, 开始位置={preserve_start}"
        )

        for i in range(preserve_start, -1, -1):
            if i >= len(self.chat_history):
                continue
            is_safe = self._is_safe_cut_point(i)
            ic(f"检查位置 {i}: 安全={is_safe}")
            if is_safe:
                ic(f"找到安全保留点: {i}")
                return i

        fallback = len(self.chat_history) - 1
        ic(f"未找到安全点，使用备用位置: {fallback}")
        return fallback

    def _is_safe_cut_point(self, start_idx: int) -> bool:
        if start_idx >= len(self.chat_history):
            ic(f"切割点 {start_idx} >= 历史长度，安全")
            return True

        tool_messages = []
        for i in range(start_idx, len(self.chat_history)):
            msg = self.chat_history[i]
            if isinstance(msg, dict) and msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id")
                tool_messages.append((i, tool_call_id))
                ic(f"发现tool消息在位置 {i}, tool_call_id={tool_call_id}")

                if tool_call_id:
                    found_tool_call = False
                    for j in range(start_idx, i):
                        prev_msg = self.chat_history[j]
                        if (
                            isinstance(prev_msg, dict)
                            and "tool_calls" in prev_msg
                            and prev_msg["tool_calls"]
                        ):
                            for tool_call in prev_msg["tool_calls"]:
                                if tool_call.get("id") == tool_call_id:
                                    found_tool_call = True
                                    ic(f"找到对应的tool_call在位置 {j}")
                                    break
                            if found_tool_call:
                                break

                    if not found_tool_call:
                        ic(
                            f"❌ tool消息 {tool_call_id} 没有找到对应的tool_call，切割点不安全"
                        )
                        return False

        ic(f"切割点 {start_idx} 安全，检查了 {len(tool_messages)} 个tool消息")
        return True

    def _get_safe_fallback_history(self) -> list:
        if not self.chat_history:
            return []

        safe_history = []
        if self.chat_history and self.chat_history[0]["role"] == "system":
            safe_history.append(self.chat_history[0])

        for preserve_count in range(1, min(4, len(self.chat_history)) + 1):
            start_idx = len(self.chat_history) - preserve_count
            if self._is_safe_cut_point(start_idx):
                safe_history.extend(self.chat_history[start_idx:])
                return safe_history

        for i in range(len(self.chat_history) - 1, -1, -1):
            msg = self.chat_history[i]
            if isinstance(msg, dict) and msg.get("role") != "tool":
                safe_history.append(msg)
                break

        return safe_history

    def _find_last_unmatched_tool_call(self) -> int | None:
        ic("开始查找未匹配的tool_call")

        for i in range(len(self.chat_history) - 1, -1, -1):
            msg = self.chat_history[i]
            if isinstance(msg, dict) and "tool_calls" in msg and msg["tool_calls"]:
                ic(f"在位置 {i} 发现tool_calls消息")
                for tool_call in msg["tool_calls"]:
                    tool_call_id = tool_call.get("id")
                    ic(f"检查tool_call_id: {tool_call_id}")
                    if tool_call_id:
                        response_found = False
                        for j in range(i + 1, len(self.chat_history)):
                            response_msg = self.chat_history[j]
                            if (
                                isinstance(response_msg, dict)
                                and response_msg.get("role") == "tool"
                                and response_msg.get("tool_call_id") == tool_call_id
                            ):
                                ic(f"找到匹配的tool响应在位置 {j}")
                                response_found = True
                                break
                        if not response_found:
                            ic(f"❌ 发现未匹配的tool_call在位置 {i}, id={tool_call_id}")
                            return i

        ic("没有发现未匹配的tool_call")
        return None
