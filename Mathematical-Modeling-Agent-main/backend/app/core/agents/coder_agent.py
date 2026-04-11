from app.core.agents.agent import Agent
from app.config.setting import settings
from app.utils.log_util import logger
from app.services.redis_manager import redis_manager
from app.schemas.response import SystemMessage, InterpreterMessage
from app.tools.base_interpreter import BaseCodeInterpreter
from app.core.llm.llm import LLM
from app.schemas.A2A import CoderToWriter, GeneratedFigure, RequiredFigure
from app.core.prompts import CODER_PROMPT
from app.utils.common_utils import get_current_files
import json
import os
from app.core.prompts import get_reflection_prompt
from app.core.functions import coder_tools

# TODO: 时间等待过久，stop 进程
# TODO: 支持 cuda
# TODO: 引入创新方案：


# 代码强
class CoderAgent(Agent):  # 同样继承自Agent类
    MIN_IMAGES_BY_SECTION = {
        "eda": 2,
        "sensitivity_analysis": 2,
    }

    def __init__(
        self,
        task_id: str,
        model: LLM,
        work_dir: str,  # 工作目录
        max_chat_turns: int = settings.MAX_CHAT_TURNS,  # 最大聊天次数
        max_retries: int = settings.MAX_RETRIES,  # 最大反思次数
        max_memory: int = 24,  # 最大记忆轮次
        code_interpreter: BaseCodeInterpreter = None,
    ) -> None:
        super().__init__(task_id, model, max_chat_turns, max_memory)
        self.work_dir = work_dir
        self.max_retries = max_retries
        self.is_first_run = True
        self.system_prompt = CODER_PROMPT
        self.code_interpreter = code_interpreter

    @classmethod
    def _get_min_expected_images(cls, subtask_title: str) -> int:
        if subtask_title.startswith("ques"):
            return 3
        return cls.MIN_IMAGES_BY_SECTION.get(subtask_title, 0)

    @staticmethod
    def _format_image_list(images: list[str]) -> str:
        return ", ".join(images) if images else "无"

    @staticmethod
    def _normalize_image_name(image_name: str) -> str:
        return os.path.basename((image_name or "").strip())

    @classmethod
    def _normalize_required_figures(
        cls, required_figures: list[RequiredFigure] | list[dict] | None
    ) -> list[RequiredFigure]:
        normalized: list[RequiredFigure] = []
        for figure in required_figures or []:
            try:
                figure_obj = figure if isinstance(figure, RequiredFigure) else RequiredFigure(**figure)
            except Exception:
                continue
            figure_obj.filename = cls._normalize_image_name(figure_obj.filename)
            if not figure_obj.filename:
                continue
            normalized.append(figure_obj)
        return normalized

    @classmethod
    def _get_required_filenames(
        cls, required_figures: list[RequiredFigure] | list[dict] | None
    ) -> list[str]:
        filenames: list[str] = []
        for figure in cls._normalize_required_figures(required_figures):
            if not figure.required:
                continue
            if figure.filename not in filenames:
                filenames.append(figure.filename)
        return filenames

    @classmethod
    def _find_missing_required_images(
        cls,
        created_images: list[str],
        required_figures: list[RequiredFigure] | list[dict] | None,
    ) -> list[str]:
        created = {
            cls._normalize_image_name(image)
            for image in created_images or []
            if cls._normalize_image_name(image)
        }
        return [
            filename
            for filename in cls._get_required_filenames(required_figures)
            if filename not in created
        ]

    @classmethod
    def _build_generated_figures_manifest(
        cls,
        required_figures: list[RequiredFigure] | list[dict] | None,
        created_images: list[str],
    ) -> list[GeneratedFigure]:
        normalized_required_figures = cls._normalize_required_figures(required_figures)
        created = []
        created_set = set()
        for image in created_images or []:
            normalized = cls._normalize_image_name(image)
            if normalized and normalized not in created_set:
                created.append(normalized)
                created_set.add(normalized)

        manifest: list[GeneratedFigure] = []
        seen_filenames: set[str] = set()

        for figure in normalized_required_figures:
            manifest.append(
                GeneratedFigure(
                    figure_id=figure.figure_id,
                    filename=figure.filename,
                    purpose=figure.purpose,
                    section_hint=figure.section_hint,
                    caption_hint=figure.caption_hint,
                    required=figure.required,
                    generated=figure.filename in created_set,
                )
            )
            seen_filenames.add(figure.filename)

        for image_name in created:
            if image_name in seen_filenames:
                continue
            figure_id = os.path.splitext(image_name)[0]
            manifest.append(
                GeneratedFigure(
                    figure_id=figure_id,
                    filename=image_name,
                    generated=True,
                )
            )
            seen_filenames.add(image_name)

        return manifest

    def _build_required_figures_prompt(
        self,
        subtask_title: str,
        required_figures: list[RequiredFigure] | list[dict] | None,
    ) -> str:
        figures = self._normalize_required_figures(required_figures)
        if not figures:
            return ""

        lines = [
            "【本题图片交付清单】",
            f"当前章节: {subtask_title}",
            "你必须严格按以下清单生成论文主图，文件名必须完全一致：",
        ]
        for index, figure in enumerate(figures, start=1):
            requirement = "必须生成" if figure.required else "可选补充"
            detail_parts = [
                f"{index}. {requirement}: {figure.filename}",
                f"figure_id={figure.figure_id}",
            ]
            if figure.purpose:
                detail_parts.append(f"用途={figure.purpose}")
            if figure.section_hint:
                detail_parts.append(f"插入位置={figure.section_hint}")
            if figure.caption_hint:
                detail_parts.append(f"图注提示={figure.caption_hint}")
            lines.append("；".join(detail_parts))
        lines.extend(
            [
                "禁止自创新的论文主图文件名，也不要把这些图保存成 step0_xxx、fig1.png、image.png、plot.png 之类的临时名字。",
                "每张图保存后，继续按既有规范打印关键数据特征。",
                "最终总结时，请明确列出已经生成的图片文件名，并与上面的清单逐一对应。",
            ]
        )
        return "\n".join(lines)

    def _build_image_completion_prompt(
        self,
        subtask_title: str,
        created_images: list[str],
        min_expected_images: int,
    ) -> str:
        missing_count = max(min_expected_images - len(created_images), 0)
        return (
            "系统校验发现当前子任务的论文图表产出不足，请继续补充最能支撑结论的图表，而不是重写已有分析。\n"
            f"当前章节: {subtask_title}\n"
            f"当前已有图片({len(created_images)}张): {self._format_image_list(created_images)}\n"
            f"至少需要图片数: {min_expected_images}\n"
            f"至少还需新增: {missing_count} 张\n"
            "必须满足以下要求：\n"
            "1. 所有新图片都保存到当前工作目录。\n"
            "2. 文件名必须语义化且稳定，采用 `章节_语义名.png` 风格，例如 "
            f"`{subtask_title}_correlation_heatmap.png`、`{subtask_title}_residual_diagnostics.png`。\n"
            "3. 严禁使用 `fig1.png`、`image.png`、`plot.png`、`step0_xxx.png` 这类泛化或临时文件名。\n"
            "4. 每张图保存后，必须立刻用 print() 输出该图的关键数据特征，方便后续写作手准确解读。\n"
            "5. 优先补充能够支撑模型结论的结果图、诊断图、对比图、敏感性图，而不是重复画同类图。\n"
            "6. 完成后请明确总结所有图片文件名，再结束。"
        )

    def _build_required_figure_completion_prompt(
        self,
        subtask_title: str,
        created_images: list[str],
        required_figures: list[RequiredFigure] | list[dict] | None,
        missing_required_images: list[str],
    ) -> str:
        figures = self._normalize_required_figures(required_figures)
        lines = [
            "系统校验发现当前子任务缺少建模手指定的必需图片，请补齐这些图片，不要重写已有分析。",
            f"当前章节: {subtask_title}",
            f"当前已有图片({len(created_images)}张): {self._format_image_list(created_images)}",
            f"缺少的必需图片: {self._format_image_list(missing_required_images)}",
            "必须满足以下要求：",
            "1. 新补充的图片必须保存到当前工作目录，文件名与缺失清单完全一致。",
            "2. 不要把图片保存成 step0_xxx、fig1.png、image.png、plot.png 等临时名字。",
            "3. 已经生成正确文件名的图片不要改名重做，优先补齐缺失项。",
            "4. 每张补充图片保存后，必须立刻用 print() 输出图中关键数据特征。",
            "5. 完成后请明确列出全部已生成图片文件名，再结束。",
        ]
        if figures:
            lines.append("参考图片清单如下：")
            for figure in figures:
                requirement = "必须生成" if figure.required else "可选补充"
                detail = [f"- {requirement}: {figure.filename}"]
                if figure.purpose:
                    detail.append(f"用途={figure.purpose}")
                if figure.caption_hint:
                    detail.append(f"图注提示={figure.caption_hint}")
                lines.append("；".join(detail))
        return "\n".join(lines)

    async def _build_terminal_result(
        self,
        subtask_title: str,
        required_figures: list[RequiredFigure] | list[dict] | None,
        fallback_message: str,
        preferred_response: str | None = None,
    ) -> CoderToWriter:
        created_images = await self.code_interpreter.get_created_images(subtask_title)
        response_text = preferred_response or fallback_message
        return CoderToWriter(
            code_response=response_text,
            code_output=self.code_interpreter.get_code_output(subtask_title),
            created_images=created_images,
            generated_figures=self._build_generated_figures_manifest(
                required_figures,
                created_images,
            ),
        )

    async def run(
        self,
        prompt: str,
        subtask_title: str,
        required_figures: list[RequiredFigure] | list[dict] | None = None,
    ) -> CoderToWriter:
        logger.info(f"{self.__class__.__name__}:开始:执行子任务: {subtask_title}")
        self.code_interpreter.add_section(subtask_title)
        normalized_required_figures = self._normalize_required_figures(required_figures)
        required_filenames = self._get_required_filenames(normalized_required_figures)

        # 如果是第一次运行，则添加系统提示
        if self.is_first_run:
            logger.info("首次运行，添加系统提示和数据集文件信息")
            self.is_first_run = False
            await self.append_chat_history(
                {"role": "system", "content": self.system_prompt}
            )
            # 当前数据集文件
            await self.append_chat_history(
                {
                    "role": "user",
                    "content": f"当前文件夹下的数据集文件{get_current_files(self.work_dir, 'data')}",
                }
            )

        # 添加 sub_task
        prompt_parts = [prompt]
        required_figures_prompt = self._build_required_figures_prompt(
            subtask_title, normalized_required_figures
        )
        if required_figures_prompt:
            prompt_parts.append(required_figures_prompt)
        task_prompt = "\n\n".join(part for part in prompt_parts if part)
        logger.info(f"添加子任务提示: {task_prompt}")
        await self.append_chat_history({"role": "user", "content": task_prompt})

        retry_count = 0
        last_error_message = ""
        last_successful_response: str | None = None

        while True:
            if retry_count >= self.max_retries:
                logger.error(f"超过最大尝试次数: {self.max_retries}")
                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content="超过最大尝试次数", type="error"),
                )
                logger.warning(f"任务失败，超过最大尝试次数{self.max_retries}, 最后错误信息: {last_error_message}")
                return await self._build_terminal_result(
                    subtask_title=subtask_title,
                    required_figures=normalized_required_figures,
                    fallback_message=(
                        f"任务失败，超过最大尝试次数{self.max_retries}, "
                        f"最后错误信息: {last_error_message}"
                    ),
                    preferred_response=last_successful_response,
                )
                

            if self.current_chat_turns >= self.max_chat_turns:
                logger.error(f"超过最大聊天次数: {self.max_chat_turns}")
                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content="超过最大聊天次数", type="error"),
                )
                raise Exception(
                    f"Reached maximum number of chat turns ({self.max_chat_turns}). Task incomplete."
                )

            self.current_chat_turns += 1
            logger.info(f"当前对话轮次: {self.current_chat_turns}")
            
            try:
                response = await self.model.chat(
                    history=self.chat_history,
                    tools=coder_tools,
                    tool_choice="auto",
                    agent_name=self.__class__.__name__,
                )

                # 如果有工具调用
                if (
                    hasattr(response.choices[0].message, "tool_calls")
                    and response.choices[0].message.tool_calls
                ):
                    logger.info("检测到工具调用")
                    tool_call = response.choices[0].message.tool_calls[0]
                    tool_id = tool_call.id
                    
                    if tool_call.function.name == "execute_code":
                        logger.info(f"调用工具: {tool_call.function.name}")
                        await redis_manager.publish_message(
                            self.task_id,
                            SystemMessage(
                                content=f"代码手调用{tool_call.function.name}工具"
                            ),
                        )

                        code = json.loads(tool_call.function.arguments)["code"]

                        await redis_manager.publish_message(
                            self.task_id,
                            InterpreterMessage(
                                input={"code": code},
                            ),
                        )

                        # 更新对话历史 - 添加助手的响应
                        await self.append_chat_history(
                            response.choices[0].message.model_dump()
                        )
                        logger.info(response.choices[0].message.model_dump())

                        # 执行工具调用
                        logger.info("执行工具调用")
                        (
                            text_to_gpt,
                            error_occurred,
                            error_message,
                        ) = await self.code_interpreter.execute_code(code)

                        # 添加工具执行结果
                        if error_occurred:
                            # 即使发生错误也要添加tool响应
                            await self.append_chat_history(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": "execute_code",
                                    "content": error_message,
                                }
                            )

                            logger.warning(f"代码执行错误: {error_message}")
                            retry_count += 1
                            logger.info(f"当前尝试次:{retry_count} / {self.max_retries}")
                            last_error_message = error_message
                            reflection_prompt = get_reflection_prompt(error_message, code)

                            await redis_manager.publish_message(
                                self.task_id,
                                SystemMessage(content="代码手反思纠正错误", type="error"),
                            )

                            await self.append_chat_history(
                                {"role": "user", "content": reflection_prompt}
                            )
                            continue
                        else:
                            # 成功执行的tool响应
                            await self.append_chat_history(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": "execute_code",
                                    "content": text_to_gpt,
                                }
                            )
                            # 成功执行后继续循环，等待下一步指令
                            continue
                else:
                    # 没有工具调用，表示任务完成
                    logger.info("没有工具调用，任务完成")
                    response_content = response.choices[0].message.content
                    if response_content:
                        last_successful_response = response_content
                    created_images = await self.code_interpreter.get_created_images(
                        subtask_title
                    )
                    missing_required_images = self._find_missing_required_images(
                        created_images,
                        normalized_required_figures,
                    )
                    # 当建模手已经给出结构化必需图片清单时，优先遵守该清单；
                    # 通用的最少图片数仅作为没有明确清单时的兜底规则。
                    min_expected_images = (
                        len(required_filenames)
                        if required_filenames
                        else self._get_min_expected_images(subtask_title)
                    )

                    if missing_required_images:
                        logger.warning(
                            f"{subtask_title} 缺少必需图片: {missing_required_images}"
                        )
                        retry_count += 1
                        last_error_message = (
                            f"{subtask_title} 缺少必需图片："
                            f"{self._format_image_list(missing_required_images)}"
                        )
                        await redis_manager.publish_message(
                            self.task_id,
                            SystemMessage(
                                content=f"代码手正在补齐 {subtask_title} 必需图表",
                                type="error",
                            ),
                        )
                        if response_content:
                            await self.append_chat_history(
                                {"role": "assistant", "content": response_content}
                            )
                        await self.append_chat_history(
                            {
                                "role": "user",
                                "content": self._build_required_figure_completion_prompt(
                                    subtask_title=subtask_title,
                                    created_images=created_images,
                                    required_figures=normalized_required_figures,
                                    missing_required_images=missing_required_images,
                                ),
                            }
                        )
                        continue

                    if len(created_images) < min_expected_images:
                        logger.warning(
                            f"{subtask_title} 图片产出不足，当前 {len(created_images)} 张，至少需要 {min_expected_images} 张"
                        )
                        retry_count += 1
                        last_error_message = (
                            f"{subtask_title} 图片产出不足："
                            f"{len(created_images)}/{min_expected_images}"
                        )
                        await redis_manager.publish_message(
                            self.task_id,
                            SystemMessage(
                                content=f"代码手正在补充 {subtask_title} 图表",
                                type="error",
                            ),
                        )
                        if response_content:
                            await self.append_chat_history(
                                {"role": "assistant", "content": response_content}
                            )
                        await self.append_chat_history(
                            {
                                "role": "user",
                                "content": self._build_image_completion_prompt(
                                    subtask_title=subtask_title,
                                    created_images=created_images,
                                    min_expected_images=min_expected_images,
                                ),
                            }
                        )
                        continue

                    return await self._build_terminal_result(
                        subtask_title=subtask_title,
                        required_figures=normalized_required_figures,
                        fallback_message=response_content or "",
                        preferred_response=response_content,
                    )
                    
            except Exception as e:
                logger.error(f"执行过程中发生异常: {str(e)}")
                retry_count += 1
                last_error_message = str(e)
                continue
            logger.info(f"{self.__class__.__name__}:完成:执行子任务: {subtask_title}")
