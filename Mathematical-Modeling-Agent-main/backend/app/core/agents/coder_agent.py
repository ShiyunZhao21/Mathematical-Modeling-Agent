from app.core.agents.agent import Agent
from app.config.setting import settings
from app.utils.log_util import logger
from app.services.redis_manager import redis_manager
from app.schemas.response import SystemMessage, InterpreterMessage
from app.tools.base_interpreter import BaseCodeInterpreter
from app.core.llm.llm import LLM
from app.schemas.A2A import CoderToWriter
from app.core.prompts import CODER_PROMPT
from app.utils.common_utils import get_current_files
import json
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
        code_interpreter: BaseCodeInterpreter = None,
    ) -> None:
        super().__init__(task_id, model, max_chat_turns)
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

    async def run(self, prompt: str, subtask_title: str) -> CoderToWriter:
        logger.info(f"{self.__class__.__name__}:开始:执行子任务: {subtask_title}")
        self.code_interpreter.add_section(subtask_title)

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
        logger.info(f"添加子任务提示: {prompt}")
        await self.append_chat_history({"role": "user", "content": prompt})

        retry_count = 0
        last_error_message = ""

        while True:
            if retry_count >= self.max_retries:
                logger.error(f"超过最大尝试次数: {self.max_retries}")
                await redis_manager.publish_message(
                    self.task_id,
                    SystemMessage(content="超过最大尝试次数", type="error"),
                )
                logger.warning(f"任务失败，超过最大尝试次数{self.max_retries}, 最后错误信息: {last_error_message}")
                return CoderToWriter(
                    code_response=f"任务失败，超过最大尝试次数{self.max_retries}, 最后错误信息: {last_error_message}",
                    code_output=self.code_interpreter.get_code_output(subtask_title),
                    created_images=[])
                

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
                    created_images = await self.code_interpreter.get_created_images(
                        subtask_title
                    )
                    min_expected_images = self._get_min_expected_images(subtask_title)

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

                    return CoderToWriter(
                        code_response=response_content,
                        code_output=self.code_interpreter.get_code_output(subtask_title),
                        created_images=created_images,
                    )
                    
            except Exception as e:
                logger.error(f"执行过程中发生异常: {str(e)}")
                retry_count += 1
                last_error_message = str(e)
                continue
            logger.info(f"{self.__class__.__name__}:完成:执行子任务: {subtask_title}")
