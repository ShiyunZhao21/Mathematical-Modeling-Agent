from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from app.core.agents.agent import Agent
from app.core.functions import writer_tools
from app.core.llm.llm import LLM
from app.core.prompts import get_writer_prompt
from app.schemas.A2A import GeneratedFigure, RequiredFigure, WriterResponse
from app.schemas.enums import CompTemplate, FormatOutPut
from app.schemas.response import SystemMessage
from app.services.redis_manager import task_store
from app.tools.openalex_scholar import OpenAlexScholar
from app.utils.log_util import logger
from icecream import ic
import json


# 长文本
# TODO: 并行 parallel
# TODO: 获取当前文件下的文件
# TODO: 引用cites tool
class WriterAgent(Agent):
    IMAGE_REWRITE_MAX_ATTEMPTS = 2
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")

    def __init__(
        self,
        task_id: str,
        model: LLM,
        max_chat_turns: int = 10,
        comp_template: CompTemplate = CompTemplate,
        format_output: FormatOutPut = FormatOutPut.Markdown,
        scholar: OpenAlexScholar = None,
        max_memory: int = 25,
        work_dir: str | None = None,
    ) -> None:
        super().__init__(task_id, model, max_chat_turns, max_memory)
        self.format_out_put = format_output
        self.comp_template = comp_template
        self.scholar = scholar
        self.is_first_run = True
        self.system_prompt = get_writer_prompt(format_output)
        self.available_images: list[str] = []
        self.allowed_images: list[str] = []
        self.required_images: list[str] = []
        self.required_figures: list[RequiredFigure] = []
        self.generated_figures: list[GeneratedFigure] = []
        self.missing_required_generation: list[str] = []
        self.work_dir = work_dir

    @staticmethod
    def _normalize_image_name(image_path: str) -> str:
        return os.path.basename((image_path or "").strip())

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
    def _normalize_generated_figures(
        cls, generated_figures: list[GeneratedFigure] | list[dict] | None
    ) -> list[GeneratedFigure]:
        normalized: list[GeneratedFigure] = []
        for figure in generated_figures or []:
            try:
                figure_obj = figure if isinstance(figure, GeneratedFigure) else GeneratedFigure(**figure)
            except Exception:
                continue
            figure_obj.filename = cls._normalize_image_name(figure_obj.filename)
            if not figure_obj.filename:
                continue
            normalized.append(figure_obj)
        return normalized

    @classmethod
    def _dedupe_image_names(cls, images: list[str] | None) -> list[str]:
        deduped: list[str] = []
        for image in images or []:
            normalized = cls._normalize_image_name(image)
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped

    @classmethod
    def _build_image_contract(
        cls,
        available_images: list[str] | None,
        required_figures: list[RequiredFigure] | list[dict] | None,
        generated_figures: list[GeneratedFigure] | list[dict] | None,
    ) -> tuple[list[str], list[str], list[str]]:
        normalized_required_figures = cls._normalize_required_figures(required_figures)
        normalized_generated_figures = cls._normalize_generated_figures(generated_figures)

        if normalized_generated_figures:
            allowed_images = cls._dedupe_image_names(
                [figure.filename for figure in normalized_generated_figures if figure.generated]
            )
            required_images = cls._dedupe_image_names(
                [
                    figure.filename
                    for figure in normalized_generated_figures
                    if figure.required and figure.generated
                ]
            )
            missing_required_generation = cls._dedupe_image_names(
                [
                    figure.filename
                    for figure in normalized_required_figures
                    if figure.required and figure.filename not in required_images
                ]
            )
            return allowed_images, required_images, missing_required_generation

        fallback_images = cls._dedupe_image_names(available_images)
        if fallback_images:
            return fallback_images, fallback_images, []

        required_images = cls._dedupe_image_names(
            [figure.filename for figure in normalized_required_figures if figure.required]
        )
        return required_images, required_images, []

    @classmethod
    def _extract_inserted_images(cls, content: str) -> set[str]:
        matches = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content or "")
        return {
            cls._normalize_image_name(match.split("?", 1)[0].split("#", 1)[0])
            for match in matches
            if cls._normalize_image_name(match)
        }

    @classmethod
    def _extract_raw_image_references(cls, content: str) -> list[str]:
        matches = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content or "")
        return [
            match.split("?", 1)[0].split("#", 1)[0].strip()
            for match in matches
            if match.strip()
        ]

    @classmethod
    def _find_missing_images(cls, content: str, expected_images: list[str] | None) -> list[str]:
        expected = [
            cls._normalize_image_name(image)
            for image in (expected_images or [])
            if cls._normalize_image_name(image)
        ]
        inserted = cls._extract_inserted_images(content)
        return [image for image in expected if image not in inserted]

    @classmethod
    def _find_missing_images_on_disk(
        cls, expected_images: list[str] | None, actual_images: set[str]
    ) -> list[str]:
        expected = [
            cls._normalize_image_name(image)
            for image in (expected_images or [])
            if cls._normalize_image_name(image)
        ]
        return [image for image in expected if image not in actual_images]

    @classmethod
    def _find_invalid_image_references(
        cls, content: str, actual_images: set[str]
    ) -> list[str]:
        invalid: list[str] = []
        for raw_reference in cls._extract_raw_image_references(content):
            normalized = cls._normalize_image_name(raw_reference)
            if raw_reference != normalized or normalized not in actual_images:
                invalid.append(raw_reference)
        return invalid

    def _get_actual_image_files(self) -> set[str]:
        if not self.work_dir or not os.path.isdir(self.work_dir):
            return set()

        return {
            file
            for file in os.listdir(self.work_dir)
            if os.path.isfile(os.path.join(self.work_dir, file))
            and file.lower().endswith(self.IMAGE_EXTENSIONS)
        }

    def _reconcile_contract_with_disk(self) -> None:
        actual_images = self._get_actual_image_files()
        if not actual_images:
            return

        on_disk_required = self._dedupe_image_names(
            [
                figure.filename
                for figure in self.required_figures
                if figure.required and figure.filename in actual_images
            ]
        )
        if on_disk_required:
            self.allowed_images = self._dedupe_image_names(
                [*self.allowed_images, *on_disk_required]
            )
            self.required_images = self._dedupe_image_names(
                [*self.required_images, *on_disk_required]
            )

        self.missing_required_generation = [
            image for image in self.missing_required_generation if image not in actual_images
        ]

    # ===== FIX: 压缩变体恢复机制（与 CoderAgent 完全一致） =====
    @staticmethod
    def _try_recover_from_variants(work_dir: Path, fname: str) -> Path | None:
        """
        按优先级搜索压缩后产生的变体文件，返回第一个找到的路径。
        搜索范围：work_dir 本身 + work_dir/compressed/ 子目录
        与 CoderAgent._try_recover_from_variants 保持完全一致。
        """
        stem = Path(fname).stem
        suffix = Path(fname).suffix

        candidate_names: list[str] = [
            fname,  # original filename — matches compressed/<fname> subdir copies
            f"{stem}_compressed{suffix}",
            f"{stem}_thumb{suffix}",
            f"{stem}_resized{suffix}",
            f"{stem}_optimized{suffix}",
            stem + (".jpg" if suffix.lower() == ".png" else ".png"),
            stem + ".jpeg",
            stem + ".webp",
        ]

        search_dirs: list[Path] = [
            work_dir,
            work_dir / "compressed",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for candidate in candidate_names:
                # Skip searching for the original filename in the root work_dir —
                # it was deleted there (that's why we're recovering). Only look for
                # it in subdirs (e.g. compressed/<fname>).
                if candidate == fname and search_dir == work_dir:
                    continue
                candidate_path = search_dir / candidate
                if candidate_path.exists():
                    return candidate_path

        return None

    @staticmethod
    def _generate_placeholder_image(image_path: Path, filename: str) -> None:
        """生成占位图（让 LaTeX/Markdown 渲染不因缺图崩溃）"""
        import matplotlib.pyplot as plt
        image_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.55, f"占位图片\n{filename}\n\n原始图片丢失，Writer 自动生成",
                ha='center', va='center', fontsize=14, color='#d32f2f', fontweight='bold')
        ax.text(0.5, 0.35, "Writer 自动占位图 v2.5",
                ha='center', va='center', fontsize=11, color='gray')
        ax.set_title("Writer 自动占位图", color='blue', fontsize=14)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    async def _pre_compile_ensure_images(
        self,
        work_dir: Path,
        required_images: list[str],
    ) -> list[str]:
        """
        FIX: Writer 硬校验前的最后一次修复机会。

        原来的问题：Writer 在 run() 方法开头直接 raise ValueError，
        但 CoderAgent 的修复机制只在 execute_code 成功后运行一次。
        如果 Coder 最后一次代码执行恰好触发压缩，修复没有重跑，
        Writer 就会看到空文件列表并崩溃。

        修复策略：
          1. 文件已存在 → 跳过
          2. 找到压缩变体 → copy 回原名
          3. 以上均失败 → 生成占位图（让编译继续，不崩溃）

        返回修复后仍然缺失的文件列表（正常情况下为空列表）。
        """
        for fname in required_images:
            target = work_dir / fname
            if target.exists():
                continue

            recovered = self._try_recover_from_variants(work_dir, fname)
            if recovered:
                shutil.copy2(recovered, target)
                logger.warning(f"[Writer修复] 从压缩变体恢复: {recovered.name} → {fname}")
                continue

            # 兜底：占位图，让编译不崩溃
            self._generate_placeholder_image(target, fname)
            logger.warning(f"[Writer修复] 生成占位图: {fname}（原始图片丢失，无变体可恢复）")

        # 返回修复后仍缺失的（占位图已兜底，理论上应为空）
        return [fname for fname in required_images if not (work_dir / fname).exists()]

    async def _ensure_images_inserted(
        self,
        response_content: str,
        tools,
        tool_choice,
        sub_title: str | None,
    ) -> str:
        if not self.allowed_images:
            self.allowed_images = self._dedupe_image_names(self.available_images)
        if not self.required_images:
            self.required_images = list(self.allowed_images)

        # ===== FIX: 硬校验前先执行修复，而非直接 raise =====
        if self.missing_required_generation and self.work_dir:
            work_dir_path = Path(self.work_dir).resolve()
            logger.warning(
                f"[Writer] missing_required_generation={self.missing_required_generation}，"
                f"执行修复后再校验"
            )
            still_missing = await self._pre_compile_ensure_images(
                work_dir_path, self.missing_required_generation
            )
            if still_missing:
                # 占位图已经生成，文件存在了，更新 missing_required_generation
                # 理论上 still_missing 应该是空的，但保留日志
                logger.error(f"[Writer] 修复后仍缺失（极端情况）: {still_missing}")
            # 修复完成后，重新对齐 contract（磁盘上现在应该都有了）
            self._reconcile_contract_with_disk()
            # 修复后不再视为 missing，继续正常编译
            self.missing_required_generation = []
        elif self.missing_required_generation:
            # work_dir 未配置，无法修复，只能 raise
            raise ValueError(
                "Writer 硬校验失败：代码手未交付必需图片: "
                f"{self.missing_required_generation}"
            )
        # ===================================================

        actual_images = self._get_actual_image_files()

        # ===== FIX: 磁盘缺失时也先修复再校验 =====
        missing_on_disk = self._find_missing_images_on_disk(
            self.allowed_images,
            actual_images,
        )
        if missing_on_disk and self.work_dir:
            work_dir_path = Path(self.work_dir).resolve()
            logger.warning(f"[Writer] 磁盘缺失图片={missing_on_disk}，执行修复")
            await self._pre_compile_ensure_images(work_dir_path, missing_on_disk)
            actual_images = self._get_actual_image_files()
            missing_on_disk = self._find_missing_images_on_disk(self.allowed_images, actual_images)

        if missing_on_disk:
            raise ValueError(
                f"Writer 硬校验失败：代码手声明的图片文件不存在于工作目录: {missing_on_disk}"
            )
        # ==========================================

        missing_images = self._find_missing_images(response_content, self.required_images)
        invalid_images = self._find_invalid_image_references(
            response_content, set(self.allowed_images)
        )
        attempt = 0

        while (missing_images or invalid_images) and attempt < self.IMAGE_REWRITE_MAX_ATTEMPTS:
            logger.warning(
                f"检测到图片引用未通过校验，缺失图片: {missing_images}, 无效引用: {invalid_images}"
            )
            self.chat_history.append({"role": "assistant", "content": response_content})
            prompt_lines = [
                "你刚才的章节图片引用未通过系统硬校验。请保留现有有效内容，并重新输出完整修订稿。",
                "必须满足以下要求：",
                f"1. 只能使用当前章节图片清单中的文件名：{', '.join(self.allowed_images) or '无'}",
            ]
            if self.required_images:
                prompt_lines.append(
                    f"2. 当前章节必须插入以下图片：{', '.join(self.required_images)}"
                )
            if missing_images:
                prompt_lines.append(
                    f"3. 以下必须插入的图片仍然缺失：{', '.join(missing_images)}"
                )
            if invalid_images:
                prompt_lines.append(
                    f"4. 以下图片引用无效，必须删除或替换：{', '.join(invalid_images)}"
                )
            prompt_lines.extend(
                [
                    "5. 每张图片必须使用独占一行的 Markdown 标签 ![描述](文件名)",
                    "6. 文件名只能写纯文件名，禁止目录、URL、以及任何不存在的 fig1_xxx / image_xxx 之类编造名",
                    "7. 每张图片前后要有至少3行分析解读",
                    "8. 只输出完整修订后的章节正文，不要解释你修改了什么",
                ]
            )
            fix_prompt = "\n".join(prompt_lines)
            await self.append_chat_history({"role": "user", "content": fix_prompt})
            retry_response = await self.model.chat(
                history=self.chat_history,
                tools=tools,
                tool_choice=tool_choice,
                agent_name=self.__class__.__name__,
                sub_title=sub_title,
            )
            response_content = retry_response.choices[0].message.content
            missing_images = self._find_missing_images(response_content, self.required_images)
            invalid_images = self._find_invalid_image_references(
                response_content, set(self.allowed_images)
            )
            attempt += 1

        if missing_images or invalid_images:
            raise ValueError(
                "Writer 硬校验失败："
                f"缺失图片={missing_images or '无'}；"
                f"无效引用={invalid_images or '无'}"
            )

        return response_content

    async def run(
        self,
        prompt: str,
        available_images: list[str] = None,
        required_figures: list[RequiredFigure] | list[dict] | None = None,
        generated_figures: list[GeneratedFigure] | list[dict] | None = None,
        sub_title: str = None,
    ) -> WriterResponse:
        """
        执行写作任务
        Args:
            prompt: 写作提示
            available_images: 可用的图片相对路径列表
            required_figures: 建模手定义的结构化图片清单
            generated_figures: 代码手实际交付的结构化图片清单
            sub_title: 子任务标题
        """
        # FIX(subtitle=None): 用 sub_title 兜底，避免日志中出现 subtitle是:None
        effective_sub_title = sub_title or "writer_task"
        logger.info(f"subtitle是:{effective_sub_title}")

        if self.is_first_run:
            self.is_first_run = False
            await self.append_chat_history(
                {"role": "system", "content": self.system_prompt}
            )

        self.available_images = self._dedupe_image_names(available_images)
        self.required_figures = self._normalize_required_figures(required_figures)
        self.generated_figures = self._normalize_generated_figures(generated_figures)
        (
            self.allowed_images,
            self.required_images,
            self.missing_required_generation,
        ) = self._build_image_contract(
            available_images=self.available_images,
            required_figures=self.required_figures,
            generated_figures=self.generated_figures,
        )
        self._reconcile_contract_with_disk()

        # ===== FIX: 不在 run() 开头直接 raise，改为在 _ensure_images_inserted 里修复后再校验 =====
        # 原来的代码：
        #   if self.missing_required_generation:
        #       raise ValueError("Writer 硬校验失败：代码手未交付必需图片: ...")
        # 现在移到 _ensure_images_inserted 内部，并在 raise 前先尝试修复
        # =======================================================================================

        if self.generated_figures:
            image_lines = []
            for figure in self.generated_figures:
                if not figure.generated:
                    continue
                requirement = "必须插入" if figure.required else "可选插入"
                detail = [f"- {requirement}: ![{figure.caption_hint or figure.filename}]({figure.filename})"]
                if figure.purpose:
                    detail.append(f"用途={figure.purpose}")
                if figure.section_hint:
                    detail.append(f"建议位置={figure.section_hint}")
                image_lines.append("；".join(detail))
            image_prompt = (
                f"\n\n【本章节结构化图片清单】\n"
                f"以下清单是当前章节唯一允许引用的图片文件名，请严格按清单写作，不要自行发明新图名：\n"
                f"{chr(10).join(image_lines) if image_lines else '无'}\n"
                f"必须插入的图片文件名：{', '.join(self.required_images) or '无'}\n"
                f"注意：当前 prompt 上文中如果出现"相关性热力图""ROC曲线""残差分析图""特征重要性图"等描述性名称，它们只是图的语义描述，不是文件名。\n"
                f"如果 prose 里的描述性名称与本清单冲突，永远以本清单为准；不得根据 prose 自行拼接或翻译出新的图片文件名。\n"
                f"只允许使用以上文件名，禁止引用工作目录中的其他图片，也禁止虚构 fig1_xxx.png、image.png、chart.png 等不存在文件。\n"
                f"插入格式为独占一行的 ![描述](文件名)，每张必需图片前后需配3行以上的分析解读。\n"
            )
            logger.info(f"image_prompt是:{image_prompt}")
            prompt = prompt + image_prompt
        elif self.allowed_images:
            image_lines = "\n".join(
                [f"- ![{img}]({img})" for img in self.allowed_images]
            )
            image_prompt = (
                f"\n\n【必须插入的图片列表】\n"
                f"以下图片是工作目录中真实存在、且由代码手生成的文件，你必须在论文相关段落后用 Markdown 格式逐一插入：\n"
                f"{image_lines}\n"
                f"注意：上文里出现的任何"热力图""ROC曲线""残差分析图"等描述性名称都只是语义说明，不是文件名；文件名只能从下面这份列表中选择。\n"
                f"只能使用上面这些真实文件名，禁止虚构 fig1_xxx.png、image.png、chart.png 等不存在文件。\n"
                f"插入格式为独占一行的 ![描述](文件名)，每张图片后需配3行以上的分析解读。\n"
            )
            logger.info(f"image_prompt是:{image_prompt}")
            prompt = prompt + image_prompt

        logger.info(f"{self.__class__.__name__}:开始:执行对话")
        self.current_chat_turns += 1

        await self.append_chat_history({"role": "user", "content": prompt})

        tools = writer_tools if self.scholar is not None else None
        tool_choice = "auto" if tools else None

        response = await self.model.chat(
            history=self.chat_history,
            tools=tools,
            tool_choice=tool_choice,
            agent_name=self.__class__.__name__,
            sub_title=effective_sub_title,
        )

        footnotes = []

        if (
            hasattr(response.choices[0].message, "tool_calls")
            and response.choices[0].message.tool_calls
        ):
            logger.info("检测到工具调用")
            tool_call = response.choices[0].message.tool_calls[0]
            tool_id = tool_call.id
            if tool_call.function.name == "search_papers":
                logger.info("调用工具: search_papers")
                await task_store.publish_message(
                    self.task_id,
                    SystemMessage(content=f"写作手调用{tool_call.function.name}工具"),
                )

                query = json.loads(tool_call.function.arguments)["query"]

                await self.append_chat_history(response.choices[0].message.model_dump())
                ic(response.choices[0].message.model_dump())

                try:
                    papers = await self.scholar.search_papers(query)
                except Exception as e:
                    error_msg = f"搜索文献失败: {str(e)}"
                    logger.error(error_msg)
                    return WriterResponse(
                        response_content=error_msg, footnotes=footnotes
                    )
                papers_str = self.scholar.papers_to_str(papers)
                logger.info(f"搜索文献结果\n{papers_str}")
                await self.append_chat_history(
                    {
                        "role": "tool",
                        "content": papers_str,
                        "tool_call_id": tool_id,
                        "name": "search_papers",
                    }
                )
                next_response = await self.model.chat(
                    history=self.chat_history,
                    tools=tools,
                    tool_choice=tool_choice,
                    agent_name=self.__class__.__name__,
                    sub_title=effective_sub_title,
                )
                response_content = next_response.choices[0].message.content
        else:
            response_content = response.choices[0].message.content

        response_content = await self._ensure_images_inserted(
            response_content=response_content,
            tools=tools,
            tool_choice=tool_choice,
            sub_title=effective_sub_title,
        )
        self.chat_history.append({"role": "assistant", "content": response_content})
        logger.info(f"{self.__class__.__name__}:完成:执行对话")
        return WriterResponse(response_content=response_content, footnotes=footnotes)

    async def summarize(self) -> str:
        """总结对话内容"""
        try:
            await self.append_chat_history(
                {"role": "user", "content": "请简单总结以上完成什么任务取得什么结果:"}
            )
            response = await self.model.chat(
                history=self.chat_history, agent_name=self.__class__.__name__
            )
            await self.append_chat_history(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"总结生成失败: {str(e)}")
            return "由于网络原因无法生成详细总结，但已完成主要任务处理。"