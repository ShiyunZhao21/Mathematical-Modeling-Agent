# base_interpreter.py
import abc
import os
import re
from app.tools.notebook_serializer import NotebookSerializer
from app.services.redis_manager import redis_manager
from app.utils.log_util import logger
from app.schemas.response import (
    OutputItem,
    InterpreterMessage,
)


class BaseCodeInterpreter(abc.ABC):
    CHINESE_FONT_FAMILIES = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
    GENERIC_IMAGE_STEM_PATTERN = re.compile(
        r"^(?:fig(?:ure)?|image|img|plot|chart|graph|visual(?:ization)?|"
        r"output|result|photo|pic|picture|tmp|temp)(?:[_-]?\d+)?$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        task_id: str,
        work_dir: str,
        notebook_serializer: NotebookSerializer,
    ):
        self.task_id = task_id
        self.work_dir = work_dir
        self.notebook_serializer = notebook_serializer
        self.section_output: dict[str, dict[str, list[str]]] = {}
        self.last_created_images = set()
        self.current_section: str | None = None

    @abc.abstractmethod
    async def initialize(self):
        """初始化解释器，必要时上传文件、启动内核等"""
        ...

    @abc.abstractmethod
    async def _pre_execute_code(self):
        """执行初始化代码"""
        ...

    @abc.abstractmethod
    async def execute_code(self, code: str) -> tuple[str, bool, str]:
        """执行一段代码，返回 (输出文本, 是否出错, 错误信息)"""
        ...

    @abc.abstractmethod
    async def cleanup(self):
        """清理资源，比如关闭沙箱或内核"""
        ...

    @abc.abstractmethod
    async def get_created_images(self, section: str) -> list[str]:
        """获取当前 section 创建的图片列表"""
        ...

    async def _push_to_websocket(self, content_to_display: list[OutputItem] | None):
        logger.info("执行结果已记录到任务消息")

        agent_msg = InterpreterMessage(
            output=content_to_display,
        )
        logger.debug(f"发送消息: {agent_msg.model_dump_json()}")
        await redis_manager.publish_message(
            self.task_id,
            agent_msg,
        )

    def add_section(self, section_name: str) -> None:
        """确保添加的section结构正确"""

        if section_name not in self.section_output:
            self.section_output[section_name] = {"content": [], "images": []}
        self.current_section = section_name

    def add_content(self, section: str, text: str) -> None:
        """向指定section添加文本内容"""
        if not text:
            return
        self.add_section(section)
        self.section_output[section]["content"].append(text)

    def get_code_output(self, section: str) -> str:
        """获取指定section的代码输出"""
        if section not in self.section_output:
            return ""
        return "\n".join(self.section_output[section]["content"])

    def record_execution_output(self, text: str) -> None:
        """记录当前 section 的代码输出，供写作阶段引用。"""
        if self.current_section and text:
            self.add_content(self.current_section, text)

    def delete_color_control_char(self, string):
        ansi_escape = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")
        return ansi_escape.sub("", string)

    def _truncate_text(self, text: str, max_length: int = 1000) -> str:
        """截断文本，保留开头和结尾的重要信息"""
        if len(text) <= max_length:
            return text

        half_length = max_length // 2
        return text[:half_length] + "\n... (内容已截断) ...\n" + text[-half_length:]

    @classmethod
    def _is_supported_image(cls, filename: str) -> bool:
        return filename.lower().endswith(cls.IMAGE_EXTENSIONS)

    @staticmethod
    def _merge_unique(existing: list[str], new_items: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()

        for item in [*existing, *new_items]:
            if not item or item in seen:
                continue
            seen.add(item)
            merged.append(item)
        return merged

    @staticmethod
    def _sanitize_filename_component(value: str) -> str:
        sanitized = (value or "").strip()
        sanitized = re.sub(r"[\\/]+", "_", sanitized)
        sanitized = sanitized.replace(" ", "_")
        sanitized = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", sanitized, flags=re.UNICODE)
        sanitized = re.sub(r"_+", "_", sanitized)
        return sanitized.strip("._-").lower()

    @classmethod
    def _is_generic_image_stem(cls, stem: str) -> bool:
        return not stem or cls.GENERIC_IMAGE_STEM_PATTERN.fullmatch(stem) is not None

    def _strip_unstable_prefixes(self, section_slug: str, stem: str) -> str:
        cleaned = stem
        if section_slug:
            cleaned = re.sub(
                rf"^{re.escape(section_slug)}[_-]*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
        cleaned = re.sub(
            r"^(?:step[_-]?\d+|step\d+|fig(?:ure)?[_-]?\d*|image[_-]?\d*|img[_-]?\d*|"
            r"plot[_-]?\d*|chart[_-]?\d*|graph[_-]?\d*|output[_-]?\d*|result[_-]?\d*|"
            r"tmp[_-]?\d*|temp[_-]?\d*)[_-]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned

    def _build_stable_image_name(
        self,
        section: str,
        original_name: str,
        index: int,
        occupied_names: set[str],
    ) -> str:
        original_base = os.path.basename(original_name)
        stem, ext = os.path.splitext(original_base)
        section_slug = self._sanitize_filename_component(section) or "section"
        semantic_stem = self._sanitize_filename_component(
            self._strip_unstable_prefixes(section_slug, stem)
        )

        if self._is_generic_image_stem(semantic_stem):
            semantic_stem = f"figure_{index:02d}"

        candidate_stem = f"{section_slug}_{semantic_stem}"
        ext = ext.lower() or ".png"
        candidate = f"{candidate_stem}{ext}"

        suffix = 2
        while candidate in occupied_names and candidate != original_base:
            candidate = f"{candidate_stem}_{suffix:02d}{ext}"
            suffix += 1
        return candidate

    def _list_local_images(self) -> list[str]:
        if not os.path.exists(self.work_dir):
            return []
        return sorted(
            file
            for file in os.listdir(self.work_dir)
            if self._is_supported_image(file)
            and os.path.isfile(os.path.join(self.work_dir, file))
        )

    def _normalize_local_created_images(
        self, section: str, image_names: list[str]
    ) -> list[str]:
        normalized: list[str] = []
        occupied_names = set(self._list_local_images())

        for index, image_name in enumerate(sorted(image_names), start=1):
            original_base = os.path.basename(image_name)
            original_path = os.path.join(self.work_dir, original_base)
            if not os.path.exists(original_path):
                continue

            stable_name = self._build_stable_image_name(
                section=section,
                original_name=original_base,
                index=index,
                occupied_names=occupied_names,
            )

            if stable_name != original_base:
                stable_path = os.path.join(self.work_dir, stable_name)
                os.replace(original_path, stable_path)
                logger.info(f"重命名图片: {original_base} -> {stable_name}")
                occupied_names.discard(original_base)
                occupied_names.add(stable_name)
                normalized.append(stable_name)
            else:
                normalized.append(original_base)

        return normalized

    def collect_created_images(self, section: str) -> list[str]:
        """收集当前 section 新增图片，并统一整理为稳定文件名。"""
        self.add_section(section)
        current_images = set(self._list_local_images())
        new_images = sorted(current_images - self.last_created_images)
        normalized_new_images = self._normalize_local_created_images(section, new_images)
        refreshed_images = set(self._list_local_images())

        existing_images = [
            image
            for image in self.section_output[section]["images"]
            if image in refreshed_images
        ]
        self.section_output[section]["images"] = self._merge_unique(
            existing_images,
            normalized_new_images,
        )
        self.last_created_images = refreshed_images

        logger.info(
            f"{section}-当前累计图片列表: {self.section_output[section]['images']}"
        )
        return list(self.section_output[section]["images"])

    def _font_list_literal(self) -> str:
        return repr(self.CHINESE_FONT_FAMILIES)

    def get_matplotlib_font_setup_code(self) -> str:
        return (
            "import matplotlib as mpl\n"
            "import matplotlib.pyplot as plt\n"
            f"_codex_cn_fonts = {self._font_list_literal()}\n"
            "mpl.rcParams['font.family'] = ['sans-serif']\n"
            "mpl.rcParams['font.sans-serif'] = _codex_cn_fonts\n"
            "plt.rcParams['font.family'] = ['sans-serif']\n"
            "plt.rcParams['font.sans-serif'] = _codex_cn_fonts\n"
            "mpl.rcParams['axes.unicode_minus'] = False\n"
            "plt.rcParams['axes.unicode_minus'] = False\n"
        )

    def _normalize_matplotlib_font_config(self, code: str) -> str:
        normalized = code
        normalized = re.sub(
            r"(['\"])font\.family\1\s*:\s*(['\"])Arial\2",
            r"\1font.family\1: 'sans-serif'",
            normalized,
        )
        normalized = re.sub(
            r"plt\.rcParams\[\s*(['\"])font\.family\1\s*\]\s*=\s*(['\"])Arial\2",
            "plt.rcParams['font.family'] = ['sans-serif']",
            normalized,
        )
        normalized = re.sub(
            r"mpl\.rcParams\[\s*(['\"])font\.family\1\s*\]\s*=\s*(['\"])Arial\2",
            "mpl.rcParams['font.family'] = ['sans-serif']",
            normalized,
        )
        normalized = re.sub(
            r"font\.sans-serif'\s*:\s*\[[^\]]*\]",
            f"'font.sans-serif': {self._font_list_literal()}",
            normalized,
        )
        normalized = re.sub(
            r'font\.sans-serif"\s*:\s*\[[^\]]*\]',
            f'"font.sans-serif": {self._font_list_literal()}',
            normalized,
        )
        normalized = re.sub(
            r"plt\.rcParams\[\s*(['\"])font\.sans-serif\1\s*\]\s*=\s*\[[^\]]*\]",
            f"plt.rcParams['font.sans-serif'] = {self._font_list_literal()}",
            normalized,
        )
        normalized = re.sub(
            r"mpl\.rcParams\[\s*(['\"])font\.sans-serif\1\s*\]\s*=\s*\[[^\]]*\]",
            f"mpl.rcParams['font.sans-serif'] = {self._font_list_literal()}",
            normalized,
        )
        return normalized

    def prepare_code_for_execution(self, code: str) -> str:
        normalized = self._normalize_matplotlib_font_config(code)
        prelude = (
            "# Auto-injected matplotlib Chinese font setup\n"
            + self.get_matplotlib_font_setup_code()
            + "\n"
        )
        return prelude + normalized
