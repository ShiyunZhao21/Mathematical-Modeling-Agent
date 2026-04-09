import os
import re
import json
import uuid

from app.utils.data_recorder import DataRecorder
from app.schemas.A2A import WriterResponse


class UserOutput:
    def __init__(
        self, work_dir: str, ques_count: int, data_recorder: DataRecorder | None = None
    ):
        self.work_dir = work_dir
        self.res: dict[str, dict] = {}
        self.latex_sections: dict[str, str] = {}
        self.data_recorder = data_recorder
        self.cost_time = 0.0
        self.initialized = True
        self.ques_count: int = ques_count
        self.footnotes = {}
        self._init_seq()

    def _init_seq(self):
        ques_str = [f"ques{i}" for i in range(1, self.ques_count + 1)]
        self.seq = [
            "firstPage",
            "RepeatQues",
            "analysisQues",
            "modelAssumption",
            "symbol",
            "eda",
            *ques_str,
            "sensitivity_analysis",
            "judge",
        ]

    def set_res(self, key: str, writer_response: WriterResponse):
        self.res[key] = {
            "response_content": writer_response.response_content,
            "footnotes": writer_response.footnotes,
        }

    def set_latex_section(self, key: str, content: str) -> None:
        self.latex_sections[key] = content

    def get_res(self):
        return self.res

    def get_model_build_solve(self) -> str:
        return ",".join(
            f"{key}-{value}"
            for key, value in self.res.items()
            if key.startswith("ques") and key != "ques_count"
        )

    def replace_references_with_uuid(self, text: str) -> str:
        references = re.findall(r"\{\[\^(\d+)\]:\s*(.*?)\}", text, re.DOTALL)

        for ref_num, ref_content in references:
            ref_content = ref_content.strip().rstrip(".")

            existing_uuid = None
            for uuid_key, footnote_data in self.footnotes.items():
                if footnote_data["content"] == ref_content:
                    existing_uuid = uuid_key
                    break

            if existing_uuid:
                text = re.sub(
                    rf"\{{\[\^{ref_num}\]:.*?\}}",
                    f"[{existing_uuid}]",
                    text,
                    flags=re.DOTALL,
                )
            else:
                new_uuid = str(uuid.uuid4())
                self.footnotes[new_uuid] = {"content": ref_content}
                text = re.sub(
                    rf"\{{\[\^{ref_num}\]:.*?\}}",
                    f"[{new_uuid}]",
                    text,
                    flags=re.DOTALL,
                )

        return text

    def sort_text_with_footnotes(self, replace_res: dict) -> dict:
        sort_res = {}
        ref_index = 1

        for seq_key in self.seq:
            text = replace_res.get(seq_key, {}).get("response_content", "")
            uuid_list = re.findall(r"\[([a-f0-9-]{36})\]", text)
            for uid in uuid_list:
                text = text.replace(f"[{uid}]", f"[^{ref_index}]")
                if self.footnotes[uid].get("number") is None:
                    self.footnotes[uid]["number"] = ref_index
                ref_index += 1
            sort_res[seq_key] = {"response_content": text}

        return sort_res

    def append_footnotes_to_text(self, text: str) -> str:
        text += "\n\n ## 参考文献"
        sorted_footnotes = sorted(self.footnotes.items(), key=lambda x: x[1]["number"])
        for _, footnote in sorted_footnotes:
            text += f"\n\n[^{footnote['number']}]: {footnote['content']}"
        return text

    def get_result_to_save(self) -> str:
        replace_res = {}

        for key, value in self.res.items():
            new_text = self.replace_references_with_uuid(value["response_content"])
            replace_res[key] = {"response_content": new_text}

        sort_res = self.sort_text_with_footnotes(replace_res)
        full_res_1 = "\n\n".join(
            [sort_res.get(key, {"response_content": ""})["response_content"] for key in self.seq]
        )
        return self.append_footnotes_to_text(full_res_1)

    def save_result(self):
        with open(os.path.join(self.work_dir, "paper.json"), "w", encoding="utf-8") as f:
            json.dump(self.res, f, ensure_ascii=False, indent=4)

        with open(os.path.join(self.work_dir, "paper.md"), "w", encoding="utf-8") as f:
            f.write(self.get_result_to_save())

    def build_paper_tex(self) -> str:
        ordered_sections = [
            self.latex_sections.get(key, "") for key in self.seq if self.latex_sections.get(key)
        ]
        return "\n\n".join(ordered_sections)
