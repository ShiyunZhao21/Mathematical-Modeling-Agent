from app.models.user_output import UserOutput
from app.tools.base_interpreter import BaseCodeInterpreter
from app.schemas.A2A import (
    GeneratedFigure,
    ModelerToCoder,
    ProblemAnalysis,
    ProblemDigest,
    QuestionModelPlan,
)


class Flows:
    def __init__(self, questions: dict[str, str | int]):
        self.flows: dict[str, dict] = {}
        self.questions: dict[str, str | int] = questions

    def set_flows(self, ques_count: int):
        ques_str = [f"ques{i}" for i in range(1, ques_count + 1)]
        seq = [
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
        self.flows = {key: {} for key in seq}

    def get_solution_flows(
        self, questions: dict[str, str | int], modeler_response: ModelerToCoder
    ):
        questions_quesx = {
            key: value
            for key, value in questions.items()
            if key.startswith("ques") and key != "ques_count"
        }
        solutions = modeler_response.questions_solution
        ques_flow = {
            key: {
                "coder_prompt": modeler_response.question_model_docs.get(key, QuestionModelPlan(question_key=key)).coder_prompt
                or f"参考建模手给出的解决方案{solutions.get(key, '')}\n完成如下问题{value}",
                "required_figures": modeler_response.question_model_docs.get(
                    key, QuestionModelPlan(question_key=key)
                ).required_figures,
            }
            for key, value in questions_quesx.items()
        }
        flows = {
            "eda": {
                "coder_prompt": f"参考建模手给出的解决方案{solutions.get('eda', '对数据进行探索性分析')}\n对当前目录下数据进行EDA分析(数据清洗,可视化),清洗后的数据保存当前目录下,**不需要复杂的模型**",
                "required_figures": [],
            },
            **ques_flow,
            "sensitivity_analysis": {
                "coder_prompt": f"参考建模手给出的解决方案{solutions.get('sensitivity_analysis', '对模型进行灵敏度分析')}\n完成敏感性分析",
                "required_figures": [],
            },
        }
        return flows

    def get_write_flows(
        self,
        user_output: UserOutput,
        config_template: dict,
        problem_digest: ProblemDigest | None = None,
        problem_analysis: ProblemAnalysis | None = None,
        conclusion_memory_markdown: str = "",
        bg_ques_all: str = "",
    ):
        model_build_solve = user_output.get_model_build_solve()
        background = bg_ques_all or (problem_digest.background if problem_digest else "")
        overall_analysis = problem_analysis.overall_analysis if problem_analysis else ""
        shared_context = f"问题背景{background}；整体分析{overall_analysis}；结论库{conclusion_memory_markdown}；模型求解信息{model_build_solve}"
        flows = {
            "firstPage": f"{shared_context}，不需要编写代码，按照如下模板撰写：{config_template['firstPage']}，撰写标题，摘要，关键词",
            "RepeatQues": f"{shared_context}，不需要编写代码，按照如下模板撰写：{config_template['RepeatQues']}，撰写问题重述",
            "analysisQues": f"{shared_context}，不需要编写代码，按照如下模板撰写：{config_template['analysisQues']}，撰写问题分析，突出问题间衔接",
            "modelAssumption": f"{shared_context}，不需要编写代码，按照如下模板撰写：{config_template['modelAssumption']}，撰写模型假设与参数估计依据",
            "symbol": f"{shared_context}，不需要编写代码，按照如下模板撰写：{config_template['symbol']}，撰写符号说明部分",
            "judge": f"{shared_context}，不需要编写代码，按照如下模板撰写：{config_template['judge']}，撰写模型的评价部分",
        }
        return flows

    def _build_writer_asset_contract(
        self,
        key: str,
        available_images: list[str] | None,
        generated_figures: list[GeneratedFigure] | list[dict] | None,
    ) -> str:
        normalized_images = []
        seen_images: set[str] = set()
        for image in available_images or []:
            normalized = (image or "").strip()
            if normalized and normalized not in seen_images:
                seen_images.add(normalized)
                normalized_images.append(normalized)

        generated_lines: list[str] = []
        required_images: list[str] = []
        for figure in generated_figures or []:
            figure_obj = figure if isinstance(figure, GeneratedFigure) else GeneratedFigure(**figure)
            if not figure_obj.generated:
                continue
            if figure_obj.filename and figure_obj.filename not in seen_images:
                seen_images.add(figure_obj.filename)
                normalized_images.append(figure_obj.filename)
            if figure_obj.required and figure_obj.filename and figure_obj.filename not in required_images:
                required_images.append(figure_obj.filename)
            details = [f"文件名={figure_obj.filename}"]
            if figure_obj.required:
                details.append("要求=必须插入")
            else:
                details.append("要求=可选插入")
            if figure_obj.purpose:
                details.append(f"用途={figure_obj.purpose}")
            if figure_obj.section_hint:
                details.append(f"建议位置={figure_obj.section_hint}")
            if figure_obj.caption_hint:
                details.append(f"图注提示={figure_obj.caption_hint}")
            generated_lines.append("- " + "；".join(details))

        contract_lines = [
            "【章节资产合同】",
            f"当前章节={key}",
            f"唯一允许引用的图片文件名={', '.join(normalized_images) if normalized_images else '无'}",
            f"必须插入的图片文件名={', '.join(required_images) if required_images else '无'}",
            "规则：文件名只认本合同，不认上下文中的描述性图名；若 prose 与本合同冲突，永远以本合同为准。",
            "规则：禁止把“热力图”“残差图”“ROC曲线”等语义描述改写成新的文件名。",
            "规则：禁止引用未出现在本合同中的其他图片，也禁止编造 fig1.png、image.png、chart.png 等名字。",
        ]
        if generated_lines:
            contract_lines.append("结构化交付清单：")
            contract_lines.extend(generated_lines)
        return "\n".join(contract_lines)

    def get_writer_prompt(
        self,
        key: str,
        coder_response: str,
        code_interpreter: BaseCodeInterpreter,
        config_template: dict,
        question_plan: QuestionModelPlan | None = None,
        problem_digest: ProblemDigest | None = None,
        problem_analysis: ProblemAnalysis | None = None,
        conclusion_memory_markdown: str = "",
        available_images: list[str] | None = None,
        generated_figures: list[GeneratedFigure] | list[dict] | None = None,
    ) -> str:
        code_output = code_interpreter.get_code_output(key)
        bgc = problem_digest.background if problem_digest else self.questions.get("background", "")
        overall_analysis = problem_analysis.overall_analysis if problem_analysis else ""
        question_context = question_plan.writer_context if question_plan else ""
        asset_contract = self._build_writer_asset_contract(
            key=key,
            available_images=available_images,
            generated_figures=generated_figures,
        )

        if key == "eda":
            return (
                f"{asset_contract}\n\n问题背景{bgc}，不需要编写代码，"
                f"代码手得到的结果{coder_response},{code_output}，"
                f"按照如下模板撰写：{config_template['eda']}"
            )

        if key == "sensitivity_analysis":
            return (
                f"{asset_contract}\n\n问题背景{bgc}，不需要编写代码，"
                f"代码手得到的结果{coder_response},{code_output}，"
                f"按照如下模板撰写：{config_template['sensitivity_analysis']}"
            )

        questions_quesx_keys = self.get_questions_quesx_keys()
        if key in questions_quesx_keys:
            return (
                f"{asset_contract}\n\n问题背景{bgc}；整体分析{overall_analysis}；"
                f"结论库{conclusion_memory_markdown}；当前问题文档3上下文{question_context}；"
                f"代码手得到的结果{coder_response},{code_output}，按照如下模板撰写：{config_template[key]}"
            )

        raise ValueError(f"未知的任务类型: {key}")

    def get_questions_quesx_keys(self) -> list[str]:
        return list(self.get_questions_quesx().keys())

    def get_questions_quesx(self) -> dict[str, str]:
        return {
            key: value
            for key, value in self.questions.items()
            if key.startswith("ques") and key != "ques_count"
        }

    def get_seq(self, ques_count: int) -> dict[str, str]:
        ques_str = [f"ques{i}" for i in range(1, ques_count + 1)]
        seq = [
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
        return {key: "" for key in seq}
